
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import json
from datetime import datetime, timezone

from sklearn.pipeline import Pipeline


class LightGBMShapInterpreter:
    """
    Interpret SHAP values for a scikit-learn Pipeline that contains a
    preprocessing step and a LightGBM-like tree model.

    This helper supports:
    - global feature importance
    - SHAP summary plots
    - SHAP dependence plots for threshold analysis
    - identification of records where the hybrid model corrected
      large baseline errors
    - SHAP force plots for critical corrected cases
    """

    def __init__(self, pipeline: Pipeline):
        """
        Initialize the interpreter.

        Parameters
        ----------
        pipeline : Pipeline
            A fitted scikit-learn Pipeline with 'preprocessor' and 'model' steps.
        """
        if pipeline is None:
            raise ValueError("The pipeline cannot be None.")

        if not isinstance(pipeline, Pipeline):
            raise ValueError("Expected a scikit-learn Pipeline.")

        if "preprocessor" not in pipeline.named_steps:
            raise ValueError("Pipeline must contain a 'preprocessor' step.")

        if "model" not in pipeline.named_steps:
            raise ValueError("Pipeline must contain a 'model' step.")

        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.model = pipeline.named_steps["model"]
        self.explainer = None

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the pipeline preprocessor and return a DataFrame with feature names.

        Parameters
        ----------
        df : pd.DataFrame
            Input features before preprocessing.

        Returns
        -------
        pd.DataFrame
            Transformed features with the same index as the input DataFrame.
        """
        X_transformed = self.preprocessor.transform(df)

        try:
            feature_names = self.preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        return pd.DataFrame(X_transformed, columns=feature_names, index=df.index)

    def create_explainer(self):
        """
        Create the SHAP TreeExplainer for the model.

        Returns
        -------
        shap.TreeExplainer
            SHAP explainer instance.
        """
        self.explainer = shap.TreeExplainer(self.model)
        return self.explainer

    def compute_shap_values(self, df: pd.DataFrame):
        """
        Compute SHAP values for the given dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input features before preprocessing.

        Returns
        -------
        tuple
            (shap_values, transformed_features_dataframe)
        """
        if self.explainer is None:
            self.create_explainer()

        X_transformed_df = self.transform_features(df)
        shap_values = self.explainer.shap_values(X_transformed_df)

        return shap_values, X_transformed_df

    def _get_shap_array(self, shap_values) -> np.ndarray:
        """
        Normalize SHAP output into a single numpy array.

        Parameters
        ----------
        shap_values : Any
            SHAP values returned by the explainer.

        Returns
        -------
        np.ndarray
            SHAP values as a 2D array for regression or the first class when
            a list is returned.
        """
        if isinstance(shap_values, list):
            return np.array(shap_values[0])
        return np.array(shap_values)

    def _get_expected_value(self):
        """
        Return the SHAP expected value in a scalar form when possible.

        Returns
        -------
        float or Any
            Expected value used by local explanation plots.
        """
        if self.explainer is None:
            self.create_explainer()

        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = np.array(expected_value).flatten()[0]
        return expected_value

    def _resolve_feature_name(self, columns, feature_name: str) -> str:
        """
        Resolve a user-provided feature name to an existing transformed column.

        This is useful when the transformed feature is named like
        'numeric__anomaly_score' but the user passes 'anomaly_score'.

        Parameters
        ----------
        columns : iterable
            Available transformed column names.
        feature_name : str
            Requested feature name.

        Returns
        -------
        str
            The resolved column name.

        Raises
        ------
        ValueError
            If the feature cannot be found.
        """
        if feature_name in columns:
            return feature_name

        matches = [col for col in columns if col.endswith(f"__{feature_name}")]
        if len(matches) == 1:
            return matches[0]

        contains_matches = [col for col in columns if feature_name in col]
        if len(contains_matches) == 1:
            return contains_matches[0]

        raise ValueError(
            f"Feature '{feature_name}' was not found. "
            f"Available examples: {list(columns)[:10]}"
        )

    def plot_summary(self, df: pd.DataFrame, plot_type: str = "dot") -> None:
        """
        Plot a SHAP summary chart.

        Parameters
        ----------
        df : pd.DataFrame
            Input features before preprocessing.
        plot_type : str, default='dot'
            SHAP summary plot type, such as 'dot' or 'bar'.
        """
        shap_values, X_transformed_df = self.compute_shap_values(df)
        shap.summary_plot(self._get_shap_array(shap_values), X_transformed_df, plot_type=plot_type)

    def get_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean absolute SHAP importance for all transformed features.

        Parameters
        ----------
        df : pd.DataFrame
            Input features before preprocessing.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'feature' and 'mean_abs_shap' columns sorted by importance.
        """
        shap_values, X_transformed_df = self.compute_shap_values(df)
        shap_array = np.abs(self._get_shap_array(shap_values))

        importance = pd.DataFrame({
            "feature": X_transformed_df.columns,
            "mean_abs_shap": shap_array.mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)

        return importance.reset_index(drop=True)

    def plot_dependence(
        self,
        df: pd.DataFrame,
        feature_name: str = "anomaly_score",
        interaction_index="auto"
    ) -> None:
        """
        Plot a SHAP dependence plot for a selected feature.

        This plot is useful for threshold analysis because it can show
        where the effect of a feature starts to change more strongly.

        Parameters
        ----------
        df : pd.DataFrame
            Input features before preprocessing.
        feature_name : str, default='anomaly_score'
            Feature to analyze. The method tries to resolve both raw names
            and transformed names such as 'numeric__anomaly_score'.
        interaction_index : str or int, default='auto'
            Interaction feature passed to shap.dependence_plot.
        """
        shap_values, X_transformed_df = self.compute_shap_values(df)
        shap_array = self._get_shap_array(shap_values)
        resolved_feature = self._resolve_feature_name(X_transformed_df.columns, feature_name)

        shap.dependence_plot(
            resolved_feature,
            shap_array,
            X_transformed_df,
            interaction_index=interaction_index
        )

    def find_critical_corrections(
        self,
        df: pd.DataFrame,
        y_true_col: str,
        baseline_pred_col: str,
        hybrid_pred_col: str,
        top_n: int = 3
    ) -> pd.DataFrame:
        """
        Identify records where the hybrid model corrected the largest
        baseline errors.

        The ranking is based on:
        1. large baseline absolute error
        2. positive improvement from baseline to hybrid model

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ground truth and both model predictions.
        y_true_col : str
            Column with the true target values.
        baseline_pred_col : str
            Column with baseline model predictions.
        hybrid_pred_col : str
            Column with hybrid model predictions.
        top_n : int, default=3
            Number of critical corrected cases to return.

        Returns
        -------
        pd.DataFrame
            Top corrected cases with baseline and hybrid errors.
        """
        required_cols = [y_true_col, baseline_pred_col, hybrid_pred_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        analysis_df = df.copy()
        analysis_df["baseline_abs_error"] = np.abs(
            analysis_df[y_true_col] - analysis_df[baseline_pred_col]
        )
        analysis_df["hybrid_abs_error"] = np.abs(
            analysis_df[y_true_col] - analysis_df[hybrid_pred_col]
        )
        analysis_df["error_reduction"] = (
            analysis_df["baseline_abs_error"] - analysis_df["hybrid_abs_error"]
        )

        corrected = analysis_df[analysis_df["error_reduction"] > 0].copy()
        corrected = corrected.sort_values(
            by=["error_reduction", "baseline_abs_error"],
            ascending=False
        )

        return corrected.head(top_n)

    def plot_force_for_critical_cases(
        self,
        X_df: pd.DataFrame,
        analysis_df: pd.DataFrame,
        y_true_col: str,
        baseline_pred_col: str,
        hybrid_pred_col: str,
        top_n: int = 3,
        max_display: int = 5,
        matplotlib: bool = True
    ):
        """
        Generate SHAP force plots for the most relevant corrected cases.

        Parameters
        ----------
        X_df : pd.DataFrame
            Feature-only DataFrame used by the hybrid model.
        analysis_df : pd.DataFrame
            DataFrame with the same index as X_df and containing the true value,
            baseline prediction, and hybrid prediction columns.
        y_true_col : str
            Column with the true target values.
        baseline_pred_col : str
            Column with baseline model predictions.
        hybrid_pred_col : str
            Column with hybrid model predictions.
        top_n : int, default=3
            Number of corrected cases to explain.
        matplotlib : bool, default=True
            Whether to render matplotlib-based force plots.

        Returns
        -------
        list
            A list of SHAP force plot objects, one for each selected record.
        """
        shap_values, X_transformed_df = self.compute_shap_values(X_df)

        critical_cases = self.find_critical_corrections(
            df=analysis_df,
            y_true_col=y_true_col,
            baseline_pred_col=baseline_pred_col,
            hybrid_pred_col=hybrid_pred_col,
            top_n=top_n
        )

        # Se shap_values vier como lista, ajustar
        if isinstance(shap_values, list):
            shap_array = shap_values[0]
        else:
            shap_array = shap_values

        for row_index in critical_cases.index:
            row_pos = X_transformed_df.index.get_loc(row_index)

            shap_row = shap_array[row_pos]
            x_row = X_transformed_df.iloc[row_pos]

            # seleciona as top N features por valor absoluto de SHAP
            top_idx = np.argsort(np.abs(shap_row))[-max_display:]
            top_idx = top_idx[np.argsort(np.abs(shap_row[top_idx]))]  # ordena
            top_idx = top_idx[::-1]  # maior para menor

            shap_row_top = shap_row[top_idx]
            x_row_top = x_row.iloc[top_idx]

            print(f"\nCritical case index: {row_index}")
            print(f"Actual: {analysis_df.loc[row_index, y_true_col]:.4f}")
            print(f"Baseline prediction: {analysis_df.loc[row_index, baseline_pred_col]:.4f}")
            print(f"Hybrid prediction: {analysis_df.loc[row_index, hybrid_pred_col]:.4f}")

            shap.force_plot(
                self.explainer.expected_value,
                shap_row_top,
                x_row_top,
                matplotlib=matplotlib,
                show=True
            )

    def plot_waterfall_for_critical_cases(
            self,
            X_df: pd.DataFrame,
            analysis_df: pd.DataFrame,
            y_true_col: str,
            baseline_pred_col: str,
            hybrid_pred_col: str,
            top_n: int = 3,
            max_display: int = 5
        ):
            """
            Generate SHAP waterfall plots for the most relevant corrected cases.

            This method identifies the cases where the hybrid model reduced the
            absolute prediction error the most compared to the baseline model,
            and then generates one SHAP waterfall plot for each selected case.

            Parameters
            ----------
            X_df : pd.DataFrame
                Feature-only DataFrame used by the hybrid model.
            analysis_df : pd.DataFrame
                DataFrame with the same index as X_df and containing the true value,
                baseline prediction, and hybrid prediction columns.
            y_true_col : str
                Column with the true target values.
            baseline_pred_col : str
                Column with baseline model predictions.
            hybrid_pred_col : str
                Column with hybrid model predictions.
            top_n : int, default=3
                Number of corrected cases to explain.
            max_display : int, default=5
                Maximum number of features displayed in each waterfall plot.

            Returns
            -------
            list
                A list of dictionaries with the selected case metadata.
            """
            shap_values, X_transformed_df = self.compute_shap_values(X_df)

            critical_cases = self.find_critical_corrections(
                df=analysis_df,
                y_true_col=y_true_col,
                baseline_pred_col=baseline_pred_col,
                hybrid_pred_col=hybrid_pred_col,
                top_n=top_n
            )

            if isinstance(shap_values, list):
                shap_array = shap_values[0]
            else:
                shap_array = shap_values

            expected_value = self.explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = np.array(expected_value).reshape(-1)[0]

            results = []

            for row_index in critical_cases.index:
                row_pos = X_transformed_df.index.get_loc(row_index)

                shap_row = shap_array[row_pos]
                x_row = X_transformed_df.iloc[row_pos]

                print(f"\nCritical case index: {row_index}")
                print(f"Actual: {analysis_df.loc[row_index, y_true_col]:.4f}")
                print(f"Baseline prediction: {analysis_df.loc[row_index, baseline_pred_col]:.4f}")
                print(f"Hybrid prediction: {analysis_df.loc[row_index, hybrid_pred_col]:.4f}")
                print(f"Error reduction: {critical_cases.loc[row_index, 'error_reduction']:.4f}")

                exp = shap.Explanation(
                    values=shap_row,
                    base_values=expected_value,
                    data=x_row.values,
                    feature_names=X_transformed_df.columns.tolist()
                )

                shap.plots.waterfall(exp, max_display=max_display, show=True)

                results.append({
                    "row_index": row_index,
                    "actual": float(analysis_df.loc[row_index, y_true_col]),
                    "baseline_prediction": float(analysis_df.loc[row_index, baseline_pred_col]),
                    "hybrid_prediction": float(analysis_df.loc[row_index, hybrid_pred_col]),
                    "error_reduction": float(critical_cases.loc[row_index, "error_reduction"])
                })

            return results
    
    