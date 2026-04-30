import pandas as pd
import numpy as np
import shap

from sklearn.pipeline import Pipeline


class LightGBMShapInterpreter:

    def __init__(self, pipeline: Pipeline):
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

        X_transformed = self.preprocessor.transform(df)

        try:
            feature_names = self.preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        return pd.DataFrame(X_transformed, columns=feature_names, index=df.index)

    def create_explainer(self):
        self.explainer = shap.TreeExplainer(self.model)
        return self.explainer

    def compute_shap_values(self, df: pd.DataFrame):
        if self.explainer is None:
            self.create_explainer()

        X_transformed_df = self.transform_features(df)
        shap_values = self.explainer.shap_values(X_transformed_df)

        return shap_values, X_transformed_df

    def plot_summary(self, df: pd.DataFrame, plot_type: str = "dot") -> None:
        shap_values, X_transformed_df = self.compute_shap_values(df)
        shap.summary_plot(shap_values, X_transformed_df, plot_type=plot_type)

    def get_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        shap_values, X_transformed_df = self.compute_shap_values(df)

        if isinstance(shap_values, list):
            shap_array = np.abs(shap_values[0])
        else:
            shap_array = np.abs(shap_values)

        importance = pd.DataFrame({
            "feature": X_transformed_df.columns,
            "mean_abs_shap": shap_array.mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)

        return importance.reset_index(drop=True)