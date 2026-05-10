import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from src.config.config import MODELS_ANOMALYD


class IsolationForestAnalyzer:
    """
    Classe para detecção de anomalias em dados agregados de vendas usando Isolation Forest.

    Objetivo
    --------
    Detectar observações anômalas com base em sinais derivados do comportamento
    temporal e de negócio do dataframe.

    Features utilizadas no modelo
    -----------------------------
    - if_qty_signal:
        Desvio percentual da quantidade versus média móvel de 7 dias.
        Origem: quantity_pct_vs_mean_7d

    - if_sales_signal:
        Desvio do valor total versus média de 30 dias.
        Origem: total_price_vs_mean_30d

    - if_discount_signal:
        Comportamento médio recente do desconto.
        Origem: discount_mean_mean_14d

    - if_ticket_signal:
        Ticket médio da observação.
        Origem: avg_ticket
    """

    def __init__(self, base_path: str = None, random_state=42, cv=3, n_jobs=-1, verbose=0):
        self.random_state = random_state
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.feature_cols_ = [
            "if_qty_signal",
            "if_sales_signal",
            "if_discount_signal",
            "if_ticket_signal"
        ]

        self.pipeline_ = None
        self.grid_search_ = None
        self.best_estimator_ = None
        self.best_params_ = None
        self.fitted_ = False

        # Se base_path for None, é definida o caminho relativo ao arquivo atual
        if base_path is None:
            self.base_path = os.path.join(MODELS_ANOMALYD)
        else:
            self.base_path = base_path

    # =========================================================
    # VALIDAÇÃO E UTILITÁRIOS
    # =========================================================
    @staticmethod
    def _validate_required_columns(df: pd.DataFrame, required_cols: list):
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    def _check_is_fitted(self):
        if not self.fitted_ or self.best_estimator_ is None:
            raise ValueError("O modelo ainda não foi treinado. Execute fit(df) antes.")

    @staticmethod
    def _normalize_filter_values(values):
        """
        Normaliza filtros para lista.
        Aceita:
        - None
        - valor único: "Water"
        - lista: ["Water", "Juices"]
        """
        if values is None:
            return None
        if isinstance(values, (str, int, float)):
            return [values]
        return list(values)
    
    @staticmethod
    def _get_full_path(self, relative_path: str) -> Path:
        """
        Constrói o caminho completo do arquivo.

        Parameters
        ----------
        relative_path : str
            Caminho relativo dentro do base_path.

        Returns
        -------
        Path
            Caminho completo do arquivo.
        """
        return self.base_path / relative_path
    
    def set_base_path (self, base_path: str = "data"):
        """
        redefine a pasta onde os arquivos serão armazenados.

        Parameters
        ----------
        base_path : str
            Diretório base onde os arquivos serão armazenados.
        """
        self.base_path = Path(base_path)

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        product_filter=None,
        region_filter=None
    ) -> pd.DataFrame:
        """
        Filtra o DataFrame por Product e/ou Region.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada.

        product_filter : None, str, list, default=None
            Produto único ou lista de produtos.

        region_filter : None, str, list, default=None
            Região única ou lista de regiões.

        Returns
        -------
        pd.DataFrame
            DataFrame filtrado.
        """
        df_out = df.copy()

        if product_filter is not None:
            self._validate_required_columns(df_out, ["Product"])
            product_filter = self._normalize_filter_values(product_filter)
            df_out = df_out[df_out["Product"].isin(product_filter)]

        if region_filter is not None:
            self._validate_required_columns(df_out, ["Region"])
            region_filter = self._normalize_filter_values(region_filter)
            df_out = df_out[df_out["Region"].isin(region_filter)]

        return df_out.copy()

    # =========================================================
    # FEATURES DO MODELO
    # =========================================================
    def create_if_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = [
            "quantity_pct_vs_mean_7d",
            "total_price_vs_mean_30d",
            "discount_mean_mean_14d",
            "avg_ticket"
        ]
        self._validate_required_columns(df, required_cols)

        df_out = df.copy()

        df_out["if_qty_signal"] = (
            pd.to_numeric(df_out["quantity_pct_vs_mean_7d"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .clip(-5, 5)
        )

        df_out["if_sales_signal"] = (
            pd.to_numeric(df_out["total_price_vs_mean_30d"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .clip(-5, 5)
        )

        df_out["if_discount_signal"] = (
            pd.to_numeric(df_out["discount_mean_mean_14d"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .clip(-1, 1)
        )

        df_out["if_ticket_signal"] = (
            pd.to_numeric(df_out["avg_ticket"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=0)
        )

        return df_out

    # =========================================================
    # SCORER CUSTOMIZADO
    # =========================================================
    @staticmethod
    def _unsupervised_extreme_separation_scorer(estimator, X, y=None):
        scores = estimator.decision_function(X)

        if len(scores) < 10:
            return 0.0

        cutoff = np.quantile(scores, 0.10)
        extreme_mask = scores <= cutoff
        normal_mask = scores > cutoff

        if extreme_mask.sum() == 0 or normal_mask.sum() == 0:
            return 0.0

        extreme_mean = scores[extreme_mask].mean()
        normal_mean = scores[normal_mask].mean()

        return float(normal_mean - extreme_mean)

    # =========================================================
    # PIPELINE
    # =========================================================
    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "model",
                    IsolationForest(
                        random_state=self.random_state,
                        bootstrap=False
                    )
                )
            ]
        )

    def _build_param_grid(self) -> dict:
        return {
            "model__n_estimators": [100, 200, 300],
            "model__max_samples": [512, 1024, 2048],
            "model__contamination": [0.01, 0.03, 0.05, 0.10],
            "model__max_features": [0.75, 1.0]
        }

    # =========================================================
    # TREINO E PREDIÇÃO
    # =========================================================
    def fit(self, df: pd.DataFrame):
        df_model = self.create_if_features(df)
        X = df_model[self.feature_cols_].copy()

        self.pipeline_ = self._build_pipeline()

        self.grid_search_ = GridSearchCV(
            estimator=self.pipeline_,
            param_grid=self._build_param_grid(),
            scoring=self._unsupervised_extreme_separation_scorer,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            refit=True
        )

        y_dummy = np.zeros(len(X))
        self.grid_search_.fit(X, y_dummy)

        self.best_estimator_ = self.grid_search_.best_estimator_
        self.best_params_ = self.grid_search_.best_params_
        self.fitted_ = True

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()

        df_model = self.create_if_features(df)
        X = df_model[self.feature_cols_].copy()

        raw_pred = self.best_estimator_.predict(X)
        decision_score = self.best_estimator_.decision_function(X)

        df_out = df_model.copy()
        df_out["anomaly_flag"] = np.where(raw_pred == -1, 1, 0)
        df_out["anomaly_label"] = np.where(df_out["anomaly_flag"] == 1, "Anomaly", "Normal")
        df_out["anomaly_score"] = decision_score

        return df_out

    def get_best_params(self) -> dict:
        self._check_is_fitted()
        return self.best_params_

    # =========================================================
    # TOP N ANOMALIAS COM FILTRO
    # =========================================================
    def top_n_anomalies(
        self,
        df_pred: pd.DataFrame,
        n: int = 10,
        product_filter=None,
        region_filter=None,
        columns_to_show: list = None
    ) -> pd.DataFrame:
        """
        Retorna as N anomalias mais severas, com filtro opcional por produto/região.
        """
        required_cols = ["anomaly_flag", "anomaly_score"]
        self._validate_required_columns(df_pred, required_cols)

        filtered_df = self.filter_dataframe(
            df_pred,
            product_filter=product_filter,
            region_filter=region_filter
        )

        anomaly_df = filtered_df[filtered_df["anomaly_flag"] == 1].copy()
        anomaly_df = anomaly_df.sort_values("anomaly_score", ascending=True)

        if columns_to_show is None:
            default_cols = [
                "Order_Date",
                "Product",
                "Region",
                "total_price_sum",
                "quantity_sum",
                "avg_ticket",
                "if_qty_signal",
                "if_sales_signal",
                "if_discount_signal",
                "if_ticket_signal",
                "anomaly_score"
            ]
            columns_to_show = [col for col in default_cols if col in anomaly_df.columns]

        return anomaly_df[columns_to_show].head(n)

    # =========================================================
    # PLOT SIMPLES COM FILTRO
    # =========================================================
    def plot_anomalies_over_time(
        self,
        df_pred: pd.DataFrame,
        date_col: str = "Order_Date",
        value_col: str = "total_price_sum",
        title: str = "Anomalies Over Time",
        figsize: tuple = (14, 6),
        product_filter=None,
        region_filter=None
    ):
        """
        Plota a série temporal destacando anomalias, com filtro opcional por Product/Region.
        """
        required_cols = [date_col, value_col, "anomaly_flag"]
        self._validate_required_columns(df_pred, required_cols)

        plot_df = self.filter_dataframe(
            df_pred,
            product_filter=product_filter,
            region_filter=region_filter
        ).copy()

        if plot_df.empty:
            raise ValueError("Nenhum dado encontrado após aplicar os filtros.")

        plot_df = plot_df.sort_values(date_col)

        normal_df = plot_df[plot_df["anomaly_flag"] == 0]
        anomaly_df = plot_df[plot_df["anomaly_flag"] == 1]

        plt.figure(figsize=figsize)
        plt.plot(normal_df[date_col], normal_df[value_col], label="Normal", alpha=0.8)
        plt.scatter(
            anomaly_df[date_col],
            anomaly_df[value_col],
            label="Anomaly",
            s=60,
            color="orange",
            edgecolors="black",
            linewidths=0.8
        )
        plt.title(title)
        plt.xlabel(date_col)
        plt.ylabel(value_col)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # =========================================================
    # SUBPLOTS POR PRODUTO OU REGIÃO
    # =========================================================
    def plot_anomalies_subplots(
        self,
        df_pred: pd.DataFrame,
        group_by: str = "Product",
        date_col: str = "Order_Date",
        value_col: str = "total_price_sum",
        product_filter=None,
        region_filter=None,
        max_groups: int = 6,
        figsize_per_plot: tuple = (14, 4)
    ):
        """
        Plota subplots de anomalias ao longo do tempo por Product ou Region.

        Parameters
        ----------
        df_pred : pd.DataFrame
            DataFrame já processado por `predict`.

        group_by : str, default="Product"
            Agrupador do subplot.
            Valores esperados: "Product" ou "Region".

        date_col : str, default="Order_Date"
            Coluna de data.

        value_col : str, default="total_price_sum"
            Métrica do eixo Y.

        product_filter : None, str, list, default=None
            Filtro opcional por produto.

        region_filter : None, str, list, default=None
            Filtro opcional por região.

        max_groups : int, default=6
            Número máximo de grupos a exibir.

        figsize_per_plot : tuple, default=(14, 4)
            Tamanho base por subplot.

        Returns
        -------
        None
            Exibe os gráficos.
        """
        if group_by not in ["Product", "Region"]:
            raise ValueError("group_by deve ser 'Product' ou 'Region'.")

        required_cols = [group_by, date_col, value_col, "anomaly_flag"]
        self._validate_required_columns(df_pred, required_cols)

        plot_df = self.filter_dataframe(
            df_pred,
            product_filter=product_filter,
            region_filter=region_filter
        ).copy()

        if plot_df.empty:
            raise ValueError("Nenhum dado encontrado após aplicar os filtros.")

        plot_df = plot_df.sort_values(date_col)

        groups = (
            plot_df[group_by]
            .dropna()
            .astype(str)
            .value_counts()
            .index
            .tolist()
        )[:max_groups]

        if len(groups) == 0:
            raise ValueError(f"Não há grupos válidos na coluna {group_by}.")

        n_groups = len(groups)
        fig, axes = plt.subplots(
            n_groups,
            1,
            figsize=(figsize_per_plot[0], figsize_per_plot[1] * n_groups),
            squeeze=False
        )

        axes = axes.flatten()

        for ax, group_value in zip(axes, groups):
            temp_df = plot_df[plot_df[group_by].astype(str) == str(group_value)].copy()
            temp_df = temp_df.sort_values(date_col)

            normal_df = temp_df[temp_df["anomaly_flag"] == 0]
            anomaly_df = temp_df[temp_df["anomaly_flag"] == 1]

            ax.plot(normal_df[date_col], normal_df[value_col], alpha=0.8, label="Normal")
            ax.scatter(
                anomaly_df[date_col],
                anomaly_df[value_col],
                s=50,
                label="Anomaly",
                color="orange",
                edgecolors="black",
                linewidths=0.8
            )

            ax.set_title(f"{group_by}: {group_value}")
            ax.set_xlabel(date_col)
            ax.set_ylabel(value_col)
            ax.tick_params(axis="x", rotation=45)
            ax.legend()

        plt.tight_layout()
        plt.show()


    # =========================================================
    # XAI / INTERPRETABILIDADE DO ISOLATION FOREST
    # =========================================================
    def transform_if_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the Isolation Forest feature matrix and apply the fitted preprocessing
        steps (imputer and scaler), returning a DataFrame with aligned feature names.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe containing the source business columns.

        Returns
        -------
        pd.DataFrame
            Transformed feature matrix used by the fitted Isolation Forest model.
        """
        self._check_is_fitted()

        df_model = self.create_if_features(df)
        X_raw = df_model[self.feature_cols_].copy()

        imputer = self.best_estimator_.named_steps["imputer"]
        scaler = self.best_estimator_.named_steps["scaler"]

        X_imputed = imputer.transform(X_raw)
        X_scaled = scaler.transform(X_imputed)

        return pd.DataFrame(X_scaled, columns=self.feature_cols_, index=df.index)

    def create_shap_explainer(self):
        """
        Create and return a SHAP TreeExplainer for the fitted Isolation Forest model.

        Returns
        -------
        shap.TreeExplainer
            SHAP explainer bound to the fitted tree model.
        """
        self._check_is_fitted()
        model = self.best_estimator_.named_steps["model"]
        return shap.TreeExplainer(model)

    def compute_shap_values(self, df: pd.DataFrame):
        """
        Compute SHAP values for the fitted Isolation Forest using the transformed
        anomaly feature matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe.

        Returns
        -------
        tuple
            (shap_values, X_transformed_df)
        """
        X_transformed_df = self.transform_if_features(df)
        explainer = self.create_shap_explainer()
        shap_values = explainer.shap_values(X_transformed_df)

        return shap_values, X_transformed_df

    @staticmethod
    def _normalize_shap_array(shap_values):
        """
        Normalize SHAP output to a 2D numpy array.
        """
        if isinstance(shap_values, list):
            shap_array = np.asarray(shap_values[0])
        else:
            shap_array = np.asarray(shap_values)

        if shap_array.ndim == 1:
            shap_array = shap_array.reshape(1, -1)

        return shap_array

    def plot_shap_summary(self, df: pd.DataFrame, plot_type: str = "dot") -> None:
        """
        Plot a SHAP summary chart for the Isolation Forest anomaly signals.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe.
        plot_type : str, default="dot"
            SHAP summary plot type ("dot", "bar", etc.).
        """
        shap_values, X_transformed_df = self.compute_shap_values(df)
        shap.summary_plot(shap_values, X_transformed_df, plot_type=plot_type)

    def plot_shap_dependence(
        self,
        df: pd.DataFrame,
        feature: str = "if_sales_signal",
        interaction_index="auto"
    ) -> None:
        """
        Plot a SHAP dependence plot for one anomaly signal.

        This is useful to inspect threshold-like behavior, such as whether
        very high sales deviation or quantity deviation starts pushing the
        anomaly explanation more strongly.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe.
        feature : str, default="if_sales_signal"
            Feature name used in the dependence plot.
        interaction_index : str or int, default="auto"
            SHAP interaction setting.
        """
        shap_values, X_transformed_df = self.compute_shap_values(df)

        if feature not in X_transformed_df.columns:
            raise ValueError(f"Feature '{feature}' not found. Available: {list(X_transformed_df.columns)}")

        shap.dependence_plot(
            feature,
            shap_values,
            X_transformed_df,
            interaction_index=interaction_index
        )

    def get_anomaly_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return global SHAP importance for the Isolation Forest anomaly signals.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe.

        Returns
        -------
        pd.DataFrame
            Feature importance sorted by mean absolute SHAP value.
        """
        shap_values, X_transformed_df = self.compute_shap_values(df)
        shap_array = self._normalize_shap_array(shap_values)

        importance = pd.DataFrame({
            "feature": X_transformed_df.columns,
            "mean_abs_shap": np.abs(shap_array).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)

        return importance.reset_index(drop=True)

    def explain_anomaly_drivers(
        self,
        df_pred: pd.DataFrame,
        top_n: int = 10,
        top_features: int = 2,
        product_filter=None,
        region_filter=None
    ) -> pd.DataFrame:
        """
        Explain the strongest drivers behind the most severe anomalies.

        The method computes SHAP values for the fitted Isolation Forest and
        returns the top contributing anomaly signals for each selected row.

        Parameters
        ----------
        df_pred : pd.DataFrame
            DataFrame already enriched with anomaly_flag and anomaly_score.
        top_n : int, default=10
            Number of anomalies to explain.
        top_features : int, default=2
            Number of top local drivers to keep per anomaly.
        product_filter : None, str, list, default=None
            Optional product filter.
        region_filter : None, str, list, default=None
            Optional region filter.

        Returns
        -------
        pd.DataFrame
            Table with anomaly metadata and local explanation columns.
        """
        required_cols = ["anomaly_flag", "anomaly_score"]
        self._validate_required_columns(df_pred, required_cols)

        filtered_df = self.filter_dataframe(
            df_pred,
            product_filter=product_filter,
            region_filter=region_filter
        ).copy()

        anomaly_df = (
            filtered_df[filtered_df["anomaly_flag"] == 1]
            .sort_values("anomaly_score", ascending=True)
            .head(top_n)
            .copy()
        )

        if anomaly_df.empty:
            return anomaly_df

        shap_values, X_transformed_df = self.compute_shap_values(anomaly_df)
        shap_array = self._normalize_shap_array(shap_values)

        explanation_rows = []
        for row_pos, (idx, row) in enumerate(anomaly_df.iterrows()):
            shap_row = shap_array[row_pos]
            feature_row = X_transformed_df.loc[idx]

            ranked_idx = np.argsort(np.abs(shap_row))[::-1][:top_features]
            explanation = {
                "row_index": idx,
                "anomaly_score": row.get("anomaly_score", np.nan),
                "anomaly_label": row.get("anomaly_label", "Anomaly")
            }

            for rank, feat_idx in enumerate(ranked_idx, start=1):
                feat_name = X_transformed_df.columns[feat_idx]
                explanation[f"top_driver_{rank}"] = feat_name
                explanation[f"top_driver_{rank}_value"] = float(feature_row.iloc[feat_idx])
                explanation[f"top_driver_{rank}_shap"] = float(shap_row[feat_idx])

            for meta_col in ["Order_Date", "Product", "Region", "quantity_sum", "total_price_sum", "avg_ticket"]:
                if meta_col in anomaly_df.columns:
                    explanation[meta_col] = row[meta_col]

            explanation_rows.append(explanation)

        ordered_cols = [c for c in [
            "row_index", "Order_Date", "Product", "Region",
            "quantity_sum", "total_price_sum", "avg_ticket",
            "anomaly_label", "anomaly_score"
        ] if c in explanation_rows[0]]

        dynamic_cols = []
        for rank in range(1, top_features + 1):
            dynamic_cols.extend([
                f"top_driver_{rank}",
                f"top_driver_{rank}_value",
                f"top_driver_{rank}_shap"
            ])

        result = pd.DataFrame(explanation_rows)
        final_cols = [c for c in ordered_cols + dynamic_cols if c in result.columns]
        return result[final_cols]

    def plot_force_for_anomaly(
        self,
        df_pred: pd.DataFrame,
        row_index,
        matplotlib: bool = True
    ):
        """
        Plot a SHAP force plot for one anomaly row.

        Parameters
        ----------
        df_pred : pd.DataFrame
            DataFrame containing the row to explain.
        row_index : int or label
            Row index from df_pred.
        matplotlib : bool, default=True
            Use the matplotlib backend when possible.

        Returns
        -------
        object
            SHAP force plot object.
        """
        if row_index not in df_pred.index:
            raise ValueError(f"Row index '{row_index}' not found in df_pred.")

        row_df = df_pred.loc[[row_index]].copy()
        shap_values, X_transformed_df = self.compute_shap_values(row_df)
        shap_array = self._normalize_shap_array(shap_values)

        explainer = self.create_shap_explainer()
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = np.asarray(expected_value).flatten()[0]

        return shap.force_plot(
            expected_value,
            shap_array[0],
            X_transformed_df.iloc[0],
            matplotlib=matplotlib
        )

    # =========================================================
    # RESUMO
    # =========================================================
    def anomaly_summary(
        self,
        df_pred: pd.DataFrame,
        product_filter=None,
        region_filter=None
    ) -> pd.DataFrame:
        """
        Resume as anomalias por Product e Region, com filtros opcionais.
        """
        required_cols = ["Product", "Region", "anomaly_flag"]
        self._validate_required_columns(df_pred, required_cols)

        summary_df = self.filter_dataframe(
            df_pred,
            product_filter=product_filter,
            region_filter=region_filter
        ).copy()

        summary = (
            summary_df.groupby(["Product", "Region"], dropna=False)["anomaly_flag"]
            .agg(total_rows="count", anomaly_count="sum")
            .reset_index()
        )

        summary["anomaly_rate"] = summary["anomaly_count"] / summary["total_rows"]

        return summary.sort_values(
            by=["anomaly_count", "anomaly_rate"],
            ascending=[False, False]
        )

    # =========================================================
    # EXPORTAÇÃO
    # =========================================================
    def export_anomaly_report(
        self,
        df_pred: pd.DataFrame,
        folder_path: str = "reports",
        file_name: str = None,
        top_n: int = 20,
        product_filter=None,
        region_filter=None
    ) -> str:
        """
        Exporta CSV com top anomalias, com filtros opcionais por produto/região.
        """
        os.makedirs(folder_path, exist_ok=True)

        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"anomaly_report_top_{top_n}_{timestamp}.csv"

        full_path = os.path.join(folder_path, file_name)

        report_df = self.top_n_anomalies(
            df_pred=df_pred,
            n=top_n,
            product_filter=product_filter,
            region_filter=region_filter
        )

        report_df.to_csv(full_path, index=False, encoding="utf-8-sig")
        return full_path

    # =========================================================
    # SAVE / LOAD
    # =========================================================
    def save_model(self, folder_path=None, file_name=None) -> str:
        self._check_is_fitted()

        if folder_path is None:
            folder_path = self.base_path

        os.makedirs(folder_path, exist_ok=True)

        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"isolation_forest_pipeline_{timestamp}.joblib"

        full_path = os.path.join(folder_path, file_name)
        joblib.dump(self.best_estimator_, full_path)

        return full_path

    def save_best_params_json(
        self,
        folder_path=None,
        file_name="isolation_forest_best_params.json"
    ) -> str:
        
        self._check_is_fitted()

        if folder_path is None:
            folder_path = self.base_path

        os.makedirs(folder_path, exist_ok=True)
        full_path = os.path.join(folder_path, file_name)

        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(self.best_params_, f, indent=4, ensure_ascii=False)

        return full_path

    @staticmethod
    def load_model(model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {model_path}")

        return joblib.load(model_path)