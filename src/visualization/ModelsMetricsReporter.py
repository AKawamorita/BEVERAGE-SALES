import pandas as pd
import numpy as np


class ModelsMetricsReporter:
    def __init__(self, df_baseline: pd.DataFrame,df_with_anomaly: pd.DataFrame ):
        self.df_baseline = df_baseline.copy()
        self.df_with_anomaly = df_with_anomaly.copy()

    def compare_models_by_anomaly_flag(self,
        baseline_model,
        anomaly_model,
        target_col: str = "quantity_sum",
        anomaly_flag_col: str = "anomaly_flag",
        id_columns: list = None
    ):
        df_base = self.df_baseline.copy().reset_index(drop=True)
        df_anom = self.df_with_anomaly.copy().reset_index(drop=True)

        if target_col not in df_base.columns:
            raise ValueError(f"'{target_col}' not found in df_baseline")

        if target_col not in df_anom.columns:
            raise ValueError(f"'{target_col}' not found in df_with_anomaly")

        if anomaly_flag_col not in df_anom.columns:
            raise ValueError(f"'{anomaly_flag_col}' not found in df_with_anomaly")

        if id_columns is not None:
            required_base = set(id_columns + [target_col])
            required_anom = set(id_columns + [target_col, anomaly_flag_col])

            missing_base = [c for c in required_base if c not in df_base.columns]
            missing_anom = [c for c in required_anom if c not in df_anom.columns]

            if missing_base:
                raise ValueError(f"Missing columns in df_baseline: {missing_base}")

            if missing_anom:
                raise ValueError(f"Missing columns in df_with_anomaly: {missing_anom}")

            df_compare = df_anom[id_columns + [target_col, anomaly_flag_col]].merge(
                df_base[id_columns],
                on=id_columns,
                how="inner"
            )

            if df_compare.empty:
                raise ValueError("No matching rows found between df_baseline and df_with_anomaly using id_columns.")

            matched_base = df_base.merge(df_compare[id_columns], on=id_columns, how="inner")
            matched_anom = df_anom.merge(df_compare[id_columns], on=id_columns, how="inner")

            baseline_pred = baseline_model.predict(matched_base)
            anomaly_pred = anomaly_model.predict(matched_anom)

            df_compare = matched_anom[id_columns + [target_col, anomaly_flag_col]].copy()
            df_compare["baseline_pred"] = baseline_pred
            df_compare["anomaly_pred"] = anomaly_pred

        else:
            if len(df_base) != len(df_anom):
                raise ValueError("DataFrames have different lengths. Provide id_columns to align them safely.")

            df_compare = pd.DataFrame({
                target_col: df_anom[target_col].values,
                anomaly_flag_col: df_anom[anomaly_flag_col].values
            })

            df_compare["baseline_pred"] = baseline_model.predict(df_base)
            df_compare["anomaly_pred"] = anomaly_model.predict(df_anom)

        df_compare["baseline_abs_error"] = np.abs(df_compare[target_col] - df_compare["baseline_pred"])
        df_compare["anomaly_abs_error"] = np.abs(df_compare[target_col] - df_compare["anomaly_pred"])

        df_compare["baseline_sq_error"] = (df_compare[target_col] - df_compare["baseline_pred"]) ** 2
        df_compare["anomaly_sq_error"] = (df_compare[target_col] - df_compare["anomaly_pred"]) ** 2

        report = df_compare.groupby(anomaly_flag_col).agg(
            count=(target_col, "count"),
            baseline_mae=("baseline_abs_error", "mean"),
            anomaly_model_mae=("anomaly_abs_error", "mean"),
            baseline_median_ae=("baseline_abs_error", "median"),
            anomaly_model_median_ae=("anomaly_abs_error", "median"),
            baseline_rmse=("baseline_sq_error", lambda x: np.sqrt(np.mean(x))),
            anomaly_model_rmse=("anomaly_sq_error", lambda x: np.sqrt(np.mean(x))),
            baseline_max_ae=("baseline_abs_error", "max"),
            anomaly_model_max_ae=("anomaly_abs_error", "max")
        ).reset_index()

        report["mae_improvement"] = report["baseline_mae"] - report["anomaly_model_mae"]
        report["rmse_improvement"] = report["baseline_rmse"] - report["anomaly_model_rmse"]

        report["mae_improvement_pct"] = np.where(
            report["baseline_mae"] != 0,
            (report["mae_improvement"] / report["baseline_mae"]) * 100,
            0.0
        )

        report["rmse_improvement_pct"] = np.where(
            report["baseline_rmse"] != 0,
            (report["rmse_improvement"] / report["baseline_rmse"]) * 100,
            0.0
        )

        return report, df_compare