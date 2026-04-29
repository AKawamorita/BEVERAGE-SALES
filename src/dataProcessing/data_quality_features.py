import pandas as pd
import numpy as np


class DataQualityFeatures:
    """
    Build and validate time-based sales features for anomaly detection and
    temporal behavior analysis.

    Main goals
    ----------
    - Aggregate transactional sales data into daily business-level metrics
    - Create rolling-window features using only past information
    - Handle missing values caused by insufficient initial history
    - Provide a simple feature-quality report
    - Keep feature generation, validation, and correction separated

    Expected input columns
    ----------------------
    - Order_ID
    - Customer_ID
    - Customer_Type
    - Product
    - Category
    - Unit_Price
    - Quantity
    - Discount
    - Total_Price
    - Region
    - Order_Date

    Generated features
    ------------------
    Base daily features:
    - quantity_sum
    - total_price_sum
    - unit_price_mean
    - discount_mean
    - order_count
    - customer_count
    - avg_ticket

    Calendar features:
    - day_of_week
    - month
    - is_weekend

    Rolling features:
    - quantity_sum_mean_7d
    - quantity_sum_std_7d
    - quantity_sum_sum_7d
    - total_price_sum_mean_7d
    - total_price_sum_std_7d
    - total_price_sum_mean_30d
    - unit_price_mean_mean_7d
    - discount_mean_mean_14d

    Deviation features:
    - quantity_vs_mean_7d
    - total_price_vs_mean_30d
    - quantity_pct_vs_mean_7d

    History flags:
    - history_less_than_7d
    - history_less_than_30d

    Notes
    -----
    - Rolling features use shift(1) to avoid data leakage.
    - Initial missing values caused by insufficient history can be corrected
      with fill_initial_feature_nans() or fix_missing_feature_values().
    """

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = "Order_Date",
        group_cols: list | None = None,
        fill_missing_days: bool = True
    ):
        self.original_df = df.copy()
        self.df = df.copy()
        self.features_df = None
        self.report = None
        self.date_col = date_col
        self.group_cols = group_cols if group_cols is not None else ["Product", "Region"]
        self.fill_missing_days = fill_missing_days

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================
    def _safe_numeric(self, series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def _safe_datetime(self, series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce")

    def _build_report_row(
        self,
        rule_name: str,
        issue_found: bool,
        issue_count: int,
        severity: str,
        recommendation: str
    ) -> dict:
        return {
            "rule_name": rule_name,
            "issue_found": "YES" if issue_found else "NO",
            "issue_count": int(issue_count),
            "severity": severity,
            "recommendation": recommendation
        }

    def _validate_required_columns(self):
        required_cols = {
            "Order_ID", "Customer_ID", "Customer_Type", "Product", "Category",
            "Unit_Price", "Quantity", "Discount", "Total_Price", "Region", self.date_col
        }

        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # =========================================================
    # FEATURE GENERATION
    # =========================================================
    def build_base_daily_features(self) -> pd.DataFrame:
        """
        Aggregate the raw transactional dataset into daily metrics by business group.
        """
        self._validate_required_columns()

        dfx = self.df.copy()

        dfx[self.date_col] = self._safe_datetime(dfx[self.date_col])
        dfx = dfx.dropna(subset=[self.date_col]).copy()
        dfx[self.date_col] = dfx[self.date_col].dt.normalize()

        dfx["Quantity"] = self._safe_numeric(dfx["Quantity"])
        dfx["Unit_Price"] = self._safe_numeric(dfx["Unit_Price"])
        dfx["Discount"] = self._safe_numeric(dfx["Discount"])
        dfx["Total_Price"] = self._safe_numeric(dfx["Total_Price"])

        daily = (
            dfx.groupby(self.group_cols + [self.date_col], as_index=False)
            .agg(
                quantity_sum=("Quantity", "sum"),
                total_price_sum=("Total_Price", "sum"),
                unit_price_mean=("Unit_Price", "mean"),
                discount_mean=("Discount", "mean"),
                order_count=("Order_ID", "nunique"),
                customer_count=("Customer_ID", "nunique")
            )
        )

        daily["avg_ticket"] = np.where(
            daily["order_count"] > 0,
            daily["total_price_sum"] / daily["order_count"],
            0
        )

        self.features_df = daily.copy()
        return self.features_df

    def fill_missing_days_by_group(self) -> pd.DataFrame:
        """
        Fill missing calendar days inside each group to enable continuous daily rolling windows.
        """
        if self.features_df is None:
            self.build_base_daily_features()

        frames = []

        for keys, g in self.features_df.groupby(self.group_cols):
            g = g.sort_values(self.date_col).copy()

            full_dates = pd.date_range(
                g[self.date_col].min(),
                g[self.date_col].max(),
                freq="D"
            )

            g = (
                g.set_index(self.date_col)
                .reindex(full_dates)
                .rename_axis(self.date_col)
                .reset_index()
            )

            if len(self.group_cols) == 1:
                g[self.group_cols[0]] = keys
            else:
                for col, val in zip(self.group_cols, keys):
                    g[col] = val

            fill_zero_cols = [
                "quantity_sum",
                "total_price_sum",
                "unit_price_mean",
                "discount_mean",
                "order_count",
                "customer_count",
                "avg_ticket"
            ]

            for col in fill_zero_cols:
                g[col] = g[col].fillna(0)

            frames.append(g)

        self.features_df = pd.concat(frames, ignore_index=True)
        self.features_df = self.features_df.sort_values(
            self.group_cols + [self.date_col]
        ).reset_index(drop=True)

        return self.features_df

    def create_calendar_features(self) -> pd.DataFrame:
        """
        Create simple calendar-based seasonal features.
        """
        if self.features_df is None:
            self.build_base_daily_features()

        self.features_df["day_of_week"] = self.features_df[self.date_col].dt.dayofweek
        self.features_df["month"] = self.features_df[self.date_col].dt.month
        self.features_df["year"] = self.features_df[self.date_col].dt.year
        self.features_df["is_weekend"] = (
            self.features_df["day_of_week"].isin([5, 6]).astype(int)
        )

        return self.features_df

    def create_rolling_features(self) -> pd.DataFrame:
        """
        Create rolling-window features using only historical data.
        """
        if self.features_df is None:
            self.build_base_daily_features()

        g = self.features_df.groupby(self.group_cols)

        self.features_df["quantity_sum_mean_7d"] = (
            g["quantity_sum"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        )

        self.features_df["quantity_sum_std_7d"] = (
            g["quantity_sum"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).std())
        )

        self.features_df["quantity_sum_sum_7d"] = (
            g["quantity_sum"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
        )

        self.features_df["total_price_sum_mean_7d"] = (
            g["total_price_sum"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        )

        self.features_df["total_price_sum_std_7d"] = (
            g["total_price_sum"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).std())
        )

        self.features_df["total_price_sum_mean_30d"] = (
            g["total_price_sum"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
        )

        self.features_df["unit_price_mean_mean_7d"] = (
            g["unit_price_mean"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        )

        self.features_df["discount_mean_mean_14d"] = (
            g["discount_mean"].transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
        )

        self.features_df["quantity_vs_mean_7d"] = (
            self.features_df["quantity_sum"] - self.features_df["quantity_sum_mean_7d"]
        )

        self.features_df["total_price_vs_mean_30d"] = (
            self.features_df["total_price_sum"] - self.features_df["total_price_sum_mean_30d"]
        )

        self.features_df["quantity_pct_vs_mean_7d"] = np.where(
            self.features_df["quantity_sum_mean_7d"].fillna(0) != 0,
            (
                self.features_df["quantity_sum"] -
                self.features_df["quantity_sum_mean_7d"]
            ) / self.features_df["quantity_sum_mean_7d"],
            np.nan
        )

        return self.features_df

    def create_future_quantity_targets(self) -> pd.DataFrame:
        """
        Create future quantity features for each Product and Region group.

        The method calculates the total quantity sold in the next 7, 14, and 30 days.
        These features can be used as future targets or supporting variables for
        supervised modeling, but they should not be used as input features for
        anomaly detection to avoid data leakage.

        quantity_sum_next_14d, quantity_sum_next_30d
        Features are created for future supervised modeling, but they are not currently 
        used in the anomaly detection pipeline.
        """
        if self.features_df is None:
            self.build_base_daily_features()

        self.features_df = self.features_df.sort_values(["Product", "Region", "Order_Date"])

        self.features_df["quantity_sum_next_7d"] = (
            self.features_df
            .groupby(["Product", "Region"])["quantity_sum"]
            .transform(
                lambda s: s[::-1].rolling(window=7, min_periods=1).sum()[::-1].shift(-1)
            )
        )

        # This feature is available for future supervised modeling, but it is not currently used for anomaly detection.
        self.features_df["quantity_sum_next_14d"] = (
            self.features_df
            .groupby(["Product", "Region"])["quantity_sum"]
            .transform(
                lambda s: s[::-1].rolling(window=14, min_periods=1).sum()[::-1].shift(-1)
            )
        )

        # This feature is available for future supervised modeling, but it is not currently used for anomaly detection.
        self.features_df["quantity_sum_next_30d"] = (
            self.features_df
            .groupby(["Product", "Region"])["quantity_sum"]
            .transform(
                lambda s: s[::-1].rolling(window=30, min_periods=1).sum()[::-1].shift(-1)
            )
        )

        return self.features_df
    
    def create_history_flags(self) -> pd.DataFrame:
        """
        Create flags that indicate insufficient historical depth for each group.
        """
        if self.features_df is None:
            self.build_base_daily_features()

        row_number = self.features_df.groupby(self.group_cols).cumcount()

        self.features_df["history_less_than_7d"] = (row_number < 7).astype(int)
        self.features_df["history_less_than_30d"] = (row_number < 30).astype(int)

        return self.features_df

    def fill_initial_feature_nans(self) -> pd.DataFrame:
        """
        Fill NaN values in derived features caused by insufficient initial history.

        Strategy
        --------
        - rolling means, sums and deviations are filled with 0
        - rolling std is filled with 0
        - percentage deviation is filled with 0
        """
        if self.features_df is None:
            raise ValueError("No features were created yet.")

        cols_fill_zero = [
            "quantity_sum_mean_7d",
            "quantity_sum_std_7d",
            "quantity_sum_sum_7d",
            "total_price_sum_mean_7d",
            "total_price_sum_std_7d",
            "total_price_sum_mean_30d",
            "unit_price_mean_mean_7d",
            "discount_mean_mean_14d",
            "quantity_vs_mean_7d",
            "total_price_vs_mean_30d",
            "quantity_pct_vs_mean_7d"
        ]

        existing_cols = [col for col in cols_fill_zero if col in self.features_df.columns]
        self.features_df[existing_cols] = self.features_df[existing_cols].fillna(0)

        return self.features_df

    def build_all_features(self) -> pd.DataFrame:
        """
        Run the full feature-engineering pipeline.
        """
        self.build_base_daily_features()

        if self.fill_missing_days:
            self.fill_missing_days_by_group()

        self.create_calendar_features()
        self.create_rolling_features()
        self.create_history_flags()
        self.create_future_quantity_targets()
        self.fill_initial_feature_nans()

        return self.features_df.copy()

    # =========================================================
    # FEATURE QUALITY ANALYSIS
    # =========================================================
    def analyze_missing_feature_values(self) -> dict:
        """
        Check whether derived features still contain missing values.
        """
        if self.features_df is None:
            self.build_all_features()

        feature_cols = [
            "quantity_sum_mean_7d",
            "quantity_sum_std_7d",
            "quantity_sum_sum_7d",
            "total_price_sum_mean_7d",
            "total_price_sum_std_7d",
            "total_price_sum_mean_30d",
            "unit_price_mean_mean_7d",
            "discount_mean_mean_14d",
            "quantity_vs_mean_7d",
            "total_price_vs_mean_30d",
            "quantity_pct_vs_mean_7d"
        ]

        existing_cols = [c for c in feature_cols if c in self.features_df.columns]
        issue_count = self.features_df[existing_cols].isna().sum().sum()

        return self._build_report_row(
            rule_name="Missing Derived Feature Values",
            issue_found=issue_count > 0,
            issue_count=issue_count,
            severity="High",
            recommendation="Apply fix_missing_feature_values()."
        )

    def analyze_negative_base_metrics(self) -> dict:
        """
        Check negative values in aggregated business metrics.
        """
        if self.features_df is None:
            self.build_all_features()

        cols = ["quantity_sum", "total_price_sum", "order_count", "customer_count", "avg_ticket"]
        issue_count = 0

        for col in cols:
            if col in self.features_df.columns:
                issue_count += (self.features_df[col] < 0).sum()

        return self._build_report_row(
            rule_name="Negative Aggregated Metrics",
            issue_found=issue_count > 0,
            issue_count=issue_count,
            severity="Medium",
            recommendation="Review raw transactional data before feature generation."
        )

    def analyze_invalid_calendar_features(self) -> dict:
        """
        Validate expected ranges for calendar features.
        """
        if self.features_df is None:
            self.build_all_features()

        issue_count = 0

        if "day_of_week" in self.features_df.columns:
            issue_count += (~self.features_df["day_of_week"].between(0, 6)).sum()

        if "month" in self.features_df.columns:
            issue_count += (~self.features_df["month"].between(1, 12)).sum()

        if "is_weekend" in self.features_df.columns:
            issue_count += (~self.features_df["is_weekend"].isin([0, 1])).sum()

        return self._build_report_row(
            rule_name="Invalid Calendar Features",
            issue_found=issue_count > 0,
            issue_count=issue_count,
            severity="Low",
            recommendation="Rebuild calendar features from parsed Order_Date."
        )

    def run_full_analysis(self) -> pd.DataFrame:
        """
        Run a compact quality analysis over the generated features.
        """
        rows = [
            self.analyze_missing_feature_values(),
            self.analyze_negative_base_metrics(),
            self.analyze_invalid_calendar_features()
        ]

        self.report = pd.DataFrame(rows).drop(columns=["recommendation"], errors="ignore")
        return self.report

    # =========================================================
    # FIX METHODS
    # =========================================================
    def fix_missing_feature_values(self) -> pd.DataFrame:
        """
        Reapply missing-value correction to derived temporal features.
        """
        self.fill_initial_feature_nans()
        return self.features_df

    def rebuild_calendar_features(self) -> pd.DataFrame:
        """
        Recompute calendar features.
        """
        self.create_calendar_features()
        return self.features_df

    def rebuild_rolling_features(self) -> pd.DataFrame:
        """
        Recompute rolling features and reapply missing-value correction.
        """
        self.create_rolling_features()
        self.fill_initial_feature_nans()
        return self.features_df

    def rebuild_history_flags(self) -> pd.DataFrame:
        """
        Recompute historical depth flags.
        """
        self.create_history_flags()
        return self.features_df

    # =========================================================
    # REPORTING AND ACCESS
    # =========================================================
    def generate_executive_summary(self) -> str:
        """
        Return a compact executive summary of the feature-quality checks.
        """
        if self.report is None:
            self.run_full_analysis()

        total_rules = len(self.report)
        issues_found = (self.report["issue_found"] == "YES").sum()
        total_issues = self.report["issue_count"].sum()

        high_issues = self.report[
            (self.report["issue_found"] == "YES") &
            (self.report["severity"] == "High")
        ]["rule_name"].tolist()

        high_issues_text = ", ".join(high_issues) if high_issues else "No high-severity issues were detected"

        summary = (
            f"Feature Quality Executive Summary\n"
            f"--------------------------------\n"
            f"Rules evaluated: {total_rules}\n"
            f"Rules with issues found: {issues_found}\n"
            f"Total issue count: {total_issues}\n"
            f"High-severity issues: {high_issues_text}\n"
        )
        return summary

    def get_feature_dataframe(self) -> pd.DataFrame:
        """
        Return the engineered feature DataFrame.
        """
        if self.features_df is None:
            self.build_all_features()
        return self.features_df.copy()

    def reset_to_original(self):
        """
        Reset the object to its initial state.
        """
        self.df = self.original_df.copy()
        self.features_df = None
        self.report = None