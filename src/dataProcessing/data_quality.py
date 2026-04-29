import pandas as pd
import numpy as np


class DataQuality:
    """
    Classe para análise e correção de qualidade de dados em datasets de vendas de bebidas.

    Objetivos:
    - Analisar problemas de qualidade de dados
    - Gerar report profissional com status e volume de inconsistências
    - Aplicar correções de forma controlada
    - Manter funções de análise e correção separadas

    Regras avaliadas:
    - Quantity: valores nulos ou negativos
    - Discount: valores nulos, negativos ou acima de 1
    - Customer_Type: validação da regra B2B/B2C
    - Order_Date: parsing e datas inválidas
    - Total_Price: consistência com preço, quantidade e desconto
    - Duplicatas exatas
    """

    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        self.df = df.copy()
        self.report = None

    # =========================================================
    # UTILITÁRIOS
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

    # =========================================================
    # ANÁLISE
    # =========================================================
    def analyze_quantity_missing(self) -> dict:
        qty = self._safe_numeric(self.df["Quantity"])
        mask = qty.isna()
        return self._build_report_row(
            rule_name="Missing Quantity",
            issue_found=mask.any(),
            issue_count=mask.sum(),
            severity="Medium",
            recommendation="Impute missing Quantity with zero if business meaning is 'no units sold'."
        )

    def analyze_quantity_negative(self) -> dict:
        qty = self._safe_numeric(self.df["Quantity"])
        mask = qty < 0
        return self._build_report_row(
            rule_name="Negative Quantity",
            issue_found=mask.any(),
            issue_count=mask.sum(),
            severity="High",
            recommendation="Replace negative Quantity with zero or review source transactions."
        )

    def analyze_discount_missing(self) -> dict:
        disc = self._safe_numeric(self.df["Discount"])
        mask = disc.isna()
        return self._build_report_row(
            rule_name="Missing Discount",
            issue_found=mask.any(),
            issue_count=mask.sum(),
            severity="Low",
            recommendation="Set missing Discount to zero."
        )

    def analyze_discount_negative(self) -> dict:
        disc = self._safe_numeric(self.df["Discount"])
        mask = disc < 0
        return self._build_report_row(
            rule_name="Negative Discount",
            issue_found=mask.any(),
            issue_count=mask.sum(),
            severity="High",
            recommendation="Set negative Discount to zero."
        )

    def analyze_discount_above_one(self) -> dict:
        disc = self._safe_numeric(self.df["Discount"])
        mask = disc > 1
        return self._build_report_row(
            rule_name="Discount Above One",
            issue_found=mask.any(),
            issue_count=mask.sum(),
            severity="High",
            recommendation="Clip Discount to valid range [0, 1]."
        )

    def analyze_customer_type_invalid(self) -> dict:
        valid_types = {"B2B", "B2C"}
        mask = ~self.df["Customer_Type"].isin(valid_types)
        return self._build_report_row(
            rule_name="Invalid Customer Type",
            issue_found=mask.any(),
            issue_count=mask.sum(),
            severity="Medium",
            recommendation="Review or standardize Customer_Type values to B2B/B2C."
        )

    def analyze_b2c_discount_rule(self) -> dict:
        disc = self._safe_numeric(self.df["Discount"]).fillna(0)
        mask = (self.df["Customer_Type"] == "B2C") & (disc > 0)
        return self._build_report_row(
            rule_name="B2C Discount Rule Violation",
            issue_found=mask.any(),
            issue_count=mask.sum(),
            severity="High",
            recommendation="Set Discount to zero for B2C records."
        )

    def analyze_order_date_invalid(self) -> dict:
        dates = self._safe_datetime(self.df["Order_Date"])
        mask = dates.isna()
        return self._build_report_row(
            rule_name="Invalid Order Date",
            issue_found=mask.any(),
            issue_count=mask.sum(),
            severity="High",
            recommendation="Parse Order_Date and review invalid date records."
        )

    def analyze_duplicates(self) -> dict:
        mask = self.df.duplicated()
        return self._build_report_row(
            rule_name="Exact Duplicate Rows",
            issue_found=mask.any(),
            issue_count=mask.sum(),
            severity="Medium",
            recommendation="Remove exact duplicate rows."
        )

    def analyze_total_price_consistency(self, tolerance: float = 0.1) -> dict:
        unit_price = self._safe_numeric(self.df["Unit_Price"])
        qty = self._safe_numeric(self.df["Quantity"])
        disc = self._safe_numeric(self.df["Discount"]).fillna(0).clip(0, 1)
        total_price = self._safe_numeric(self.df["Total_Price"]).round(2)

        expected_total = (unit_price * qty * (1 - disc)).round(2)
        diff = (expected_total - total_price).abs().round(2)

        mask = (
            unit_price.notna() &
            qty.notna() &
            total_price.notna() &
            (diff > tolerance)
        )

        return self._build_report_row(
            rule_name="Total Price Consistency",
            issue_found=bool(mask.any()),
            issue_count=int(mask.sum()),
            severity="High",
            recommendation="Recalculate Total_Price."
        )

    def run_full_analysis(self, tolerance: float = 0.01) -> pd.DataFrame:
        rows = [
            self.analyze_quantity_missing(),
            self.analyze_quantity_negative(),
            self.analyze_discount_missing(),
            self.analyze_discount_negative(),
            self.analyze_discount_above_one(),
            self.analyze_customer_type_invalid(),
            self.analyze_b2c_discount_rule(),
            self.analyze_order_date_invalid(),
            self.analyze_duplicates(),
            self.analyze_total_price_consistency(tolerance=tolerance)
        ]
        self.report = self.report = pd.DataFrame(rows).drop(columns=["recommendation"], errors="ignore")
        return self.report
    
    def get_total_price_inconsistencies_sample(self, tolerance: float = 0.1, n: int = 20) -> pd.DataFrame:
        unit_price = self._safe_numeric(self.df["Unit_Price"])
        qty = self._safe_numeric(self.df["Quantity"])
        disc = self._safe_numeric(self.df["Discount"]).fillna(0).clip(0, 1)
        total_price = self._safe_numeric(self.df["Total_Price"]).round(2)

        expected_total = (unit_price * qty * (1 - disc)).round(2)
        diff = (expected_total - total_price).abs().round(2)

        mask = (
            unit_price.notna() &
            qty.notna() &
            total_price.notna() &
            (diff > tolerance)
        )

        sample = self.df.copy()
        sample["Unit_Price_num"] = unit_price
        sample["Quantity_num"] = qty
        sample["Discount_num"] = disc
        sample["Total_Price_num"] = total_price
        sample["Expected_Total"] = expected_total
        sample["Diff"] = diff

        return sample.loc[mask, [
            "Product", "Region", "Order_Date",
            "Unit_Price_num", "Quantity_num", "Discount_num",
            "Total_Price_num", "Expected_Total", "Diff"
        ]].sort_values("Diff", ascending=False).head(n)

    # =========================================================
    # CORREÇÕES
    # =========================================================
    def fix_quantity_missing_with_zero(self):
        self.df["Quantity"] = self._safe_numeric(self.df["Quantity"]).fillna(0)
        return self.df

    def fix_quantity_negative_with_zero(self):
        self.df["Quantity"] = self._safe_numeric(self.df["Quantity"])
        self.df.loc[self.df["Quantity"] < 0, "Quantity"] = 0
        return self.df

    def fix_discount_missing_with_zero(self):
        self.df["Discount"] = self._safe_numeric(self.df["Discount"]).fillna(0)
        return self.df

    def fix_discount_negative_with_zero(self):
        self.df["Discount"] = self._safe_numeric(self.df["Discount"])
        self.df.loc[self.df["Discount"] < 0, "Discount"] = 0
        return self.df

    def fix_discount_clip_range(self, lower: float = 0, upper: float = 1):
        self.df["Discount"] = self._safe_numeric(self.df["Discount"]).fillna(0)
        self.df["Discount"] = self.df["Discount"].clip(lower=lower, upper=upper)
        return self.df

    def fix_b2c_discount_rule(self):
        self.df["Discount"] = self._safe_numeric(self.df["Discount"]).fillna(0)
        self.df.loc[self.df["Customer_Type"] == "B2C", "Discount"] = 0
        return self.df

    def fix_order_date_parse(self):
        self.df["Order_Date"] = self._safe_datetime(self.df["Order_Date"])
        return self.df

    def fix_remove_duplicates(self):
        self.df = self.df.drop_duplicates().copy()
        return self.df

    def fix_total_price_recalculate(self, round_digits: int = 2):
        self.df["Unit_Price"] = self._safe_numeric(self.df["Unit_Price"])
        self.df["Quantity"] = self._safe_numeric(self.df["Quantity"]).fillna(0)
        self.df["Discount"] = self._safe_numeric(self.df["Discount"]).fillna(0).clip(0, 1)

        self.df["Total_Price"] = (
            self.df["Unit_Price"] * self.df["Quantity"] * (1 - self.df["Discount"])
        ).round(round_digits)

        return self.df

    def fix_sales_value_by_neighbor_mean(
        self,
        group_cols=("Product", "Region"),
        target_col="Total_Price"
    ):
        """
        Corrige valores ausentes em target_col usando média entre d-1 e d+1 dentro do grupo.
        Requer ordenação temporal por Order_Date.
        """
        self.df["Order_Date"] = self._safe_datetime(self.df["Order_Date"])
        self.df[target_col] = self._safe_numeric(self.df[target_col])

        self.df = self.df.sort_values(list(group_cols) + ["Order_Date"]).copy()

        def _fill_group(s):
            prev_val = s.shift(1)
            next_val = s.shift(-1)
            neighbor_mean = (prev_val + next_val) / 2
            return s.fillna(neighbor_mean)

        self.df[target_col] = (
            self.df.groupby(list(group_cols))[target_col]
            .transform(_fill_group)
        )

        return self.df

    def apply_all_fixes(self, recalc_total_price: bool = True):
        self.fix_quantity_missing_with_zero()
        self.fix_quantity_negative_with_zero()
        self.fix_discount_missing_with_zero()
        self.fix_discount_negative_with_zero()
        self.fix_discount_clip_range()
        self.fix_b2c_discount_rule()
        self.fix_order_date_parse()
        self.fix_remove_duplicates()

        if recalc_total_price:
            self.fix_total_price_recalculate()

        return self.df

    # =========================================================
    # RELATÓRIO PROFISSIONAL
    # =========================================================
    def generate_executive_summary(self) -> str:
        if self.report is None:
            self.run_full_analysis()

        total_rules = len(self.report)
        issues_found = (self.report["issue_found"] == "YES").sum()
        total_issues = self.report["issue_count"].sum()

        high_issues = self.report[
            (self.report["issue_found"] == "YES") &
            (self.report["severity"] == "High")
        ]["rule_name"].tolist()

        if high_issues:
            high_issues_text = ", ".join(high_issues)
        else:
            high_issues_text = "No high-severity issues were detected"

        summary = (
            f"Data Quality Executive Summary\n"
            f"------------------------------\n"
            f"Rules evaluated: {total_rules}\n"
            f"Rules with issues found: {issues_found}\n"
            f"Total issue count: {total_issues}\n"
            f"High-severity issues: {high_issues_text}\n"
        )
        return summary

    def get_clean_dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def reset_to_original(self):
        self.df = self.original_df.copy()
        self.report = None