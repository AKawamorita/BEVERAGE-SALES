import pandas as pd
import matplotlib.pyplot as plt


class SalesTimeViolin:
    def __init__(self, df: pd.DataFrame, date_col: str = "Order_Date"):
        self.df = df.copy()
        self.date_col = date_col
        self._prepare_dataframe()

    def _prepare_dataframe(self):
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce")
        self.df = self.df.dropna(subset=[self.date_col]).copy()
        self.df = self.df.sort_values(self.date_col).reset_index(drop=True)

        self.df["YearMonth"] = self.df[self.date_col].dt.to_period("M").astype(str)
        self.df["Quarter"] = self.df[self.date_col].dt.to_period("Q").astype(str)
        self.df["Semester"] = self.df[self.date_col].apply(
            lambda x: f"{x.year}-S1" if x.month <= 6 else f"{x.year}-S2"
        )
        self.df["Year"] = self.df[self.date_col].dt.year.astype(str)

    def _filter_data(
        self,
        value_col: str,
        group_col: str = None,
        group_value=None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        if value_col not in self.df.columns:
            raise ValueError(f"A coluna '{value_col}' não existe no DataFrame.")

        filtered_df = self.df.copy()

        if start_date is not None:
            filtered_df = filtered_df[filtered_df[self.date_col] >= pd.to_datetime(start_date)]

        if end_date is not None:
            filtered_df = filtered_df[filtered_df[self.date_col] <= pd.to_datetime(end_date)]

        if group_col is not None:
            if group_col not in filtered_df.columns:
                raise ValueError(f"A coluna '{group_col}' não existe no DataFrame.")
            if group_value is None:
                raise ValueError("Se 'group_col' for informado, 'group_value' também deve ser informado.")
            filtered_df = filtered_df[filtered_df[group_col] == group_value]

        filtered_df = filtered_df.dropna(subset=[value_col])

        return filtered_df
    
    def _filter_data(
        self,
        value_col: str,
        filters: dict = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        if value_col not in self.df.columns:
            raise ValueError(f"A coluna '{value_col}' não existe no DataFrame.")

        filtered_df = self.df.copy()

        if start_date is not None:
            filtered_df = filtered_df[filtered_df[self.date_col] >= pd.to_datetime(start_date)]

        if end_date is not None:
            filtered_df = filtered_df[filtered_df[self.date_col] <= pd.to_datetime(end_date)]

        if filters is not None:
            for col, values in filters.items():
                if col not in filtered_df.columns:
                    raise ValueError(f"A coluna '{col}' não existe no DataFrame.")

                if values is None:
                    continue

                if not isinstance(values, (list, tuple, set)):
                    values = [values]

                filtered_df = filtered_df[filtered_df[col].isin(values)]

        filtered_df = filtered_df.dropna(subset=[value_col])

        return filtered_df

    def plot_violin_by_period(
        self,
        value_col: str,
        period: str = "year",
        filters: dict = None,
        start_date: str = None,
        end_date: str = None,
        figsize=(9, 6),
        rotation=45,
        pastel_color="#C7DCEB",
        show_medians=True,
        use_log_scale=False
    ):
        period_map = {
            "month": "YearMonth",
            "quarter": "Quarter",
            "semester": "Semester",
            "year": "Year"
        }

        if period not in period_map:
            raise ValueError("period deve ser: 'month', 'quarter', 'semester' ou 'year'.")

        period_col = period_map[period]

        plot_df = self._filter_data(
            value_col=value_col,
            filters=filters,
            start_date=start_date,
            end_date=end_date
        )

        if plot_df.empty:
            print("Nenhum dado disponível após os filtros aplicados.")
            return

        grouped = plot_df.groupby(period_col)[value_col].apply(list)

        if grouped.empty:
            print("Nenhum dado agrupado disponível para plotagem.")
            return

        fig, ax = plt.subplots(figsize=figsize)

        violin = ax.violinplot(
            grouped.values,
            showmeans=False,
            showmedians=show_medians,
            showextrema=False
        )

        for body in violin["bodies"]:
            body.set_facecolor(pastel_color)
            body.set_edgecolor("#6B7280")
            body.set_alpha(0.8)

        if show_medians and "cmedians" in violin:
            violin["cmedians"].set_color("#D97706")
            violin["cmedians"].set_linewidth(2)

        ax.set_xticks(range(1, len(grouped.index) + 1))
        ax.set_xticklabels(grouped.index, rotation=rotation)

        title = f"Violin plot de {value_col} por {period_col}"
        if filters:
            filtros_txt = " | ".join([f"{k}={v}" for k, v in filters.items()])
            title += f" | {filtros_txt}"
        if start_date or end_date:
            title += " | período filtrado"

        ax.set_title(title)
        ax.set_xlabel("Período")
        ax.set_ylabel(value_col)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if use_log_scale:
            ax.set_yscale("log")

        plt.tight_layout()
        plt.show()