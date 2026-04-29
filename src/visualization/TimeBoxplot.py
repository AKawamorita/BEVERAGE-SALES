import pandas as pd
import matplotlib.pyplot as plt


class SalesTimeBoxplot:
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

    def plot_boxplot_by_period(
        self,
        value_col: str,
        period: str = "quarter",
        filters: dict = None,
        start_date: str = None,
        end_date: str = None,
        figsize=(8, 6),
        rotation=45,
        showfliers=True,
        pastel_color="#C7DCEB"
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

        bp = ax.boxplot(
            grouped.values,
            labels=grouped.index,
            patch_artist=True,
            showfliers=showfliers
        )

        for box in bp["boxes"]:
            box.set(facecolor=pastel_color, edgecolor="#6B7280", linewidth=1.2)

        for whisker in bp["whiskers"]:
            whisker.set(color="#6B7280", linewidth=1.1)

        for cap in bp["caps"]:
            cap.set(color="#6B7280", linewidth=1.1)

        for median in bp["medians"]:
            median.set(color="#D97706", linewidth=1.8)

        for flier in bp["fliers"]:
            flier.set(
                marker="o",
                markerfacecolor="#F4A6A6",
                markeredgecolor="#A66",
                markersize=4,
                alpha=0.35
            )

        title = f"Boxplot de {value_col} por {period_col}"
        if filters:
            filtros_txt = " | ".join([f"{k}={v}" for k, v in filters.items()])
            title += f" | {filtros_txt}"
        if start_date or end_date:
            title += " | período filtrado"

        ax.set_title(title)
        ax.set_xlabel("Período")
        ax.set_ylabel(value_col)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.xticks(rotation=rotation)
        plt.tight_layout()
        plt.show()