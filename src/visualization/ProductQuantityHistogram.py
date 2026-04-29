import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, List


class ProductQuantityHistogram:
    """
    Create a clean grouped bar chart for product quantity over time.

    This class filters a sales dataframe by year, category, month, and product.
    The chart shows quantity_sum by year/month on the X axis and total quantity
    on the Y axis. Products are displayed as grouped bars inside the selected
    category.

    Expected dataframe columns:
        - Product: product name
        - Region: region name
        - Category: product category
        - quantity_sum: numeric quantity
        - month: month number
        - year: year number

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with sales features.
    """

    REQUIRED_COLUMNS = [
        "Product",
        "Region",
        "Category",
        "quantity_sum",
        "month",
        "year"
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._validate_dataframe()
        self._prepare_columns()

    def _validate_dataframe(self) -> None:
        """
        Validate if the dataframe has all required columns.

        Raises
        ------
        ValueError
            If one or more required columns are missing.
        """
        missing_columns = [
            col for col in self.REQUIRED_COLUMNS
            if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"The dataframe is missing required columns: {missing_columns}"
            )

    def _prepare_columns(self) -> None:
        """
        Prepare columns used in the visualization.

        This method ensures that year and month are numeric and creates
        a year_month column in YYYY-MM format.
        """
        self.df["year"] = self.df["year"].astype(int)
        self.df["month"] = self.df["month"].astype(int)

        self.df["year_month"] = pd.to_datetime(
            self.df["year"].astype(str) + "-" +
            self.df["month"].astype(str).str.zfill(2) + "-01"
        )

        self.df["year_month_label"] = self.df["year_month"].dt.strftime("%Y-%m")

    def _normalize_to_list(
        self,
        value: Optional[Union[str, int, List[Union[str, int]]]]
    ) -> Optional[List[Union[str, int]]]:
        """
        Convert a single value into a list.

        Parameters
        ----------
        value : str, int, list or None
            Filter value.

        Returns
        -------
        list or None
            Normalized list of values.
        """
        if value is None:
            return None

        if isinstance(value, list):
            return value

        return [value]

    def filter_data(
        self,
        year: Union[int, List[int]],
        category: Union[str, List[str]],
        month: Optional[Union[int, List[int]]] = None,
        product: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Filter dataframe by year, category, month and product.

        Year and category are mandatory filters. Month and product are optional.

        Parameters
        ----------
        year : int or list of int
            Year or years to filter.
        category : str or list of str
            Category or categories to filter.
        month : int or list of int, optional
            Month or months to filter.
        product : str or list of str, optional
            Product or products to filter.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """
        if year is None:
            raise ValueError("The parameter 'year' is mandatory.")

        if category is None:
            raise ValueError("The parameter 'category' is mandatory.")

        years = self._normalize_to_list(year)
        categories = self._normalize_to_list(category)
        months = self._normalize_to_list(month)
        products = self._normalize_to_list(product)

        filtered_df = self.df[
            (self.df["year"].isin(years)) &
            (self.df["Category"].isin(categories))
        ].copy()

        if months is not None:
            filtered_df = filtered_df[
                filtered_df["month"].isin(months)
            ]

        if products is not None:
            filtered_df = filtered_df[
                filtered_df["Product"].isin(products)
            ]

        return filtered_df

    def aggregate_data(
        self,
        filtered_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate quantity_sum by period, category and product.

        Parameters
        ----------
        filtered_df : pd.DataFrame
            Filtered dataframe.

        Returns
        -------
        pd.DataFrame
            Aggregated dataframe ready for plotting.
        """
        if filtered_df.empty:
            return pd.DataFrame()

        grouped_df = (
            filtered_df
            .groupby(
                ["year_month", "year_month_label", "Category", "Product"],
                as_index=False
            )
            .agg(quantity_sum=("quantity_sum", "sum"))
            .sort_values(["year_month", "Category", "Product"])
        )

        return grouped_df

    def plot(
        self,
        year: Union[int, List[int]],
        category: Union[str, List[str]],
        month: Optional[Union[int, List[int]]] = None,
        product: Optional[Union[str, List[str]]] = None,
        figsize: tuple = (14, 6),
        title: Optional[str] = None,
        rotate_xticks: int = 45
    ) -> pd.DataFrame:
        """
        Plot a clean grouped bar chart using pastel colors.

        Parameters
        ----------
        year : int or list of int
            Mandatory year filter.
        category : str or list of str
            Mandatory category filter.
        month : int or list of int, optional
            Optional month filter.
        product : str or list of str, optional
            Optional product filter.
        figsize : tuple, default=(14, 6)
            Figure size.
        title : str, optional
            Custom chart title.
        rotate_xticks : int, default=45
            Rotation angle for X axis labels.

        Returns
        -------
        pd.DataFrame
            Aggregated dataframe used in the chart.
        """
        filtered_df = self.filter_data(
            year=year,
            category=category,
            month=month,
            product=product
        )

        grouped_df = self.aggregate_data(filtered_df)

        if grouped_df.empty:
            print("No data found for the selected filters.")
            return grouped_df

        pivot_df = grouped_df.pivot_table(
            index="year_month_label",
            columns="Product",
            values="quantity_sum",
            aggfunc="sum",
            fill_value=0
        )

        pivot_df = pivot_df.sort_index()

        products = pivot_df.columns.tolist()
        x = np.arange(len(pivot_df.index))

        total_products = len(products)
        bar_width = min(0.8 / max(total_products, 1), 0.25)

        pastel_colors = [
            "#AEC6CF",  # pastel blue
            "#FFB7B2",  # pastel red
            "#B5EAD7",  # pastel green
            "#C7CEEA",  # pastel purple
            "#FFDAC1",  # pastel orange
            "#E2F0CB",  # pastel lime
            "#F6C6EA",  # pastel pink
            "#D5AAFF",  # pastel violet
            "#FFFFD1",  # pastel yellow
            "#BDE0FE"   # light blue
        ]

        plt.figure(figsize=figsize)

        for i, prod in enumerate(products):
            offset = (i - total_products / 2) * bar_width + bar_width / 2

            plt.bar(
                x + offset,
                pivot_df[prod],
                width=bar_width,
                label=prod,
                color=pastel_colors[i % len(pastel_colors)],
                edgecolor="#FFFFFF",
                linewidth=0.8
            )

        if title is None:
            category_text = ", ".join(self._normalize_to_list(category))
            title = f"Quantity Sum by Product - Category: {category_text}"

        plt.title(title, fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Year / Month", fontsize=11)
        plt.ylabel("Quantity Sum", fontsize=11)

        plt.xticks(
            x,
            pivot_df.index,
            rotation=rotate_xticks,
            ha="right"
        )

        plt.grid(
            axis="y",
            linestyle="--",
            alpha=0.25
        )

        plt.legend(
            title="Product",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False
        )

        plt.tight_layout()
        plt.show()

        return grouped_df