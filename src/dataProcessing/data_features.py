import pandas as pd
import numpy as np


def DataFeatures(
    df: pd.DataFrame,
    date_col: str = "Order_Date",
    group_cols: list = None,
    windows: tuple = (7, 14, 30),
    min_periods: int = 1,
    fill_missing_days: bool = True
) -> pd.DataFrame:
    """
    Cria features temporais com sliding window para um dataset de vendas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame original contendo as colunas:
        Order_ID, Customer_ID, Customer_Type, Product, Category, Unit_Price,
        Quantity, Discount, Total_Price, Region, Order_Date

    date_col : str, default="Order_Date"
        Nome da coluna de data.

    group_cols : list, default=["Product", "Region"]
        Granularidade para cálculo das janelas.
        Ex.: ["Product", "Region"], ["Product"], ["Category", "Region"].

    windows : tuple, default=(7, 14, 30)
        Janelas móveis em dias.

    min_periods : int, default=1
        Número mínimo de observações na janela para calcular a métrica.

    fill_missing_days : bool, default=True
        Se True, cria datas faltantes dentro de cada grupo para permitir
        rolling diário contínuo.

    Retorno
    -------
    pd.DataFrame
        DataFrame agregado por dia e por grupo, contendo novas features temporais.
    """

    if group_cols is None:
        group_cols = ["Product", "Region"]

    required_cols = {
        "Order_ID", "Customer_ID", "Customer_Type", "Product", "Category",
        "Unit_Price", "Quantity", "Discount", "Total_Price", "Region", date_col
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    dfx = df.copy()

    # 1) Conversão de data
    dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce")
    dfx = dfx.dropna(subset=[date_col])

    # 2) Normalização da data para nível diário
    dfx[date_col] = dfx[date_col].dt.normalize()

    # 3) Agregação diária por granularidade de negócio
    daily = (
        dfx.groupby(group_cols + [date_col], as_index=False)
           .agg(
               quantity_sum=("Quantity", "sum"),
               quantity_mean=("Quantity", "mean"),
               quantity_std=("Quantity", "std"),
               total_price_sum=("Total_Price", "sum"),
               total_price_mean=("Total_Price", "mean"),
               unit_price_mean=("Unit_Price", "mean"),
               discount_mean=("Discount", "mean"),
               order_count=("Order_ID", "nunique"),
               customer_count=("Customer_ID", "nunique")
           )
    )

    # ticket médio do dia
    daily["avg_ticket"] = np.where(
        daily["order_count"] > 0,
        daily["total_price_sum"] / daily["order_count"],
        0
    )

    # Preenche std nulo
    daily["quantity_std"] = daily["quantity_std"].fillna(0)

    # 4) Completa dias faltantes por grupo, se desejado
    if fill_missing_days:
        frames = []

        for keys, g in daily.groupby(group_cols):
            g = g.sort_values(date_col).copy()

            full_dates = pd.date_range(g[date_col].min(), g[date_col].max(), freq="D")
            g = g.set_index(date_col).reindex(full_dates).rename_axis(date_col).reset_index()

            # recoloca as chaves do grupo
            if len(group_cols) == 1:
                g[group_cols[0]] = keys
            else:
                for col, val in zip(group_cols, keys):
                    g[col] = val

            # métricas de dias sem venda
            numeric_fill_zero = [
                "quantity_sum", "quantity_mean", "quantity_std",
                "total_price_sum", "total_price_mean",
                "unit_price_mean", "discount_mean",
                "order_count", "customer_count", "avg_ticket"
            ]

            for col in numeric_fill_zero:
                if col in g.columns:
                    g[col] = g[col].fillna(0)

            frames.append(g)

        daily = pd.concat(frames, ignore_index=True)

    # 5) Ordenação
    daily = daily.sort_values(group_cols + [date_col]).reset_index(drop=True)

    # 6) Features de calendário
    daily["day_of_week"] = daily[date_col].dt.dayofweek
    daily["day"] = daily[date_col].dt.day
    daily["month"] = daily[date_col].dt.month
    daily["year"] = daily[date_col].dt.year
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)

    # 7) Rolling features usando apenas o passado
    # shift(1) evita usar o valor do próprio dia na janela
    metric_cols = [
        "quantity_sum",
        "total_price_sum",
        "unit_price_mean",
        "discount_mean",
        "order_count",
        "avg_ticket"
    ]

    for w in windows:
        for metric in metric_cols:
            daily[f"{metric}_mean_{w}d"] = (
                daily.groupby(group_cols)[metric]
                     .transform(lambda s: s.shift(1).rolling(window=w, min_periods=min_periods).mean())
            )

            daily[f"{metric}_std_{w}d"] = (
                daily.groupby(group_cols)[metric]
                     .transform(lambda s: s.shift(1).rolling(window=w, min_periods=min_periods).std())
            )

            daily[f"{metric}_sum_{w}d"] = (
                daily.groupby(group_cols)[metric]
                     .transform(lambda s: s.shift(1).rolling(window=w, min_periods=min_periods).sum())
            )

    # 8) Features de comparação com histórico recente
    for w in windows:
        daily[f"quantity_vs_mean_{w}d"] = (
            daily["quantity_sum"] - daily[f"quantity_sum_mean_{w}d"]
        )

        daily[f"total_price_vs_mean_{w}d"] = (
            daily["total_price_sum"] - daily[f"total_price_sum_mean_{w}d"]
        )

        daily[f"unit_price_vs_mean_{w}d"] = (
            daily["unit_price_mean"] - daily[f"unit_price_mean_mean_{w}d"]
        )

        daily[f"discount_vs_mean_{w}d"] = (
            daily["discount_mean"] - daily[f"discount_mean_mean_{w}d"]
        )

    # 9) Variação percentual vs histórico
    for w in windows:
        mean_col = f"quantity_sum_mean_{w}d"
        daily[f"quantity_pct_vs_mean_{w}d"] = np.where(
            daily[mean_col].fillna(0) != 0,
            (daily["quantity_sum"] - daily[mean_col]) / daily[mean_col],
            np.nan
        )

    # 10) Preencher std rolling nulo
    rolling_std_cols = [c for c in daily.columns if "_std_" in c]
    for col in rolling_std_cols:
        daily[col] = daily[col].fillna(0)

    return daily