import pandas as pd
import matplotlib.pyplot as plt


class SalesTimeOutlier:
    """
    Gera gráfico temporal para análise de possíveis outliers em vendas.

    A classe foi pensada para análise de séries temporais em datasets de vendas,
    permitindo filtrar os dados por colunas categóricas como Product, Category
    e Region, além de restringir o período por data inicial e final.

    O gráfico resultante combina:
    - linha temporal da métrica agregada por data
    - pontos das observações agregadas
    - faixa de referência estatística
    - destaque visual para possíveis outliers

    A detecção de outliers pode ser feita por:
    - IQR (Intervalo Interquartil)
    - Z-score

    A faixa de referência pode ser:
    - global no período filtrado
    - móvel/rolling, caso seja informado `moving_window`

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original contendo os dados de vendas.
    date_col : str, default="Order_Date"
        Nome da coluna de data usada no eixo temporal.

    Example
    -------
    >>> analyzer = SalesTimeOutlierPlot(df)
    >>> result = analyzer.plot_outliers_by_period(
    ...     value_col="Quantity",
    ...     filters={
    ...         "Category": ["Water"],
    ...         "Product": ["Evian"],
    ...         "Region": ["North"]
    ...     },
    ...     start_date="2022-12-01",
    ...     end_date="2022-12-31",
    ...     agg="sum",
    ...     method="iqr"
    ... )
    """

    def __init__(self, df: pd.DataFrame, date_col: str = "Order_Date"):
        """
        Inicializa a classe e prepara o DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame com os dados brutos.
        date_col : str, default="Order_Date"
            Nome da coluna que representa a data.
        """
        self.df = df.copy()
        self.date_col = date_col
        self._prepare_dataframe()

    def _prepare_dataframe(self):
        """
        Prepara o DataFrame para análise temporal.

        Etapas executadas:
        - converte a coluna de data para datetime
        - remove linhas com datas inválidas
        - ordena os registros pela data
        """
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce")
        self.df = self.df.dropna(subset=[self.date_col]).copy()
        self.df = self.df.sort_values(self.date_col).reset_index(drop=True)

    def _filter_data(
        self,
        value_col: str,
        filters: dict = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Filtra o DataFrame com base em coluna numérica, período e filtros categóricos.

        Parameters
        ----------
        value_col : str
            Coluna numérica a ser analisada, por exemplo:
            'Quantity', 'Total_Price', 'Discount', 'Unit_Price'.
        filters : dict, optional
            Dicionário com filtros por coluna.
            Cada chave é o nome da coluna e o valor pode ser:
            - valor único
            - lista de valores

            Exemplo:
            {
                "Category": ["Water", "Juices"],
                "Region": ["North", "South"],
                "Product": ["Evian"]
            }

        start_date : str, optional
            Data inicial do filtro, no formato compatível com pandas datetime.
        end_date : str, optional
            Data final do filtro, no formato compatível com pandas datetime.

        Returns
        -------
        pd.DataFrame
            DataFrame filtrado.

        Raises
        ------
        ValueError
            Se a coluna `value_col` não existir.
            Se alguma coluna de filtro não existir no DataFrame.
        """
        if value_col not in self.df.columns:
            raise ValueError(f"A coluna '{value_col}' não existe no DataFrame.")

        filtered_df = self.df.copy()

        if start_date is not None:
            filtered_df = filtered_df[
                filtered_df[self.date_col] >= pd.to_datetime(start_date)
            ]

        if end_date is not None:
            filtered_df = filtered_df[
                filtered_df[self.date_col] <= pd.to_datetime(end_date)
            ]

        if filters is not None:
            for col, values in filters.items():
                if col not in filtered_df.columns:
                    raise ValueError(f"A coluna '{col}' não existe no DataFrame.")

                if values is None:
                    continue

                if not isinstance(values, (list, tuple, set)):
                    values = [values]

                filtered_df = filtered_df[filtered_df[col].isin(values)]

        filtered_df = filtered_df.dropna(subset=[value_col]).copy()
        return filtered_df

    def plot_outliers_by_period(
        self,
        value_col: str,
        filters: dict = None,
        start_date: str = None,
        end_date: str = None,
        agg: str = "sum",
        method: str = "iqr",
        iqr_multiplier: float = 1.5,
        z_threshold: float = 3.0,
        moving_window: int = None,
        figsize=(13, 6),
        line_alpha: float = 0.75,
        point_size: int = 45,
        rotation: int = 45,
        pastel_band_color: str = "#D9EAF7",
        line_color: str = "#6B7280",
        point_color: str = "#5B8DB8",
        outlier_color: str = "#D97706",
        show_reference_line: bool = True
    ) -> pd.DataFrame:
        """
        Plota série temporal com faixa de referência e destaque de outliers.

        O método:
        1. filtra os dados
        2. agrega por data
        3. calcula faixa de referência estatística
        4. identifica outliers
        5. desenha gráfico com linha, pontos e banda

        Parameters
        ----------
        value_col : str
            Coluna numérica a ser analisada.
        filters : dict, optional
            Filtros categóricos em formato de dicionário.
        start_date : str, optional
            Data inicial do período.
        end_date : str, optional
            Data final do período.
        agg : str, default="sum"
            Tipo de agregação por dia:
            - "sum"    : soma diária
            - "mean"   : média diária
            - "median" : mediana diária
            - "count"  : contagem diária
        method : str, default="iqr"
            Método estatístico para detectar outliers:
            - "iqr"
            - "zscore"
        iqr_multiplier : float, default=1.5
            Multiplicador usado no cálculo do limite IQR.
            Regra clássica:
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
        z_threshold : float, default=3.0
            Número de desvios padrão para definir outlier no método z-score.
        moving_window : int, optional
            Janela móvel para cálculo dinâmico da faixa de referência.
            Exemplo: 7 cria uma faixa baseada nos últimos 7 pontos.
            Se None, a faixa será global no período.
        figsize : tuple, default=(13, 6)
            Tamanho da figura.
        line_alpha : float, default=0.75
            Transparência da linha principal.
        point_size : int, default=45
            Tamanho dos pontos.
        rotation : int, default=45
            Rotação dos rótulos do eixo X.
        pastel_band_color : str, default="#D9EAF7"
            Cor da faixa de referência.
        line_color : str, default="#6B7280"
            Cor da linha principal.
        point_color : str, default="#5B8DB8"
            Cor dos pontos normais.
        outlier_color : str, default="#D97706"
            Cor dos pontos classificados como outliers.
        show_reference_line : bool, default=True
            Se True, exibe a linha central de referência.

        Returns
        -------
        pd.DataFrame
            DataFrame agregado por data com colunas adicionais:
            - value_col
            - lower_limit
            - upper_limit
            - reference
            - is_outlier

        Raises
        ------
        ValueError
            Se `agg` ou `method` forem inválidos.
        """
        plot_df = self._filter_data(
            value_col=value_col,
            filters=filters,
            start_date=start_date,
            end_date=end_date
        )

        if plot_df.empty:
            print("Nenhum dado disponível após os filtros aplicados.")
            return pd.DataFrame()

        # Agrega os valores por data.
        if agg == "sum":
            daily = plot_df.groupby(self.date_col, as_index=False)[value_col].sum()
        elif agg == "mean":
            daily = plot_df.groupby(self.date_col, as_index=False)[value_col].mean()
        elif agg == "median":
            daily = plot_df.groupby(self.date_col, as_index=False)[value_col].median()
        elif agg == "count":
            daily = plot_df.groupby(self.date_col, as_index=False)[value_col].count()
        else:
            raise ValueError("agg deve ser: 'sum', 'mean', 'median' ou 'count'.")

        daily = daily.sort_values(self.date_col).reset_index(drop=True)

        if daily.empty:
            print("Nenhum dado disponível para plotagem após agregação.")
            return pd.DataFrame()

        y = daily[value_col]

        # =========================================================
        # Cálculo da faixa de referência
        # =========================================================
        # Caso moving_window seja informado, a faixa será dinâmica.
        # Caso contrário, a faixa será única para todo o período.
        # =========================================================
        if moving_window is not None and moving_window >= 2:

            if method == "iqr":
                ref_center = y.rolling(window=moving_window, min_periods=1).median()
                q1 = y.rolling(window=moving_window, min_periods=1).quantile(0.25)
                q3 = y.rolling(window=moving_window, min_periods=1).quantile(0.75)
                iqr = q3 - q1
                lower = q1 - iqr_multiplier * iqr
                upper = q3 + iqr_multiplier * iqr

            elif method == "zscore":
                rolling_mean = y.rolling(window=moving_window, min_periods=1).mean()
                rolling_std = y.rolling(window=moving_window, min_periods=1).std().fillna(0)
                ref_center = rolling_mean
                lower = rolling_mean - z_threshold * rolling_std
                upper = rolling_mean + z_threshold * rolling_std

            else:
                raise ValueError("method deve ser 'iqr' ou 'zscore'.")

        else:
            if method == "iqr":
                q1 = y.quantile(0.25)
                q3 = y.quantile(0.75)
                iqr = q3 - q1

                lower = pd.Series(
                    [q1 - iqr_multiplier * iqr] * len(daily),
                    index=daily.index
                )
                upper = pd.Series(
                    [q3 + iqr_multiplier * iqr] * len(daily),
                    index=daily.index
                )
                ref_center = pd.Series([y.median()] * len(daily), index=daily.index)

            elif method == "zscore":
                mean_ = y.mean()
                std_ = y.std()

                if pd.isna(std_) or std_ == 0:
                    std_ = 0.0

                lower = pd.Series(
                    [mean_ - z_threshold * std_] * len(daily),
                    index=daily.index
                )
                upper = pd.Series(
                    [mean_ + z_threshold * std_] * len(daily),
                    index=daily.index
                )
                ref_center = pd.Series([mean_] * len(daily), index=daily.index)

            else:
                raise ValueError("method deve ser 'iqr' ou 'zscore'.")

        # Salva os limites e a classificação no DataFrame final.
        daily["lower_limit"] = lower
        daily["upper_limit"] = upper
        daily["reference"] = ref_center
        daily["is_outlier"] = (
            (daily[value_col] < daily["lower_limit"]) |
            (daily[value_col] > daily["upper_limit"])
        )

        # =========================================================
        # Plot
        # =========================================================
        fig, ax = plt.subplots(figsize=figsize)

        # Faixa de referência
        ax.fill_between(
            daily[self.date_col],
            daily["lower_limit"],
            daily["upper_limit"],
            alpha=0.35,
            color=pastel_band_color,
            label="Faixa de referência"
        )

        # Linha principal
        ax.plot(
            daily[self.date_col],
            daily[value_col],
            linestyle="-",
            alpha=line_alpha,
            color=line_color,
            linewidth=1.5,
            label=f"{value_col} ({agg} diário)"
        )

        # Pontos normais
        normal_df = daily[~daily["is_outlier"]]
        ax.scatter(
            normal_df[self.date_col],
            normal_df[value_col],
            s=point_size,
            color=point_color,
            alpha=0.85,
            label="Pontos"
        )

        # Pontos outliers
        out_df = daily[daily["is_outlier"]]
        if not out_df.empty:
            ax.scatter(
                out_df[self.date_col],
                out_df[value_col],
                s=point_size * 1.4,
                color=outlier_color,
                alpha=0.95,
                label="Outliers"
            )

        # Linha central de referência
        if show_reference_line:
            ax.plot(
                daily[self.date_col],
                daily["reference"],
                linestyle="--",
                linewidth=1.4,
                color="#7C3AED",
                alpha=0.85,
                label="Linha de referência"
            )

        # Monta o título dinamicamente
        title = f"Outliers de {value_col} por data"
        if filters:
            filtros_txt = " | ".join([f"{k}={v}" for k, v in filters.items()])
            title += f" | {filtros_txt}"
        if start_date or end_date:
            title += " | período filtrado"

        ax.set_title(title)
        ax.set_xlabel("Data")
        ax.set_ylabel(value_col)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.xticks(rotation=rotation)
        ax.legend()
        plt.tight_layout()
        plt.show()

        return daily