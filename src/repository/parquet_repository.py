from pathlib import Path
import pandas as pd
import pyarrow

class ParquetRepository:
    """
    Classe responsável por leitura e escrita de arquivos Parquet.

    Esta classe abstrai operações comuns de persistência de DataFrames
    em formato Parquet, garantindo:

    - Padronização de caminhos
    - Criação automática de diretórios
    - Controle de sobrescrita
    - Logging básico

    Ideal para pipelines de Machine Learning e Data Engineering.

    Exemplo de uso:
    ----------------
    repo = ParquetRepository(base_path="data")

    repo.save(df, "features/sales_features.parquet")
    df_loaded = repo.load("features/sales_features.parquet")
    """

    def __init__(self, base_path: str = "data"):
        """
        Inicializa o repositório.

        Parameters
        ----------
        base_path : str
            Diretório base onde os arquivos serão armazenados.
        """
        self.base_path = Path(base_path)

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

    def save(
        self,
        df: pd.DataFrame,
        relative_path: str,
        compression: str = "snappy",
        index: bool = False,
        overwrite: bool = True
    ) -> None:
        """
        Salva um DataFrame em formato Parquet.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame a ser salvo.

        relative_path : str
            Caminho relativo do arquivo (ex: "features/data.parquet").

        compression : str
            Tipo de compressão (default: "snappy").

        index : bool
            Se deve salvar o índice.

        overwrite : bool
            Se False, evita sobrescrever arquivo existente.

        Raises
        ------
        FileExistsError
            Caso overwrite=False e o arquivo já exista.
        """
        full_path = self._get_full_path(relative_path)

        # cria diretórios automaticamente
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if full_path.exists() and not overwrite:
            raise FileExistsError(f"Arquivo já existe: {full_path}")

        df.to_parquet(
            full_path,
            compression=compression,
            index=index
        )

        print(f"[OK] Arquivo salvo em: {full_path}")

    def load(
        self,
        relative_path: str,
        columns: list = None
    ) -> pd.DataFrame:
        """
        Carrega um arquivo Parquet como DataFrame.

        Parameters
        ----------
        relative_path : str
            Caminho relativo do arquivo.

        columns : list, opcional
            Lista de colunas para leitura parcial.

        Returns
        -------
        pd.DataFrame
            DataFrame carregado.

        Raises
        ------
        FileNotFoundError
            Caso o arquivo não exista.
        """
        full_path = self._get_full_path(relative_path)

        if not full_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {full_path}")

        df = pd.read_parquet(full_path, columns=columns)

        print(f"[OK] Arquivo carregado: {full_path}")

        return df

    def exists(self, relative_path: str) -> bool:
        """
        Verifica se o arquivo existe.

        Parameters
        ----------
        relative_path : str

        Returns
        -------
        bool
        """
        return self._get_full_path(relative_path).exists()

    def delete(self, relative_path: str) -> None:
        """
        Remove um arquivo Parquet.

        Parameters
        ----------
        relative_path : str
        """
        full_path = self._get_full_path(relative_path)

        if full_path.exists():
            full_path.unlink()
            print(f"[OK] Arquivo removido: {full_path}")
        else:
            print(f"[WARN] Arquivo não encontrado: {full_path}")