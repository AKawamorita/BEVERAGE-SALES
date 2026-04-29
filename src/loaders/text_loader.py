from pathlib import Path
import pandas as pd
import os
import glob
from datetime import datetime
from src import features as ft

class TextLoader:
    """
    Responsável pelo carregamento de dados brutos e extração inicial de características.

    Esta classe gerencia a leitura de arquivos de sinais de vibração (geralmente do 
    Bearing Dataset), a organização por experimentos (runs) e o cálculo do 
    RUL (Remaining Useful Life) teórico.
    """

    # Metodos protegidos 
    def _extract_features_from_file(self, file_path, run_id, time_index, max_sensors=4):
        """
        Lê um arquivo individual de vibração e extrai as estatísticas do sinal.

        Este método é protegido e coordena a chamada para o extrator de features 
        para cada sensor disponível no arquivo.

        Args:
            file_path (str): Caminho completo para o arquivo de texto bruto.
            run_id (str): Identificador do experimento atual.
            time_index (int): Índice temporal baseado na ordem de leitura do arquivo.
            max_sensors (int, opcional): Número máximo de colunas (sensores) a processar. 
                Padrão é 4.

        Returns:
            dict: Um dicionário contendo metadados (run_id, time_index) e todas 
                as features estatísticas calculadas para os sensores.
        """

        df = self.read_bearing_file(file_path)
        row_features = {
            "run_id": run_id,
            "time_index": time_index,
            "file_name": os.path.basename(file_path)
        }

        n_sensors = min(df.shape[1], max_sensors)

        for sensor_idx in range(n_sensors):
            signal = df.iloc[:, sensor_idx].values
            sensor_prefix = f"s{sensor_idx+1}_"
            sensor_feats = ft.extract_signal_features(signal, prefix=sensor_prefix)
            row_features.update(sensor_feats)

        return row_features

    # Metodos publicos
    def read_bearing_file(self, file_path):
        """
        Executa a leitura bruta de um arquivo de vibração usando delimitadores de espaço.

        No contexto do IMS Bearing Dataset, os arquivos não possuem cabeçalho e 
        os valores são separados por tabulações ou espaços múltiplos.

        Args:
            file_path (str): Caminho para o arquivo.

        Returns:
            pd.DataFrame: DataFrame bruto com as leituras dos sensores.
        """
        df = pd.read_csv(file_path, sep=r"\s+", header=None)
        return df


    def create_dataframe_old(self, run_folder, run_id, max_sensors=4):
        """
        Processa uma pasta inteira de arquivos, consolidando-os em um único DataFrame.

        Além de extrair as características de cada arquivo, este método calcula a 
        coluna alvo 'RUL' (Remaining Useful Life) de forma linear, baseada no 
        tempo restante até o último arquivo da pasta.

        Args:
            run_folder (str): Caminho da pasta que contém os arquivos do experimento.
            run_id (str): Nome identificador para este conjunto de dados.
            max_sensors (int, opcional): Limite de sensores a serem lidos por arquivo.

        Returns:
            pd.DataFrame: DataFrame estruturado com features e a coluna alvo 'RUL'.
        """

        files = sorted(glob.glob(os.path.join(run_folder, "*")))
        print(f"\nRun: {run_id}")
        print(f"Pasta: {run_folder}")
        print(f"Qtd. arquivos encontrados: {len(files)}")

        rows = []
        for time_index, file_path in enumerate(files):
            row = self._extract_features_from_file(
                file_path=file_path,
                run_id=run_id,
                time_index=time_index,
                max_sensors=max_sensors
            )
            rows.append(row)

        print(f"Qtd. linhas geradas: {len(rows)}")

        df_run = pd.DataFrame(rows)
        print("Colunas do df_run:", df_run.columns.tolist())

        max_time = df_run["time_index"].max()
        df_run["RUL"] = max_time - df_run["time_index"]

        return df_run
    
    def create_dataframe(self, run_folder, run_id, max_sensors=4):
        """
        Processa uma pasta inteira de arquivos, consolidando-os em um único DataFrame.

        Além de extrair as características de cada arquivo, este método calcula a 
        coluna alvo 'RUL' (Remaining Useful Life) de forma linear, baseada no 
        tempo restante até o último arquivo da pasta.

        Args:
            run_folder (str): Caminho da pasta que contém os arquivos do experimento.
            run_id (str): Nome identificador para este conjunto de dados.
            max_sensors (int, opcional): Limite de sensores a serem lidos por arquivo.

        Returns:
            pd.DataFrame: DataFrame estruturado com features e a coluna alvo 'RUL'.
        """

        files = sorted(glob.glob(os.path.join(run_folder, "*")))
        print(f"\nRun: {run_id}")
        print(f"Pasta: {run_folder}")
        print(f"Qtd. arquivos encontrados: {len(files)}")

        rows = []
        for time_index, file_path in enumerate(files):

            # EXTRAÇÃO DA DATA: Pega o nome do arquivo (ex: 2003.10.22.12.06.24)
            file_name = os.path.basename(file_path)
            try:
                # Converte o nome do arquivo em um objeto datetime real
                timestamp = datetime.strptime(file_name, '%Y.%m.%d.%H.%M.%S')
            except ValueError:
                # Caso o nome do arquivo não seja uma data (ex: .DS_Store ou logs)
                timestamp = None

            row = self._extract_features_from_file(
                file_path=file_path,
                run_id=run_id,
                time_index=time_index,
                max_sensors=max_sensors
            )
            # Adiciona a data extraída ao dicionário da linha
            row['timestamp'] = timestamp
            rows.append(row)

        print(f"Qtd. linhas geradas: {len(rows)}")
        df_run = pd.DataFrame(rows)

        print("Colunas do df_run:", df_run.columns.tolist())

        # CONVERSÃO E INDICE: Essencial para a função de alerta
        if 'timestamp' in df_run.columns:
            df_run['timestamp'] = pd.to_datetime(df_run['timestamp'])
            df_run.set_index('timestamp', inplace=True)

        max_time = df_run["time_index"].max()
        df_run["RUL"] = max_time - df_run["time_index"]

        return df_run
    
    