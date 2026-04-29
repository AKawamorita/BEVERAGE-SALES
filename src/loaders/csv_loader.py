from pathlib import Path
import pandas as pd
import os
import glob
from datetime import datetime

class CSVLoader:
    """
    Responsável pelo carregamento de dados brutos e extração inicial de características.

    Esta classe gerencia a leitura de arquivos de sinais de vibração (geralmente do 
    Bearing Dataset), a organização por experimentos (runs) e o cálculo do 
    RUL (Remaining Useful Life) teórico.
    """

    #df = pd.read_csv("DataSet/creditcard.csv")

    # Metodos protegidos 
    def _extract_features_from_csvfile(self, file_path, csv_file):
        """
        Lê um arquivo individual , do tipo csv

        Este método é protegido e coordena a chamada para o extrator de features 

        Args:
            file_path (str): Caminho completo para o arquivo de texto bruto.
            csv_file (str): Caminho da pasta que contém os arquivos do experimento.

        Returns:
            dict: Um dicionário contendo metadados e todas as features especificas para esta situacao
        """

        df = pd.read_csv(file_path)
        return df
    
    def create_dataframe(self, file_path, csv_file):
        """
        Processa um arquivo csv, consolidando-os em um único DataFrame.
        
        Nota
            Caminho da pasta que contém os arquivos do experimento, se encontra no arquivo config.py

        Args:
            file_path (str): Caminho completo para o arquivo de texto bruto.
            csv_file (str): Caminho da pasta que contém os arquivos do experimento.

        Returns:
            pd.DataFrame: DataFrame estruturado com features 
        """
        arquivo = Path(file_path / csv_file) 

        df = pd.read_csv(arquivo)
        return df