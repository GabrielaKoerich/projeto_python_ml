"""Aquisição, estruturação e divisão dos dados de biomarcadores."""

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import validate_call
from sklearn.model_selection import train_test_split

CAMINHO_PADRAO = Path(__file__).resolve().parent.parent / "dados" / "biomarcadores.csv"

# Variáveis escolhidas na seleção de variáveis
VARIAVEIS_NUMERICAS = ["TP53", "IFN-γ", "CD3_2", "CD8", "PPD_log", "CD7"]
VARIAVEIS_CATEGORICAS = ["FOXP3"]
ALVO_CLASSIFICACAO = "classe"
ALVO_REGRESSAO = "IFN-γ"
VARIAVEIS_CLUSTER = ["PPD_log", "IFN-γ", "CD3_2"]


@validate_call
def carregar_dados(caminho: str = str(CAMINHO_PADRAO)) -> pd.DataFrame:
    """Lê o CSV bruto de biomarcadores.

    Parametros:
        caminho (str): caminho do arquivo CSV.
    Retorna:
        pd.DataFrame: dados brutos.
    """
    return pd.read_csv(caminho)


def remover_duplicatas(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém uma única linha por paciente.

    O banco tem 1000 linhas, mas só 400 pacientes únicos. Usada apenas na
    seção 9 da AED, para mostrar o efeito das duplicatas. Os modelos seguem
    a metodologia da disciplina e usam as 1000 linhas (limitação discutida
    na AED).

    Parametros:
        df (pd.DataFrame): dados brutos.
    Retorna:
        pd.DataFrame: dados sem linhas repetidas.
    """
    return df.drop_duplicates().reset_index(drop=True)


def estruturar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Remove constantes, trata faltantes e cria PPD_log.

    Parametros:
        df (pd.DataFrame): dados brutos.
    Retorna:
        pd.DataFrame: dados prontos para modelagem.
    """
    df = df.copy()

    # 1. remove colunas constantes (apenas 1 valor distinto)
    constantes = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    df = df.drop(columns=constantes)

    # 2. PPD é muito assimétrica -> log, e faltantes preenchidos com a média
    df["PPD_log"] = np.log1p(df["PPD"])
    df["PPD_log"] = df["PPD_log"].fillna(df["PPD_log"].mean())

    # 3. demais numéricas faltantes -> mediana
    for coluna in df.select_dtypes(include="number").columns:
        df[coluna] = df[coluna].fillna(df[coluna].median())

    # 4. mantme apenas as variáveis selecionadas + alvo
    colunas = VARIAVEIS_NUMERICAS + VARIAVEIS_CATEGORICAS + [ALVO_CLASSIFICACAO]
    return df[colunas].reset_index(drop=True)


def dividir_dados(
    df: pd.DataFrame, semente: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide em treino (70%), validação (15%) e teste (15%), estratificado.

    Parametros:
        df (pd.DataFrame): dados estruturados.
        semente (int): semente aleatória (reprodutibilidade).
    Retorna:
        tuple: (treino, validacao, teste)
    """
    treino, resto = train_test_split(
        df, test_size=0.30, stratify=df[ALVO_CLASSIFICACAO], random_state=semente
    )
    validacao, teste = train_test_split(
        resto, test_size=0.50, stratify=resto[ALVO_CLASSIFICACAO], random_state=semente
    )
    return treino, validacao, teste
