"""Aquisição, estruturação e divisão dos dados de biomarcadores."""

from pathlib import Path

import pandas as pd
from pydantic import validate_call
from sklearn.model_selection import train_test_split

CAMINHO_PADRAO = Path(__file__).resolve().parent.parent / "dados" / "biomarcadores.csv"

# As variáveis de entrada NÃO são fixadas aqui: são escolhidas a partir do
# treino em src/selecao.py. Aqui ficam só os alvos da pergunta do projeto.
ALVO_CLASSIFICACAO = "classe"
ALVO_REGRESSAO = "IFN-γ"


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
    seção 9 da AED, para mostrar o efeito das duplicatas.

    Parametros:
        df (pd.DataFrame): dados brutos.
    Retorna:
        pd.DataFrame: dados sem linhas repetidas.
    """
    return df.drop_duplicates().reset_index(drop=True)


def estruturar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Remove constantes e converte as variáveis lógicas em 0/1.

    Faltantes, transformações e escolha de variáveis NÃO são feitos aqui,
    porque dependem de estatísticas (mediana, assimetria, correlação) que
    devem ser calculadas só no treino, depois da divisão.

    Parametros:
        df (pd.DataFrame): dados brutos.
    Retorna:
        pd.DataFrame: dados com todas as variáveis candidatas.
    """
    df = df.copy()

    # 1. remove colunas constantes (apenas 1 valor distinto): não distinguem
    #    nenhum paciente
    constantes = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    df = df.drop(columns=constantes)

    # 2. variáveis lógicas (True/False) viram 0/1, para entrarem nos cálculos
    for coluna in df.columns:
        if set(df[coluna].dropna().unique()) <= {True, False}:
            df[coluna] = df[coluna].astype(float)

    return df.reset_index(drop=True)


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
