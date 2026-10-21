"""Orquestra todo o fluxo: dados -> divisão -> treino -> métricas."""

from dataclasses import dataclass

import pandas as pd

from src.dados import carregar_dados, dividir_dados, estruturar_dados
from src.modelos import Classificador, Clusterizador, Regressor


@dataclass
class Resultado:
    treino: pd.DataFrame
    validacao: pd.DataFrame
    teste: pd.DataFrame
    clusterizador: Clusterizador
    classificador: Classificador
    regressor: Regressor
    selecao_cluster: dict[str, float]
    selecao_classif: dict[str, float]
    selecao_regr: dict[str, float]


def treinar_tudo() -> Resultado:
    """Executa o projeto de ponta a ponta e devolve modelos treinados."""
    # as 1000 linhas são usadas (sem remover duplicatas)
    df = estruturar_dados(carregar_dados())
    treino, validacao, teste = dividir_dados(df)

    clusterizador = Clusterizador()
    selecao_cluster = clusterizador.treinar(treino, validacao)

    classificador = Classificador()
    selecao_classif = classificador.treinar(treino, validacao)

    regressor = Regressor()
    selecao_regr = regressor.treinar(treino, validacao)

    return Resultado(
        treino,
        validacao,
        teste,
        clusterizador,
        classificador,
        regressor,
        selecao_cluster,
        selecao_classif,
        selecao_regr,
    )
