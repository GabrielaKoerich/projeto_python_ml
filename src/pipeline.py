"""Orquestra todo o fluxo: dados -> divisão -> seleção -> treino -> métricas."""

from dataclasses import dataclass

import pandas as pd

from src.dados import carregar_dados, dividir_dados, estruturar_dados
from src.modelos import Classificador, Clusterizador, Regressor
from src.selecao import Selecao, aplicar_log, selecionar


@dataclass
class Resultado:
    treino: pd.DataFrame
    validacao: pd.DataFrame
    teste: pd.DataFrame
    selecao: Selecao
    clusterizador: Clusterizador
    classificador: Classificador
    regressor: Regressor
    selecao_cluster: dict[str, float]
    selecao_classif: dict[str, float]
    selecao_regr: dict[str, float]


def treinar_tudo() -> Resultado:
    """Executa o projeto e devolve modelos treinados."""
    # as 1000 linhas são usadas (sem remover duplicatas; limitação na AED)
    df = estruturar_dados(carregar_dados())
    treino, validacao, teste = dividir_dados(df)

    # a seleção de variáveis olha SÓ o treino; a mesma transformação
    # (log nas variáveis assimétricas do treino) vale para as três partes
    selecao = selecionar(treino)
    treino, validacao, teste = (
        aplicar_log(parte, selecao.assimetricas) for parte in (treino, validacao, teste)
    )

    clusterizador = Clusterizador(selecao.numericas)
    selecao_cluster = clusterizador.treinar(treino, validacao)

    classificador = Classificador(selecao.numericas, selecao.categoricas)
    selecao_classif = classificador.treinar(treino, validacao)

    regressor = Regressor(selecao.regressao)
    selecao_regr = regressor.treinar(treino, validacao)

    return Resultado(
        treino,
        validacao,
        teste,
        selecao,
        clusterizador,
        classificador,
        regressor,
        selecao_cluster,
        selecao_classif,
        selecao_regr,
    )
