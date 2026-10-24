"""Testes unitários das funções de dados."""

import pandas as pd
import pytest
from pydantic import ValidationError

from src.dados import (
    ALVO_CLASSIFICACAO,
    carregar_dados,
    dividir_dados,
    estruturar_dados,
    remover_duplicatas,
)


@pytest.fixture(scope="module")
def dados_brutos() -> pd.DataFrame:
    return carregar_dados()


def test_carregar_dados_formato(dados_brutos):
    # Given / When: o CSV é carregado
    # Then: tem 1000 linhas e a coluna alvo
    assert dados_brutos.shape[0] == 1000
    assert ALVO_CLASSIFICACAO in dados_brutos.columns


def test_carregar_dados_tipo_errado():
    # um número no lugar do caminho deve ser barrado pelo pydantic
    with pytest.raises(ValidationError):
        carregar_dados(caminho=10)


def test_estruturar_remove_constantes_e_converte_logicas(dados_brutos):
    df = estruturar_dados(dados_brutos)
    # nenhuma coluna constante sobra
    assert (df.nunique(dropna=True) > 1).all()
    # variáveis lógicas viram 0/1
    assert set(df["TdT"].unique()) <= {0.0, 1.0}
    # faltantes NÃO são preenchidos aqui (isso é feito no treino, no Pipeline)
    assert df.isna().sum().sum() > 0


def test_estruturar_nao_altera_original(dados_brutos):
    copia = dados_brutos.copy()
    estruturar_dados(dados_brutos)
    pd.testing.assert_frame_equal(copia, dados_brutos)


def test_remover_duplicatas(dados_brutos):
    # Given: 1000 linhas brutas, mas só 400 pacientes únicos
    assert dados_brutos.duplicated().sum() == 600
    # When / Then: sobra 1 linha por paciente, 100 por classe
    df = remover_duplicatas(dados_brutos)
    assert len(df) == 400
    assert (df["classe"].value_counts() == 100).all()


def test_estruturar_nao_remove_linhas(dados_brutos):
    # estruturar_dados não mexe nas linhas: a remoção é feita à parte
    assert len(estruturar_dados(dados_brutos)) == 1000


def test_dividir_proporcoes(dados_brutos):
    # 1000 linhas, sem remover duplicatas
    treino, validacao, teste = dividir_dados(estruturar_dados(dados_brutos))
    assert len(treino) == 700
    assert len(validacao) == 150
    assert len(teste) == 150
    # Nenhuma linha repetida entre os conjuntos (pelos índices)
    assert set(treino.index).isdisjoint(teste.index)
    assert set(validacao.index).isdisjoint(teste.index)


def test_sem_vazamento_apos_remover_duplicatas(dados_brutos):
    # Se as duplicatas forem removidas, nenhum
    # paciente do teste ou da validação tem cópia idêntica no treino
    df = estruturar_dados(remover_duplicatas(dados_brutos))
    treino, validacao, teste = dividir_dados(df)
    juntos = pd.concat([treino, validacao, teste])
    assert not juntos.duplicated().any()
