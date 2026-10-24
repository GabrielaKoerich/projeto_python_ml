"""Testes da seleção de variáveis feita a partir do treino."""

import numpy as np
import pandas as pd
import pytest

from src.dados import carregar_dados, dividir_dados, estruturar_dados
from src import selecao as sel
from src.modelos import _preprocessador
from src.selecao import (
    agrupar_categoricas,
    aplicar_log,
    grupos_correlacionados,
    selecionar,
    v_de_cramer,
    variaveis_assimetricas,
)


@pytest.fixture(scope="module")
def treino() -> pd.DataFrame:
    return dividir_dados(estruturar_dados(carregar_dados()))[0]


@pytest.fixture(scope="module")
def selecao(treino):
    return selecionar(treino)


def test_assimetricas_recebem_log():
    # Given: uma variável com cauda longa e outra simétrica
    df = pd.DataFrame(
        {"cauda": [1, 1, 1, 2, 2, 3, 50, 100], "simetrica": [1, 2, 3, 4, 5, 6, 7, 8]}
    )
    # Then: só a assimétrica é escolhida para o log
    assert variaveis_assimetricas(df) == ["cauda"]
    resultado = aplicar_log(df, ["cauda"])
    assert "cauda" not in resultado.columns
    assert resultado["cauda_log"].iloc[-1] == pytest.approx(np.log1p(100))


def test_alvo_da_regressao_continua_na_escala_original():
    df = pd.DataFrame({"IFN-γ": [1.0, 2.0, 100.0]})
    resultado = aplicar_log(df, ["IFN-γ"])
    assert {"IFN-γ", "IFN-γ_log"} <= set(resultado.columns)


def test_grupos_correlacionados_em_cadeia():
    # a ~ b e b ~ c formam um único grupo; d fica sozinho
    base = np.arange(20, dtype=float)
    x = pd.DataFrame(
        {
            "a": base,
            "b": base * 2,
            "c": base**2,
            "d": np.random.default_rng(0).random(20),
        }
    )
    grupos = grupos_correlacionados(x)
    assert len(grupos) == 1
    assert set(grupos[0]) == {"a", "b", "c"}


def test_teste_f_ordena_pela_separacao():
    y = pd.Series(["A"] * 5 + ["B"] * 5)
    x = pd.DataFrame({"separa": [0] * 5 + [10] * 5, "ruido": [1, 2, 3, 4, 5] * 2})
    tabela = sel.teste_f(x + np.arange(10).reshape(-1, 1) * 0.01, y)
    assert tabela.index[0] == "separa"


def test_v_de_cramer_extremos():
    y = pd.Series(["A", "A", "B", "B"])
    assert v_de_cramer(pd.Series(["x", "x", "y", "y"]), y) == pytest.approx(1.0)
    assert v_de_cramer(pd.Series(["x", "x", "x", "x"]), y) == 0.0


def test_categoricas_com_mesmo_padrao_ficam_juntas(treino):
    grupos = agrupar_categoricas(treino)
    # dois padrões: um marca o BCG e outro a imunodeficiência
    assert len(grupos) == 2
    assert sum(len(g) for g in grupos.values()) == 7


def test_selecao_usa_variaveis_do_treino(selecao, treino):
    assert len(selecao.variaveis) >= 2
    # toda variável escolhida separa as classes ou é categórica representante
    for v in selecao.numericas:
        assert v in selecao.separam
    # do bloco correlacionado, sobra no máximo uma variável
    for representante, grupo in selecao.representantes.items():
        assert len(set(selecao.variaveis) & set(grupo)) <= 1
        assert representante in grupo


def test_regressao_nao_usa_o_alvo(selecao):
    assert "IFN-γ" not in selecao.regressao
    assert "IFN-γ_log" not in selecao.regressao


def test_preprocessador_com_categoricas():
    # categóricas viram dummies e faltantes são preenchidos
    df = pd.DataFrame({"num": [1.0, np.nan, 3.0], "cat": ["x", "y", np.nan]})
    resultado = _preprocessador(["num"], ["cat"]).fit_transform(df)
    assert resultado.shape == (3, 3)
    assert not np.isnan(resultado).any()
