"""Testes dos modelos e de integração do pipeline completo."""

import numpy as np
import pandas as pd
import pytest

from src.modelos import Classificador, Clusterizador, Regressor
from src.paginas import conclusao, interpretar
from src.pipeline import treinar_tudo


@pytest.fixture(scope="module")
def resultado():
    # Integração: dados -> divisão -> treino de todos os algoritmos
    return treinar_tudo()


def test_prever_sem_treinar_gera_erro():
    for modelo in (Clusterizador(), Classificador(), Regressor()):
        with pytest.raises(RuntimeError):
            modelo.prever(pd.DataFrame())


def test_modelo_inexistente_gera_erro(resultado):
    for modelo in (
        resultado.classificador,
        resultado.regressor,
        resultado.clusterizador,
    ):
        with pytest.raises(KeyError):
            modelo.prever(resultado.teste, "Modelo que não existe")


def test_todos_os_classificadores_foram_treinados(resultado):
    assert set(resultado.classificador.modelos) == set(Classificador.CANDIDATOS)


def test_tabela_classificacao_tem_todos_os_modelos(resultado):
    tabela = resultado.classificador.tabela_resultados(resultado.teste)
    assert len(tabela) == len(Classificador.CANDIDATOS)
    assert tabela["p-valor (qui²)"].between(0, 1).all()


def test_classificadores_superam_acaso(resultado):
    # 4 classes -> acaso = 0.25; todo algoritmo deve ser bem melhor
    for nome in resultado.classificador.modelos:
        metricas = resultado.classificador.avaliar(resultado.teste, nome)
        assert metricas["acuracia"] > 0.5, nome


def test_matriz_confusao_soma_teste(resultado):
    for nome in resultado.classificador.modelos:
        matriz = resultado.classificador.matriz_confusao(resultado.teste, nome)
        assert matriz.values.sum() == len(resultado.teste)


def test_tabela_regressao_tem_todos_os_modelos(resultado):
    tabela = resultado.regressor.tabela_resultados(resultado.teste)
    assert len(tabela) == len(Regressor.CANDIDATOS)
    assert (tabela["RMSE teste"] >= 0).all()
    assert tabela["p-valor (Mann-Whitney)"].between(0, 1).all()


def test_tabela_clusterizacao_tem_todos_os_modelos(resultado):
    tabela = resultado.clusterizador.tabela_resultados(resultado.teste)
    assert set(tabela["Modelo"]) == set(Clusterizador.GRADES)
    assert tabela["Silhueta teste"].between(-1, 1).all()
    assert tabela["p-valor (qui²)"].between(0, 1).all()


def test_melhores_parametros_sem_prefixo(resultado):
    params = resultado.classificador.melhores_parametros("k-NN")
    assert "n_neighbors" in params


def test_novo_paciente(resultado):
    novo = resultado.teste.head(1)
    assert len(resultado.classificador.prever(novo)) == 1
    assert len(resultado.regressor.prever(novo)) == 1
    assert len(resultado.clusterizador.prever(novo)) == 1


def test_silhueta_com_um_grupo_so():
    # Given: todos os pontos no mesmo grupo
    # Then: silhueta não é definida e retornamos -1
    assert Clusterizador._silhueta(pd.DataFrame({"a": [1, 2]}).values, [0, 0]) == -1


def test_interpretar_p_valor():
    assert "não rejeita" in interpretar(0.5)
    assert "rejeita H0 (predito difere" in interpretar(0.01)


def test_pipeline_usa_as_1000_linhas(resultado):
    total = len(resultado.treino) + len(resultado.validacao) + len(resultado.teste)
    assert total == 1000


def test_contingencia_e_homogeneidade(resultado):
    c = resultado.classificador
    tabela = c.tabela_contingencia(resultado.treino, resultado.teste, "GBM")
    assert list(tabela.columns) == ["Teste", "Treino"]
    assert tabela["Teste"].sum() == len(resultado.teste)
    assert tabela["Treino"].sum() == len(resultado.treino)
    teste = c.teste_homogeneidade(resultado.treino, resultado.teste, "GBM")
    assert teste["graus_liberdade"] == 3
    assert 0.0 <= teste["p_valor"] <= 1.0


def test_relatorio_classificacao(resultado):
    relatorio = resultado.classificador.relatorio(resultado.teste, "k-NN")
    assert {"precision", "recall", "f1-score"} <= set(relatorio.columns)
    assert "accuracy" in relatorio.index


def test_quatro_testes_da_regressao(resultado):
    testes = resultado.regressor.testes_treino_teste(
        resultado.treino, resultado.teste, "Random Forest"
    )
    assert set(testes) == {
        "Kolmogorov-Smirnov",
        "Teste T (Médias)",
        "Mann-Whitney (Medianas)",
        "Levene (Variâncias)",
    }
    for valores in testes.values():
        assert 0.0 <= valores["p_valor"] <= 1.0


def test_cluster_treino_validacao(resultado):
    k = resultado.clusterizador
    for nome in k.modelos:
        comp = k.comparar_treino_validacao(resultado.validacao, nome)
        assert sum(comp["observadas"]) == len(resultado.validacao)
        assert -1.0 <= comp["ari"] <= 1.0


def test_cluster_com_um_grupo_so(resultado):
    # um único grupo no treino: o qui-quadrado não se aplica (p = nan)
    k = Clusterizador()
    k.treinar(resultado.treino, resultado.validacao)
    k.rotulos_treino["K-means"] = np.zeros(len(resultado.treino), dtype=int)
    k.modelos["K-means"].predict = lambda x: np.zeros(len(x), dtype=int)
    comp = k.comparar_treino_validacao(resultado.validacao, "K-means")
    assert np.isnan(comp["p_valor"])


def test_conclusao_colorida():
    assert "green" in conclusao(0.5, "IGUAL", "DIFERENTE")
    assert "red" in conclusao(0.01, "IGUAL", "DIFERENTE")
    assert "não se aplica" in conclusao(float("nan"), "IGUAL", "DIFERENTE")
