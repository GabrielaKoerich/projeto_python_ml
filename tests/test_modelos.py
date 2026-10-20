"""Testes dos modelos e de integração do pipeline completo."""

import pandas as pd
import pytest

from src.modelos import Classificador, Clusterizador, Regressor
from src.paginas import interpretar
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


def test_predicao_novo_paciente(resultado):
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
