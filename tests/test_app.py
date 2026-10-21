"""Teste: abre o app Streamlit sem navegador."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def test_app_abre_sem_erro():
    app = AppTest.from_file(APP, default_timeout=300).run()
    assert not app.exception
    assert app.title[0].value == "Métricas de Modelos de Machine Learning"


def test_botao_prever_mostra_resultado():
    app = AppTest.from_file(APP, default_timeout=300).run()
    app.button[0].click().run()
    assert not app.exception
    rotulos = [m.label for m in app.metric]
    assert any(r.startswith("Classe prevista") for r in rotulos)
    # tabela com a predição de cada algoritmo
    assert len(app.dataframe) >= 3


def test_clusterizacao_troca_de_variaveis():
    app = AppTest.from_file(APP, default_timeout=300).run()
    # escolher a mesma variável nas duas opções mostra um aviso
    app.radio[1].set_value(app.radio[0].value).run()
    assert not app.exception
    assert len(app.warning) == 1
    # escolher outro par refaz o agrupamento sem erro
    app.radio[1].set_value("IFN-γ").run()
    assert not app.exception
    assert len(app.warning) == 0
