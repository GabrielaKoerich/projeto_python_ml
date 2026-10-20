"""Teste ponta a ponta: abre o app Streamlit sem navegador."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def test_app_abre_sem_erro():
    app = AppTest.from_file(APP, default_timeout=300).run()
    assert not app.exception
    assert app.title[0].value == "Sistema de Predição com Biomarcadores"


def test_botao_prever_mostra_resultado():
    app = AppTest.from_file(APP, default_timeout=300).run()
    app.button[0].click().run()
    assert not app.exception
    rotulos = [m.label for m in app.metric]
    assert any(r.startswith("Classe prevista") for r in rotulos)
    # tabela com a predição de cada algoritmo
    assert len(app.dataframe) >= 3
