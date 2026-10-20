"""Páginas do Streamlit (front end).

Todas as abas mostram TODOS os algoritmos testados: uma tabela-resumo
comparativa e, abaixo, um cartão por algoritmo com métricas, teste de
hipótese e gráfico.
"""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.dados import ALVO_REGRESSAO, VARIAVEIS_CLUSTER
from src.pipeline import Resultado, treinar_tudo

COLUNAS_POR_LINHA = 3


@st.cache_resource(show_spinner="Treinando todos os modelos (cerca de 1 minuto)...")
def obter_resultado() -> Resultado:
    """Treina uma única vez e reaproveita entre as interações."""
    return treinar_tudo()


def interpretar(
    p_valor: float,
    igual: str = "predito ≈ observado",
    diferente: str = "predito difere do observado",
) -> str:
    """Texto padrão para o resultado de um teste de hipótese (alfa = 0,05)."""
    if p_valor > 0.05:
        return f"p = {p_valor:.3f} → não rejeita H0 ({igual})"
    return f"p = {p_valor:.3f} → rejeita H0 ({diferente})"


def grade(nomes: list[str]) -> list[tuple[str, Any]]:
    """Distribui os modelos em linhas de COLUNAS_POR_LINHA colunas."""
    pares = []
    for inicio in range(0, len(nomes), COLUNAS_POR_LINHA):
        colunas = st.columns(COLUNAS_POR_LINHA)
        for nome, coluna in zip(nomes[inicio : inicio + COLUNAS_POR_LINHA], colunas):
            pares.append((nome, coluna))
    return pares


class Paginas:
    def __init__(self) -> None:
        self.r = obter_resultado()

    def cabecalho(self) -> str:
        st.title("Sistema de Predição com Biomarcadores")
        st.caption(
            "Projeto da disciplina Python Machine Learning — "
            "Clusterização, Classificação e Regressão. "
            "Todas as métricas abaixo são do conjunto de TESTE (150 pacientes), "
            "exceto quando indicado 'validação'."
        )
        return "Cabeçalho criado"

    def aba_predicao(self) -> str:
        st.subheader("Faça uma predição para um novo paciente")
        base = self.r.treino
        col = st.columns(3)
        entrada = {
            "TP53": col[0].number_input("TP53", value=float(base["TP53"].median())),
            "IFN-γ": col[1].number_input("IFN-γ", value=float(base["IFN-γ"].median())),
            "CD3_2": col[2].number_input("CD3_2", value=float(base["CD3_2"].median())),
            "CD8": col[0].number_input("CD8", value=float(base["CD8"].median())),
            "PPD_log": col[1].number_input(
                "PPD_log", value=float(base["PPD_log"].median())
            ),
            "CD7": col[2].number_input("CD7", value=float(base["CD7"].median())),
            "FOXP3": col[0].selectbox("FOXP3", ["Normal", "Alterado"]),
        }
        novo = pd.DataFrame([entrada])
        if st.button("Prever", type="primary"):
            r = self.r
            res = st.columns(3)
            res[0].metric(
                f"Classe prevista ({r.classificador.nome})",
                str(r.classificador.prever(novo)[0]),
            )
            res[1].metric(
                f"{ALVO_REGRESSAO} previsto ({r.regressor.nome})",
                f"{r.regressor.prever(novo)[0]:.3f}",
            )
            res[2].metric(
                f"Grupo ({r.clusterizador.nome})",
                f"Grupo {r.clusterizador.prever(novo)[0]}",
            )
            st.write("**Predição de cada algoritmo:**")
            tab = st.columns(3)
            tab[0].dataframe(
                pd.DataFrame(
                    {
                        "Classificador": list(r.classificador.modelos),
                        "Classe": [
                            r.classificador.prever(novo, n)[0]
                            for n in r.classificador.modelos
                        ],
                    }
                ),
                hide_index=True,
            )
            tab[1].dataframe(
                pd.DataFrame(
                    {
                        "Regressor": list(r.regressor.modelos),
                        ALVO_REGRESSAO: [
                            round(float(r.regressor.prever(novo, n)[0]), 3)
                            for n in r.regressor.modelos
                        ],
                    }
                ),
                hide_index=True,
            )
            tab[2].dataframe(
                pd.DataFrame(
                    {
                        "Agrupamento": list(r.clusterizador.modelos),
                        "Grupo": [
                            int(r.clusterizador.prever(novo, n)[0])
                            for n in r.clusterizador.modelos
                        ],
                    }
                ),
                hide_index=True,
            )
        return "Predição exibida"

    def aba_classificacao(self) -> str:
        r = self.r
        c = r.classificador
        st.subheader(f"Classificação — {len(c.modelos)} algoritmos comparados")
        st.write(
            f"Tabela ordenada pelo F1 na validação. Destaque: **{c.nome}**. "
            "Teste de hipótese: qui-quadrado entre a distribuição de classes "
            "predita e a observada."
        )
        st.dataframe(c.tabela_resultados(r.teste).round(3), hide_index=True)
        st.divider()
        for nome, coluna in grade(list(c.modelos)):
            with coluna:
                st.markdown(f"#### {nome}")
                metricas = c.avaliar(r.teste, nome)
                st.write(
                    f"Acurácia: **{metricas['acuracia']:.3f}** · "
                    f"F1: **{metricas['f1_macro']:.3f}**"
                )
                st.caption(interpretar(c.testar_hipotese(r.teste, nome)["p_valor"]))
                st.caption(f"Hiperparâmetros: {c.melhores_parametros(nome) or '—'}")
                fig, ax = plt.subplots(figsize=(4, 3.4))
                sns.heatmap(
                    c.matriz_confusao(r.teste, nome),
                    annot=True,
                    fmt="d",
                    cbar=False,
                    cmap="Blues",
                    ax=ax,
                )
                ax.set_xlabel("Predito")
                ax.set_ylabel("Observado")
                st.pyplot(fig)
                plt.close(fig)
        return "Classificação exibida"

    def aba_regressao(self) -> str:
        r = self.r
        g = r.regressor
        st.subheader(
            f"Regressão de {ALVO_REGRESSAO} — {len(g.modelos)} algoritmos comparados"
        )
        st.write(
            f"Tabela ordenada pelo RMSE na validação. Destaque: **{g.nome}**. "
            "Teste de hipótese: Mann-Whitney entre valores preditos e observados."
        )
        st.dataframe(g.tabela_resultados(r.teste).round(3), hide_index=True)
        st.divider()
        observado = r.teste[ALVO_REGRESSAO]
        limites = [observado.min(), observado.max()]
        for nome, coluna in grade(list(g.modelos)):
            with coluna:
                st.markdown(f"#### {nome}")
                metricas = g.avaliar(r.teste, nome)
                st.write(
                    f"R²: **{metricas['R2']:.3f}** · "
                    f"RMSE: **{metricas['RMSE']:.3f}** · "
                    f"MAE: **{metricas['MAE']:.3f}**"
                )
                st.caption(interpretar(g.testar_hipotese(r.teste, nome)["p_valor"]))
                fig, ax = plt.subplots(figsize=(4, 3.4))
                ax.scatter(observado, g.prever(r.teste, nome), alpha=0.6, s=14)
                ax.plot(limites, limites, color="gray", linestyle="--", linewidth=1)
                ax.set_xlabel("Observado")
                ax.set_ylabel("Predito")
                st.pyplot(fig)
                plt.close(fig)
        return "Regressão exibida"

    def aba_clusterizacao(self) -> str:
        r = self.r
        k = r.clusterizador
        st.subheader(f"Clusterização — {len(k.modelos)} algoritmos comparados")
        st.write(
            f"Variáveis: {', '.join(k.variaveis)} (padronizadas). "
            f"Destaque: **{k.nome}**. ARI compara os grupos com a classe real. "
            "Teste de hipótese: qui-quadrado de aderência entre a proporção dos "
            "grupos no teste e no treino."
        )
        st.dataframe(k.tabela_resultados(r.teste).round(3), hide_index=True)
        st.divider()
        x, y = VARIAVEIS_CLUSTER[0], VARIAVEIS_CLUSTER[1]
        for nome, coluna in grade(list(k.modelos)):
            with coluna:
                st.markdown(f"#### {nome}")
                metricas = k.avaliar(r.teste, nome)
                st.write(
                    f"Silhueta: **{metricas['silhueta']:.3f}** · "
                    f"ARI: **{metricas['ARI_vs_classe']:.3f}**"
                )
                st.caption(
                    interpretar(
                        k.testar_hipotese(r.teste, nome)["p_valor"],
                        igual="proporção dos grupos no teste ≈ treino",
                        diferente="proporção dos grupos mudou",
                    )
                )
                fig, ax = plt.subplots(figsize=(4, 3.4))
                ax.scatter(
                    r.teste[x],
                    r.teste[y],
                    c=k.prever(r.teste, nome),
                    cmap="viridis",
                    s=14,
                )
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                st.pyplot(fig)
                plt.close(fig)
        return "Clusterização exibida"

    def aba_dados(self) -> str:
        r = self.r
        st.write(
            pd.DataFrame(
                {"linhas": [len(r.treino), len(r.validacao), len(r.teste)]},
                index=["Treino", "Validação", "Teste"],
            )
        )
        st.dataframe(r.treino.head(50))
        return "Dados exibidos"

    def montar(self) -> str:
        self.cabecalho()
        abas = st.tabs(
            ["Predição", "Classificação", "Regressão", "Clusterização", "Dados"]
        )
        with abas[0]:
            self.aba_predicao()
        with abas[1]:
            self.aba_classificacao()
        with abas[2]:
            self.aba_regressao()
        with abas[3]:
            self.aba_clusterizacao()
        with abas[4]:
            self.aba_dados()
        return "Página montada"
