"""Páginas do Streamlit (front end).

em cada aba, um cartão por
algoritmo com os testes de hipótese e suas conclusões; uma tabela comparando todos os algoritmos e uma aba para prever novos pacientes.
"""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.dados import ALVO_REGRESSAO, VARIAVEIS_CLUSTER
from src.modelos import Clusterizador
from src.pipeline import Resultado, treinar_tudo

COLUNAS_POR_LINHA = 3
ALFA = 0.05


@st.cache_resource(show_spinner="Treinando todos os modelos (cerca de 1 minuto)...")
def obter_resultado() -> Resultado:
    """Treina uma única vez e reaproveita entre as interações."""
    return treinar_tudo()


@st.cache_resource(show_spinner="Agrupando com as variáveis escolhidas...")
def obter_clusterizador(variavel_1: str, variavel_2: str) -> Clusterizador:
    """Treina os 3 algoritmos de agrupamento para o par de variáveis escolhido."""
    r = obter_resultado()
    clusterizador = Clusterizador(variaveis=[variavel_1, variavel_2])
    clusterizador.treinar(r.treino, r.validacao)
    return clusterizador


def conclusao(p_valor: float, igual: str, diferente: str) -> str:
    """Texto colorido (verde/vermelho) com a decisão do teste (alfa = 0,05)."""
    if pd.isna(p_valor):
        return ":orange[**Teste não se aplica (apenas um grupo)**]"
    if p_valor > ALFA:
        return f":green[**{igual} (Falha ao rejeitar H0)**]"
    return f":red[**{diferente} (Rejeita H0)**]"


def interpretar(
    p_valor: float,
    igual: str = "predito ≈ observado",
    diferente: str = "predito difere do observado",
) -> str:
    """Texto curto para o resultado de um teste de hipótese (alfa = 0,05)."""
    if p_valor > ALFA:
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


def titulo_modelo(nome: str) -> None:
    st.markdown("---")
    st.markdown(f"<h4 style='text-align: center;'>{nome}</h4>", unsafe_allow_html=True)
    st.markdown("---")


class Paginas:
    def __init__(self) -> None:
        self.r = obter_resultado()

    def cabecalho(self) -> str:
        st.title("Métricas de Modelos de Machine Learning")
        st.subheader("Projeto da disciplina Python Machine Learning — Clusterização, Classificação e Regressão")
        st.caption(
            "Nome: Gabriela Marino Koerich"
            f"Treino: {len(self.r.treino)} | Validação: {len(self.r.validacao)} | "
            f"Teste: {len(self.r.teste)} pacientes."
        )
        return "Cabeçalho criado"

    # ------------------------------------------------------------------ #
    # Clusterização
    # ------------------------------------------------------------------ #
    def _cartao_cluster(self, k: Clusterizador, nome: str) -> None:
        titulo_modelo(nome)
        comp = k.comparar_treino_validacao(self.r.validacao, nome)
        st.write(f"Parâmetros: {k.parametros[nome]}")
        st.write(f"Clusters identificados no Treino: {comp['grupos_treino']}")
        st.write(f"Frequências Esperadas na Validação (Treino): {comp['esperadas']}")
        st.write(f"Frequências Observadas na Validação: {comp['observadas']}")
        st.write(
            f"Estatística Qui-Quadrado (χ²): {comp['estatistica']:.4f} | "
            f"p-valor: {comp['p_valor']:.4f}"
        )
        st.markdown(
            "Distribuição dos clusters na validação x treino: "
            + conclusao(comp["p_valor"], "Igual", "Diferente")
        )
        semelhantes = comp["ari"] >= 0.5
        cor = "green" if semelhantes else "red"
        texto = "Semelhantes" if semelhantes else "Diferentes"
        st.markdown(
            "Modelo do Treino x modelo da Validação: "
            f":{cor}[**{texto}**] (ARI = {comp['ari']:.4f})"
        )
        dados = self.r.validacao[k.variaveis].copy()
        dados["grupo"] = k.prever(self.r.validacao, nome).astype(str)
        grafico = sns.jointplot(
            data=dados,
            x=k.variaveis[0],
            y=k.variaveis[1],
            hue="grupo",
            height=4,
        )
        st.pyplot(grafico.figure)
        plt.close(grafico.figure)

    def aba_clusterizacao(self) -> str:
        st.subheader("Clusterização — K-means, DBSCAN e Hierárquico")
        col = st.columns((1, 4))
        with col[0]:
            variavel_1 = st.radio("Primeira variável:", VARIAVEIS_CLUSTER, index=0)
            variavel_2 = st.radio("Segunda variável:", VARIAVEIS_CLUSTER, index=2)
        with col[1]:
            if variavel_1 == variavel_2:
                st.warning("Escolha duas variáveis DIFERENTES.")
                return "Variáveis iguais"
            k = obter_clusterizador(variavel_1, variavel_2)
            st.write(
                "Parâmetros escolhidos pela silhueta na validação. "
                "Qui-quadrado: H0 = a distribuição dos grupos na validação é "
                "igual à do treino. ARI ≥ 0,5: os modelos de treino e validação "
                "formam grupos semelhantes."
            )
            st.dataframe(k.tabela_resultados(self.r.teste).round(3), hide_index=True)
        for nome, coluna in grade(list(k.modelos)):
            with coluna:
                self._cartao_cluster(k, nome)
        return "Clusterização exibida"

    # ------------------------------------------------------------------ #
    # Classificação
    # ------------------------------------------------------------------ #
    def _cartao_classificador(self, nome: str) -> None:
        c = self.r.classificador
        titulo_modelo(nome)
        st.write("Tabela de Contingência (Observada):")
        st.dataframe(c.tabela_contingencia(self.r.treino, self.r.teste, nome))
        teste = c.teste_homogeneidade(self.r.treino, self.r.teste, nome)
        st.write(f"Estatística Qui-Quadrado: {teste['estatistica']:.4f}")
        st.write(f"Graus de Liberdade: {teste['graus_liberdade']}")
        st.write(f"Valor-p: {teste['p_valor']:.4f}")
        st.markdown(
            "Distribuição das classes no treino e no teste: "
            + conclusao(
                teste["p_valor"],
                "Estatisticamente Igual (Homogênea)",
                "Estatisticamente Diferente (Heterogênea)",
            )
        )
        acuracia = c.avaliar(self.r.teste, nome)["acuracia"]
        fig, ax = plt.subplots(figsize=(4, 3.4))
        sns.heatmap(
            c.matriz_confusao(self.r.teste, nome),
            annot=True,
            fmt="d",
            cbar=False,
            cmap="Greens",
            ax=ax,
        )
        ax.set_title(f"Acurácia: {acuracia:.3f}")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Observado")
        st.pyplot(fig)
        plt.close(fig)
        st.write("Classification Report:")
        st.dataframe(c.relatorio(self.r.teste, nome).round(4))
        st.caption(f"Hiperparâmetros: {c.melhores_parametros(nome) or '—'}")

    def aba_classificacao(self) -> str:
        c = self.r.classificador
        st.subheader(f"Classificação — {len(c.modelos)} algoritmos")
        st.write(
            f"Resumo ordenado pelo F1 na validação (destaque: **{c.nome}**). "
            "Nos cartões, o qui-quadrado compara as classes previstas no treino "
            "e no teste (H0 = distribuição homogênea)."
        )
        st.dataframe(c.tabela_resultados(self.r.teste).round(3), hide_index=True)
        for nome, coluna in grade(list(c.modelos)):
            with coluna:
                self._cartao_classificador(nome)
        return "Classificação exibida"

    # ------------------------------------------------------------------ #
    # Regressão
    # ------------------------------------------------------------------ #
    def _cartao_regressor(self, nome: str) -> None:
        g = self.r.regressor
        titulo_modelo(nome)
        metricas = g.avaliar(self.r.teste, nome)
        st.write(
            f"R²: **{metricas['R2']:.3f}** · RMSE: **{metricas['RMSE']:.3f}** · "
            f"MAE: **{metricas['MAE']:.3f}**"
        )
        testes = g.testes_treino_teste(self.r.treino, self.r.teste, nome)
        for teste_nome, resultado in testes.items():
            st.markdown(f"**{teste_nome}**")
            st.markdown(conclusao(resultado["p_valor"], "Igual", "Diferente"))
            st.dataframe(
                pd.DataFrame(
                    {"Valores": [resultado["estatistica"], resultado["p_valor"]]},
                    index=["Estatistica", "p-valor"],
                ).round(4)
            )
        observado = self.r.teste[ALVO_REGRESSAO]
        limites = [observado.min(), observado.max()]
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.scatter(observado, g.prever(self.r.teste, nome), alpha=0.6, s=14)
        ax.plot(limites, limites, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Observado")
        ax.set_ylabel("Predito")
        st.pyplot(fig)
        plt.close(fig)

    def aba_regressao(self) -> str:
        g = self.r.regressor
        st.subheader(f"Regressão de {ALVO_REGRESSAO} — {len(g.modelos)} algoritmos")
        st.write(
            f"Resumo ordenado pelo RMSE na validação (destaque: **{g.nome}**). "
            "Nos cartões, os 4 testes comparam as previsões do treino com as "
            "do teste (H0 = as duas amostras são iguais)."
        )
        st.dataframe(g.tabela_resultados(self.r.teste).round(3), hide_index=True)
        for nome, coluna in grade(list(g.modelos)):
            with coluna:
                self._cartao_regressor(nome)
        return "Regressão exibida"

    # ------------------------------------------------------------------ #
    # Dados e Predição
    # ------------------------------------------------------------------ #
    def aba_dados(self) -> str:
        r = self.r
        col = st.columns(3)
        for coluna, titulo, dados in zip(
            col,
            ["Dados de Treino", "Dados de Validação", "Dados de Teste"],
            [r.treino, r.validacao, r.teste],
        ):
            with coluna:
                st.markdown(
                    f"<h4 style='text-align: center;'>{titulo}</h4>",
                    unsafe_allow_html=True,
                )
                st.dataframe(dados)
        st.write("Tamanho dos datasets:")
        st.dataframe(
            pd.DataFrame(
                {
                    "Métrica": ["Treino", "Validação", "Teste"],
                    "Tamanho": [len(r.treino), len(r.validacao), len(r.teste)],
                }
            ),
            hide_index=True,
        )
        return "Dados exibidos"

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
            self._resultado_predicao(novo)
        return "Predição exibida"

    def _resultado_predicao(self, novo: pd.DataFrame) -> None:
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

    def montar(self) -> str:
        self.cabecalho()
        abas = st.tabs(
            ["Clustering", "Classificação", "Regressão", "Datasets", "Predição"]
        )
        with abas[0]:
            self.aba_clusterizacao()
        with abas[1]:
            self.aba_classificacao()
        with abas[2]:
            self.aba_regressao()
        with abas[3]:
            self.aba_dados()
        with abas[4]:
            self.aba_predicao()
        return "Página montada"
