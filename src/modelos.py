"""Modelos de Clusterização, Classificação e Regressão (POO).

Cada classe treina VÁRIOS algoritmos, guarda todos e permite avaliar
qualquer um deles pelo nome. O "melhor" é só um destaque: todos os
resultados ficam disponíveis para comparação no app.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    chisquare,
    ks_2samp,
    levene,
    mannwhitneyu,
    ttest_ind,
)
from sklearn.base import clone
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.dados import (
    ALVO_CLASSIFICACAO,
    ALVO_REGRESSAO,
    VARIAVEIS_CATEGORICAS,
    VARIAVEIS_CLUSTER,
    VARIAVEIS_NUMERICAS,
)


def _preprocessador(numericas: list[str]) -> ColumnTransformer:
    """Padroniza numéricas e transforma categóricas em dummies."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numericas),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                VARIAVEIS_CATEGORICAS,
            ),
        ]
    )


class _BaseSupervisionado:
    """Comportamento comum a Classificador e Regressor."""

    CANDIDATOS: dict[str, tuple[Any, dict[str, list]]] = {}

    def __init__(self) -> None:
        self.modelos: dict[str, GridSearchCV] = {}
        self.selecao: dict[str, float] = {}
        self.nome: str | None = None

    def _modelo(self, nome: str | None) -> GridSearchCV:
        if not self.modelos:
            raise RuntimeError("Treine o modelo antes de prever.")
        chave = nome or self.nome
        if chave not in self.modelos:
            raise KeyError(f"Modelo desconhecido: {chave}")
        return self.modelos[chave]

    def prever(self, df: pd.DataFrame, nome: str | None = None) -> np.ndarray:
        """Prevê com o modelo `nome` (padrão: o melhor na validação)."""
        modelo = self._modelo(nome)
        return modelo.predict(df[self.variaveis])

    def melhores_parametros(self, nome: str) -> dict[str, Any]:
        params = self._modelo(nome).best_params_
        return {k.replace("modelo__", ""): v for k, v in params.items()}

    @property
    def variaveis(self) -> list[str]:  # pragma: no cover - sobrescrito
        raise NotImplementedError


class Classificador(_BaseSupervisionado):
    """Treina 8 classificadores com GridSearchCV e guarda todos."""

    CANDIDATOS = {
        "k-NN": (KNeighborsClassifier(), {"modelo__n_neighbors": [3, 5, 11]}),
        "Reg. Logística": (
            LogisticRegression(max_iter=2000),
            {"modelo__C": [0.1, 1.0, 10.0]},
        ),
        "SVM": (
            SVC(random_state=42),
            {"modelo__C": [0.1, 1.0, 10.0], "modelo__kernel": ["rbf", "linear"]},
        ),
        "Árvore de Decisão": (
            DecisionTreeClassifier(random_state=42),
            {"modelo__max_depth": [3, 5, None]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"modelo__n_estimators": [100, 200], "modelo__max_depth": [None, 5]},
        ),
        "GBM": (
            GradientBoostingClassifier(random_state=42),
            {"modelo__n_estimators": [100], "modelo__learning_rate": [0.05, 0.1]},
        ),
        "Naive Bayes": (GaussianNB(), {}),
        "Rede Neural": (
            MLPClassifier(max_iter=2000, random_state=42),
            {"modelo__hidden_layer_sizes": [(16,), (32, 16)]},
        ),
    }

    @property
    def variaveis(self) -> list[str]:
        return VARIAVEIS_NUMERICAS + VARIAVEIS_CATEGORICAS

    def treinar(
        self, treino: pd.DataFrame, validacao: pd.DataFrame
    ) -> dict[str, float]:
        """Ajusta hiperparâmetros (CV 5-fold no treino) e mede F1 na validação."""
        for nome, (estimador, grade) in self.CANDIDATOS.items():
            pipe = Pipeline(
                [
                    ("prep", _preprocessador(VARIAVEIS_NUMERICAS)),
                    ("modelo", clone(estimador)),
                ]
            )
            busca = GridSearchCV(pipe, grade, cv=5, scoring="f1_macro")
            busca.fit(treino[self.variaveis], treino[ALVO_CLASSIFICACAO])
            pred = busca.predict(validacao[self.variaveis])
            self.selecao[nome] = float(
                f1_score(validacao[ALVO_CLASSIFICACAO], pred, average="macro")
            )
            self.modelos[nome] = busca
        self.nome = max(self.selecao, key=lambda chave: self.selecao[chave])
        return self.selecao

    def avaliar(self, df: pd.DataFrame, nome: str | None = None) -> dict[str, float]:
        pred = self.prever(df, nome)
        obs = df[ALVO_CLASSIFICACAO]
        return {
            "acuracia": float(accuracy_score(obs, pred)),
            "f1_macro": float(f1_score(obs, pred, average="macro")),
        }

    def matriz_confusao(
        self, df: pd.DataFrame, nome: str | None = None
    ) -> pd.DataFrame:
        pred = self.prever(df, nome)
        rotulos = sorted(df[ALVO_CLASSIFICACAO].unique())
        matriz = confusion_matrix(df[ALVO_CLASSIFICACAO], pred, labels=rotulos)
        return pd.DataFrame(matriz, index=rotulos, columns=rotulos)

    def testar_hipotese(
        self, df: pd.DataFrame, nome: str | None = None
    ) -> dict[str, float]:
        """Qui-quadrado: H0 = distribuição predita igual à observada."""
        pred = pd.Series(self.prever(df, nome), name="predito")
        obs = df[ALVO_CLASSIFICACAO].reset_index(drop=True)
        tabela = pd.DataFrame(
            {
                "observado": obs.value_counts().sort_index(),
                "predito": pred.value_counts().sort_index(),
            }
        ).fillna(0)
        _, p_valor, _, _ = chi2_contingency(tabela.T)
        return {"p_valor": float(p_valor)}

    def tabela_contingencia(
        self, treino: pd.DataFrame, teste: pd.DataFrame, nome: str | None = None
    ) -> pd.DataFrame:
        """Quantas vezes cada classe foi PREVISTA no teste e no treino."""
        pred_teste = pd.Series(self.prever(teste, nome)).value_counts()
        pred_treino = pd.Series(self.prever(treino, nome)).value_counts()
        tabela = pd.DataFrame({"Teste": pred_teste, "Treino": pred_treino})
        tabela = tabela.fillna(0).astype(int).sort_index()
        tabela.index.name = "Classe_Predita"
        return tabela

    def teste_homogeneidade(
        self, treino: pd.DataFrame, teste: pd.DataFrame, nome: str | None = None
    ) -> dict[str, float]:
        """Qui-quadrado (como no app da disciplina): H0 = a distribuição das
        classes previstas é a mesma no treino e no teste (homogênea)."""
        tabela = self.tabela_contingencia(treino, teste, nome)
        estatistica, p_valor, graus, _ = chi2_contingency(tabela)
        return {
            "estatistica": float(estatistica),
            "graus_liberdade": int(graus),
            "p_valor": float(p_valor),
        }

    def relatorio(self, df: pd.DataFrame, nome: str | None = None) -> pd.DataFrame:
        """Classification report do scikit-learn como tabela."""
        dados = classification_report(
            df[ALVO_CLASSIFICACAO], self.prever(df, nome), output_dict=True
        )
        return pd.DataFrame(dados).T

    def tabela_resultados(self, teste: pd.DataFrame) -> pd.DataFrame:
        """Uma linha por algoritmo: validação, teste e teste de hipótese."""
        linhas = []
        for nome in self.modelos:
            metricas = self.avaliar(teste, nome)
            linhas.append(
                {
                    "Modelo": nome,
                    "F1 validação": self.selecao[nome],
                    "Acurácia teste": metricas["acuracia"],
                    "F1 teste": metricas["f1_macro"],
                    "p-valor (qui²)": self.testar_hipotese(teste, nome)["p_valor"],
                }
            )
        return (
            pd.DataFrame(linhas)
            .sort_values("F1 validação", ascending=False)
            .reset_index(drop=True)
        )


class Regressor(_BaseSupervisionado):
    """Prevê IFN-γ com 6 algoritmos e guarda todos."""

    CANDIDATOS = {
        "Regressão Linear": (LinearRegression(), {}),
        "k-NN": (KNeighborsRegressor(), {"modelo__n_neighbors": [5, 11, 21]}),
        "SVM": (SVR(), {"modelo__C": [0.1, 1.0, 10.0]}),
        "Árvore de Decisão": (
            DecisionTreeRegressor(random_state=42),
            {"modelo__max_depth": [3, 5, None]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=42),
            {"modelo__n_estimators": [100, 200], "modelo__max_depth": [None, 5]},
        ),
        "Rede Neural": (
            MLPRegressor(max_iter=3000, random_state=42),
            {"modelo__hidden_layer_sizes": [(16,), (32, 16)]},
        ),
    }

    @property
    def numericas(self) -> list[str]:
        return [v for v in VARIAVEIS_NUMERICAS if v != ALVO_REGRESSAO]

    @property
    def variaveis(self) -> list[str]:
        return self.numericas + VARIAVEIS_CATEGORICAS

    def treinar(
        self, treino: pd.DataFrame, validacao: pd.DataFrame
    ) -> dict[str, float]:
        """Ajusta hiperparâmetros (CV 5-fold) e mede RMSE na validação."""
        for nome, (estimador, grade) in self.CANDIDATOS.items():
            pipe = Pipeline(
                [
                    ("prep", _preprocessador(self.numericas)),
                    ("modelo", clone(estimador)),
                ]
            )
            busca = GridSearchCV(
                pipe, grade, cv=5, scoring="neg_root_mean_squared_error"
            )
            busca.fit(treino[self.variaveis], treino[ALVO_REGRESSAO])
            pred = busca.predict(validacao[self.variaveis])
            self.selecao[nome] = float(
                root_mean_squared_error(validacao[ALVO_REGRESSAO], pred)
            )
            self.modelos[nome] = busca
        self.nome = min(self.selecao, key=lambda chave: self.selecao[chave])
        return self.selecao

    def avaliar(self, df: pd.DataFrame, nome: str | None = None) -> dict[str, float]:
        pred = self.prever(df, nome)
        obs = df[ALVO_REGRESSAO]
        return {
            "R2": float(r2_score(obs, pred)),
            "MAE": float(mean_absolute_error(obs, pred)),
            "RMSE": float(root_mean_squared_error(obs, pred)),
        }

    def testar_hipotese(
        self, df: pd.DataFrame, nome: str | None = None
    ) -> dict[str, float]:
        """Mann-Whitney: H0 = predito e observado têm a mesma distribuição."""
        _, p_valor = mannwhitneyu(self.prever(df, nome), df[ALVO_REGRESSAO])
        return {"p_valor": float(p_valor)}

    def testes_treino_teste(
        self, treino: pd.DataFrame, teste: pd.DataFrame, nome: str | None = None
    ) -> dict[str, dict[str, float]]:
        """Compara as PREVISÕES do treino com as do teste (como no app da
        disciplina). H0 de cada teste: as duas amostras são iguais em
        distribuição (KS), média (t), mediana (Mann-Whitney) e variância
        (Levene)."""
        pred_treino = self.prever(treino, nome)
        pred_teste = self.prever(teste, nome)
        testes = {
            "Kolmogorov-Smirnov": ks_2samp(pred_treino, pred_teste),
            "Teste T (Médias)": ttest_ind(pred_treino, pred_teste),
            "Mann-Whitney (Medianas)": mannwhitneyu(pred_treino, pred_teste),
            "Levene (Variâncias)": levene(pred_treino, pred_teste),
        }
        return {
            teste_nome: {
                "estatistica": float(resultado[0]),
                "p_valor": float(resultado[1]),
            }
            for teste_nome, resultado in testes.items()
        }

    def tabela_resultados(self, teste: pd.DataFrame) -> pd.DataFrame:
        linhas = []
        for nome in self.modelos:
            metricas = self.avaliar(teste, nome)
            linhas.append(
                {
                    "Modelo": nome,
                    "RMSE validação": self.selecao[nome],
                    "R² teste": metricas["R2"],
                    "MAE teste": metricas["MAE"],
                    "RMSE teste": metricas["RMSE"],
                    "p-valor (Mann-Whitney)": self.testar_hipotese(teste, nome)[
                        "p_valor"
                    ],
                }
            )
        return pd.DataFrame(linhas).sort_values("RMSE validação").reset_index(drop=True)


class Clusterizador:
    """K-means, DBSCAN e Agrupamento Hierárquico.

    Cada algoritmo é ajustado no treino; os hiperparâmetros são escolhidos
    pela silhueta na validação. DBSCAN e Hierárquico não têm `predict`,
    então pontos novos recebem o grupo do vizinho mais próximo do treino.
    """

    GRADES: dict[str, list[dict[str, Any]]] = {
        "K-means": [{"n_clusters": k} for k in range(2, 7)],
        "DBSCAN": [
            {"eps": e, "min_samples": m} for e in (0.3, 0.5, 0.7) for m in (5, 10)
        ],
        "Hierárquico": [{"n_clusters": k} for k in range(2, 7)],
    }

    def __init__(self, variaveis: list[str] | None = None) -> None:
        self.variaveis = variaveis or VARIAVEIS_CLUSTER
        self.escala = StandardScaler()
        self.modelos: dict[str, KNeighborsClassifier] = {}
        self.parametros: dict[str, dict[str, Any]] = {}
        self.rotulos_treino: dict[str, np.ndarray] = {}
        self.selecao: dict[str, float] = {}
        self.nome: str | None = None

    @staticmethod
    def _criar(algoritmo: str, params: dict[str, Any]) -> Any:
        if algoritmo == "K-means":
            return KMeans(n_init=10, random_state=42, **params)
        if algoritmo == "DBSCAN":
            return DBSCAN(**params)
        return AgglomerativeClustering(linkage="ward", **params)

    @staticmethod
    def _silhueta(x: np.ndarray, rotulos: np.ndarray) -> float:
        if len(set(rotulos)) < 2:
            return -1.0
        return float(silhouette_score(x, rotulos))

    def treinar(
        self, treino: pd.DataFrame, validacao: pd.DataFrame
    ) -> dict[str, float]:
        x_treino = self.escala.fit_transform(treino[self.variaveis])
        x_valid = self.escala.transform(validacao[self.variaveis])
        for algoritmo, grade in self.GRADES.items():
            melhor: tuple[float, dict[str, Any], np.ndarray, Any] = (
                -2.0,
                {},
                np.array([]),
                None,
            )
            for params in grade:
                rotulos = self._criar(algoritmo, params).fit_predict(x_treino)
                atribuidor = KNeighborsClassifier(n_neighbors=1).fit(x_treino, rotulos)
                nota = self._silhueta(x_valid, atribuidor.predict(x_valid))
                if nota > melhor[0]:
                    melhor = (nota, params, rotulos, atribuidor)
            nota, params, rotulos, atribuidor = melhor
            self.selecao[algoritmo] = nota
            self.parametros[algoritmo] = params
            self.rotulos_treino[algoritmo] = rotulos
            self.modelos[algoritmo] = atribuidor
        self.nome = max(self.selecao, key=lambda chave: self.selecao[chave])
        return self.selecao

    def prever(self, df: pd.DataFrame, nome: str | None = None) -> np.ndarray:
        if not self.modelos:
            raise RuntimeError("Treine o modelo antes de prever.")
        chave = nome or self.nome
        if chave not in self.modelos:
            raise KeyError(f"Modelo desconhecido: {chave}")
        x = self.escala.transform(df[self.variaveis])
        return self.modelos[chave].predict(x)

    def avaliar(self, df: pd.DataFrame, nome: str | None = None) -> dict[str, float]:
        """Silhueta e concordância (ARI) com a classe real."""
        rotulos = self.prever(df, nome)
        x = self.escala.transform(df[self.variaveis])
        return {
            "silhueta": self._silhueta(x, rotulos),
            "ARI_vs_classe": float(
                adjusted_rand_score(df[ALVO_CLASSIFICACAO], rotulos)
            ),
        }

    def testar_hipotese(
        self, df: pd.DataFrame, nome: str | None = None
    ) -> dict[str, float]:
        """Qui-quadrado de aderência: H0 = proporção dos grupos no teste
        igual à do treino (o agrupamento generalizou)."""
        chave = nome or self.nome
        assert chave is not None
        grupos_treino = pd.Series(self.rotulos_treino[chave]).value_counts(
            normalize=True
        )
        grupos_teste = pd.Series(self.prever(df, chave)).value_counts()
        observados = grupos_teste.reindex(grupos_treino.index, fill_value=0)
        esperados = grupos_treino * observados.sum()
        _, p_valor = chisquare(observados, esperados)
        # com um único grupo o teste não se aplica (p = nan) -> trata como 1
        return {"p_valor": float(np.nan_to_num(p_valor, nan=1.0))}

    def comparar_treino_validacao(
        self, validacao: pd.DataFrame, nome: str
    ) -> dict[str, Any]:
        """- qui-quadrado de aderência: as frequências dos grupos na validação
          (pelo modelo do treino) seguem as proporções do treino?
        - ARI: um modelo ajustado só na validação forma os mesmos grupos que
          o modelo do treino??????"""
        grupos_treino = pd.Series(self.rotulos_treino[nome]).value_counts(
            normalize=True
        )
        rotulos_valid = pd.Series(self.prever(validacao, nome))
        observadas = (
            rotulos_valid.value_counts()
            .reindex(grupos_treino.index, fill_value=0)
            .sort_index()
        )
        esperadas = (grupos_treino * len(validacao)).sort_index()
        if len(observadas) > 1:
            estatistica, p_valor = chisquare(observadas, esperadas)
        else:
            estatistica, p_valor = 0.0, float("nan")

        x_valid = self.escala.transform(validacao[self.variaveis])
        rotulos_modelo_valid = self._criar(nome, self.parametros[nome]).fit_predict(
            x_valid
        )
        ari = adjusted_rand_score(rotulos_valid, rotulos_modelo_valid)
        return {
            "grupos_treino": sorted(grupos_treino.index.tolist()),
            "esperadas": esperadas.round(2).tolist(),
            "observadas": observadas.tolist(),
            "estatistica": float(estatistica),
            "p_valor": float(p_valor),
            "ari": float(ari),
        }

    def tabela_resultados(self, teste: pd.DataFrame) -> pd.DataFrame:
        linhas = []
        for nome in self.modelos:
            metricas = self.avaliar(teste, nome)
            linhas.append(
                {
                    "Modelo": nome,
                    "Parâmetros": str(self.parametros[nome]),
                    "Grupos": len(set(self.rotulos_treino[nome])),
                    "Silhueta validação": self.selecao[nome],
                    "Silhueta teste": metricas["silhueta"],
                    "ARI vs classe": metricas["ARI_vs_classe"],
                    "p-valor (qui²)": self.testar_hipotese(teste, nome)["p_valor"],
                }
            )
        return (
            pd.DataFrame(linhas)
            .sort_values("Silhueta de validação", ascending=False)
            .reset_index(drop=True)
        )
