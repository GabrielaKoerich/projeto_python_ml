"""Seleção de variáveis feita a partir dos dados de TREINO.

Cada etapa usa um critério objetivo, calculado só com o treino, para que a
validação e o teste não influenciem nenhuma escolha:

1. Assimetria: variáveis numéricas positivas muito assimétricas
   (assimetria > 1) recebem log, para reduzir a cauda longa.
2. Redundância: variáveis com correlação de Spearman |r| > 0,7 formam um
   grupo; o SequentialFeatureSelector (eliminação para trás, Random
   Forest) escolhe um representante de cada grupo.
3. Separação: ficam as numéricas que diferem entre as classes no teste F
   da ANOVA (p < 0,05).
4. Categóricas: as que têm o mesmo padrão por classe são equivalentes;
   fica uma por padrão (a de maior V de Cramér).
5. Conjunto final: o SequentialFeatureSelector para a frente acrescenta
   variáveis enquanto o F1-macro (validação cruzada no treino) subir pelo
   menos TOLERANCIA.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector, f_classif
from sklearn.model_selection import StratifiedKFold

from src.dados import ALVO_CLASSIFICACAO, ALVO_REGRESSAO

LIMITE_ASSIMETRIA = 1.0
LIMITE_CORRELACAO = 0.7
ALFA = 0.05
TOLERANCIA = 0.005
SEMENTE = 42


@dataclass
class Selecao:
    """Resultado de cada etapa da seleção (tudo calculado no treino)."""

    assimetricas: list[str] = field(default_factory=list)
    grupos: list[list[str]] = field(default_factory=list)
    representantes: dict[str, list[str]] = field(default_factory=dict)
    tabela_f: pd.DataFrame = field(default_factory=pd.DataFrame)
    separam: list[str] = field(default_factory=list)
    padroes_categoricos: dict[str, list[str]] = field(default_factory=dict)
    candidatas: list[str] = field(default_factory=list)
    numericas: list[str] = field(default_factory=list)
    categoricas: list[str] = field(default_factory=list)
    regressao: list[str] = field(default_factory=list)

    @property
    def variaveis(self) -> list[str]:
        """Variáveis de entrada da classificação."""
        return self.numericas + self.categoricas


def _floresta() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=50, random_state=SEMENTE, n_jobs=-1)


def _dobras() -> StratifiedKFold:
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMENTE)


def colunas_numericas(df: pd.DataFrame) -> list[str]:
    """Numéricas candidatas (sem o alvo da classificação)."""
    return [c for c in df.select_dtypes("number").columns if c != ALVO_CLASSIFICACAO]


def colunas_categoricas(df: pd.DataFrame) -> list[str]:
    """Categóricas candidatas (texto, sem o alvo)."""
    return [c for c in df.select_dtypes("object").columns if c != ALVO_CLASSIFICACAO]


def variaveis_assimetricas(
    treino: pd.DataFrame, limite: float = LIMITE_ASSIMETRIA
) -> list[str]:
    """Numéricas não negativas com assimetria à direita maior que o limite.

    Variáveis 0/1 (lógicas) ficam de fora, porque log não faz sentido nelas.
    """
    assimetricas = []
    for coluna in colunas_numericas(treino):
        valores = treino[coluna].dropna()
        if valores.nunique() <= 2 or (valores < 0).any():
            continue
        if valores.skew() > limite:
            assimetricas.append(coluna)
    return assimetricas


def aplicar_log(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """Troca cada coluna por log(1 + x), com o sufixo _log.

    O alvo da regressão é mantido na escala original e ganha uma cópia _log.
    """
    df = df.copy()
    for coluna in colunas:
        df[f"{coluna}_log"] = np.log1p(df[coluna])
        if coluna != ALVO_REGRESSAO:
            df = df.drop(columns=coluna)
    return df


def grupos_correlacionados(
    x: pd.DataFrame, limite: float = LIMITE_CORRELACAO
) -> list[list[str]]:
    """Agrupa variáveis ligadas por |r de Spearman| > limite (em cadeia)."""
    correlacao = x.corr(method="spearman").abs()
    lider = {c: c for c in x.columns}

    def raiz(coluna: str) -> str:
        while lider[coluna] != coluna:
            coluna = lider[coluna]
        return coluna

    for a, b in combinations(x.columns, 2):
        if correlacao.loc[a, b] > limite:
            lider[raiz(a)] = raiz(b)
    grupos: dict[str, list[str]] = {}
    for coluna in x.columns:
        grupos.setdefault(raiz(coluna), []).append(coluna)
    return [g for g in grupos.values() if len(g) > 1]


def escolher_representante(x: pd.DataFrame, y: pd.Series, grupo: list[str]) -> str:
    """Eliminação para trás até sobrar 1 variável do grupo."""
    seletor = SequentialFeatureSelector(
        _floresta(),
        n_features_to_select=1,
        direction="backward",
        scoring="f1_macro",
        cv=_dobras(),
    )
    seletor.fit(x[grupo], y)
    return str(np.array(grupo)[seletor.get_support()][0])


def teste_f(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Teste F da ANOVA de cada variável contra a classe."""
    estatistica, p_valor = f_classif(x, y)
    return pd.DataFrame(
        {"F": estatistica, "p_valor": p_valor}, index=x.columns
    ).sort_values("F", ascending=False)


def v_de_cramer(coluna: pd.Series, y: pd.Series) -> float:
    """Associação entre uma categórica e a classe (0 = nenhuma, 1 = total)."""
    tabela = pd.crosstab(coluna, y)
    if min(tabela.shape) < 2:
        return 0.0
    # sem a correção de Yates, que reduz o valor em tabelas pequenas
    estatistica = chi2_contingency(tabela, correction=False)[0]
    return float(np.sqrt(estatistica / (tabela.values.sum() * (min(tabela.shape) - 1))))


def agrupar_categoricas(treino: pd.DataFrame) -> dict[str, list[str]]:
    """Agrupa categóricas com o mesmo padrão por classe; chave = representante."""
    y = treino[ALVO_CLASSIFICACAO]
    por_padrao: dict[tuple, list[str]] = {}
    for coluna in colunas_categoricas(treino):
        moda = treino.groupby(ALVO_CLASSIFICACAO)[coluna].agg(lambda s: s.mode()[0])
        padrao = tuple(moda == moda.iloc[0])
        por_padrao.setdefault(padrao, []).append(coluna)
    grupos = {}
    for colunas in por_padrao.values():
        forca = {c: v_de_cramer(treino[c], y) for c in colunas}
        representante = max(colunas, key=lambda c: forca[c])
        grupos[representante] = colunas
    return grupos


def _codificar(df: pd.DataFrame, categoricas: list[str]) -> pd.DataFrame:
    """Categóricas binárias viram 0/1 (1 = valor mais comum no treino)."""
    df = df.copy()
    for coluna in categoricas:
        df[coluna] = (df[coluna] == df[coluna].mode()[0]).astype(float)
    return df


def selecionar(treino: pd.DataFrame) -> Selecao:
    """Executa todas as etapas da seleção, só com os dados de treino."""
    selecao = Selecao()
    y = treino[ALVO_CLASSIFICACAO]

    # 1. Assimetria -> log
    selecao.assimetricas = variaveis_assimetricas(treino)
    dados = aplicar_log(treino, selecao.assimetricas)
    # o alvo da regressão fica na escala original só para ser previsto;
    # como entrada da classificação, vale a versão _log (quando existe)
    numericas = [
        c
        for c in colunas_numericas(dados)
        if not (c == ALVO_REGRESSAO and f"{c}_log" in dados.columns)
    ]
    x = dados[numericas].fillna(dados[numericas].median())

    # 2. Redundância -> um representante por grupo correlacionado
    selecao.grupos = grupos_correlacionados(x)
    removidas: set[str] = set()
    for grupo in selecao.grupos:
        representante = escolher_representante(x, y, grupo)
        selecao.representantes[representante] = grupo
        removidas.update(c for c in grupo if c != representante)
    restantes = [c for c in numericas if c not in removidas]

    # 3. Separação entre as classes (teste F)
    selecao.tabela_f = teste_f(x[restantes], y)
    selecao.separam = [
        c for c in selecao.tabela_f.index if selecao.tabela_f.loc[c, "p_valor"] < ALFA
    ]

    # 4. Categóricas: uma por padrão
    selecao.padroes_categoricos = agrupar_categoricas(dados)
    categoricas = list(selecao.padroes_categoricos)

    # 5. Conjunto final: seleção para a frente
    selecao.candidatas = selecao.separam + categoricas
    candidatos = _codificar(
        pd.concat([x[selecao.separam], dados[categoricas]], axis=1), categoricas
    )
    seletor = SequentialFeatureSelector(
        _floresta(),
        n_features_to_select="auto",
        tol=TOLERANCIA,
        direction="forward",
        scoring="f1_macro",
        cv=_dobras(),
    )
    seletor.fit(candidatos, y)
    escolhidas = [
        c for c, usar in zip(candidatos.columns, seletor.get_support()) if usar
    ]
    selecao.numericas = [c for c in escolhidas if c in selecao.separam]
    selecao.categoricas = [c for c in escolhidas if c in categoricas]

    # Regressão: candidatas são todas as numéricas, menos o alvo e sua versão log
    selecao.regressao = [
        c for c in numericas if c not in (ALVO_REGRESSAO, f"{ALVO_REGRESSAO}_log")
    ]
    return selecao
