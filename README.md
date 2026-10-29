# Métricas de Modelos de Machine Learning com Biomarcadores

![CI](https://github.com/GabrielaKoerich/projeto_python_ml/actions/workflows/ci.yml/badge.svg)

Projeto final da disciplina **Python Machine Learning** — Gabriela Marino Koerich.

| Entrega | Link |
| --- | --- |
| 1ª — AED comentada (notebook) | [Abrir no GitHub](notebooks/01_AED_relatorio_comentado.ipynb) · [Abrir no nbviewer](https://nbviewer.org/github/GabrielaKoerich/projeto_python_ml/blob/main/notebooks/01_AED_relatorio_comentado.ipynb) |
| 1ª — AED no Jupyter Lab (local) | [http://localhost:8888/lab/tree/notebooks/01_AED_relatorio_comentado.ipynb?token=pythonml](http://localhost:8888/lab/tree/notebooks/01_AED_relatorio_comentado.ipynb?token=pythonml) — com o container ligado (`.\dev\subir_docker.ps1`) |
| 2ª — App (deploy) | https://disciplinapythonml-jbqmcul4xbvnx3t4ugmhjn.streamlit.app/ |
| 3ª — Testes automatizados | [`tests/`](tests) · [Execuções do CI](https://github.com/GabrielaKoerich/projeto_python_ml/actions) |
| 1ª — Relatório ydata-profiling | [Abrir relatório](https://GabrielaKoerich.github.io/projeto_python_ml/notebooks/relatorio_aed.html) |

## Pergunta e objetivos
A partir de biomarcadores de 1000 pacientes:
1. **Classificação** — prever a `classe` (BCG, cancer, imunodeficiencia, leucemia). Sucesso: F1-macro ≥ 0,75.
2. **Regressão** — prever o valor de `IFN-γ`. Sucesso: R² ≥ 0,70.
3. **Clusterização** — encontrar grupos naturais com K-means, DBSCAN e hierárquico (parâmetros escolhidos pela silhueta). Sucesso: silhueta ≥ 0,50.

## Como as variáveis são escolhidas
Nenhuma variável é fixada à mão. A função `selecionar` (`src/selecao.py`) usa **só o treino** (700 linhas):
1. **Assimetria > 1 → log(1 + x)** (PPD, CH50, Th1, CD19_2, IFN-γ).
2. **Redundância:** variáveis com |r de Spearman| > 0,7 formam um grupo; o `SequentialFeatureSelector` (eliminação para trás, Random Forest) mantém um representante (CD3_1, de um grupo de 10).
3. **Separação:** ficam as numéricas que diferem entre as classes no teste F (p < 0,05); TP53 e NMP22 saem.
4. **Categóricas:** as que têm o mesmo padrão por classe são equivalentes; fica uma por padrão.
5. **Conjunto final:** seleção para a frente, enquanto o F1-macro (validação cruzada no treino) sobe ≥ 0,005 → **PPD_log, CD3_1, CD8 e CD56**.

Na regressão, cada algoritmo escolhe suas variáveis com `SelectKBest` + `f_regression` dentro do Pipeline. Faltantes são preenchidos pela mediana do treino, também dentro do Pipeline. O passo a passo está na seção 10 da AED e na aba Datasets do app.

## Estrutura
```
dados/biomarcadores.csv      dados brutos
notebooks/                   AED comentada
src/dados.py                 aquisição, estruturação, divisão 70/15/15
src/selecao.py               seleção de variáveis feita no treino
src/modelos.py               classes Clusterizador, Classificador, Regressor
src/pipeline.py              fluxo completo: divisão, seleção e treino
src/paginas.py               páginas do Streamlit
streamlit_app.py             ponto de entrada do app (2ª entrega)
docker/ + docker-compose.yml ambiente Docker (Python 3.12, Jupyter Lab, Streamlit)
dev/*.ps1                    atalhos: subir container, testes, app
tests/                       testes unitários, integração e ponta a ponta (3ª entrega)
.github/workflows/ci.yml     CI: ruff, flake8, mypy, pytest + cobertura
```

## Resultados (conjunto de teste, 150 pacientes)
O app mostra todos os algoritmos, cada um com métricas, teste de hipótese e gráfico. Destaques (escolhidos pela validação):

| Tarefa | Melhor na validação | Resultado no teste | Meta |
| --- | --- | --- | --- |
| Classificação | Árvore de Decisão | F1-macro 0,866 (acurácia 0,873); os 8 algoritmos ficam acima de 0,75 | F1 ≥ 0,75 ✔ |
| Regressão de IFN-γ | Random Forest | R² 0,941; os 6 algoritmos ficam acima de 0,70 | R² ≥ 0,70 ✔ |
| Clusterização (PPD_log × CD3_1) | Hierárquico, 2 grupos | silhueta 0,519 | silhueta ≥ 0,50 ✔ |


