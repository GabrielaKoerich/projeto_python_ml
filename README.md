# Sistema de Predição com Biomarcadores

![CI](https://github.com/GabrielaKoerich/projeto_python_ML/actions/workflows/ci.yml/badge.svg)

Projeto final da disciplina **Python Machine Learning**.

- **App público (deploy):** https://SEU-APP.streamlit.app
- **AED comentada:** [`notebooks/01_AED_relatorio_comentado.ipynb`](notebooks/01_AED_relatorio_comentado.ipynb)
- **Testes automatizados:** [`tests/`](tests) — rodam a cada push no GitHub Actions

## Pergunta e objetivos
A partir de biomarcadores de 1000 pacientes:
1. **Classificação** — prever a `classe` (BCG, cancer, imunodeficiencia, leucemia). Sucesso: F1-macro ≥ 0,75.
2. **Regressão** — prever o valor de `IFN-γ`. Sucesso: R² ≥ 0,70.
3. **Clusterização** — encontrar grupos naturais com K-means (k escolhido pela silhueta).

## Estrutura
```
dados/biomarcadores.csv      dados brutos
notebooks/                   AED comentada (1ª entrega)
src/dados.py                 aquisição, estruturação, divisão 70/15/15
src/modelos.py               classes Clusterizador, Classificador, Regressor
src/pipeline.py              fluxo completo de treino
src/paginas.py               páginas do Streamlit
streamlit_app.py             ponto de entrada do app (2ª entrega)
docker/ + docker-compose.yml ambiente Docker (Python 3.12, Jupyter Lab, Streamlit)
dev/*.ps1                    atalhos: subir container, testes, app
tests/                       testes unitários, integração e ponta a ponta (3ª entrega)
.github/workflows/ci.yml     CI: ruff, flake8, mypy, pytest + cobertura
```

## Como rodar (Docker)
Pré-requisito: Docker Desktop deve estar aberto. No PowerShell, na pasta do projeto:
```powershell
.\dev\subir_docker.ps1      # constrói a imagem e sobe o container "python-ml"
.\dev\testes.ps1            # pytest + cobertura, ruff, mypy, flake8 no container
.\dev\roda_streamlit.ps1    # app em http://localhost:8501
```
Jupyter Lab (AED) (relatório comentado) fica em http://localhost:8888. Para desligar: `docker compose down`.

Arquivos: `docker/Dockerfile`, `docker/requirements-docker.txt`, `docker-compose.yml`.
O CI do GitHub também constrói a imagem e roda os testes dentro do container.

## Resultados (conjunto de teste, 150 pacientes)
O app mostra todos os algoritmos, cada um com métricas, teste de hipótese e gráfico.

