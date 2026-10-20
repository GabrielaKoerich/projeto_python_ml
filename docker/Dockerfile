# Imagem do projeto: Python 3.12 + bibliotecas do app, dos testes e da AED
FROM python:3.12-slim

LABEL maintainer="Seu Nome"

# Pasta de trabalho dentro do container
WORKDIR /app

# Instala as dependências primeiro (aproveita o cache do Docker)
COPY requirements.txt requirements-dev.txt docker/requirements-docker.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copia o projeto (no docker compose a pasta também é montada como volume)
COPY . .

# 8888 = Jupyter Lab (AED) | 8501 = Streamlit (app)
EXPOSE 8888 8501

# Ao subir, o container abre o Jupyter Lab (igual ao container da disciplina)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", \
     "--no-browser", "--ServerApp.token=pythonml"]
