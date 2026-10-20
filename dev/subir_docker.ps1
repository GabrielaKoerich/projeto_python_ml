# Constrói a imagem (na 1a vez) e sobe o container em segundo plano.
# Uso:  .\dev\subir_docker.ps1
docker compose up -d --build
Write-Host ""
Write-Host "Container 'python-ml' no ar." -ForegroundColor Green
Write-Host "Jupyter Lab (AED): http://localhost:8888/lab?token=pythonml"
Write-Host "Para o app:        .\dev\roda_streamlit.ps1"
