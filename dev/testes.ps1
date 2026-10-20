# Roda testes e checagens de qualidade DENTRO do container
# (equivalente ao testes_python.sh da disciplina).
# Uso:  .\dev\testes.ps1
$c = "python-ml"

Write-Host "==== PyTest (unitários, integração e cobertura) ====" -ForegroundColor Green
docker exec $c pytest --disable-warnings --cov=src --cov-report=term --cov-report=html

Write-Host "==== Ruff (formatação) ====" -ForegroundColor Green
docker exec $c ruff format .

Write-Host "==== MyPy (tipos) ====" -ForegroundColor Green
docker exec $c mypy src --ignore-missing-imports

Write-Host "==== Flake8 (estilo e complexidade) ====" -ForegroundColor Green
docker exec $c flake8 --max-complexity 12 src tests streamlit_app.py
