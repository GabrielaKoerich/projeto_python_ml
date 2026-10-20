# Roda o app Streamlit DENTRO do container
# (equivalente ao roda_streamlit.sh da disciplina).
# Uso:  .\dev\roda_streamlit.ps1   -> abra http://localhost:8501
# Para parar: Ctrl+C
docker exec -it python-ml streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
