import streamlit as st

from src.paginas import Paginas

st.set_page_config(
    page_title="Métricas de ML com Biomarcadores", page_icon="🧬", layout="wide"
)
Paginas().montar()
