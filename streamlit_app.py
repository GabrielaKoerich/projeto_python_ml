import streamlit as st

from src.paginas import Paginas

st.set_page_config(page_title="Predição com Biomarcadores", layout="wide")
Paginas().montar()
