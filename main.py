import pandas as pd
import streamlit as st
import pygwalker as pyg
import streamlit.components.v1 as components
from pygwalker.api.streamlit import StreamlitRenderer

df = pd.read_csv('diamond_data.csv')

st.set_page_config(
    page_title="Streamlit Using Pygwalker",
    layout="wide"
)

st.title("Streamlit Using Pygwalker")

pyg_app = StreamlitRenderer(df)

pyg_app.explorer()


