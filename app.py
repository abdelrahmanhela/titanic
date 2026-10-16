"""Titanic Explorer — entry point (router).

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Titanic Explorer",
    page_icon=":material/directions_boat:",
    layout="wide",
)

pages = [
    st.Page("app_pages/overview.py", title="Overview", icon=":material/home:"),
    st.Page("app_pages/data_prep.py", title="Data prep", icon=":material/filter_alt:"),
    st.Page("app_pages/eda.py", title="Survival patterns", icon=":material/query_stats:"),
    st.Page("app_pages/model.py", title="Model", icon=":material/model_training:"),
    st.Page("app_pages/predict.py", title="Predict", icon=":material/upload:"),
]

pg = st.navigation(pages)
pg.run()
