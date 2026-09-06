import streamlit as st


def apply_theme():
    st.markdown(
        """<style>
    .block-container {max-width:1250px; padding-top:2rem; padding-bottom:3rem;}
    h1,h2,h3 {letter-spacing:-.035em;}
    [data-testid="stMetric"] {padding:1rem 1.1rem; background:#151e2b; border:1px solid #253246; border-radius:12px;}
    [data-testid="stSidebar"] {border-right:1px solid #253246;}
    [data-testid="stMetricValue"] {font-size:1.65rem;}
    .eyebrow {color:#55d6be; font-size:.75rem; letter-spacing:.16em; font-weight:700; margin-bottom:.4rem;}
    </style>""",
        unsafe_allow_html=True,
    )
