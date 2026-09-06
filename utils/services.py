"""Bounded provider caches; explicit refresh applies to every provider."""

import streamlit as st

from .data_loader import load_market, load_news
from .zoya_api import fetch_compliance


@st.cache_data(ttl=900, max_entries=100, show_spinner=False)
def market(symbol, years, provider, key, secret, refresh=0):
    return load_market(symbol, years, provider, key, secret)


@st.cache_data(ttl=600, max_entries=100, show_spinner=False)
def news(symbol, refresh=0):
    return load_news(symbol)


@st.cache_data(ttl=3600, max_entries=100, show_spinner=False)
def compliance(symbol, api_key, refresh=0):
    return fetch_compliance(symbol, api_key)
