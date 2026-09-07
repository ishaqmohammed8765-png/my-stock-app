"""Stock Research: transparent signals and reproducible historical evaluation."""

import os
import time

import streamlit as st

from ui.settings import settings
from ui.theme import apply_theme
from ui.views import charts, news_view, number, overview, research, screener
from utils.data_loader import demo_market, is_stale, valid_symbol
from utils.indicators import add_indicators
from utils.services import market

st.set_page_config(page_title="Stock Research", page_icon="◈", layout="wide")
apply_theme()


def credential(name):
    try:
        return str(st.secrets.get(name, os.getenv(name, "")))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return os.getenv(name, "")


with st.sidebar:
    st.markdown("## ◈ Stock Research")
    st.caption("Observe. Test. Paper trade.")
    demo = st.checkbox("Explore synthetic demo", key="demo")
    symbol = st.text_input("Ticker", "AAPL", key="ticker").strip().upper()
    years = st.selectbox("Historical years", [5, 10, 2], key="years")
    provider = st.selectbox("Daily data provider", ["Yahoo", "Alpaca"], key="provider")
    load = st.button("Load / refresh data", width="stretch", type="primary")
strategy, execution = settings()
st.markdown('<div class="eyebrow">RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
st.title("Make the evidence visible.")
st.caption(
    "Transparent rules, realistic costs and a separate historical holdout. No automated orders or guaranteed returns."
)
page = st.radio(
    "Workspace",
    ["Overview", "Charts", "Evaluate", "Screener", "News"],
    horizontal=True,
    label_visibility="collapsed",
    key="page",
)
zoya_key = credential("ZOYA_API_KEY")
request = (symbol, years, provider, demo)
if st.session_state.get("market_request") != request or load:
    st.session_state.pop("market", None)
    st.session_state.pop("evaluation", None)
    st.session_state["market_request"] = request
if load:
    st.session_state["refresh"] = time.time_ns()
refresh = st.session_state.get("refresh", 0)
if page == "Screener":
    screener(strategy, zoya_key, refresh, demo)
    st.stop()
if not valid_symbol(symbol):
    st.error("Enter a valid ticker, such as AAPL, MSFT or BRK-B.")
    st.stop()
if demo and "market" not in st.session_state:
    st.session_state["market"] = demo_market(symbol)
elif load:
    try:
        with st.spinner(f"Loading completed sessions for {symbol}…"):
            st.session_state["market"] = market(
                symbol,
                years,
                provider,
                credential("ALPACA_KEY"),
                credential("ALPACA_SECRET"),
                refresh,
            )
    except Exception:
        st.error(
            "Historical data could not be loaded or failed validation. Try another ticker or provider, or use the offline demo."
        )
if "market" not in st.session_state:
    st.info(
        "Choose a ticker and select Load / refresh data. You can also explore the synthetic demo without API access."
    )
    for col, title, text in zip(
        st.columns(3),
        ["01 · Understand", "02 · Challenge", "03 · Observe forward"],
        [
            "See setup conditions, with unknown and stale data clearly marked.",
            "Compare later-period results with buy & hold, doubled costs and separate time windows.",
            "Export a paper-trade plan and record real observations before drawing conclusions.",
        ],
    ):
        col.markdown("### " + title)
        col.write(text)
    st.stop()
data = st.session_state["market"]
bars = add_indicators(data.bars)
if data.demo:
    st.warning(
        f"SYNTHETIC DEMO · These are generated prices, not observations of {symbol}. Results say nothing about this stock's performance."
    )
last = bars.iloc[-1]
cols = st.columns(4)
cols[0].metric(
    symbol + (" · DEMO" if data.demo else ""),
    number(last.close) + " " + (data.currency or "unverified"),
)
cols[1].metric("RSI (14)", number(last.rsi14, decimals=1))
cols[2].metric("Relative volume", number(last.rvol, "×"))
cols[3].metric("ADX (14)", number(last.adx14, decimals=1))
st.caption(
    f"{data.source} · Completed data through {bars.index[-1].date()} · {len(bars):,} sessions · USD research account"
)
if is_stale(data):
    st.warning(
        "Stale data: current setups are blocked. Historical evaluation remains available with dates shown."
    )
if data.currency != "USD":
    st.warning("Only verified USD instruments are supported for trade planning and evaluation.")
if data.warnings:
    with st.expander("Data quality & provider notes"):
        for warning in data.warnings:
            st.write(warning)
st.divider()
if page == "Overview":
    overview(data, bars, strategy, execution, zoya_key, refresh)
elif page == "Charts":
    charts(bars)
elif page == "Evaluate":
    research(data, bars, strategy, execution)
else:
    news_view(data, refresh)
