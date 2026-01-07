"""
Stock Calculator Pro — Smart Dashboard
Features:
- Auto buy/sell targets with probability
- Suggested action (Buy / Hold / Sell)
- Random opportunity finder
- Candlestick chart with expected moves
- Portfolio tracking (in-memory + CSV download)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go
import random

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="Stock Calculator Pro",
    page_icon="📈",
    layout="wide"
)

# ==========================
# SESSION STATE INIT
# ==========================
def init_state():
    defaults = dict(
        ticker=None,
        price=None,
        hist=None,
        volatility=0.3,
        vol_method="Historical",
        buy_price=None,
        sell_price=None,
        qty=100,
        horizon=30,
        portfolio=[]
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()

# ==========================
# DATA FUNCTIONS
# ==========================
@st.cache_data(ttl=300)
def load_stock(ticker):
    """Load historical data safely"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty:
            hist = pd.DataFrame({
                "Open": [1], "High": [1], "Low": [1], "Close": [1]
            }, index=[pd.Timestamp.today()])
            price = 1.0
        else:
            hist.index = hist.index.tz_localize(None)
            hist = hist.tail(120)  # limit to last ~6 months
            price = hist["Close"].iloc[-1]
        return price, hist
    except Exception:
        hist = pd.DataFrame({
            "Open": [1], "High": [1], "Low": [1], "Close": [1]
        }, index=[pd.Timestamp.today()])
        return 1.0, hist

def hist_vol(hist, days=60):
    r = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    if r.empty: return 0.3
    return np.clip(r.tail(days).std() * np.sqrt(252), 0.05, 3.0)

def ewma_vol(hist, span=20):
    r = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    if r.empty: return 0.3
    return np.clip(r.ewm(span=span).std().iloc[-1] * np.sqrt(252), 0.05, 3.0)

# ==========================
# FINANCE FUNCTIONS
# ==========================
def bs_prob(S, K, sigma, T):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0.5
    d2 = (np.log(S / K) - 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return float(np.clip(norm.cdf(d2), 0.0001, 0.9999))

def interpret_prob(p):
    if p < 0.2: return "Very unlikely"
    if p < 0.4: return "Unlikely"
    if p < 0.6: return "Plausible"
    if p < 0.8: return "Likely"
    return "Very likely"

def calculate_expected_move(S, sigma, T):
    return S * sigma * np.sqrt(T)

def calculate_pnl(buy, sell, qty):
    return (sell - buy) * qty

def suggested_action(buy_prob, sell_prob):
    if buy_prob > 0.7: return "Buy", "green"
    elif sell_prob > 0.7: return "Sell", "red"
    else: return "Hold", "orange"

# ==========================
# RANDOM OPPORTUNITY
# ==========================
WATCHLIST = [
    "AAPL","MSFT","TSLA","GOOGL","AMZN","NVDA","META","NFLX",
    "AMD","INTC","PYPL","UBER","SQ","SHOP","ADBE","CRM","ORCL",
    "PEP","KO","MCD","DIS"
]

def find_opportunity():
    ticker = random.choice(WATCHLIST)
    price, hist = load_stock(ticker)
    sigma = hist_vol(hist)
    T = 30 / 365  # 30-day horizon
    buy_target = price - calculate_expected_move(price, sigma, T)
    sell_target = price + calculate_expected_move(price, sigma, T)
    buy_prob = 1 - bs_prob(price, buy_target, sigma, T)
    sell_prob = bs_prob(price, sell_target, sigma, T)
    action, color = suggested_action(buy_prob, sell_prob)
    return {
        "Ticker": ticker,
        "Price": price,
        "Buy Target": buy_target,
        "Sell Target": sell_target,
        "Buy Prob": buy_prob,
        "Sell Prob": sell_prob,
        "Action": action,
        "Color": color,
        "Hist": hist
    }

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.header("🔍 Stock Input")
    ticker_input = st.text_input("Ticker", placeholder="AAPL").upper()
    load_stock_btn = st.button("Load Stock")

# ==========================
# MAIN
# ==========================
st.title("📈 Stock Calculator Pro — Smart Dashboard")
S = None
hist = None

# --- Load ticker ---
if load_stock_btn and ticker_input:
    with st.spinner("Fetching stock data..."):
        S, hist = load_stock(ticker_input)
        st.session_state.ticker = ticker_input
        st.session_state.price = S
        st.session_state.hist = hist
        # compute volatility
        st.session_state.volatility = hist_vol(hist)
        st.session_state.buy_price = S - calculate_expected_move(S, st.session_state.volatility, 30/365)
        st.session_state.sell_price = S + calculate_expected_move(S, st.session_state.volatility, 30/365)

# fallback
if st.session_state.hist is not None:
    S = st.session_state.price
    hist = st.session_state.hist

tab1, tab2, tab3, tab4 = st.tabs(["💰 Trade", "📊 Chart", "📁 Portfolio", "🚀 Opportunity"])

# ==========================
# TAB 1 — TRADE
# ==========================
with tab1:
    if hist is None:
        st.info("Load a ticker to start")
    else:
        st.subheader(f"📌 {st.session_state.ticker} — Recommended Trade")
        sigma = st.session_state.volatility
        T = st.session_state.horizon / 365
        buy_target = st.session_state.buy_price
        sell_target = st.session_state.sell_price
        buy_prob = 1 - bs_prob(S, buy_target, sigma, T)
        sell_prob = bs_prob(S, sell_target, sigma, T)
        action, color = suggested_action(buy_prob, sell_prob)
        pnl = calculate_pnl(buy_target, sell_target, st.session_state.qty)

        # --- Metrics layout ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Price", f"${S:.2f}")
        c2.metric("Buy Target", f"${buy_target:.2f}", f"{buy_prob:.1%}")
        c3.metric("Sell Target", f"${sell_target:.2f}", f"{sell_prob:.1%}")

        st.markdown(f"### Suggested Action: <span style='color:{color}'>{action}</span>", unsafe_allow_html=True)
        st.metric("Expected P&L", f"${pnl:,.2f}")

        st.divider()
        with st.form("manual_trade"):
            qty = st.number_input("Quantity", min_value=1, value=st.session_state.qty)
            submit_trade = st.form_submit_button("Add Trade to Portfolio")
        if submit_trade:
            st.session_state.portfolio.append({
                "Ticker": st.session_state.ticker,
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Buy": buy_target,
                "Sell": sell_target,
                "Qty": qty,
                "PnL": pnl
            })
            st.success("Trade added!")

# ==========================
# TAB 2 — CHART
# ==========================
with tab2:
    if hist is not None:
        move = calculate_expected_move(S, st.session_state.volatility, st.session_state.horizon/365)
        fig = go.Figure()
        fig.add_candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"]
        )
        fig.add_hrect(y0=S - move, y1=S + move, fillcolor="green", opacity=0.06, line_width=0)
        fig.add_hrect(y0=S - 2*move, y1=S + 2*move, fillcolor="red", opacity=0.04, line_width=0)
        fig.add_hline(y=st.session_state.buy_price, line_dash="dash", line_color="blue")
        fig.add_hline(y=st.session_state.sell_price, line_dash="dash", line_color="green")
        fig.update_layout(height=650, title=f"{st.session_state.ticker} — Expected Move", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load a ticker to view chart")

# ==========================
# TAB 3 — PORTFOLIO
# ==========================
with tab3:
    if st.session_state.portfolio:
        df = pd.DataFrame(st.session_state.portfolio)
        st.dataframe(df, use_container_width=True)
        st.metric("Total P&L", f"${df['PnL'].sum():,.2f}")
        csv = df.to_csv(index=False)
        st.download_button("Download Portfolio CSV", csv, "portfolio.csv")
    else:
        st.info("No trades yet")

# ==========================
# TAB 4 — OPPORTUNITY
# ==========================
with tab4:
    st.subheader("🚀 Random Opportunity Finder")
    if st.button("Find Opportunity"):
        with st.spinner("Fetching random stock..."):
            opp = find_opportunity()
        S = opp["Price"]
        hist = opp["Hist"]
        st.session_state.ticker = opp["Ticker"]
        st.session_state.price = S
        st.session_state.hist = hist
        st.session_state.buy_price = opp["Buy Target"]
        st.session_state.sell_price = opp["Sell Target"]
        st.session_state.volatility = hist_vol(hist)
        # display
        st.metric(f"{opp['Ticker']} Price", f"${S:.2f}")
        st.metric("Buy Target", f"${opp['Buy Target']:.2f}", f"{opp['BuyProb']:.1%}")
        st.metric("Sell Target", f"${opp['Sell Target']:.2f}", f"{opp['SellProb']:.1%}")
        st.markdown(f"### Suggested Action: <span style='color:{opp['Color']}'>{opp['Action']}</span>", unsafe_allow_html=True)
