"""
Stock Calculator Pro — Enhanced Trading Dashboard
Features: Auto buy/sell targets, probability indicators, random opportunity finder.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go
import json
import os
import random

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="Stock Calculator Pro",
    page_icon="📈",
    layout="wide"
)

PORTFOLIO_FILE = "portfolio.json"

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
# PORTFOLIO PERSISTENCE
# ==========================
def save_portfolio():
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(st.session_state.portfolio, f, indent=2)

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            st.session_state.portfolio = json.load(f)

load_portfolio()

# ==========================
# DATA FUNCTIONS
# ==========================
@st.cache_data(ttl=300)
def load_stock(ticker):
    """Load 6 months of historical stock data."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty:
            return None, None
        hist.index = hist.index.tz_localize(None)
        price = hist["Close"].iloc[-1]
        return price, hist
    except Exception:
        return None, None

def hist_vol(hist, days=60):
    r = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    return np.clip(r.tail(days).std() * np.sqrt(252), 0.05, 3.0)

def ewma_vol(hist, span=20):
    r = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    return np.clip(r.ewm(span=span).std().iloc[-1] * np.sqrt(252), 0.05, 3.0)

# ==========================
# FINANCE FUNCTIONS
# ==========================
def bs_prob(S, K, sigma, T):
    """Estimate probability of hitting target price using Black-Scholes approximation."""
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

# ==========================
# RANDOM OPPORTUNITY SCANNER
# ==========================
WATCHLIST = [
    "AAPL","MSFT","TSLA","GOOGL","AMZN","NVDA","META","NFLX",
    "AMD","INTC","PYPL","UBER","SQ","SHOP","ADBE","CRM","ORCL",
    "PEP","KO","MCD","DIS"
]

def find_opportunity():
    ticker = random.choice(WATCHLIST)
    price, hist = load_stock(ticker)
    if hist is None:
        return None
    sigma = hist_vol(hist)
    T = 30 / 365  # assume 30-day horizon
    buy_target = price - calculate_expected_move(price, sigma, T)
    sell_target = price + calculate_expected_move(price, sigma, T)
    buy_prob = 1 - bs_prob(price, buy_target, sigma, T)
    sell_prob = bs_prob(price, sell_target, sigma, T)
    return {
        "Ticker": ticker,
        "Price": price,
        "Buy Target": buy_target,
        "Sell Target": sell_target,
        "Buy Prob": buy_prob,
        "Sell Prob": sell_prob,
        "Buy Indicator": interpret_prob(buy_prob)
    }

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.header("🔍 Stock Selection")

    t = st.text_input("Ticker", placeholder="AAPL").upper()
    if st.button("Load"):
        price, hist = load_stock(t)
        if hist is None:
            st.error("Invalid ticker or network issue")
        else:
            st.session_state.ticker = t
            st.session_state.price = price
            st.session_state.hist = hist
            st.session_state.buy_price = price * 0.95
            st.session_state.sell_price = price * 1.05

    if st.session_state.hist is not None:
        st.divider()
        st.metric("Price", f"${st.session_state.price:.2f}")

        st.session_state.vol_method = st.radio(
            "Volatility method",
            ["Historical", "EWMA"]
        )

        if st.session_state.vol_method == "Historical":
            st.session_state.volatility = hist_vol(st.session_state.hist)
        else:
            st.session_state.volatility = ewma_vol(st.session_state.hist)

        st.metric("σ (annual)", f"{st.session_state.volatility*100:.1f}%")

# ==========================
# MAIN LAYOUT
# ==========================
st.title("📈 Stock Calculator Pro — Enhanced")

if st.session_state.hist is None:
    st.info("Load a ticker to begin")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["💰 Trade", "📊 Chart", "📁 Portfolio", "🚀 Opportunity Finder"])

# ==========================
# TAB 1 — TRADE
# ==========================
with tab1:
    st.subheader("⚡ Trade Presets & Manual Entry")

    c1, c2, c3 = st.columns(3)
    if c1.button("Scalp (5d)"): st.session_state.horizon = 5
    if c2.button("Swing (30d)"): st.session_state.horizon = 30
    if c3.button("Position (90d)"): st.session_state.horizon = 90

    st.divider()

    with st.form("trade_form"):
        c1, c2 = st.columns(2)
        with c1:
            buy = st.number_input("Buy Price", value=st.session_state.buy_price)
            qty = st.number_input("Quantity", min_value=1, value=st.session_state.qty)
        with c2:
            sell = st.number_input("Sell Price", value=st.session_state.sell_price)
            days = st.number_input("Days", min_value=1, value=st.session_state.horizon)

        submit = st.form_submit_button("Calculate")

    if submit:
        st.session_state.buy_price = buy
        st.session_state.sell_price = sell
        st.session_state.qty = qty
        st.session_state.horizon = days

        S = st.session_state.price
        σ = st.session_state.volatility
        T = days / 365

        buy_p = 1 - bs_prob(S, buy, σ, T)
        sell_p = bs_prob(S, sell, σ, T)
        move = calculate_expected_move(S, σ, T)
        pnl = calculate_pnl(buy, sell, qty)

        c1, c2 = st.columns(2)
        c1.metric("Buy Probability", f"{buy_p:.1%}", interpret_prob(buy_p))
        c2.metric("Sell Probability", f"{sell_p:.1%}", interpret_prob(sell_p))

        st.divider()
        st.metric("Expected P&L", f"${pnl:,.2f}")

        if abs(buy - S) > 2 * move:
            st.warning("Buy target > 2σ away")
        if abs(sell - S) > 2 * move:
            st.warning("Sell target > 2σ away")

        if st.button("Add to Portfolio"):
            st.session_state.portfolio.append({
                "Ticker": st.session_state.ticker,
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Buy": buy,
                "Sell": sell,
                "Qty": qty,
                "PnL": pnl
            })
            save_portfolio()
            st.success("Trade added to portfolio")

# ==========================
# TAB 2 — CHART
# ==========================
with tab2:
    hist = st.session_state.hist
    S = st.session_state.price
    σ = st.session_state.volatility
    T = st.session_state.horizon / 365
    move = calculate_expected_move(S, σ, T)

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

# ==========================
# TAB 3 — PORTFOLIO
# ==========================
with tab3:
    if st.session_state.portfolio:
        df = pd.DataFrame(st.session_state.portfolio)
        st.dataframe(df, use_container_width=True)
        st.metric("Total P&L", f"${df['PnL'].sum():,.2f}")
        if st.button("Export Portfolio to CSV"):
            df.to_csv("portfolio_export.csv", index=False)
            st.success("Exported to portfolio_export.csv")
    else:
        st.info("No trades yet")

# ==========================
# TAB 4 — OPPORTUNITY FINDER
# ==========================
with tab4:
    st.subheader("🚀 Random Opportunity Scanner")
    if st.button("Find Opportunity"):
        opp = find_opportunity()
        if opp:
            st.metric(f"{opp['Ticker']} Price", f"${opp['Price']:.2f}")
            st.metric(f"Suggested Buy", f"${opp['Buy Target']:.2f}", opp['Buy Indicator'])
            st.metric(f"Suggested Sell", f"${opp['Sell Target']:.2f}", f"{opp['Sell Prob']:.1%} chance")
            if opp['Buy Prob'] > 0.7:
                st.success("Strong Buy Opportunity!")
        else:
            st.error("Could not fetch stock data. Try again.")
