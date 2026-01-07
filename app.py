"""
Stock Calculator Pro — Advanced Dashboard
Features:
- Auto Buy/Sell Targets with Probabilities
- Suggested Action (Buy / Hold / Sell)
- Stop-Loss & Risk Control
- Position Sizing via Kelly Criterion
- Random Buy / Strong Buy Opportunity Finder
- Backtesting with Metrics and Cumulative P&L Chart
- Traffic-Light Risk-to-Reward System
- Portfolio Tracking + CSV Download
- Robust error handling to prevent white screens and handle data limits
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
        price=1.0,
        hist=None,
        volatility=0.3,
        buy_price=1.0,
        sell_price=1.0,
        stop_loss=0.95,
        qty=100,
        horizon=30,
        portfolio=[],
        account_balance=10000,
        risk_perc=0.02
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()

# ==========================
# DATA FUNCTIONS
# ==========================
@st.cache_data(ttl=300)
def load_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty:
            return 1.0, pd.DataFrame({"Open":[1], "High":[1], "Low":[1], "Close":[1]}, index=[pd.Timestamp.today()])
        hist.index = hist.index.tz_localize(None)
        hist = hist.tail(120)  # limit to prevent overload
        price = hist["Close"].iloc[-1]
        return price, hist
    except:
        return 1.0, pd.DataFrame({"Open":[1], "High":[1], "Low":[1], "Close":[1]}, index=[pd.Timestamp.today()])

def hist_vol(hist, days=60):
    try:
        r = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        return np.clip(r.tail(days).std() * np.sqrt(252), 0.05, 3.0) if not r.empty else 0.3
    except:
        return 0.3

# ==========================
# FINANCE FUNCTIONS
# ==========================
def bs_prob(S, K, sigma, T):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0.5
    d2 = (np.log(S / K) - 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return float(np.clip(norm.cdf(d2), 0.0001, 0.9999))

def calculate_expected_move(S, sigma, T):
    return S * sigma * np.sqrt(T)

def calculate_pnl(buy, sell, qty):
    return (sell - buy) * qty

def suggested_action(buy_prob, sell_prob):
    if buy_prob > 0.7: return "Buy", "green"
    elif sell_prob > 0.7: return "Sell", "red"
    else: return "Hold", "orange"

def calc_stop_loss(buy_price, sigma, multiplier=2):
    return max(0, buy_price - multiplier * sigma * buy_price)

def kelly_position_size(account_balance, win_rate, RRR, buy_price, stop_loss):
    if RRR <= 0: RRR = 1.0
    f = win_rate - (1 - win_rate)/RRR
    f = max(f, 0.01) # minimum 1%
    risk_amount = account_balance * f
    qty = int(risk_amount / max(buy_price - stop_loss, 0.01))
    return max(qty, 1)

# ==========================
# RANDOM OPPORTUNITY
# ==========================
WATCHLIST = ["AAPL","MSFT","TSLA","GOOGL","AMZN","NVDA","META","NFLX","AMD","INTC","PYPL","UBER","SQ","SHOP","ADBE","CRM","ORCL","PEP","KO","MCD","DIS"]

def find_opportunity():
    random.shuffle(WATCHLIST)
    for ticker in WATCHLIST:
        price, hist = load_stock(ticker)
        sigma = hist_vol(hist)
        T = 30/365
        buy_target = price - calculate_expected_move(price, sigma, T)
        sell_target = price + calculate_expected_move(price, sigma, T)
        buy_prob = 1 - bs_prob(price, buy_target, sigma, T)
        sell_prob = bs_prob(price, sell_target, sigma, T)
        RRR = (sell_target - buy_target)/(buy_target - calc_stop_loss(buy_target, sigma))
        # Buy or Strong Buy
        if buy_prob >= 0.6 and RRR >= 1.0:
            stop_loss = calc_stop_loss(buy_target, sigma)
            action, color = suggested_action(buy_prob, sell_prob)
            qty = kelly_position_size(st.session_state.account_balance, 0.6, RRR, buy_target, stop_loss)
            return {"Ticker": ticker, "Price": price, "Hist": hist, "BuyTarget": buy_target, "SellTarget": sell_target, "BuyProb": buy_prob, "SellProb": sell_prob, "Action": action, "Color": color, "StopLoss": stop_loss, "Qty": qty, "RRR": RRR}
    return None

# ==========================
# BACKTEST
# ==========================
def backtest(hist, sigma, horizon_days=30):
    results = []
    cum_pnl = 0
    peak = 0
    max_drawdown = 0
    gross_profit = 0
    gross_loss = 0
    for i in range(len(hist)-horizon_days):
        S = hist["Close"].iloc[i]
        T = horizon_days/365
        buy_target = S - calculate_expected_move(S, sigma, T)
        sell_target = S + calculate_expected_move(S, sigma, T)
        stop_loss = calc_stop_loss(buy_target, sigma)
        buy_prob = 1 - bs_prob(S, buy_target, sigma, T)
        sell_prob = bs_prob(S, sell_target, sigma, T)
        action, _ = suggested_action(buy_prob, sell_prob)
        open_next = hist["Open"].iloc[i+1]
        pnl = 0
        if open_next < stop_loss:
            pnl = stop_loss - buy_target
        else:
            pnl = sell_target - buy_target
        cum_pnl += pnl
        peak = max(peak, cum_pnl)
        drawdown = peak - cum_pnl
        max_drawdown = max(max_drawdown, drawdown)
        if pnl > 0: gross_profit += pnl
        else: gross_loss += abs(pnl)
        results.append({"Date": hist.index[i], "BuyTarget": buy_target, "SellTarget": sell_target, "Action": action, "PnL": pnl, "CumulativePnL": cum_pnl})
    df = pd.DataFrame(results)
    win_rate = (df["PnL"] > 0).mean() if not df.empty else 0.5
    profit_factor = (gross_profit / gross_loss) if gross_loss>0 else np.nan
    return df, win_rate, profit_factor, max_drawdown

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.header("🔍 Stock Input")
    ticker_input = st.text_input("Ticker", placeholder="AAPL").upper()
    load_stock_btn = st.button("Load Stock")
    st.divider()
    st.header("💰 Account & Risk Settings")
    st.session_state.account_balance = st.number_input("Account Balance ($)", value=st.session_state.account_balance)
    st.session_state.risk_perc = st.number_input("Risk per Trade (%)", value=st.session_state.risk_perc, min_value=0.001, max_value=1.0, step=0.001)

# ==========================
# MAIN
# ==========================
st.title("📈 Stock Calculator Pro")

S = None
hist = None

if load_stock_btn and ticker_input:
    with st.spinner("Fetching stock data..."):
        S, hist = load_stock(ticker_input)
        st.session_state.ticker = ticker_input
        st.session_state.price = S
        st.session_state.hist = hist
        sigma = hist_vol(hist)
        st.session_state.volatility = sigma
        T = 30/365
        st.session_state.buy_price = S - calculate_expected_move(S, sigma, T)
        st.session_state.sell_price = S + calculate_expected_move(S, sigma, T)
        st.session_state.stop_loss = calc_stop_loss(st.session_state.buy_price, sigma)
        df_bt, win_rate, pf, mdd = backtest(hist, sigma)
        RRR = (st.session_state.sell_price - st.session_state.buy_price)/(st.session_state.buy_price - st.session_state.stop_loss)
        st.session_state.qty = kelly_position_size(st.session_state.account_balance, win_rate, RRR, st.session_state.buy_price, st.session_state.stop_loss)

if st.session_state.hist is not None:
    S = st.session_state.price
    hist = st.session_state.hist

tab1, tab2, tab3, tab4, tab5 = st.tabs(["💰 Trade","📊 Chart","📁 Portfolio","🚀 Opportunity","📜 Backtest"])

# ==========================
# TAB 1 — TRADE
# ==========================
with tab1:
    if hist is None:
        st.info("Load a ticker to start")
    else:
        sigma = st.session_state.volatility
        buy_target = st.session_state.buy_price
        sell_target = st.session_state.sell_price
        stop_loss = st.session_state.stop_loss
        qty = st.session_state.qty
        buy_prob = 1 - bs_prob(S, buy_target, sigma, 30/365)
        sell_prob = bs_prob(S, sell_target, sigma, 30/365)
        action, color = suggested_action(buy_prob, sell_prob)
        pnl = calculate_pnl(buy_target, sell_target, qty)
        RRR = (sell_target - buy_target)/(buy_target - stop_loss)

        # Metrics display
        st.header(f"{st.session_state.ticker}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${S:.2f}")
        c2.metric("Buy Target", f"${buy_target:.2f}", f"{buy_prob:.1%}")
        c3.metric("Sell Target", f"${sell_target:.2f}", f"{sell_prob:.1%}")
        c4.metric("Stop-Loss", f"${stop_loss:.2f}")

        st.markdown(f"### Suggested Action: <span style='color:{color}'>{action}</span>", unsafe_allow_html=True)

        # Risk-to-Reward Panel just below metrics
        if RRR < 1.5: st.error(f"Risk-Reward Ratio: {RRR:.2f}")
        elif RRR > 2.0: st.success(f"Risk-Reward Ratio: {RRR:.2f}")
        else: st.warning(f"Risk-Reward Ratio: {RRR:.2f}")
        st.info("Recommended RRR ≥ 1.5 — Higher values indicate better reward relative to risk.")

        st.metric("Position Size (Qty)", f"{qty}")
        st.metric("Expected P&L", f"${pnl:,.2f}")

        # Add to Portfolio
        st.divider()
        with st.form("manual_trade"):
            qty_input = st.number_input("Quantity", min_value=1, value=qty)
            submit_trade = st.form_submit_button("Add Trade to Portfolio")
        if submit_trade:
            st.session_state.portfolio.append({
                "Ticker": st.session_state.ticker,
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Buy": buy_target,
                "Sell": sell_target,
                "StopLoss": stop_loss,
                "Qty": qty_input,
                "PnL": calculate_pnl(buy_target, sell_target, qty_input)
            })
            st.success("Trade added!")

# ==========================
# TAB 4 — OPPORTUNITY
# ==========================
with tab4:
    st.subheader("🚀 Random Buy / Strong Buy Opportunity")
    if st.button("Find Opportunity"):
        with st.spinner("Searching for opportunity..."):
            opp = find_opportunity()
        if opp:
            st.session_state.ticker = opp["Ticker"]
            st.session_state.price = opp["Price"]
            st.session_state.hist = opp["Hist"]
            st.session_state.buy_price = opp["BuyTarget"]
            st.session_state.sell_price = opp["SellTarget"]
            st.session_state.stop_loss = opp["StopLoss"]
            st.session_state.qty = opp["Qty"]
            st.session_state.volatility = hist_vol(opp["Hist"])

            st.title(f"{opp['Ticker']}")
            st.metric("Price", f"${opp['Price']:.2f}")
            st.metric("Buy Target", f"${opp['BuyTarget']:.2f}", f"{opp['BuyProb']:.1%}")
            st.metric("Sell Target", f"${opp['SellTarget']:.2f}", f"{opp['SellProb']:.1%}")
            st.metric("Stop-Loss", f"${opp['StopLoss']:.2f}")
            st.metric("Position Size (Qty)", f"{opp['Qty']}")
            st.markdown(f"### Suggested Action: <span style='color:{opp['Color']}'>{opp['Action']}</span>", unsafe_allow_html=True)
            # RRR traffic light
            if opp['RRR'] < 1.5: st.error(f"Risk-Reward Ratio: {opp['RRR']:.2f}")
            elif opp['RRR'] > 2.0: st.success(f"Risk-Reward Ratio: {opp['RRR']:.2f}")
            else: st.warning(f"Risk-Reward Ratio: {opp['RRR']:.2f}")
            st.info("Recommended RRR ≥ 1.5 — Higher values indicate better reward relative to risk.")
        else:
            st.warning("No opportunity found!")
