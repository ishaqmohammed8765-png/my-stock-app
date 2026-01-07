import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Stock Calculator Pro", layout="wide")

# =====================================================
# SAFE DATA LOAD
# =====================================================
@st.cache_data(ttl=300)
def load_stock(ticker):
    try:
        df = yf.download(ticker, period="6mo", progress=False)
        df = df.tail(120)
        if df.empty:
            raise ValueError
        df.index = df.index.tz_localize(None)
        return df
    except:
        return pd.DataFrame()

def annual_volatility(df):
    r = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if len(r) < 10:
        return 0.3
    return max(min(r.std() * np.sqrt(252), 1.5), 0.15)

def expected_move(price, vol, days=30):
    return price * vol * np.sqrt(days / 365)

def prob_hit(S, K, vol, days=30):
    T = days / 365
    d2 = (np.log(S / K) - 0.5 * vol**2 * T) / (vol * np.sqrt(T))
    return float(np.clip(norm.cdf(d2), 0.001, 0.999))

# =====================================================
# SESSION STATE
# =====================================================
st.session_state.setdefault("portfolio", [])

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    ticker = st.text_input("Stock Symbol", value="AAPL").upper()
    load = st.button("Load Stock")

# =====================================================
# LOAD MAIN STOCK
# =====================================================
if load:
    df = load_stock(ticker)
    if df.empty:
        st.error("No data available")
        st.stop()
    st.session_state.df = df
    st.session_state.ticker = ticker

if "df" not in st.session_state:
    st.info("Load a stock to begin")
    st.stop()

df = st.session_state.df
ticker = st.session_state.ticker
price = float(df["Close"].iloc[-1])
vol = annual_volatility(df)

# =====================================================
# LEVELS
# =====================================================
days = 30
move = expected_move(price, vol, days)
buy = price - move
sell = price + move
stop = buy - move * 0.5

buy_prob = 1 - prob_hit(price, buy, vol, days)
sell_prob = prob_hit(price, sell, vol, days)
RRR = (sell - buy) / max(buy - stop, 0.01)

# =====================================================
# KELLY
# =====================================================
returns = df["Close"].pct_change().dropna()
wins = returns[returns > 0]
losses = returns[returns < 0]

if len(wins) < 10 or len(losses) < 10:
    kelly_fraction = 0.02
else:
    win_rate = len(wins) / len(returns)
    loss_rate = 1 - win_rate
    R = wins.mean() / abs(losses.mean())
    kelly_fraction = win_rate - (loss_rate / R)

kelly_fraction = max(min(kelly_fraction * 0.5, 0.15), 0.01)
account_size = 10_000
qty = int((account_size * kelly_fraction) / max(buy - stop, 0.01))

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["💰 Trade", "📊 Chart", "📜 Backtest", "📁 Portfolio", "🚀 Opportunity"]
)

# =====================================================
# TRADE TAB
# =====================================================
with tab1:
    st.title(ticker)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${price:.2f}")
    c2.metric("Buy Target", f"${buy:.2f}", f"{buy_prob:.1%}")
    c3.metric("Sell Target", f"${sell:.2f}", f"{sell_prob:.1%}")
    c4.metric("Stop-Loss", f"${stop:.2f}")

    if RRR >= 2:
        st.success(f"Risk-Reward Ratio: {RRR:.2f}")
    elif RRR >= 1.5:
        st.warning(f"Risk-Reward Ratio: {RRR:.2f}")
    else:
        st.error(f"Risk-Reward Ratio: {RRR:.2f}")

    st.info("Recommended Risk-Reward ≥ 1.5")

    st.metric("Kelly Position Size", qty)
    st.metric("Expected Profit", f"${(sell-buy)*qty:,.2f}")

# =====================================================
# CHART TAB
# =====================================================
with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
    fig.add_hline(y=buy, line_dash="dot", annotation_text="Buy")
    fig.add_hline(y=sell, line_dash="dot", annotation_text="Sell")
    fig.add_hline(y=stop, line_dash="dot", annotation_text="Stop")
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# BACKTEST TAB
# =====================================================
with tab3:
    pnl = [df["Close"].iloc[i+days] - df["Close"].iloc[i]
           for i in range(len(df)-days)]
    bt = pd.DataFrame({"PnL": pnl})
    bt["CumPnL"] = bt["PnL"].cumsum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Win Rate", f"{(bt.PnL>0).mean():.1%}")
    c2.metric("Profit Factor",
              f"{bt[bt.PnL>0].PnL.sum() / abs(bt[bt.PnL<0].PnL.sum()):.2f}")
    c3.metric("Total P&L", f"${bt.PnL.sum():,.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=bt["CumPnL"]))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PORTFOLIO TAB
# =====================================================
with tab4:
    if not st.session_state.portfolio:
        st.info("No trades yet")
    else:
        pf = pd.DataFrame(st.session_state.portfolio)
        st.dataframe(pf)
        st.metric("Total Expected P&L", f"${pf['Expected PnL'].sum():,.2f}")

# =====================================================
# OPPORTUNITY TAB (SAFE)
# =====================================================
with tab5:
    st.subheader("Best Available Opportunity")

    universe = ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL",
                "TSLA", "AMD", "NFLX", "AVGO"]

    best = None

    for sym in universe:
        df_try = load_stock(sym)
        if df_try.empty:
            continue

        p = df_try["Close"].iloc[-1]
        v = annual_volatility(df_try)
        m = expected_move(p, v)

        b = p - m
        s = p + m
        stp = b - m * 0.5

        bp = 1 - prob_hit(p, b, v)
        RRR_try = (s - b) / max(b - stp, 0.01)

        if bp >= 0.6 and RRR_try >= 1.5:
            best = {
                "Ticker": sym,
                "Price": p,
                "Buy": b,
                "Sell": s,
                "RRR": RRR_try,
                "Prob": bp
            }
            break

    if not best:
        st.warning("No opportunity found right now")
    else:
        st.title(best["Ticker"])
        st.metric("Buy Target", f"${best['Buy']:.2f}", f"{best['Prob']:.1%}")
        st.metric("Sell Target", f"${best['Sell']:.2f}")
        st.metric("Risk-Reward Ratio", f"{best['RRR']:.2f}")
