import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Calculator Pro", layout="wide")

# =====================================================
# SAFE DATA LOAD
# =====================================================
@st.cache_data(ttl=300)
def load_stock(ticker):
    try:
        df = yf.download(ticker, period="6mo", progress=False)
        df = df.tail(120)
        if df.empty or "Close" not in df:
            return pd.DataFrame()
        df.index = df.index.tz_localize(None)
        return df
    except:
        return pd.DataFrame()

# =====================================================
# SAFE VOLATILITY (FIXED)
# =====================================================
def annual_volatility(df):
    r = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if len(r) < 10:
        return 0.30
    vol = float(r.std()) * np.sqrt(252)
    return max(0.15, min(vol, 1.5))

def expected_move(price, vol, days=30):
    return price * vol * np.sqrt(days / 365)

def prob_hit(S, K, vol, days=30):
    T = days / 365
    if S <= 0 or K <= 0:
        return 0.5
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
# LOAD STOCK
# =====================================================
if load:
    df = load_stock(ticker)
    if df.empty:
        st.error("No data available for this stock.")
        st.stop()
    st.session_state.df = df
    st.session_state.ticker = ticker

if "df" not in st.session_state:
    st.info("👈 Enter a stock symbol and press **Load Stock**")
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
# KELLY POSITION SIZING
# =====================================================
returns = df["Close"].pct_change().dropna()
wins = returns[returns > 0]
losses = returns[returns < 0]

if len(wins) < 10 or len(losses) < 10:
    kelly_fraction = 0.02
else:
    win_rate = len(wins) / len(returns)
    loss_rate = 1 - win_rate
    R = abs(wins.mean() / losses.mean())
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
        st.success(f"Risk–Reward Ratio: {RRR:.2f}")
    elif RRR >= 1.5:
        st.warning(f"Risk–Reward Ratio: {RRR:.2f}")
    else:
        st.error(f"Risk–Reward Ratio: {RRR:.2f}")

    st.info("📌 **Recommended RRR ≥ 1.5** — lower ratios mean poor reward vs risk.")

    st.metric("Kelly Position Size (Shares)", qty)
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
# BACKTEST TAB (BEGINNER FRIENDLY)
# =====================================================
with tab3:
    st.info("📘 This backtest checks what would have happened if you bought and held for 30 days each time.")

    pnl = []
    for i in range(len(df)-days):
        pnl.append(df["Close"].iloc[i+days] - df["Close"].iloc[i])

    bt = pd.DataFrame({"PnL": pnl})
    bt["CumPnL"] = bt["PnL"].cumsum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Win Rate", f"{(bt.PnL>0).mean():.1%}")
    c2.metric("Profit Factor",
              f"{bt[bt.PnL>0].PnL.sum() / abs(bt[bt.PnL<0].PnL.sum()):.2f}")
    c3.metric("Total P&L", f"${bt.PnL.sum():,.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=bt["CumPnL"], name="Cumulative P&L"))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PORTFOLIO TAB
# =====================================================
with tab4:
    if not st.session_state.portfolio:
        st.info("No trades added yet.")
    else:
        pf = pd.DataFrame(st.session_state.portfolio)
        st.dataframe(pf)
        st.metric("Total Expected P&L", f"${pf['Expected PnL'].sum():,.2f}")

# =====================================================
# OPPORTUNITY TAB (RANKED + SAFE)
# =====================================================
with tab5:
    st.subheader("Best Available Opportunity")

    universe = ["AAPL","MSFT","NVDA","META","AMZN","GOOGL","TSLA","AMD","NFLX","AVGO"]
    candidates = []

    for sym in universe:
        df_try = load_stock(sym)
        if df_try.empty or len(df_try) < 60:
            continue

        price_t = df_try["Close"].iloc[-1]
        ma50 = df_try["Close"].rolling(50).mean().iloc[-1]

        if price_t < ma50:
            continue  # trend filter

        vol_t = annual_volatility(df_try)
        move_t = expected_move(price_t, vol_t)
        buy_t = price_t - move_t
        sell_t = price_t + move_t
        stop_t = buy_t - move_t * 0.5

        prob_t = 1 - prob_hit(price_t, buy_t, vol_t)
        RRR_t = (sell_t - buy_t) / max(buy_t - stop_t, 0.01)
        EV = prob_t * (sell_t - buy_t)

        if prob_t >= 0.55 and RRR_t >= 1.5:
            candidates.append((EV, sym, buy_t, sell_t, RRR_t, prob_t))

    if not candidates:
        st.warning("❌ No good opportunities found right now.")
    else:
        best = sorted(candidates, reverse=True)[0]
        _, sym, b, s, r, p = best

        st.title(sym)
        st.metric("Buy Target", f"${b:.2f}", f"{p:.1%}")
        st.metric("Sell Target", f"${s:.2f}")
        st.metric("Risk–Reward Ratio", f"{r:.2f}")
