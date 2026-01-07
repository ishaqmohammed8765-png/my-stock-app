import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Stock Calculator Pro", layout="wide")

# =====================================================
# CONSTANTS
# =====================================================
DAYS = 30
VOL_FLOOR = 0.10
VOL_CAP = 1.5
KELLY_MAX = 0.15
KELLY_MIN = 0.01
MONTE_CARLO_SIMS = 5000
ACCOUNT_SIZE = 10_000

# =====================================================
# FUNCTIONS
# =====================================================
def load_stock(ticker):
    """Download stock data safely."""
    try:
        # Use Ticker object for more reliable data access
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")
        
        if df is None or df.empty:
            st.error(f"⚠️ No data returned from Yahoo Finance for {ticker}")
            return pd.DataFrame()
        
        # The history() method returns a clean DataFrame with Close column
        if "Close" not in df.columns:
            st.error(f"⚠️ Data structure issue - no Close column found")
            return pd.DataFrame()
        
        # Remove timezone if present
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Clean and validate data
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
        
        if len(df) < 20:
            st.error(f"⚠️ Not enough historical data (only {len(df)} days)")
            return pd.DataFrame()
            
        return df.tail(120)
    except Exception as e:
        st.error(f"❌ Error loading {ticker}: {str(e)}")
        return pd.DataFrame()

def annual_volatility(df):
    """Calculate annualized volatility using EWMA."""
    if df.empty:
        return 0.30
    r = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if len(r) < 10:
        return 0.30
    vol = r.ewm(span=20).std().iloc[-1] * np.sqrt(252)
    return float(np.clip(vol, VOL_FLOOR, VOL_CAP))

def expected_move(price, vol, days=DAYS):
    """Expected price move over a given number of days."""
    return price * vol * np.sqrt(days / 252)

def prob_hit_mc(S, K, vol, days=DAYS, sims=MONTE_CARLO_SIMS):
    """Vectorized Monte Carlo probability of hitting target."""
    if vol <= 0 or S <= 0 or K <= 0:
        return 0.0
    dt = 1/252
    random_shocks = np.random.normal(size=(sims, days))
    price_paths = S * np.exp(np.cumsum((-0.5*vol**2)*dt + vol*np.sqrt(dt)*random_shocks, axis=1))
    hits = (price_paths >= K).any(axis=1)
    return float(hits.mean())

def kelly_fraction(df):
    """Calculate Kelly fraction for position sizing."""
    if df.empty:
        return 0.02
    returns = df["Close"].pct_change().dropna()
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return 0.02
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(wins) < 10 or len(losses) < 10:
        fraction = 0.02
    else:
        win_rate = len(wins) / len(returns)
        loss_rate = 1 - win_rate
        R = abs(wins.mean() / losses.mean()) if losses.mean() != 0 else 1
        fraction = win_rate - (loss_rate / R)
    return float(np.clip(fraction * 0.5, KELLY_MIN, KELLY_MAX))

def backtest(df, days=DAYS):
    """Simple 30-day buy & hold backtest with stop-loss."""
    pnl = []
    for i in range(len(df) - days):
        entry = df["Close"].iloc[i]
        window = df["Close"].iloc[i:i+days].copy()
        stop_price = entry - expected_move(entry, annual_volatility(df)) * 0.5
        window = pd.to_numeric(window, errors="coerce").dropna()
        if window.empty:
            continue
        hits = window[window <= stop_price]
        exit_price = float(hits.iloc[0]) if not hits.empty else float(window.iloc[-1])
        pnl.append(exit_price - entry)
    if not pnl:
        return pd.DataFrame({"PnL": []})
    bt = pd.DataFrame({"PnL": pnl})
    bt["PnL"] = pd.to_numeric(bt["PnL"], errors="coerce").fillna(0)
    bt["CumPnL"] = bt["PnL"].cumsum()
    return bt

# =====================================================
# SESSION STATE
# =====================================================
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "ticker" not in st.session_state:
    st.session_state.ticker = ""

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    ticker_input = st.text_input("Stock Symbol", value="AAPL").upper()
    load = st.button("Load Stock")

# =====================================================
# LOAD STOCK
# =====================================================
if load:
    df = load_stock(ticker_input)
    if df.empty:
        st.error("No data available for this stock.")
    else:
        st.session_state.df = df
        st.session_state.ticker = ticker_input

if st.session_state.df.empty:
    st.info("👈 Enter a stock symbol and press **Load Stock**")
    st.stop()

df = st.session_state.df
ticker = st.session_state.ticker
price = float(df["Close"].iloc[-1])
vol = annual_volatility(df)

# =====================================================
# CALCULATIONS
# =====================================================
move = expected_move(price, vol)
buy = price - move
sell = price + move
stop = buy - move * 0.5

buy_prob = 1 - prob_hit_mc(price, buy, vol)
sell_prob = prob_hit_mc(price, sell, vol)
RRR = (sell - buy) / max(buy - stop, 0.01)

kelly = kelly_fraction(df)
qty = int((ACCOUNT_SIZE * kelly) / max(buy - stop, 0.01))

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
    c2.metric("Buy Target", f"${buy:.2f}", f"{buy_prob*100:.1f}%")
    c3.metric("Sell Target", f"${sell:.2f}", f"{sell_prob*100:.1f}%")
    c4.metric("Stop-Loss", f"${stop:.2f}")

    if RRR >= 2:
        st.success(f"Risk–Reward Ratio: {RRR:.2f}")
    elif RRR >= 1.5:
        st.warning(f"Risk–Reward Ratio: {RRR:.2f}")
    else:
        st.error(f"Risk–Reward Ratio: {RRR:.2f}")

    st.metric("Kelly Position Size (Shares)", qty)
    expected_profit = (sell - buy) * qty
    st.metric("Expected Profit", f"${expected_profit:,.2f}")
    
    # ADD TO PORTFOLIO BUTTON
    st.markdown("---")
    if st.button("➕ Add to Portfolio", type="primary"):
        trade = {
            "Ticker": ticker,
            "Buy": buy,
            "Sell": sell,
            "Stop": stop,
            "Quantity": qty,
            "Expected PnL": expected_profit,
            "RRR": RRR
        }
        st.session_state.portfolio.append(trade)
        st.success(f"✅ Added {ticker} to portfolio!")

# =====================================================
# CHART TAB
# =====================================================
with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
    for y_val, name in [(buy, "Buy"), (sell, "Sell"), (stop, "Stop")]:
        if not np.isnan(y_val):
            fig.add_hline(y=y_val, line_dash="dot", annotation_text=name)
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# BACKTEST TAB
# =====================================================
with tab3:
    st.info("📘 Backtest: 30-day buy & hold with stop-loss")
    bt = backtest(df)
    c1, c2, c3 = st.columns(3)
    if not bt.empty:
        win_rate = (bt["PnL"] > 0).mean() * 100
        profit_factor = bt[bt["PnL"]>0].PnL.sum() / max(abs(bt[bt["PnL"]<0].PnL.sum()), 0.01)
        total_pnl = bt["PnL"].sum()
    else:
        win_rate = profit_factor = total_pnl = 0
    c1.metric("Win Rate", f"{win_rate:.1f}%")
    c2.metric("Profit Factor", f"{profit_factor:.2f}")
    c3.metric("Total P&L", f"${total_pnl:,.2f}")
    if not bt.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=bt["CumPnL"], name="Cumulative P&L"))
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PORTFOLIO TAB
# =====================================================
with tab4:
    st.subheader("Portfolio")
    if not st.session_state.portfolio:
        st.info("No trades added yet. Go to the Trade tab to add positions.")
    else:
        pf = pd.DataFrame(st.session_state.portfolio)
        st.dataframe(pf, use_container_width=True)
        st.metric("Total Expected P&L", f"${pf['Expected PnL'].sum():,.2f}")
        
        # CLEAR PORTFOLIO BUTTON
        if st.button("🗑️ Clear Portfolio"):
            st.session_state.portfolio = []
            st.rerun()

# =====================================================
# OPPORTUNITY TAB
# =====================================================
with tab5:
    st.subheader("Best Available Opportunity")
    universe = ["AAPL","MSFT","NVDA","META","AMZN","GOOGL","TSLA","AMD","NFLX","AVGO"]
    candidates = []

    with st.spinner("Scanning market for opportunities..."):
        for sym in universe:
            df_try = load_stock(sym)
            if df_try.empty or len(df_try) < 60:
                continue

            price_t = df_try["Close"].iloc[-1]
            ma50 = df_try["Close"].rolling(50).mean().iloc[-1]
            if price_t < ma50:
                continue

            vol_t = annual_volatility(df_try)
            move_t = expected_move(price_t, vol_t)
            buy_t = price_t - move_t
            sell_t = price_t + move_t
            stop_t = buy_t - move_t * 0.5

            prob_t = 1 - prob_hit_mc(price_t, buy_t, vol_t)
            RRR_t = (sell_t - buy_t) / max(buy_t - stop_t, 0.01)
            EV = prob_t * (sell_t - buy_t)

            if prob_t >= 0.55 and RRR_t >= 1.5:
                candidates.append((EV, sym, buy_t, sell_t, RRR_t, prob_t))

    if not candidates:
        st.warning("❌ No good opportunities found right now.")
    else:
        best = max(candidates, key=lambda x: x[0])
        _, sym, b, s, r, p = best
        st.success(f"🎯 Best Opportunity: **{sym}**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Buy Target", f"${b:.2f}", f"{p*100:.1f}% prob")
        col2.metric("Sell Target", f"${s:.2f}")
        col3.metric("Risk–Reward", f"{r:.2f}")
