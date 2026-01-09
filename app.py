import streamlit as st
import pandas as pd

# =========================
# IMPORT ENGINES
# =========================
from utils.config import validate_keys
from utils.data_loader import load_historical, sanity_check_bars
from utils.indicators import add_indicators_inplace
from utils.backtester import backtest_strategy

# If you are not ready for live trading yet, keep this commented out
# from utils.live_stream import RealtimeStream


# =========================
# LOAD API KEYS FROM SECRETS
# =========================
api_key = st.secrets.get("ALPACA_KEY", "")
sec_key = st.secrets.get("ALPACA_SECRET", "")


# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="Pro Algo Trader",
    layout="wide",
)
st.title("📈 Modular Algorithmic Dashboard")


# =========================
# SIDEBAR CONTROLS
# =========================
with st.sidebar:
    st.header("Settings")

    symbol = st.text_input("Stock Symbol", value="AAPL").upper()

    # Backtest parameters
    mode = st.selectbox("Entry mode", ["pullback", "breakout"], index=0)
    horizon = st.number_input("Max hold (bars)", min_value=1, max_value=200, value=20)

    atr_entry = st.number_input("ATR entry", 0.0, 10.0, 1.0, 0.1)
    atr_stop = st.number_input("ATR stop", 0.1, 20.0, 2.0, 0.1)
    atr_target = st.number_input("ATR target", 0.1, 50.0, 3.0, 0.1)

    rsi_min = st.number_input("RSI min", 0.0, 100.0, 30.0)
    rsi_max = st.number_input("RSI max", 0.0, 100.0, 70.0)

    rvol_min = st.number_input("RVOL min", 0.0, 10.0, 1.2)
    vol_max = st.number_input("Max annual vol", 0.0, 5.0, 1.0)

    cooldown_bars = st.number_input("Cooldown bars", 0, 200, 5)

    include_spread_penalty = st.checkbox("Include spread penalty", value=True)
    assumed_spread_bps = st.number_input("Assumed spread (bps)", 0.0, 200.0, 5.0)

    start_equity = st.number_input(
        "Starting equity ($)",
        min_value=1_000.0,
        max_value=10_000_000.0,
        value=100_000.0,
        step=1_000.0,
    )

    st.divider()

    # API status (no inputs)
    if validate_keys(api_key, sec_key):
        st.success("Alpaca keys loaded from Secrets ✅")
    else:
        st.error("Missing Alpaca keys in Secrets ❌")

    run_backtest = st.button("🚀 Run Backtest", use_container_width=True)


# =========================
# BLOCK EXECUTION IF NO KEYS
# =========================
if not validate_keys(api_key, sec_key):
    st.stop()


# =========================
# LOAD HISTORICAL DATA
# =========================
df, debug_info = load_historical(symbol, api_key, sec_key)

if df is None or df.empty:
    st.error(f"Could not load data for {symbol}.")
    st.stop()

# Optional sanity checks
try:
    sanity = sanity_check_bars(df)
    if isinstance(sanity, dict) and not sanity.get("ok", True):
        st.warning(f"Data warnings detected: {sanity}")
except Exception:
    pass


# =========================
# PREPARE INDICATORS FOR CHARTING
# =========================
df_chart = df.copy()
add_indicators_inplace(df_chart)


# =========================
# UI TABS
# =========================
tab1, tab2 = st.tabs(["📊 Historical Analysis", "📡 Live Stream"])


# =========================
# TAB 1 — ANALYSIS
# =========================
with tab1:
    st.subheader(f"{symbol} Price & Trend")

    plot_cols = [c for c in ["close", "ma50", "ma200"] if c in df_chart.columns]
    st.line_chart(df_chart[plot_cols])

    if run_backtest:
        results, trades = backtest_strategy(
            df=df,
            market_df=None,
            horizon=int(horizon),
            mode=mode,
            atr_entry=float(atr_entry),
            atr_stop=float(atr_stop),
            atr_target=float(atr_target),
            require_risk_on=False,   # market_df=None
            rsi_min=float(rsi_min),
            rsi_max=float(rsi_max),
            rvol_min=float(rvol_min),
            vol_max=float(vol_max),
            cooldown_bars=int(cooldown_bars),
            include_spread_penalty=bool(include_spread_penalty),
            assumed_spread_bps=float(assumed_spread_bps),
            start_equity=float(start_equity),
        )

        st.success("Backtest completed")

        st.subheader("Trades")
        if trades.empty:
            st.info("No trades generated.")
        else:
            st.dataframe(trades, use_container_width=True)

        st.subheader("Latest Data Snapshot")
        st.dataframe(results.tail(20), use_container_width=True)


# =========================
# TAB 2 — LIVE PLACEHOLDER
# =========================
with tab2:
    st.info("Live streaming is disabled in backtest mode.")
    st.caption("Enable utils/live_stream.py when ready.")
