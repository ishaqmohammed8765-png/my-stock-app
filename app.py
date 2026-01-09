import streamlit as st
import pandas as pd

# IMPORT YOUR ENGINES
from utils.config import CFG, validate_keys
from utils.data_loader import load_historical, sanity_check_bars
from utils.indicators import add_indicators_inplace
from utils.backtester import backtest_strategy

# If you have this file; otherwise comment it out until ready
# from utils.live_stream import RealtimeStream


# PAGE SETUP
st.set_page_config(page_title="Pro Algo Trader", layout="wide")
st.title("📈 Modular Algorithmic Dashboard")

# SIDEBAR: CREDENTIALS & SETTINGS
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input("Alpaca API Key", type="default")
    sec_key = st.text_input("Alpaca Secret Key", type="password")
    symbol = st.text_input("Stock Symbol", value="AAPL").upper()

    st.divider()

    # Backtest controls (minimal but correct)
    mode = st.selectbox("Entry mode", ["pullback", "breakout"], index=0)
    horizon = st.number_input("Max hold (bars)", min_value=1, max_value=200, value=20, step=1)

    atr_entry = st.number_input("ATR entry", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    atr_stop = st.number_input("ATR stop", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
    atr_target = st.number_input("ATR target", min_value=0.1, max_value=50.0, value=3.0, step=0.1)

    rsi_min = st.number_input("RSI min", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    rsi_max = st.number_input("RSI max", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

    rvol_min = st.number_input("RVOL min", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
    vol_max = st.number_input("Max ann vol", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    cooldown_bars = st.number_input("Cooldown (bars)", min_value=0, max_value=200, value=5, step=1)

    include_spread_penalty = st.checkbox("Include spread penalty", value=True)
    assumed_spread_bps = st.number_input("Assumed spread (bps)", min_value=0.0, max_value=200.0, value=5.0, step=1.0)

    start_equity = st.number_input("Start equity ($)", min_value=1000.0, max_value=10_000_000.0, value=100_000.0, step=1000.0)

    run_backtest = st.button("🚀 Run Backtest", use_container_width=True)


# MAIN LOGIC
if not validate_keys(api_key, sec_key):
    st.warning("Please enter valid Alpaca API keys in the sidebar to begin.")
    st.stop()

df, debug_info = load_historical(symbol, api_key, sec_key)

if df is None or df.empty:
    st.error(f"Could not find data for {symbol}. Check your API keys or ticker.")
    st.stop()

# Optional: sanity check if you have it implemented
try:
    sanity = sanity_check_bars(df)
    if isinstance(sanity, dict) and sanity.get("ok") is False:
        st.warning(f"Data sanity check warnings: {sanity}")
except Exception:
    pass

# Compute indicators for charting (do it on a copy so df stays "raw" if you want)
df_chart = df.copy()
add_indicators_inplace(df_chart)

tab1, tab2 = st.tabs(["📊 Historical Analysis", "📡 Live Stream"])

with tab1:
    st.subheader(f"Analysis for {symbol}")
    cols_to_plot = [c for c in ["close", "ma50", "ma200"] if c in df_chart.columns]
    st.line_chart(df_chart[cols_to_plot])

    if run_backtest:
        # IMPORTANT:
        # - mode must be 'pullback' or 'breakout'
        # - require_risk_on must be False if market_df=None
        results, trades = backtest_strategy(
            df=df,
            market_df=None,
            horizon=int(horizon),
            mode=str(mode),
            atr_entry=float(atr_entry),
            atr_stop=float(atr_stop),
            atr_target=float(atr_target),
            require_risk_on=False,
            rsi_min=float(rsi_min),
            rsi_max=float(rsi_max),
            rvol_min=float(rvol_min),
            vol_max=float(vol_max),
            cooldown_bars=int(cooldown_bars),
            include_spread_penalty=bool(include_spread_penalty),
            assumed_spread_bps=float(assumed_spread_bps),
            start_equity=float(start_equity),
        )

        st.success("Backtest complete!")

        st.write("Latest rows (with indicators):")
        st.dataframe(results.tail(20), use_container_width=True)

        st.write("Trades:")
        if trades is None or trades.empty:
            st.info("No trades generated with current parameters.")
        else:
            st.dataframe(trades.tail(50), use_container_width=True)

with tab2:
    st.info("Live data will appear here once the stream is started.")
    st.caption("If your live_stream module isn't ready yet, keep this tab as a placeholder.")
