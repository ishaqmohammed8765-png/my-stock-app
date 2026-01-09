import streamlit as st
import pandas as pd
import numpy as np

from utils.config import validate_keys
from utils.data_loader import load_historical, sanity_check_bars
from utils.indicators import add_indicators_inplace
from utils.backtester import backtest_strategy

# Optional: only import live stream if the file exists and is ready
LIVE_AVAILABLE = True
try:
    from utils.live_stream import RealtimeStream  # noqa: F401
except Exception:
    LIVE_AVAILABLE = False


# ---------------------------
# Secrets (no manual key entry)
# ---------------------------
api_key = st.secrets.get("ALPACA_KEY", "")
sec_key = st.secrets.get("ALPACA_SECRET", "")

st.set_page_config(page_title="Pro Algo Trader", layout="wide")
st.title("📈 Modular Algorithmic Dashboard")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Settings")

    symbol = st.text_input("Stock Symbol", value="AAPL").upper()

    col_a, col_b = st.columns(2)
    with col_a:
        load_btn = st.button("🔄 Load/Refresh", use_container_width=True)
    with col_b:
        run_backtest = st.button("🚀 Run Backtest", use_container_width=True)

    st.divider()

    # Minimal always-visible controls
    mode = st.selectbox("Entry mode", ["pullback", "breakout"], index=0)
    horizon = st.number_input("Max hold (bars)", min_value=1, max_value=200, value=20)

    # Advanced controls hidden by default
    with st.expander("Advanced strategy params", expanded=False):
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

        # Regime filter is only meaningful if you actually pass market_df
        require_risk_on = st.checkbox("Require risk-on regime (needs market_df)", value=False)

    # Key status
    if validate_keys(api_key, sec_key):
        st.success("Alpaca keys loaded from Secrets ✅")
    else:
        st.error("Missing Alpaca keys in Secrets ❌")
        st.caption("Add ALPACA_KEY and ALPACA_SECRET in Streamlit Secrets.")


# ---------------------------
# Block if no keys (since your loader uses Alpaca)
# ---------------------------
if not validate_keys(api_key, sec_key):
    st.stop()


# ---------------------------
# Data load (cache in session_state so tabs feel “feature-rich”)
# ---------------------------
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
    st.session_state.debug_info = None
    st.session_state.sanity = None

if load_btn or (st.session_state.df_raw is None):
    df, debug_info = load_historical(symbol, api_key, sec_key)
    st.session_state.df_raw = df
    st.session_state.debug_info = debug_info

    try:
        st.session_state.sanity = sanity_check_bars(df) if df is not None else None
    except Exception:
        st.session_state.sanity = None


df = st.session_state.df_raw
debug_info = st.session_state.debug_info
sanity = st.session_state.sanity

if df is None or df.empty:
    st.error(f"Could not load data for {symbol}.")
    st.stop()


# Prep indicators for dashboard charts (use copy)
df_chart = df.copy()
add_indicators_inplace(df_chart)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧪 Backtest", "📡 Live"])

# ---------------------------
# Tab 1: Dashboard
# ---------------------------
with tab1:
    left, right = st.columns([2, 1])

    with left:
        st.subheader(f"{symbol} — Price & Trend")
        cols_to_plot = [c for c in ["close", "ma50", "ma200"] if c in df_chart.columns]
        st.line_chart(df_chart[cols_to_plot])

    with right:
        st.subheader("Data integrity")
        if isinstance(sanity, dict):
            if sanity.get("ok", True):
                st.success("Sanity checks: OK")
            else:
                st.warning("Sanity checks: warnings")
            st.json(sanity)
        else:
            st.info("No sanity report available.")

        st.subheader("Debug")
        if debug_info is not None:
            st.write(debug_info)
        else:
            st.caption("No debug info returned by loader.")

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        if "rsi14" in df_chart.columns:
            st.subheader("RSI(14)")
            st.line_chart(df_chart[["rsi14"]])
    with c2:
        if "rvol" in df_chart.columns:
            st.subheader("RVOL")
            st.line_chart(df_chart[["rvol"]])
    with c3:
        if "vol_ann" in df_chart.columns:
            st.subheader("Ann. Vol (proxy)")
            st.line_chart(df_chart[["vol_ann"]])


# ---------------------------
# Tab 2: Backtest
# ---------------------------
with tab2:
    st.subheader("Backtest")

    st.caption("Tip: if you get 0 trades, loosen filters (RVOL min down, vol max up, RSI range wider).")

    # Ensure advanced defaults exist even if expander never opened (Streamlit sometimes delays widget init)
    # We'll provide safe fallbacks.
    def _ss(name, default):
        return st.session_state.get(name, default)

    # Use the variables from sidebar if they exist, otherwise default to safe values.
    # (If sidebar expander was never opened, variables still exist because widgets were created.)
    try:
        _atr_entry = float(atr_entry)
        _atr_stop = float(atr_stop)
        _atr_target = float(atr_target)
        _rsi_min = float(rsi_min)
        _rsi_max = float(rsi_max)
        _rvol_min = float(rvol_min)
        _vol_max = float(vol_max)
        _cooldown = int(cooldown_bars)
        _spread_on = bool(include_spread_penalty)
        _spread_bps = float(assumed_spread_bps)
        _equity = float(start_equity)
        _risk_on = bool(require_risk_on)
    except Exception:
        _atr_entry, _atr_stop, _atr_target = 1.0, 2.0, 3.0
        _rsi_min, _rsi_max = 30.0, 70.0
        _rvol_min, _vol_max = 1.2, 1.0
        _cooldown = 5
        _spread_on, _spread_bps = True, 5.0
        _equity = 100000.0
        _risk_on = False

    if run_backtest:
        # NOTE: market_df=None, so regime filter should be off unless you implement market_df loading.
        results, trades = backtest_strategy(
            df=df,
            market_df=None,
            horizon=int(horizon),
            mode=str(mode),
            atr_entry=_atr_entry,
            atr_stop=_atr_stop,
            atr_target=_atr_target,
            require_risk_on=False if True else _risk_on,  # keep False unless market_df is supplied
            rsi_min=_rsi_min,
            rsi_max=_rsi_max,
            rvol_min=_rvol_min,
            vol_max=_vol_max,
            cooldown_bars=_cooldown,
            include_spread_penalty=_spread_on,
            assumed_spread_bps=_spread_bps,
            start_equity=_equity,
        )

        st.success("Backtest completed ✅")

        # Quick metrics (based on pnl_per_share; sizing not included)
        if trades is None or trades.empty:
            st.info("No trades generated with current params.")
        else:
            t = trades.copy()
            wins = (t["pnl_per_share"] > 0).sum()
            losses = (t["pnl_per_share"] <= 0).sum()
            win_rate = wins / max(1, len(t))

            avg_win = t.loc[t["pnl_per_share"] > 0, "pnl_per_share"].mean()
            avg_loss = t.loc[t["pnl_per_share"] <= 0, "pnl_per_share"].mean()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades", f"{len(t)}")
            c2.metric("Win rate", f"{win_rate:.1%}")
            c3.metric("Avg win (per share)", f"{avg_win:.3f}" if np.isfinite(avg_win) else "—")
            c4.metric("Avg loss (per share)", f"{avg_loss:.3f}" if np.isfinite(avg_loss) else "—")

            st.subheader("Trades")
            st.dataframe(t, use_container_width=True)

            # Download
            csv = t.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download trades CSV",
                data=csv,
                file_name=f"{symbol}_trades.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.subheader("Latest backtest data snapshot")
        st.dataframe(results.tail(50), use_container_width=True)
    else:
        st.info("Click **🚀 Run Backtest** in the sidebar.")


# ---------------------------
# Tab 3: Live
# ---------------------------
with tab3:
    if not LIVE_AVAILABLE:
        st.info("Live module not available (or import failed).")
        st.caption("If you want live streaming, paste your utils/live_stream.py and I’ll wire it back in safely.")
    else:
        st.info("Live streaming tab is ready to be wired up.")
        st.caption("Next step: add Start/Stop controls and stream output rendering.")
