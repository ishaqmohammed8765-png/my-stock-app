import streamlit as st
import pandas as pd
import numpy as np

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
# Helpers
# ---------------------------
def has_keys(api_key: str, sec_key: str) -> bool:
    return bool(api_key and sec_key and str(api_key).strip() and str(sec_key).strip())


def ss_get(name, default):
    return st.session_state.get(name, default)


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

    symbol = st.text_input("Stock Symbol", value="AAPL", key="symbol").upper().strip()

    col_a, col_b = st.columns(2)
    with col_a:
        load_btn = st.button("🔄 Load/Refresh", use_container_width=True, key="load_btn")
    with col_b:
        run_backtest = st.button("🚀 Run Backtest", use_container_width=True, key="run_backtest")

    st.divider()

    mode = st.selectbox("Entry mode", ["pullback", "breakout"], index=0, key="mode")
    horizon = st.number_input("Max hold (bars)", min_value=1, max_value=200, value=20, key="horizon")

    with st.expander("Advanced strategy params", expanded=False):
        atr_entry = st.number_input("ATR entry", 0.0, 10.0, 1.0, 0.1, key="atr_entry")
        atr_stop = st.number_input("ATR stop", 0.1, 20.0, 2.0, 0.1, key="atr_stop")
        atr_target = st.number_input("ATR target", 0.1, 50.0, 3.0, 0.1, key="atr_target")

        rsi_min = st.number_input("RSI min", 0.0, 100.0, 30.0, key="rsi_min")
        rsi_max = st.number_input("RSI max", 0.0, 100.0, 70.0, key="rsi_max")

        rvol_min = st.number_input("RVOL min", 0.0, 10.0, 1.2, key="rvol_min")
        vol_max = st.number_input("Max annual vol", 0.0, 5.0, 1.0, key="vol_max")

        cooldown_bars = st.number_input("Cooldown bars", 0, 200, 5, key="cooldown_bars")

        include_spread_penalty = st.checkbox("Include spread penalty", value=True, key="include_spread_penalty")
        assumed_spread_bps = st.number_input("Assumed spread (bps)", 0.0, 200.0, 5.0, key="assumed_spread_bps")

        start_equity = st.number_input(
            "Starting equity ($)",
            min_value=1_000.0,
            max_value=10_000_000.0,
            value=100_000.0,
            step=1_000.0,
            key="start_equity",
        )

        require_risk_on = st.checkbox(
            "Require risk-on regime (needs market_df)",
            value=False,
            key="require_risk_on",
        )

    st.divider()
    if has_keys(api_key, sec_key):
        st.success("Alpaca keys loaded from Secrets ✅")
    else:
        st.error("Missing Alpaca keys in Secrets ❌")
        st.caption("Add ALPACA_KEY and ALPACA_SECRET in Streamlit Secrets.")


# ---------------------------
# Block if no keys (since your loader uses Alpaca)
# ---------------------------
if not has_keys(api_key, sec_key):
    st.stop()


# ---------------------------
# Session state init
# ---------------------------
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
    st.session_state.debug_info = None
    st.session_state.sanity = None
    st.session_state.last_symbol = None


# ---------------------------
# Load / Refresh
# ---------------------------
needs_load = load_btn or (st.session_state.df_raw is None) or (st.session_state.last_symbol != symbol)

if needs_load:
    with st.spinner(f"Loading historical data for {symbol}..."):
        try:
            df, debug_info = load_historical(symbol, api_key, sec_key)
        except Exception as e:
            st.session_state.df_raw = None
            st.session_state.debug_info = {"error": str(e)}
            st.session_state.sanity = None
        else:
            st.session_state.df_raw = df
            st.session_state.debug_info = debug_info
            st.session_state.last_symbol = symbol
            try:
                st.session_state.sanity = sanity_check_bars(df) if df is not None else None
            except Exception as e:
                st.session_state.sanity = {"ok": False, "error": str(e)}


df = st.session_state.df_raw
debug_info = st.session_state.debug_info
sanity = st.session_state.sanity

if df is None or getattr(df, "empty", True):
    st.error(f"Could not load data for {symbol}.")
    if debug_info:
        st.write("Loader debug:", debug_info)
    st.stop()


# Prep indicators for dashboard charts (use copy)
df_chart = df.copy()
try:
    add_indicators_inplace(df_chart)
except Exception as e:
    st.warning("Indicators failed to compute — showing raw price only.")
    st.caption(str(e))


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
        if cols_to_plot:
            st.line_chart(df_chart[cols_to_plot])
        else:
            st.line_chart(df_chart[["close"]] if "close" in df_chart.columns else df_chart)

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
        else:
            st.caption("RSI not available.")
    with c2:
        if "rvol" in df_chart.columns:
            st.subheader("RVOL")
            st.line_chart(df_chart[["rvol"]])
        else:
            st.caption("RVOL not available.")
    with c3:
        if "vol_ann" in df_chart.columns:
            st.subheader("Ann. Vol (proxy)")
            st.line_chart(df_chart[["vol_ann"]])
        else:
            st.caption("Vol proxy not available.")


# ---------------------------
# Tab 2: Backtest
# ---------------------------
with tab2:
    st.subheader("Backtest")
    st.caption("Tip: if you get 0 trades, loosen filters (RVOL min down, vol max up, RSI range wider).")

    # Collect params safely
    _atr_entry = float(ss_get("atr_entry", 1.0))
    _atr_stop = float(ss_get("atr_stop", 2.0))
    _atr_target = float(ss_get("atr_target", 3.0))
    _rsi_min = float(ss_get("rsi_min", 30.0))
    _rsi_max = float(ss_get("rsi_max", 70.0))
    _rvol_min = float(ss_get("rvol_min", 1.2))
    _vol_max = float(ss_get("vol_max", 1.0))
    _cooldown = int(ss_get("cooldown_bars", 5))
    _spread_on = bool(ss_get("include_spread_penalty", True))
    _spread_bps = float(ss_get("assumed_spread_bps", 5.0))
    _equity = float(ss_get("start_equity", 100000.0))
    _risk_on = bool(ss_get("require_risk_on", False))
    _horizon = int(ss_get("horizon", 20))
    _mode = str(ss_get("mode", "pullback"))

    if _risk_on:
        st.warning("Risk-on regime filter is ON, but market_df is not loaded in app.py yet — results may be wrong/empty.")

    if run_backtest:
        with st.spinner("Running backtest..."):
            try:
                results, trades = backtest_strategy(
                    df=df,
                    market_df=None,               # TODO: wire this if you want regime logic
                    horizon=_horizon,
                    mode=_mode,
                    atr_entry=_atr_entry,
                    atr_stop=_atr_stop,
                    atr_target=_atr_target,
                    require_risk_on=_risk_on,     # now actually respects the checkbox
                    rsi_min=_rsi_min,
                    rsi_max=_rsi_max,
                    rvol_min=_rvol_min,
                    vol_max=_vol_max,
                    cooldown_bars=_cooldown,
                    include_spread_penalty=_spread_on,
                    assumed_spread_bps=_spread_bps,
                    start_equity=_equity,
                )
            except Exception as e:
                st.error("Backtest failed.")
                st.caption(str(e))
                st.stop()

        st.success("Backtest completed ✅")

        if trades is None or getattr(trades, "empty", True):
            st.info("No trades generated with current params.")
        else:
            t = trades.copy()

            # Basic metrics (per-share)
            if "pnl_per_share" in t.columns:
                wins = (t["pnl_per_share"] > 0).sum()
                win_rate = wins / max(1, len(t))
                avg_win = t.loc[t["pnl_per_share"] > 0, "pnl_per_share"].mean()
                avg_loss = t.loc[t["pnl_per_share"] <= 0, "pnl_per_share"].mean()
            else:
                win_rate, avg_win, avg_loss = np.nan, np.nan, np.nan

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades", f"{len(t)}")
            c2.metric("Win rate", f"{win_rate:.1%}" if np.isfinite(win_rate) else "—")
            c3.metric("Avg win (per share)", f"{avg_win:.3f}" if np.isfinite(avg_win) else "—")
            c4.metric("Avg loss (per share)", f"{avg_loss:.3f}" if np.isfinite(avg_loss) else "—")

            st.subheader("Trades")
            st.dataframe(t, use_container_width=True)

            csv = t.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download trades CSV",
                data=csv,
                file_name=f"{symbol}_trades.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.subheader("Latest backtest data snapshot")
        try:
            st.dataframe(results.tail(50), use_container_width=True)
        except Exception:
            st.caption("No results dataframe returned by backtester.")
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
        st.subheader("Live (placeholder)")

        # Minimal safe UI stub; actual streaming depends on your RealtimeStream design.
        col1, col2 = st.columns(2)
        with col1:
            st.button("▶️ Start Live", use_container_width=True, disabled=True)
        with col2:
            st.button("⏹ Stop Live", use_container_width=True, disabled=True)

        st.info("Live streaming is detected, but not wired into UI yet.")
        st.caption("Once you paste utils/live_stream.py, I’ll connect Start/Stop + live chart + last tick table.")
