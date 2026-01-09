import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
# Page config MUST be first
# ---------------------------
st.set_page_config(
    page_title="Pro Algo Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Light UI polish (CSS)
# ---------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 1.3rem; }
[data-testid="stSidebar"] .block-container { padding-top: 0.8rem; }
h1 { margin-bottom: 0.25rem; }
h2, h3 { margin-top: 0.55rem; }
[data-testid="stExpander"] details { padding: 0.15rem 0; }
[data-testid="stMetric"] { padding: 0.2rem 0.2rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------
# Helpers
# ---------------------------
def has_keys(api_key: str, sec_key: str) -> bool:
    return bool(api_key and sec_key and str(api_key).strip() and str(sec_key).strip())


def ss_get(name, default):
    return st.session_state.get(name, default)


def _nice_index(df: pd.DataFrame):
    return df.index


def plot_lines(df: pd.DataFrame, cols: list[str], *, title: str, height: int = 360) -> go.Figure:
    x = _nice_index(df)
    fig = go.Figure()
    for c in cols:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[c], mode="lines", name=c))
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True)
    return fig


def plot_indicator(df: pd.DataFrame, col: str, *, title: str, height: int = 230, hlines=None, y0=None, y1=None):
    x = _nice_index(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df[col], mode="lines", name=col))
    if hlines:
        for v in hlines:
            fig.add_hline(y=float(v), line_width=1)
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True)
    if (y0 is not None) and (y1 is not None):
        fig.update_yaxes(range=[float(y0), float(y1)])
    return fig


def compute_sr_levels(df_ind: pd.DataFrame, lookback: int) -> tuple[float, float]:
    """Simple Support/Resistance from recent lows/highs."""
    lb = int(max(10, lookback))
    tail = df_ind.tail(lb)
    support = float(np.nanmin(tail["low"].values))
    resistance = float(np.nanmax(tail["high"].values))
    return support, resistance


def compute_trade_plan_from_backtest_rules(
    df_ind: pd.DataFrame,
    *,
    mode: str,
    atr_entry: float,
    atr_stop: float,
    atr_target: float,
    assumed_spread_bps: float,
    include_spread_penalty: bool,
) -> dict:
    """
    Mirrors your backtester entry math as closely as possible for *next-bar* planning.

    Backtester entries:
      pullback: limit_px = close - atr_entry*atr; filled if next low <= limit_px, entry=limit_px
      breakout: stop_buy  = close + atr_entry*atr; filled if next high >= stop_buy, entry=stop_buy (or open if gap)

    In a dashboard, we can't know next bar. So we present the *trigger/price* it would attempt:
      pullback: planned entry = close - atr_entry*atr
      breakout: planned entry = close + atr_entry*atr
    Then stop/target derived the same.
    """
    last = df_ind.iloc[-1]
    close = float(last["close"])
    atr = float(last["atr14"])

    if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
        return {"entry": np.nan, "stop": np.nan, "target": np.nan, "rr": np.nan}

    mode_l = str(mode).lower().strip()
    if mode_l == "pullback":
        entry = close - float(atr_entry) * atr
        entry_type = "limit (pullback trigger)"
    elif mode_l == "breakout":
        entry = close + float(atr_entry) * atr
        entry_type = "stop (breakout trigger)"
    else:
        entry = np.nan
        entry_type = "—"

    # Spread penalty: backtester applies +bps on entry and -bps on exit.
    # For planning, we can show the "effective" worse entry if enabled.
    if include_spread_penalty and assumed_spread_bps > 0 and np.isfinite(entry):
        entry_eff = entry * (1.0 + assumed_spread_bps / 10000.0)
    else:
        entry_eff = entry

    stop = entry_eff - float(atr_stop) * atr
    target = entry_eff + float(atr_target) * atr

    risk = entry_eff - stop
    reward = target - entry_eff
    rr = (reward / risk) if risk > 0 else np.nan

    return {"entry": entry_eff, "stop": stop, "target": target, "rr": rr, "entry_type": entry_type}


def compute_recommendation_from_backtest_filters(
    df_ind: pd.DataFrame,
    *,
    rsi_min: float,
    rsi_max: float,
    rvol_min: float,
    vol_max: float,
) -> tuple[str, str]:
    """
    Uses the SAME filters your backtester uses (RSI/RVOL/vol_ann) to decide if a trade is even allowed.
    Then adds a small trend hint (MA50/MA200) to label BUY/SELL/HOLD.
    """
    last = df_ind.iloc[-1]

    need = ["close", "ma50", "ma200", "rsi14", "rvol", "vol_ann", "atr14"]
    missing = [c for c in need if c not in df_ind.columns]
    if missing:
        return "—", f"Missing columns: {missing}"

    close = float(last["close"])
    ma50 = float(last["ma50"]) if pd.notna(last["ma50"]) else np.nan
    ma200 = float(last["ma200"]) if pd.notna(last["ma200"]) else np.nan
    rsi = float(last["rsi14"])
    rvol = float(last["rvol"])
    vol_ann = float(last["vol_ann"])
    atr = float(last["atr14"])

    if not np.isfinite([close, rsi, rvol, vol_ann, atr]).all():
        return "HOLD", "Indicators not ready (need more history)."

    # Backtester filters
    filters_ok = True
    reasons = []

    if rsi < float(rsi_min) or rsi > float(rsi_max):
        filters_ok = False
        reasons.append(f"RSI {rsi:.1f} outside [{rsi_min:.0f}, {rsi_max:.0f}]")

    if rvol < float(rvol_min):
        filters_ok = False
        reasons.append(f"RVOL {rvol:.2f} < {rvol_min:.2f}")

    if vol_ann > float(vol_max):
        filters_ok = False
        reasons.append(f"Vol {vol_ann:.2f} > {vol_max:.2f}")

    # Trend context for label
    uptrend = np.isfinite(ma50) and np.isfinite(ma200) and (close > ma50 > ma200)
    downtrend = np.isfinite(ma50) and np.isfinite(ma200) and (close < ma50 < ma200)

    if not filters_ok:
        # Even if trend is up, backtester would not enter
        return "HOLD", " / ".join(reasons)

    # Filters pass: label as BUY in uptrend, SELL in downtrend, otherwise HOLD/WAIT
    if uptrend:
        return "BUY", "Filters pass (RSI/RVOL/Vol) + uptrend."
    if downtrend:
        return "SELL", "Filters pass but trend is down (be cautious / short logic not implemented)."
    return "HOLD", "Filters pass, but trend not clear (wait for alignment)."


# ---------------------------
# Secrets (no manual key entry)
# ---------------------------
api_key = st.secrets.get("ALPACA_KEY", "")
sec_key = st.secrets.get("ALPACA_SECRET", "")

st.title("📈 Modular Algorithmic Dashboard")


# ---------------------------
# Sidebar (form)
# ---------------------------
with st.sidebar:
    st.header("Settings")

    with st.form("settings_form", clear_on_submit=False):
        symbol = st.text_input("Stock Symbol", value=ss_get("symbol", "AAPL")).upper().strip()

        c1, c2 = st.columns(2)
        with c1:
            mode = st.selectbox("Entry mode", ["pullback", "breakout"], index=0)
        with c2:
            horizon = st.number_input("Max hold (bars)", min_value=1, max_value=200, value=int(ss_get("horizon", 20)))

        with st.expander("Advanced strategy params", expanded=False):
            atr_entry = st.number_input("ATR entry", 0.0, 10.0, float(ss_get("atr_entry", 1.0)), 0.1)
            atr_stop = st.number_input("ATR stop", 0.1, 20.0, float(ss_get("atr_stop", 2.0)), 0.1)
            atr_target = st.number_input("ATR target", 0.1, 50.0, float(ss_get("atr_target", 3.0)), 0.1)

            rsi_min = st.number_input("RSI min", 0.0, 100.0, float(ss_get("rsi_min", 30.0)))
            rsi_max = st.number_input("RSI max", 0.0, 100.0, float(ss_get("rsi_max", 70.0)))

            rvol_min = st.number_input("RVOL min", 0.0, 10.0, float(ss_get("rvol_min", 1.2)))
            vol_max = st.number_input("Max annual vol", 0.0, 5.0, float(ss_get("vol_max", 1.0)))

            cooldown_bars = st.number_input("Cooldown bars", 0, 200, int(ss_get("cooldown_bars", 5)))

            include_spread_penalty = st.checkbox(
                "Include spread penalty", value=bool(ss_get("include_spread_penalty", True))
            )
            assumed_spread_bps = st.number_input(
                "Assumed spread (bps)", 0.0, 200.0, float(ss_get("assumed_spread_bps", 5.0))
            )

            start_equity = st.number_input(
                "Starting equity ($)",
                min_value=1_000.0,
                max_value=10_000_000.0,
                value=float(ss_get("start_equity", 100_000.0)),
                step=1_000.0,
            )

            require_risk_on = st.checkbox(
                "Require risk-on regime (needs market_df)",
                value=bool(ss_get("require_risk_on", False)),
            )

            sr_lookback = st.number_input(
                "Support/Resistance lookback (bars)",
                min_value=10,
                max_value=300,
                value=int(ss_get("sr_lookback", 50)),
                step=5,
            )

        st.divider()
        b1, b2 = st.columns(2)
        with b1:
            load_btn = st.form_submit_button("🔄 Load/Refresh", use_container_width=True)
        with b2:
            run_backtest = st.form_submit_button("🚀 Run Backtest", use_container_width=True)

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


# Persist sidebar values
for k, v in {
    "symbol": symbol,
    "mode": mode,
    "horizon": horizon,
    "atr_entry": atr_entry,
    "atr_stop": atr_stop,
    "atr_target": atr_target,
    "rsi_min": rsi_min,
    "rsi_max": rsi_max,
    "rvol_min": rvol_min,
    "vol_max": vol_max,
    "cooldown_bars": cooldown_bars,
    "include_spread_penalty": include_spread_penalty,
    "assumed_spread_bps": assumed_spread_bps,
    "start_equity": start_equity,
    "require_risk_on": require_risk_on,
    "sr_lookback": sr_lookback,
}.items():
    st.session_state[k] = v


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
    with st.expander("Loader debug", expanded=True):
        st.json(debug_info or {})
    st.stop()


# ---------------------------
# Indicators
# ---------------------------
df_chart = df.copy()
if "timestamp" in df_chart.columns:
    df_chart["timestamp"] = pd.to_datetime(df_chart["timestamp"], utc=True, errors="coerce")
    df_chart = df_chart.dropna(subset=["timestamp"]).sort_values("timestamp")
    df_chart = df_chart.set_index("timestamp")
elif isinstance(df_chart.index, pd.DatetimeIndex):
    df_chart = df_chart.sort_index()

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
# Tab 1: Dashboard (Recommendation + Levels)
# ---------------------------
with tab1:
    rec, why = compute_recommendation_from_backtest_filters(
        df_chart,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        rvol_min=rvol_min,
        vol_max=vol_max,
    )

    plan = compute_trade_plan_from_backtest_rules(
        df_chart,
        mode=mode,
        atr_entry=atr_entry,
        atr_stop=atr_stop,
        atr_target=atr_target,
        assumed_spread_bps=assumed_spread_bps,
        include_spread_penalty=include_spread_penalty,
    )

    support, resistance = (np.nan, np.nan)
    if {"low", "high"}.issubset(df_chart.columns):
        support, resistance = compute_sr_levels(df_chart, int(sr_lookback))

    top_l, top_r = st.columns([2.0, 1.0], gap="large")

    with top_l:
        st.subheader(f"{symbol} — Signal & Levels")

        if rec == "BUY":
            st.success(f"**BUY** — {why}")
        elif rec == "SELL":
            st.error(f"**SELL** — {why}")
        else:
            st.info(f"**HOLD** — {why}")

        last = df_chart.iloc[-1]
        close = float(last["close"]) if "close" in df_chart.columns else np.nan
        rsi = float(last["rsi14"]) if "rsi14" in df_chart.columns else np.nan
        rvol = float(last["rvol"]) if "rvol" in df_chart.columns else np.nan
        vol_ann = float(last["vol_ann"]) if "vol_ann" in df_chart.columns else np.nan

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Close", f"{close:.2f}" if np.isfinite(close) else "—")
        m2.metric("Support", f"{support:.2f}" if np.isfinite(support) else "—")
        m3.metric("Resistance", f"{resistance:.2f}" if np.isfinite(resistance) else "—")
        m4.metric("RSI", f"{rsi:.1f}" if np.isfinite(rsi) else "—")
        m5.metric("RVOL", f"{rvol:.2f}" if np.isfinite(rvol) else "—")

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Planned entry", f"{plan['entry']:.2f}" if np.isfinite(plan.get("entry", np.nan)) else "—")
        e2.metric("Stop", f"{plan['stop']:.2f}" if np.isfinite(plan.get("stop", np.nan)) else "—")
        e3.metric("Target", f"{plan['target']:.2f}" if np.isfinite(plan.get("target", np.nan)) else "—")
        rr = plan.get("rr", np.nan)
        e4.metric("R:R", f"{rr:.2f}" if np.isfinite(rr) else "—")

        st.caption(f"Entry type: {plan.get('entry_type', '—')} (mirrors backtester trigger math)")

    with top_r:
        st.subheader("Data checks")

        if isinstance(sanity, dict):
            if sanity.get("ok", True):
                st.success("Sanity checks: OK")
            else:
                st.warning("Sanity checks: warnings")
            with st.expander("Sanity details", expanded=False):
                st.json(sanity)
        else:
            st.info("No sanity report available.")

        with st.expander("Loader debug", expanded=False):
            st.json(debug_info or {})

    st.divider()

    left, right = st.columns([2.2, 1.0], gap="large")

    with left:
        cols_to_plot = [c for c in ["close", "ma50", "ma200"] if c in df_chart.columns]
        if not cols_to_plot:
            cols_to_plot = ["close"] if "close" in df_chart.columns else list(df_chart.columns[:1])
        st.plotly_chart(plot_lines(df_chart, cols_to_plot, title=f"{symbol} Price + MAs", height=360), use_container_width=True)

    with right:
        if "rsi14" in df_chart.columns:
            st.plotly_chart(plot_indicator(df_chart, "rsi14", title="RSI(14)", height=220, hlines=[30, 70], y0=0, y1=100), use_container_width=True)
        if "rvol" in df_chart.columns:
            st.plotly_chart(plot_indicator(df_chart, "rvol", title="RVOL", height=220, hlines=[1.0]), use_container_width=True)

    if "vol_ann" in df_chart.columns:
        st.plotly_chart(plot_indicator(df_chart, "vol_ann", title="Annualized Vol (proxy)", height=220), use_container_width=True)


# ---------------------------
# Tab 2: Backtest
# ---------------------------
with tab2:
    st.subheader("Backtest")
    st.caption("Tip: if you get 0 trades, loosen filters (RVOL min down, vol max up, RSI range wider).")

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
                    require_risk_on=_risk_on,
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

            if "pnl_per_share" in t.columns:
                wins = (t["pnl_per_share"] > 0).sum()
                win_rate = wins / max(1, len(t))
                avg_win = t.loc[t["pnl_per_share"] > 0, "pnl_per_share"].mean()
                avg_loss = t.loc[t["pnl_per_share"] <= 0, "pnl_per_share"].mean()
            else:
                win_rate, avg_win, avg_loss = np.nan, np.nan, np.nan

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trades", f"{len(t)}")
            m2.metric("Win rate", f"{win_rate:.1%}" if np.isfinite(win_rate) else "—")
            m3.metric("Avg win (per share)", f"{avg_win:.3f}" if np.isfinite(avg_win) else "—")
            m4.metric("Avg loss (per share)", f"{avg_loss:.3f}" if np.isfinite(avg_loss) else "—")

            st.subheader("Trades")
            st.dataframe(t, use_container_width=True, height=420)

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
            st.dataframe(results.tail(50), use_container_width=True, height=420)
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

        col1, col2 = st.columns(2)
        with col1:
            st.button("▶️ Start Live", use_container_width=True, disabled=True)
        with col2:
            st.button("⏹ Stop Live", use_container_width=True, disabled=True)

        st.info("Live streaming is detected, but not wired into UI yet.")
        st.caption("Once you paste utils/live_stream.py, I’ll connect Start/Stop + live chart + last tick table.")
