import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.data_loader import load_historical, sanity_check_bars
from utils.indicators import add_indicators_inplace
from utils.backtester import backtest_strategy

LIVE_AVAILABLE = True
try:
    from utils.live_stream import RealtimeStream
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
# Minimal UI polish
# ---------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 1.2rem; }
[data-testid="stSidebar"] .block-container { padding-top: 0.8rem; }
h1 { margin-bottom: 0.2rem; }
h2, h3 { margin-top: 0.6rem; }
[data-testid="stMetric"] { padding: 0.15rem 0.15rem; }
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


def plot_lines(df: pd.DataFrame, cols: list[str], *, title: str, height: int = 380) -> go.Figure:
    x = df.index
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


def plot_indicator(df: pd.DataFrame, col: str, *, title: str, height: int = 240, hlines=None, y0=None, y1=None):
    x = df.index
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
    last = df_ind.iloc[-1]
    close = float(last["close"])
    atr = float(last["atr14"])

    if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
        return {"entry": np.nan, "stop": np.nan, "target": np.nan, "rr": np.nan, "entry_type": "—"}

    mode_l = str(mode).lower().strip()
    if mode_l == "pullback":
        entry = close - float(atr_entry) * atr
        entry_type = "Pullback (limit trigger)"
    elif mode_l == "breakout":
        entry = close + float(atr_entry) * atr
        entry_type = "Breakout (stop trigger)"
    else:
        return {"entry": np.nan, "stop": np.nan, "target": np.nan, "rr": np.nan, "entry_type": "—"}

    if include_spread_penalty and assumed_spread_bps > 0 and np.isfinite(entry):
        entry = entry * (1.0 + assumed_spread_bps / 10000.0)

    stop = entry - float(atr_stop) * atr
    target = entry + float(atr_target) * atr

    risk = entry - stop
    reward = target - entry
    rr = (reward / risk) if risk > 0 else np.nan

    return {"entry": entry, "stop": stop, "target": target, "rr": rr, "entry_type": entry_type}


def compute_recommendation_from_backtest_filters(
    df_ind: pd.DataFrame,
    *,
    rsi_min: float,
    rsi_max: float,
    rvol_min: float,
    vol_max: float,
) -> tuple[str, str]:
    last = df_ind.iloc[-1]
    need = ["close", "ma50", "ma200", "rsi14", "rvol", "vol_ann", "atr14"]
    missing = [c for c in need if c not in df_ind.columns]
    if missing:
        return "—", "Indicators not ready."

    close = float(last["close"])
    ma50 = float(last["ma50"]) if pd.notna(last["ma50"]) else np.nan
    ma200 = float(last["ma200"]) if pd.notna(last["ma200"]) else np.nan
    rsi = float(last["rsi14"])
    rvol = float(last["rvol"])
    vol_ann = float(last["vol_ann"])
    atr = float(last["atr14"])

    if not np.isfinite([close, rsi, rvol, vol_ann, atr]).all():
        return "HOLD", "Waiting for enough history to compute indicators."

    reasons = []
    if rsi < float(rsi_min) or rsi > float(rsi_max):
        reasons.append("RSI filter not met")
    if rvol < float(rvol_min):
        reasons.append("RVOL too low")
    if vol_ann > float(vol_max):
        reasons.append("Vol too high")

    uptrend = np.isfinite(ma50) and np.isfinite(ma200) and (close > ma50 > ma200)
    downtrend = np.isfinite(ma50) and np.isfinite(ma200) and (close < ma50 < ma200)

    if reasons:
        return "HOLD", " / ".join(reasons)

    if uptrend:
        return "BUY", "Filters pass + uptrend."
    if downtrend:
        return "SELL", "Filters pass but trend down."
    return "HOLD", "Filters pass, trend unclear."


# ---------------------------
# Secrets
# ---------------------------
api_key = st.secrets.get("ALPACA_KEY", "")
sec_key = st.secrets.get("ALPACA_SECRET", "")

st.title("📈 Modular Algorithmic Dashboard")

# ---------------------------
# Sidebar (clean)
# ---------------------------
with st.sidebar:
    st.header("Settings")

    with st.form("settings_form", clear_on_submit=False):
        symbol = st.text_input("Ticker", value=ss_get("symbol", "AAPL")).upper().strip()

        mode = st.selectbox("Entry mode", ["pullback", "breakout"], index=0)
        horizon = st.number_input("Max hold (bars)", min_value=1, max_value=200, value=int(ss_get("horizon", 20)))

        with st.expander("Advanced", expanded=False):
            atr_entry = st.number_input("ATR entry", 0.0, 10.0, float(ss_get("atr_entry", 1.0)), 0.1)
            atr_stop = st.number_input("ATR stop", 0.1, 20.0, float(ss_get("atr_stop", 2.0)), 0.1)
            atr_target = st.number_input("ATR target", 0.1, 50.0, float(ss_get("atr_target", 3.0)), 0.1)

            rsi_min = st.number_input("RSI min", 0.0, 100.0, float(ss_get("rsi_min", 30.0)))
            rsi_max = st.number_input("RSI max", 0.0, 100.0, float(ss_get("rsi_max", 70.0)))

            rvol_min = st.number_input("RVOL min", 0.0, 10.0, float(ss_get("rvol_min", 1.2)))
            vol_max = st.number_input("Max annual vol", 0.0, 5.0, float(ss_get("vol_max", 1.0)))

            include_spread_penalty = st.checkbox("Include spread penalty", value=bool(ss_get("include_spread_penalty", True)))
            assumed_spread_bps = st.number_input("Assumed spread (bps)", 0.0, 200.0, float(ss_get("assumed_spread_bps", 5.0)))

            sr_lookback = st.number_input("S/R lookback (bars)", min_value=10, max_value=300, value=int(ss_get("sr_lookback", 50)), step=5)

        st.divider()
        load_btn = st.form_submit_button("Load / Refresh", use_container_width=True)
        run_backtest = st.form_submit_button("Run Backtest", use_container_width=True)

    st.divider()
    st.caption("✅ Alpaca keys loaded" if has_keys(api_key, sec_key) else "❌ Missing Alpaca keys")


if not has_keys(api_key, sec_key):
    st.stop()


# ---------------------------
# Session init
# ---------------------------
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
    st.session_state.debug_info = None
    st.session_state.sanity = None
    st.session_state.last_symbol = None

# Persist minimal state
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
    "include_spread_penalty": include_spread_penalty,
    "assumed_spread_bps": assumed_spread_bps,
    "sr_lookback": sr_lookback,
}.items():
    st.session_state[k] = v


# ---------------------------
# Load / Refresh
# ---------------------------
needs_load = load_btn or (st.session_state.df_raw is None) or (st.session_state.last_symbol != symbol)

if needs_load:
    with st.spinner(f"Loading {symbol}..."):
        try:
            df, debug_info = load_historical(symbol, api_key, sec_key)
        except Exception:
            st.session_state.df_raw = None
            st.session_state.debug_info = None
            st.session_state.sanity = None
        else:
            st.session_state.df_raw = df
            st.session_state.debug_info = debug_info
            st.session_state.last_symbol = symbol
            try:
                st.session_state.sanity = sanity_check_bars(df) if df is not None else None
            except Exception:
                st.session_state.sanity = None

df = st.session_state.df_raw

if df is None or getattr(df, "empty", True):
    st.error(f"Could not load data for {symbol}.")
    st.stop()


# ---------------------------
# Indicators
# ---------------------------
df_chart = df.copy()
if "timestamp" in df_chart.columns:
    df_chart["timestamp"] = pd.to_datetime(df_chart["timestamp"], utc=True, errors="coerce")
    df_chart = df_chart.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
elif isinstance(df_chart.index, pd.DatetimeIndex):
    df_chart = df_chart.sort_index()

try:
    add_indicators_inplace(df_chart)
except Exception:
    pass


# ---------------------------
# Tabs (Signals separated from Charts)
# ---------------------------
tab_signal, tab_charts, tab_backtest, tab_live = st.tabs(
    ["✅ Signal", "📈 Charts", "🧪 Backtest", "📡 Live"]
)


# ---------------------------
# Signal tab (no debug / no sanity details)
# ---------------------------
with tab_signal:
    rec, why = compute_recommendation_from_backtest_filters(
        df_chart, rsi_min=rsi_min, rsi_max=rsi_max, rvol_min=rvol_min, vol_max=vol_max
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

    st.subheader(f"{symbol} — Recommendation")
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

    st.caption(f"Mode: {mode} • {plan.get('entry_type', '—')}")


# ---------------------------
# Charts tab (all charts live here)
# ---------------------------
with tab_charts:
    st.subheader(f"{symbol} — Charts")

    cols_to_plot = [c for c in ["close", "ma50", "ma200"] if c in df_chart.columns]
    if not cols_to_plot:
        cols_to_plot = ["close"] if "close" in df_chart.columns else list(df_chart.columns[:1])

    st.plotly_chart(plot_lines(df_chart, cols_to_plot, title=f"{symbol} Price + MAs", height=420), use_container_width=True)

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        if "rsi14" in df_chart.columns:
            st.plotly_chart(plot_indicator(df_chart, "rsi14", title="RSI(14)", height=240, hlines=[30, 70], y0=0, y1=100), use_container_width=True)
    with c2:
        if "rvol" in df_chart.columns:
            st.plotly_chart(plot_indicator(df_chart, "rvol", title="RVOL", height=240, hlines=[1.0]), use_container_width=True)
    with c3:
        if "vol_ann" in df_chart.columns:
            st.plotly_chart(plot_indicator(df_chart, "vol_ann", title="Annualized Vol", height=240), use_container_width=True)


# ---------------------------
# Backtest tab (kept, but no extra debug)
# ---------------------------
with tab_backtest:
    st.subheader("Backtest")

    if run_backtest:
        with st.spinner("Running backtest..."):
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
                cooldown_bars=0,
                include_spread_penalty=bool(include_spread_penalty),
                assumed_spread_bps=float(assumed_spread_bps),
                start_equity=100000.0,
            )

        if trades is None or getattr(trades, "empty", True):
            st.info("No trades generated with current params.")
        else:
            t = trades.copy()

            if "pnl_per_share" in t.columns:
                wins = (t["pnl_per_share"] > 0).sum()
                win_rate = wins / max(1, len(t))
            else:
                win_rate = np.nan

            c1, c2 = st.columns(2)
            c1.metric("Trades", f"{len(t)}")
            c2.metric("Win rate", f"{win_rate:.1%}" if np.isfinite(win_rate) else "—")

            st.dataframe(t, use_container_width=True, height=520)
    else:
        st.info("Click **Run Backtest** in the sidebar.")


# ---------------------------
# Live tab (optional)
# ---------------------------
with tab_live:
    st.subheader("Live")
    if not LIVE_AVAILABLE:
        st.info("Live module not available (or import failed).")
    else:
        st.info("Live wiring can be enabled once you confirm your Alpaca data subscription is active.")
        st.caption("If you want, I’ll plug in Start/Stop + quote table here cleanly.")
