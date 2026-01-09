# app.py
from __future__ import annotations

import queue
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
.small-muted { color: rgba(250,250,250,0.65); font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------
# Robust session init (NO attribute access surprises)
# ---------------------------
st.session_state.setdefault("df_raw", None)
st.session_state.setdefault("df_chart", None)
st.session_state.setdefault("last_symbol", None)
st.session_state.setdefault("load_error", None)
st.session_state.setdefault("sanity", None)
st.session_state.setdefault("ind_error", None)

# Live state (derive running state from stream.is_running())
st.session_state.setdefault("live_stream", None)
st.session_state.setdefault("live_rows", [])
st.session_state.setdefault("live_autorefresh", True)


# ---------------------------
# Helpers
# ---------------------------
def has_keys(api_key: str, sec_key: str) -> bool:
    return bool(api_key and sec_key and str(api_key).strip() and str(sec_key).strip())


def ss_get(name: str, default: Any) -> Any:
    return st.session_state.get(name, default)


def _safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _as_utc_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index()
    return out


def plot_lines(df: pd.DataFrame, cols: list[str], *, title: str, height: int = 420) -> go.Figure:
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


def plot_indicator(
    df: pd.DataFrame,
    col: str,
    *,
    title: str,
    height: int = 240,
    hlines=None,
    y0=None,
    y1=None,
) -> go.Figure:
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


def compute_lookback_low_high(df_ind: pd.DataFrame, lookback: int) -> tuple[float, float]:
    lb = int(max(10, lookback))
    tail = df_ind.tail(lb)
    lo = float(np.nanmin(tail["low"].values))
    hi = float(np.nanmax(tail["high"].values))
    return lo, hi


def detect_big_jump(df: pd.DataFrame, *, thresh: float = 0.18) -> Optional[dict]:
    """Flag max |close-to-close| jump. Heuristic for split/corp action/data issues."""
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return None
    c = pd.to_numeric(df["close"], errors="coerce")
    r = c.pct_change().abs()
    if r.dropna().empty:
        return None
    idx = r.idxmax()
    v = _safe_float(r.loc[idx]) if idx in r.index else np.nan
    if np.isfinite(v) and v >= float(thresh):
        return {"ts": idx, "abs_move": v}
    return None


def compute_trade_plan_breakout(
    df_ind: pd.DataFrame,
    *,
    atr_entry: float,
    atr_stop: float,
    atr_target: float,
    assumed_spread_bps: float,
    include_spread_penalty: bool,
) -> dict:
    """Breakout-only trade plan using ATR multiples."""
    last = df_ind.iloc[-1]
    close = _safe_float(last.get("close", np.nan))
    atr = _safe_float(last.get("atr14", np.nan))

    if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
        return {"entry": np.nan, "stop": np.nan, "target": np.nan, "rr": np.nan, "entry_type": "—"}

    entry = close + float(atr_entry) * atr
    entry_type = "Breakout (stop trigger)"

    # mild penalty to reflect worse-than-ideal fill
    if include_spread_penalty and assumed_spread_bps > 0 and np.isfinite(entry):
        entry = entry * (1.0 + assumed_spread_bps / 10000.0)

    stop = entry - float(atr_stop) * atr
    target = entry + float(atr_target) * atr

    risk = entry - stop
    reward = target - entry
    rr = (reward / risk) if risk > 0 else np.nan

    return {"entry": entry, "stop": stop, "target": target, "rr": rr, "entry_type": entry_type}


def compute_recommendation(
    df_ind: pd.DataFrame,
    *,
    rsi_min: float,
    rsi_max: float,
    rvol_min: float,
    vol_max: float,
) -> tuple[str, str]:
    need = ["close", "ma50", "ma200", "rsi14", "rvol", "vol_ann", "atr14"]
    missing = [c for c in need if c not in df_ind.columns]
    if missing:
        return "HOLD", "Indicators not ready yet."

    last = df_ind.iloc[-1]
    close = _safe_float(last.get("close"))
    ma50 = _safe_float(last.get("ma50"))
    ma200 = _safe_float(last.get("ma200"))
    rsi = _safe_float(last.get("rsi14"))
    rvol = _safe_float(last.get("rvol"))
    vol_ann = _safe_float(last.get("vol_ann"))
    atr = _safe_float(last.get("atr14"))

    vals = np.array([close, rsi, rvol, vol_ann, atr], dtype="float64")
    if not np.isfinite(vals).all():
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


def equity_and_drawdown_from_trades(trades: pd.DataFrame, start_equity: float = 100000.0):
    """
    Best-effort equity curve:
    - uses 'equity' if present
    - else uses cumulative PnL if possible
    """
    if trades is None or getattr(trades, "empty", True):
        return None, None

    t = trades.copy()
    eq = None

    if "equity" in t.columns:
        eq = pd.to_numeric(t["equity"], errors="coerce").dropna()
        eq.index = np.arange(len(eq))
    elif "pnl" in t.columns:
        pnl = pd.to_numeric(t["pnl"], errors="coerce").fillna(0.0)
        eq = start_equity + pnl.cumsum()
        eq.index = np.arange(len(eq))
    elif ("pnl_per_share" in t.columns) and ("qty" in t.columns):
        pps = pd.to_numeric(t["pnl_per_share"], errors="coerce").fillna(0.0)
        qty = pd.to_numeric(t["qty"], errors="coerce").fillna(0.0)
        pnl = (pps * qty).astype("float64")
        eq = start_equity + pnl.cumsum()
        eq.index = np.arange(len(eq))

    if eq is None or len(eq) < 2:
        return None, None

    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return eq, dd


def _live_running(stream_obj) -> bool:
    try:
        return bool(stream_obj is not None and getattr(stream_obj, "is_running")())
    except Exception:
        return False


def _live_msgs_to_df(rows: list[Any]) -> pd.DataFrame:
    """Convert alpaca quote objects/dicts to a tidy dataframe."""
    def to_dict(x: Any) -> dict:
        if isinstance(x, dict):
            return x
        for attr in ("model_dump", "dict"):
            m = getattr(x, attr, None)
            if callable(m):
                try:
                    return m()
                except Exception:
                    pass
        d = {}
        for k in ("symbol", "timestamp", "bid_price", "ask_price", "bid_size", "ask_size"):
            if hasattr(x, k):
                d[k] = getattr(x, k)
        if not d:
            d["message"] = str(x)
        return d

    df = pd.DataFrame([to_dict(x) for x in rows])

    # Normalize common short keys
    rename_map = {
        "bp": "bid_price",
        "ap": "ask_price",
        "bs": "bid_size",
        "as": "ask_size",
        "t": "timestamp",
        "S": "symbol",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    # Timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # Mid/spread bps
    if "bid_price" in df.columns and "ask_price" in df.columns:
        bid = pd.to_numeric(df["bid_price"], errors="coerce")
        ask = pd.to_numeric(df["ask_price"], errors="coerce")
        mid = (bid + ask) / 2.0
        spread = ask - bid
        df["mid"] = mid
        df["spread"] = spread
        df["spread_bps"] = np.where(mid > 0, (spread / mid) * 10000.0, np.nan)

    return df


# ---------------------------
# Secrets (optional; only required for live + possibly your loader)
# ---------------------------
api_key = st.secrets.get("ALPACA_KEY", "")
sec_key = st.secrets.get("ALPACA_SECRET", "")

st.title("📈 Modular Algorithmic Dashboard")


# ---------------------------
# Sidebar (NO pullback option)
# ---------------------------
with st.sidebar:
    st.header("Settings")

    with st.form("settings_form", clear_on_submit=False):
        symbol = st.text_input("Ticker", value=ss_get("symbol", "AAPL")).upper().strip()
        horizon = st.number_input("Max hold (bars)", min_value=1, max_value=200, value=int(ss_get("horizon", 20)))

        with st.expander("Advanced", expanded=False):
            # Breakout-only ATR plan
            atr_entry = st.number_input("ATR entry (breakout)", 0.0, 10.0, float(ss_get("atr_entry", 1.0)), 0.1)
            atr_stop = st.number_input("ATR stop", 0.1, 20.0, float(ss_get("atr_stop", 2.0)), 0.1)
            atr_target = st.number_input("ATR target", 0.1, 50.0, float(ss_get("atr_target", 3.0)), 0.1)

            # Filters
            rsi_min = st.number_input("RSI min", 0.0, 100.0, float(ss_get("rsi_min", 30.0)))
            rsi_max = st.number_input("RSI max", 0.0, 100.0, float(ss_get("rsi_max", 70.0)))
            rvol_min = st.number_input("RVOL min", 0.0, 10.0, float(ss_get("rvol_min", 1.2)))
            vol_max = st.number_input("Max annual vol", 0.0, 5.0, float(ss_get("vol_max", 1.0)))

            include_spread_penalty = st.checkbox(
                "Include spread penalty",
                value=bool(ss_get("include_spread_penalty", True)),
            )
            assumed_spread_bps = st.number_input(
                "Assumed spread (bps)",
                0.0,
                200.0,
                float(ss_get("assumed_spread_bps", 5.0)),
            )

            sr_lookback = st.number_input(
                "Lookback low/high (bars)",
                min_value=10,
                max_value=300,
                value=int(ss_get("sr_lookback", 50)),
                step=5,
            )

        st.divider()
        load_btn = st.form_submit_button("Load / Refresh", use_container_width=True)
        run_backtest = st.form_submit_button("Run Backtest", use_container_width=True)

    st.divider()
    if has_keys(api_key, sec_key):
        st.caption("✅ Alpaca keys loaded (Live enabled)")
    else:
        st.caption("ℹ️ No Alpaca keys (Charts/Backtest still work; Live disabled if needed)")

# Persist user inputs
for k, v in {
    "symbol": symbol,
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
# Load / Refresh (no silent failures)
# ---------------------------
needs_load = load_btn or (st.session_state.get("df_raw") is None) or (st.session_state.get("last_symbol") != symbol)

if needs_load:
    st.session_state["load_error"] = None
    st.session_state["ind_error"] = None

    with st.spinner(f"Loading {symbol}..."):
        try:
            df, _debug_info = load_historical(symbol, api_key, sec_key)
        except Exception as e:
            st.session_state["df_raw"] = None
            st.session_state["df_chart"] = None
            st.session_state["last_symbol"] = None
            st.session_state["sanity"] = None
            st.session_state["load_error"] = f"{type(e).__name__}: {e}"
        else:
            st.session_state["df_raw"] = df
            st.session_state["last_symbol"] = symbol
            try:
                st.session_state["sanity"] = sanity_check_bars(df) if df is not None else None
            except Exception as e:
                st.session_state["sanity"] = None
                st.session_state["load_error"] = f"Sanity check failed: {type(e).__name__}: {e}"

            # Normalize + indicators once; reuse everywhere
            try:
                df_chart = _as_utc_dtindex(df)
                add_indicators_inplace(df_chart)
                st.session_state["df_chart"] = df_chart
            except Exception as e:
                st.session_state["df_chart"] = _as_utc_dtindex(df)
                st.session_state["ind_error"] = f"{type(e).__name__}: {e}"

df_raw = st.session_state.get("df_raw")
df_chart = st.session_state.get("df_chart")

if df_raw is None or getattr(df_raw, "empty", True):
    st.error(f"Could not load data for {symbol}.")
    if st.session_state.get("load_error"):
        st.caption(f"Error: {st.session_state['load_error']}")
    st.stop()

if df_chart is None or getattr(df_chart, "empty", True):
    st.error("Data loaded but chart dataframe is empty.")
    if st.session_state.get("ind_error"):
        st.caption(f"Indicators error: {st.session_state['ind_error']}")
    st.stop()


# ---------------------------
# Tabs
# ---------------------------
tab_signal, tab_charts, tab_backtest, tab_live = st.tabs(
    ["✅ Signal", "📈 Charts", "🧪 Backtest", "📡 Live"]
)


# ---------------------------
# Signal tab
# ---------------------------
with tab_signal:
    if st.session_state.get("ind_error"):
        st.warning("Some indicators failed to compute. Signals may be limited.")
        st.caption(st.session_state["ind_error"])

    jump = detect_big_jump(df_chart, thresh=0.18)
    if jump:
        st.warning("Large price jump detected (possible corporate action / data issue).")
        st.caption(f"Max |close-to-close| move: {jump['abs_move']:.1%} at {jump['ts']}")

    sanity = st.session_state.get("sanity")
    if isinstance(sanity, dict):
        red_flags = []
        for k, v in sanity.items():
            if isinstance(v, (bool, int, float)) and bool(v):
                red_flags.append(k)
            elif isinstance(v, (list, tuple, set)) and len(v) > 0:
                red_flags.append(k)
            elif isinstance(v, str) and v.strip():
                red_flags.append(k)
        if red_flags:
            st.info("Data quality checks flagged potential issues.")
            st.caption(" • ".join(red_flags[:10]))

    rec, why = compute_recommendation(
        df_chart,
        rsi_min=float(rsi_min),
        rsi_max=float(rsi_max),
        rvol_min=float(rvol_min),
        vol_max=float(vol_max),
    )
    plan = compute_trade_plan_breakout(
        df_chart,
        atr_entry=float(atr_entry),
        atr_stop=float(atr_stop),
        atr_target=float(atr_target),
        assumed_spread_bps=float(assumed_spread_bps),
        include_spread_penalty=bool(include_spread_penalty),
    )

    lo, hi = (np.nan, np.nan)
    if {"low", "high"}.issubset(df_chart.columns):
        lo, hi = compute_lookback_low_high(df_chart, int(sr_lookback))

    st.subheader(f"{symbol} — Recommendation")
    if rec == "BUY":
        st.success(f"**BUY** — {why}")
    elif rec == "SELL":
        st.error(f"**SELL** — {why}")
    else:
        st.info(f"**HOLD** — {why}")

    last = df_chart.iloc[-1]
    close = _safe_float(last.get("close", np.nan))
    rsi = _safe_float(last.get("rsi14", np.nan))
    rvol = _safe_float(last.get("rvol", np.nan))
    vol_ann = _safe_float(last.get("vol_ann", np.nan))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Close", f"{close:.2f}" if np.isfinite(close) else "—")
    m2.metric("Lookback Low", f"{lo:.2f}" if np.isfinite(lo) else "—")
    m3.metric("Lookback High", f"{hi:.2f}" if np.isfinite(hi) else "—")
    m4.metric("RSI", f"{rsi:.1f}" if np.isfinite(rsi) else "—")
    m5.metric("RVOL", f"{rvol:.2f}" if np.isfinite(rvol) else "—")

    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Planned entry", f"{plan['entry']:.2f}" if np.isfinite(plan.get("entry", np.nan)) else "—")
    e2.metric("Stop", f"{plan['stop']:.2f}" if np.isfinite(plan.get("stop", np.nan)) else "—")
    e3.metric("Target", f"{plan['target']:.2f}" if np.isfinite(plan.get("target", np.nan)) else "—")
    rr = plan.get("rr", np.nan)
    e4.metric("R:R", f"{rr:.2f}" if np.isfinite(rr) else "—")
    e5.metric("Ann. Vol", f"{vol_ann:.2f}" if np.isfinite(vol_ann) else "—")

    st.caption("Plan: Breakout-only (no pullback mode).")


# ---------------------------
# Charts tab
# ---------------------------
with tab_charts:
    st.subheader(f"{symbol} — Charts")

    cols_to_plot = [c for c in ["close", "ma50", "ma200"] if c in df_chart.columns]
    if not cols_to_plot:
        cols_to_plot = ["close"] if "close" in df_chart.columns else list(df_chart.columns[:1])

    st.plotly_chart(
        plot_lines(df_chart, cols_to_plot, title=f"{symbol} Price + MAs", height=420),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        if "rsi14" in df_chart.columns:
            st.plotly_chart(
                plot_indicator(df_chart, "rsi14", title="RSI(14)", height=240, hlines=[30, 70], y0=0, y1=100),
                use_container_width=True,
            )
    with c2:
        if "rvol" in df_chart.columns:
            st.plotly_chart(
                plot_indicator(df_chart, "rvol", title="RVOL", height=240, hlines=[1.0]),
                use_container_width=True,
            )
    with c3:
        if "vol_ann" in df_chart.columns:
            st.plotly_chart(
                plot_indicator(df_chart, "vol_ann", title="Annualized Vol", height=240),
                use_container_width=True,
            )


# ---------------------------
# Backtest tab (summary + curves)
# ---------------------------
with tab_backtest:
    st.subheader("Backtest (Breakout-only)")

    if run_backtest:
        with st.spinner("Running backtest..."):
            try:
                results, trades = backtest_strategy(
                    df=df_chart,               # IMPORTANT: use same normalized df as signals
                    market_df=None,
                    horizon=int(horizon),
                    mode="breakout",           # enforced
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
            except Exception as e:
                st.error("Backtest failed.")
                st.caption(f"{type(e).__name__}: {e}")
                results, trades = None, None

        if trades is None or getattr(trades, "empty", True):
            st.info("No trades generated with current params.")
        else:
            t = trades.copy()

            win_rate = np.nan
            if "pnl_per_share" in t.columns:
                p = pd.to_numeric(t["pnl_per_share"], errors="coerce")
                win_rate = float((p > 0).sum()) / max(1, len(p))
            elif "pnl" in t.columns:
                p = pd.to_numeric(t["pnl"], errors="coerce")
                win_rate = float((p > 0).sum()) / max(1, len(p))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades", f"{len(t)}")
            c2.metric("Win rate", f"{win_rate:.1%}" if np.isfinite(win_rate) else "—")

            if isinstance(results, dict):
                total_ret = _safe_float(results.get("total_return", np.nan))
                sharpe = _safe_float(results.get("sharpe", np.nan))
                maxdd = _safe_float(results.get("max_drawdown", np.nan))
                c3.metric("Total return", f"{total_ret:.1%}" if np.isfinite(total_ret) else "—")
                c4.metric("Max drawdown", f"{maxdd:.1%}" if np.isfinite(maxdd) else "—")
                if np.isfinite(sharpe):
                    st.caption(f"Sharpe (if provided by backtester): {sharpe:.2f}")
            else:
                c3.metric("Total return", "—")
                c4.metric("Max drawdown", "—")

            eq, dd = equity_and_drawdown_from_trades(t, start_equity=100000.0)
            if eq is not None and dd is not None:
                st.plotly_chart(
                    go.Figure([go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Equity")]).update_layout(
                        title="Equity Curve (per trade)",
                        height=320,
                        margin=dict(l=10, r=10, t=45, b=10),
                        hovermode="x unified",
                    ),
                    use_container_width=True,
                )
                st.plotly_chart(
                    go.Figure([go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown")]).update_layout(
                        title="Drawdown (per trade)",
                        height=260,
                        margin=dict(l=10, r=10, t=45, b=10),
                        hovermode="x unified",
                    ),
                    use_container_width=True,
                )

            st.divider()
            st.dataframe(t, use_container_width=True, height=520)
    else:
        st.info("Click **Run Backtest** in the sidebar.")


# ---------------------------
# Live tab (Start/Stop + tidy quote table)
# ---------------------------
with tab_live:
    st.subheader("Live Quotes")

    if not LIVE_AVAILABLE:
        st.info("Live module not available (or import failed).")
    elif not has_keys(api_key, sec_key):
        st.info("Live is disabled because Alpaca keys are missing.")
    else:
        stream = st.session_state.get("live_stream")
        live_running = _live_running(stream)

        a, b, c, d = st.columns([1, 1, 1, 2])
        start_clicked = a.button("▶ Start", use_container_width=True, disabled=live_running)
        stop_clicked = b.button("⏹ Stop", use_container_width=True, disabled=not live_running)
        clear_clicked = c.button("🧹 Clear", use_container_width=True)

        st.session_state["live_autorefresh"] = d.toggle(
            "Auto refresh",
            value=bool(st.session_state.get("live_autorefresh", True)),
        )

        if clear_clicked:
            st.session_state["live_rows"] = []

        if start_clicked and not live_running:
            try:
                stream = RealtimeStream(api_key, sec_key, symbol)
                stream.start()
                st.session_state["live_stream"] = stream
                live_running = True
            except Exception as e:
                st.session_state["live_stream"] = None
                st.error("Failed to start live stream.")
                st.caption(f"{type(e).__name__}: {e}")
                live_running = False

        if stop_clicked and live_running:
            try:
                if stream is not None:
                    stream.stop()
            except Exception:
                pass
            st.session_state["live_stream"] = None
            stream = None
            live_running = False

        # Drain queue into buffer
        stream = st.session_state.get("live_stream")
        if stream is not None:
            try:
                new_msgs = stream.get_latest(max_items=250)
            except Exception:
                new_msgs = []
            if new_msgs:
                st.session_state["live_rows"].extend(new_msgs)
                st.session_state["live_rows"] = st.session_state["live_rows"][-500:]

        st.caption("Status: ✅ running" if live_running else "Status: ⏸ stopped")

        rows = st.session_state.get("live_rows", [])
        if not rows:
            st.info("No quote updates received yet.")
        else:
            df_live = _live_msgs_to_df(rows)

            # Snapshot metrics
            latest = df_live.tail(1)
            if not latest.empty and ("bid_price" in latest.columns) and ("ask_price" in latest.columns):
                lbid = _safe_float(pd.to_numeric(latest["bid_price"], errors="coerce").iloc[0])
                lask = _safe_float(pd.to_numeric(latest["ask_price"], errors="coerce").iloc[0])
                lmid = (lbid + lask) / 2.0 if np.isfinite([lbid, lask]).all() else np.nan
                lsp = (lask - lbid) if np.isfinite([lbid, lask]).all() else np.nan
                lsp_bps = (lsp / lmid) * 10000.0 if np.isfinite([lsp, lmid]).all() and lmid > 0 else np.nan

                m1, m2, m3 = st.columns(3)
                m1.metric("Bid", f"{lbid:.4f}" if np.isfinite(lbid) else "—")
                m2.metric("Ask", f"{lask:.4f}" if np.isfinite(lask) else "—")
                m3.metric("Spread (bps)", f"{lsp_bps:.1f}" if np.isfinite(lsp_bps) else "—")

            show_cols = [c for c in ["timestamp", "symbol", "bid_price", "ask_price", "mid", "spread_bps", "bid_size", "ask_size"] if c in df_live.columns]
            if show_cols:
                # show newest last
                sort_col = "timestamp" if "timestamp" in show_cols else show_cols[0]
                view = df_live.sort_values(sort_col).tail(120)
                st.dataframe(view[show_cols], use_container_width=True, height=460)
            else:
                st.dataframe(df_live.tail(120), use_container_width=True, height=460)

        # Smooth refresh without manual rerun loops
        if live_running and bool(st.session_state.get("live_autorefresh", True)):
            st.autorefresh(interval=750, key=f"live_refresh_{symbol}")
