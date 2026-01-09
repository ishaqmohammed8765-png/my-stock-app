from __future__ import annotations

import time
import queue
from dataclasses import dataclass
from typing import Optional

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
.small-muted { color: rgba(250,250,250,0.65); font-size: 0.9rem; }
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


def _safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _as_utc_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe to UTC DateTimeIndex using 'timestamp' column if present."""
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    elif isinstance(out.index, pd.DatetimeIndex):
        # try enforce utc-like behavior (won't convert naive reliably, just sort)
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
    # NOTE: These are lookback extrema; label accordingly in UI to avoid overclaiming.
    support = float(np.nanmin(tail["low"].values))
    resistance = float(np.nanmax(tail["high"].values))
    return support, resistance


def detect_big_jump(df: pd.DataFrame, *, thresh: float = 0.18) -> Optional[dict]:
    """
    Heuristic: flag large close-to-close jumps (possible split/corp action or data issue).
    thresh=0.18 means 18% overnight move.
    """
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return None
    c = df["close"].astype("float64")
    r = c.pct_change().abs()
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
    """
    Breakout-only plan:
    - entry is above close by atr_entry * ATR
    - stop/target around entry by ATR multiples
    """
    last = df_ind.iloc[-1]
    close = _safe_float(last.get("close", np.nan))
    atr = _safe_float(last.get("atr14", np.nan))

    if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
        return {"entry": np.nan, "stop": np.nan, "target": np.nan, "rr": np.nan, "entry_type": "—"}

    entry = close + float(atr_entry) * atr
    entry_type = "Breakout (stop trigger)"

    if include_spread_penalty and assumed_spread_bps > 0 and np.isfinite(entry):
        # mild penalty to reflect less-than-ideal fill
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
    - If trades has 'equity' column, use it.
    - Else if it has 'pnl' or 'pnl_per_share' and 'qty', accumulate.
    - Else return None.
    """
    if trades is None or getattr(trades, "empty", True):
        return None, None

    t = trades.copy()
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
    else:
        return None, None

    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return eq, dd


# ---------------------------
# Secrets (optional)
# ---------------------------
api_key = st.secrets.get("ALPACA_KEY", "")
sec_key = st.secrets.get("ALPACA_SECRET", "")

st.title("📈 Modular Algorithmic Dashboard")


# ---------------------------
# Sidebar (clean, no pullback)
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
                "Include spread penalty", value=bool(ss_get("include_spread_penalty", True))
            )
            assumed_spread_bps = st.number_input(
                "Assumed spread (bps)", 0.0, 200.0, float(ss_get("assumed_spread_bps", 5.0))
            )

            sr_lookback = st.number_input(
                "Lookback low/high (bars)", min_value=10, max_value=300, value=int(ss_get("sr_lookback", 50)), step=5
            )

        st.divider()
        load_btn = st.form_submit_button("Load / Refresh", use_container_width=True)
        run_backtest = st.form_submit_button("Run Backtest", use_container_width=True)

    st.divider()
    if has_keys(api_key, sec_key):
        st.caption("✅ Alpaca keys loaded (Live enabled)")
    else:
        st.caption("ℹ️ No Alpaca keys (Charts/Backtest still work; Live disabled)")


# ---------------------------
# Session init
# ---------------------------
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
    st.session_state.last_symbol = None
    st.session_state.load_error = None
    st.session_state.sanity = None

# Live state
if "live_stream" not in st.session_state:
    st.session_state.live_stream = None
    st.session_state.live_running = False
    st.session_state.live_rows = []

# Persist minimal state
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
# Load / Refresh (don’t silently swallow errors)
# ---------------------------
needs_load = load_btn or (st.session_state.df_raw is None) or (st.session_state.last_symbol != symbol)

if needs_load:
    st.session_state.load_error = None
    with st.spinner(f"Loading {symbol}..."):
        try:
            # Your loader can decide what to do with empty keys.
            df, debug_info = load_historical(symbol, api_key, sec_key)
        except Exception as e:
            st.session_state.df_raw = None
            st.session_state.last_symbol = None
            st.session_state.sanity = None
            st.session_state.load_error = f"{type(e).__name__}: {e}"
        else:
            st.session_state.df_raw = df
            st.session_state.last_symbol = symbol
            # sanity checks are allowed, but still non-intrusive
            try:
                st.session_state.sanity = sanity_check_bars(df) if df is not None else None
            except Exception as e:
                st.session_state.sanity = None
                st.session_state.load_error = f"Sanity check failed: {type(e).__name__}: {e}"

df_raw = st.session_state.df_raw
if df_raw is None or getattr(df_raw, "empty", True):
    st.error(f"Could not load data for {symbol}.")
    if st.session_state.load_error:
        st.caption(f"Error: {st.session_state.load_error}")
    st.stop()


# ---------------------------
# Normalize + Indicators (single source of truth for all tabs)
# ---------------------------
df_chart = _as_utc_dtindex(df_raw)

ind_error = None
try:
    add_indicators_inplace(df_chart)
except Exception as e:
    ind_error = f"{type(e).__name__}: {e}"

# Lightweight data alerts (no debug dump)
jump = detect_big_jump(df_chart, thresh=0.18)
sanity = st.session_state.sanity


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
    # clean, minimal banners
    if ind_error:
        st.warning("Some indicators failed to compute. Signals may be limited.")
        st.caption(ind_error)

    if jump:
        ts = jump["ts"]
        mv = jump["abs_move"]
        st.warning("Large price jump detected (possible corporate action / data issue).")
        st.caption(f"Max |close-to-close| move: {mv:.1%} at {ts}")

    # if sanity has flags, show a compact status
    if isinstance(sanity, dict):
        # show only if something looks off
        red_flags = []
        for k, v in sanity.items():
            # heuristic: anything truthy and not obviously benign
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
        df_chart, rsi_min=rsi_min, rsi_max=rsi_max, rvol_min=rvol_min, vol_max=vol_max
    )
    plan = compute_trade_plan_breakout(
        df_chart,
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
    close = _safe_float(last.get("close", np.nan))
    rsi = _safe_float(last.get("rsi14", np.nan))
    rvol = _safe_float(last.get("rvol", np.nan))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Close", f"{close:.2f}" if np.isfinite(close) else "—")
    m2.metric("Lookback Low", f"{support:.2f}" if np.isfinite(support) else "—")
    m3.metric("Lookback High", f"{resistance:.2f}" if np.isfinite(resistance) else "—")
    m4.metric("RSI", f"{rsi:.1f}" if np.isfinite(rsi) else "—")
    m5.metric("RVOL", f"{rvol:.2f}" if np.isfinite(rvol) else "—")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Planned entry", f"{plan['entry']:.2f}" if np.isfinite(plan.get("entry", np.nan)) else "—")
    e2.metric("Stop", f"{plan['stop']:.2f}" if np.isfinite(plan.get("stop", np.nan)) else "—")
    e3.metric("Target", f"{plan['target']:.2f}" if np.isfinite(plan.get("target", np.nan)) else "—")
    rr = plan.get("rr", np.nan)
    e4.metric("R:R", f"{rr:.2f}" if np.isfinite(rr) else "—")

    st.caption(f"Plan: {plan.get('entry_type', '—')} • Breakout-only mode")


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
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        if "rsi14" in df_chart.columns:
            st.plotly_chart(
                plot_indicator(df_chart, "rsi14", title="RSI(14)", height=240, hlines=[30, 70], y0=0, y1=100),
                use_container_width=True
            )
    with c2:
        if "rvol" in df_chart.columns:
            st.plotly_chart(
                plot_indicator(df_chart, "rvol", title="RVOL", height=240, hlines=[1.0]),
                use_container_width=True
            )
    with c3:
        if "vol_ann" in df_chart.columns:
            st.plotly_chart(
                plot_indicator(df_chart, "vol_ann", title="Annualized Vol", height=240),
                use_container_width=True
            )


# ---------------------------
# Backtest tab (summary + curves)
# ---------------------------
with tab_backtest:
    st.subheader("Backtest (Breakout-only)")

    if run_backtest:
        with st.spinner("Running backtest..."):
            # IMPORTANT: use df_chart (cleaned/indexed) so Signal/Charts/Backtest align
            try:
                results, trades = backtest_strategy(
                    df=df_chart,
                    market_df=None,
                    horizon=int(horizon),
                    mode="breakout",  # enforced
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
                trades = None
                results = None

        if trades is None or getattr(trades, "empty", True):
            st.info("No trades generated with current params.")
        else:
            t = trades.copy()

            # win rate best-effort
            win_rate = np.nan
            if "pnl_per_share" in t.columns:
                wins = (pd.to_numeric(t["pnl_per_share"], errors="coerce") > 0).sum()
                win_rate = wins / max(1, len(t))
            elif "pnl" in t.columns:
                wins = (pd.to_numeric(t["pnl"], errors="coerce") > 0).sum()
                win_rate = wins / max(1, len(t))

            # headline metrics (best-effort)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades", f"{len(t)}")
            c2.metric("Win rate", f"{win_rate:.1%}" if np.isfinite(win_rate) else "—")

            # try show results keys if dict-like
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

            # Equity curve + drawdown (best-effort)
            eq, dd = equity_and_drawdown_from_trades(t, start_equity=100000.0)
            if eq is not None and dd is not None and len(eq) > 1:
                st.plotly_chart(
                    go.Figure([go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Equity")]).update_layout(
                        title="Equity Curve (per trade)",
                        height=320,
                        margin=dict(l=10, r=10, t=45, b=10),
                        hovermode="x unified",
                    ),
                    use_container_width=True
                )
                st.plotly_chart(
                    go.Figure([go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown")]).update_layout(
                        title="Drawdown (per trade)",
                        height=260,
                        margin=dict(l=10, r=10, t=45, b=10),
                        hovermode="x unified",
                    ),
                    use_container_width=True
                )

            st.divider()
            st.dataframe(t, use_container_width=True, height=520)
    else:
        st.info("Click **Run Backtest** in the sidebar.")


# ---------------------------
# Live tab (Start/Stop + latest quotes)
# ---------------------------
with tab_live:
    st.subheader("Live")

    if not LIVE_AVAILABLE:
        st.info("Live module not available (or import failed).")
    elif not has_keys(api_key, sec_key):
        st.info("Live is disabled because Alpaca keys are missing.")
    else:
        # Controls
        colA, colB, colC = st.columns([1, 1, 2])
        start = colA.button("▶ Start Live", use_container_width=True, disabled=st.session_state.live_running)
        stop = colB.button("⏹ Stop Live", use_container_width=True, disabled=not st.session_state.live_running)

        if start and not st.session_state.live_running:
            try:
                st.session_state.live_stream = RealtimeStream(api_key, sec_key, symbol)
                # Expect RealtimeStream to manage its own background thread and enqueue messages.
                # If your implementation requires an explicit start(), call it here.
                if hasattr(st.session_state.live_stream, "start"):
                    st.session_state.live_stream.start()
                st.session_state.live_running = True
                st.session_state.live_rows = []
            except Exception as e:
                st.session_state.live_stream = None
                st.session_state.live_running = False
                st.error("Failed to start live stream.")
                st.caption(f"{type(e).__name__}: {e}")

        if stop and st.session_state.live_running:
            try:
                if st.session_state.live_stream and hasattr(st.session_state.live_stream, "stop"):
                    st.session_state.live_stream.stop()
            except Exception:
                pass
            st.session_state.live_stream = None
            st.session_state.live_running = False

        if st.session_state.live_running:
            st.caption("Streaming… (shows latest messages received)")

            # Pull a few messages from the queue, if available
            stream = st.session_state.live_stream
            got = 0
            if stream is not None and hasattr(stream, "msg_queue"):
                q = stream.msg_queue
                # Drain a bit each rerun to keep UI responsive
                for _ in range(50):
                    try:
                        msg = q.get_nowait()
                    except queue.Empty:
                        break
                    else:
                        got += 1
                        st.session_state.live_rows.append(msg)

                # Keep only last N rows
                st.session_state.live_rows = st.session_state.live_rows[-200:]

            if got:
                st.caption(f"Received {got} new updates.")

            # Render table best-effort
            rows = st.session_state.live_rows
            if rows:
                # Try to normalize dict-like messages to a dataframe
                if isinstance(rows[-1], dict):
                    df_live = pd.DataFrame(rows)
                else:
                    df_live = pd.DataFrame({"message": rows})

                st.dataframe(df_live.tail(50), use_container_width=True, height=460)
            else:
                st.info("No live messages received yet.")

            # auto-refresh feel without heavy loops
            time.sleep(0.15)
            st.rerun()
        else:
            st.info("Click **Start Live** to begin streaming quotes/trades (depending on your stream implementation).")
