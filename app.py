# app.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_historical, sanity_check_bars
from utils.indicators import add_indicators_inplace
from utils.backtester import backtest_strategy

# Optional: auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
except Exception:
    st_autorefresh = None  # type: ignore

# Optional: live quotes
LIVE_AVAILABLE = True
try:
    from utils.live_stream import RealtimeStream
except Exception:
    LIVE_AVAILABLE = False

# Optional: Yahoo fallback
YF_AVAILABLE = True
try:
    import yfinance as yf  # pip install yfinance
except Exception:
    YF_AVAILABLE = False


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(page_title="Pro Algo Trader", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
.block-container { padding-top: 0.9rem; padding-bottom: 1.1rem; max-width: 1320px; }
[data-testid="stSidebar"] .block-container { padding-top: 0.8rem; }
h1 { margin-bottom: 0.15rem; letter-spacing: -0.3px; }
[data-testid="stMetric"] { padding: 0.1rem 0.15rem; border-radius: 12px; }
div[data-testid="stVerticalBlockBorderWrapper"]{ border-radius: 16px; }
.stButton button { border-radius: 14px; }
.stDownloadButton button { border-radius: 14px; }
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Session helpers
# =============================================================================
def _ss_setdefault(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def ss_get(key: str, default: Any) -> Any:
    return st.session_state.get(key, default)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _has_keys_in_secrets() -> bool:
    k = str(st.secrets.get("ALPACA_KEY", "")).strip()
    s = str(st.secrets.get("ALPACA_SECRET", "")).strip()
    return bool(k and s)


def _as_utc_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index()
        out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
    return out


def _ensure_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    lower = {str(c).lower(): c for c in out.columns}
    mapping: Dict[str, str] = {}

    variants = {
        "open": ["open", "o", "opn"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "adj close", "adj_close", "adjclose"],
        "volume": ["volume", "v", "vol"],
    }

    for want, keys in variants.items():
        if want in out.columns:
            continue
        for k in keys:
            if k in lower:
                mapping[lower[k]] = want
                break

    return out.rename(columns=mapping) if mapping else out


def _coerce_ohlcv_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    price_cols = [c for c in ["open", "high", "low", "close"] if c in out.columns]
    return out.dropna(subset=price_cols) if price_cols else out


def _tail_for_plot(df: pd.DataFrame, n: int) -> pd.DataFrame:
    n = int(max(120, n))
    return df.tail(n) if len(df) > n else df


# =============================================================================
# Plots (trade markers)
# =============================================================================
def _extract_markers(trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expect trades columns: entry_ts, entry_px, exit_ts, exit_px
    """
    if trades is None or getattr(trades, "empty", True):
        return pd.DataFrame(columns=["ts", "px"]), pd.DataFrame(columns=["ts", "px"])

    t = trades.copy()
    if "entry_ts" in t.columns:
        t["entry_ts"] = pd.to_datetime(t["entry_ts"], errors="coerce", utc=True)
    if "exit_ts" in t.columns:
        t["exit_ts"] = pd.to_datetime(t["exit_ts"], errors="coerce", utc=True)

    entries = pd.DataFrame(
        {"ts": t.get("entry_ts", pd.Series(dtype="datetime64[ns, UTC]")),
         "px": pd.to_numeric(t.get("entry_px", pd.Series(dtype=float)), errors="coerce")}
    ).dropna(subset=["ts", "px"])

    exits = pd.DataFrame(
        {"ts": t.get("exit_ts", pd.Series(dtype="datetime64[ns, UTC]")),
         "px": pd.to_numeric(t.get("exit_px", pd.Series(dtype=float)), errors="coerce")}
    ).dropna(subset=["ts", "px"])

    return entries, exits


def plot_price(df: pd.DataFrame, symbol: str, trades: Optional[pd.DataFrame] = None) -> go.Figure:
    x = df.index
    fig = go.Figure()

    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        fig.add_trace(
            go.Candlestick(
                x=x,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
                showlegend=False,
            )
        )
    else:
        fig.add_trace(go.Scatter(x=x, y=df.get("close", pd.Series(index=x, dtype=float)), mode="lines", name="Close"))

    for col, label in [("ma50", "MA50"), ("ma200", "MA200")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[col], mode="lines", name=label))

    # Trade markers (color-coded)
    if isinstance(trades, pd.DataFrame) and not trades.empty:
        entries, exits = _extract_markers(trades)
        if not entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=entries["ts"],
                    y=entries["px"],
                    mode="markers",
                    name="Entry",
                    marker=dict(symbol="triangle-up", size=10, color="green"),
                )
            )
        if not exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=exits["ts"],
                    y=exits["px"],
                    mode="markers",
                    name="Exit",
                    marker=dict(symbol="triangle-down", size=10, color="red"),
                )
            )

    fig.update_layout(
        title=f"{symbol} — Price",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, rangeslider_visible=False)
    fig.update_yaxes(showgrid=True)
    return fig


def plot_indicator(
    df: pd.DataFrame,
    col: str,
    title: str,
    *,
    height: int = 260,
    hlines: Optional[List[float]] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
) -> go.Figure:
    x = df.index
    fig = go.Figure([go.Scatter(x=x, y=df[col], mode="lines")])
    if hlines:
        for v in hlines:
            fig.add_hline(y=float(v), line_width=1)
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=45, b=10), hovermode="x unified")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True)
    if ymin is not None and ymax is not None:
        fig.update_yaxes(range=[float(ymin), float(ymax)])
    return fig


# =============================================================================
# Signal helpers (simple)
# =============================================================================
@dataclass(frozen=True)
class SignalScore:
    label: str
    score: int
    summary: str
    reasons: List[str]


def compute_signal_score(df_ind: pd.DataFrame, rsi_min: float, rsi_max: float, rvol_min: float, vol_max: float) -> SignalScore:
    need = ["close", "ma50", "ma200", "rsi14", "rvol", "vol_ann", "atr14"]
    missing = [c for c in need if c not in df_ind.columns]
    if missing:
        return SignalScore("WAIT", 0, "Indicators not ready.", [f"Missing: {', '.join(missing[:6])}"])

    last = df_ind.iloc[-1]
    close = _safe_float(last.get("close"))
    ma50 = _safe_float(last.get("ma50"))
    ma200 = _safe_float(last.get("ma200"))
    rsi = _safe_float(last.get("rsi14"))
    rvol = _safe_float(last.get("rvol"))
    vol_ann = _safe_float(last.get("vol_ann"))
    atr = _safe_float(last.get("atr14"))

    vals = np.array([close, ma50, ma200, rsi, rvol, vol_ann, atr], dtype="float64")
    if not np.isfinite(vals).all():
        return SignalScore("WAIT", 0, "Waiting for enough history.", ["Non-finite indicator values"])

    reasons: List[str] = []
    score = 50

    uptrend = (close > ma50 > ma200)
    downtrend = (close < ma50 < ma200)
    if uptrend:
        score += 22
        reasons.append("Uptrend (close > MA50 > MA200)")
    elif downtrend:
        score -= 22
        reasons.append("Downtrend (close < MA50 < MA200)")
    else:
        reasons.append("Trend mixed")

    if rsi < rsi_min:
        score -= 10
        reasons.append(f"RSI low ({rsi:.1f})")
    elif rsi > rsi_max:
        score -= 10
        reasons.append(f"RSI high ({rsi:.1f})")
    else:
        score += 6
        reasons.append(f"RSI ok ({rsi:.1f})")

    if rvol < rvol_min:
        score -= 10
        reasons.append(f"RVOL low ({rvol:.2f})")
    else:
        score += 8
        reasons.append(f"RVOL ok ({rvol:.2f})")

    if vol_ann > vol_max:
        score -= 10
        reasons.append(f"Vol high ({vol_ann:.2f})")
    else:
        score += 6
        reasons.append(f"Vol ok ({vol_ann:.2f})")

    score = int(np.clip(score, 0, 100))
    if score >= 70 and uptrend:
        return SignalScore("BUY", score, "Favorable trend + filters supportive.", reasons)
    if score <= 30 and downtrend:
        return SignalScore("SELL", score, "Bearish conditions dominate.", reasons)
    return SignalScore("HOLD", score, "Mixed/neutral conditions.", reasons)


def _bt_params_signature(d: Dict[str, Any]) -> str:
    return "|".join([f"{k}={v}" for k, v in sorted((k, str(v)) for k, v in d.items())])


# =============================================================================
# Data loading (Alpaca preferred; Yahoo fallback)
# =============================================================================
@st.cache_data(ttl=15 * 60, show_spinner=False)
def _cached_load_alpaca(symbol: str, force_refresh: int) -> pd.DataFrame:
    api_key = str(st.secrets.get("ALPACA_KEY", "")).strip()
    sec_key = str(st.secrets.get("ALPACA_SECRET", "")).strip()
    if not api_key or not sec_key:
        raise RuntimeError("Missing Alpaca keys in Streamlit secrets.")
    df, _dbg = load_historical(symbol, api_key, sec_key, force_refresh=force_refresh)
    return df


@st.cache_data(ttl=30 * 60, show_spinner=False)
def _cached_load_yahoo(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance not installed.")
    t = yf.Ticker(symbol)
    hist = t.history(period=period, interval=interval, auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError("Yahoo returned no data.")
    hist = hist.rename(columns={c: c.lower() for c in hist.columns})
    hist = hist.reset_index().rename(columns={"Date": "timestamp", "Datetime": "timestamp"})
    if "timestamp" not in hist.columns:
        hist["timestamp"] = pd.to_datetime(hist.index)
    return hist


def _load_and_prepare(symbol: str, *, force_refresh: int) -> None:
    st.session_state["load_error"] = None
    st.session_state["ind_error"] = None

    df: Optional[pd.DataFrame] = None
    err_primary: Optional[str] = None

    if _has_keys_in_secrets():
        try:
            df = _cached_load_alpaca(symbol, force_refresh=force_refresh)
        except Exception as e:
            err_primary = f"{type(e).__name__}: {e}"
            df = None

    if df is None or getattr(df, "empty", True):
        try:
            df = _cached_load_yahoo(symbol, period="5y", interval="1d")
        except Exception as e:
            err_y = f"{type(e).__name__}: {e}"
            st.session_state["load_error"] = f"{err_primary} | {err_y}" if err_primary else err_y
            st.session_state["df_raw"] = None
            st.session_state["df_chart"] = None
            st.session_state["last_symbol"] = None
            return

    df = _coerce_ohlcv_numeric(_ensure_ohlcv_cols(df))
    st.session_state["df_raw"] = df
    st.session_state["last_symbol"] = symbol
    st.session_state["last_loaded_at"] = pd.Timestamp.utcnow()

    try:
        st.session_state["sanity"] = sanity_check_bars(df)
    except Exception:
        st.session_state["sanity"] = None

    try:
        df_chart = _as_utc_dtindex(df)
        add_indicators_inplace(df_chart)
        st.session_state["df_chart"] = df_chart
    except Exception as e:
        st.session_state["df_chart"] = _as_utc_dtindex(df)
        st.session_state["ind_error"] = f"{type(e).__name__}: {e}"


# =============================================================================
# Live helpers
# =============================================================================
def _live_running(stream_obj) -> bool:
    try:
        return bool(stream_obj is not None and getattr(stream_obj, "is_running")())
    except Exception:
        return False


def _stop_live_stream() -> None:
    stream = st.session_state.get("live_stream")
    try:
        if stream is not None:
            stream.stop()
    except Exception:
        pass
    st.session_state["live_stream"] = None


def _msg_to_dict(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    for attr in ("model_dump", "dict"):
        m = getattr(x, attr, None)
        if callable(m):
            try:
                return m()
            except Exception:
                pass
    d: Dict[str, Any] = {}
    for k in ("symbol", "timestamp", "bid_price", "ask_price", "bid_size", "ask_size"):
        if hasattr(x, k):
            d[k] = getattr(x, k)
    if not d:
        d["message"] = str(x)
    return d


def _live_dicts_to_df(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rename_map = {"bp": "bid_price", "ap": "ask_price", "bs": "bid_size", "as": "ask_size", "t": "timestamp", "S": "symbol"}
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    if "bid_price" in df.columns and "ask_price" in df.columns:
        bid = pd.to_numeric(df["bid_price"], errors="coerce")
        ask = pd.to_numeric(df["ask_price"], errors="coerce")
        mid = (bid + ask) / 2.0
        spread = ask - bid
        df["mid"] = mid
        df["spread_bps"] = np.where(mid > 0, (spread / mid) * 10000.0, np.nan)

    return df


def _fragment_decorator():
    frag = getattr(st, "fragment", None)
    if callable(frag):
        return frag
    def identity(fn):
        return fn
    return identity


@_fragment_decorator()
def _live_panel(symbol: str) -> None:
    stream = st.session_state.get("live_stream")
    live_running = _live_running(stream)

    if stream is not None:
        try:
            new_msgs = stream.get_latest(max_items=250)
        except Exception:
            new_msgs = []
        if new_msgs:
            st.session_state["live_rows"].extend([_msg_to_dict(x) for x in new_msgs])
            st.session_state["live_rows"] = st.session_state["live_rows"][-800:]

    rows: List[dict] = st.session_state.get("live_rows", [])
    st.caption("Status: ✅ running" if live_running else "Status: ⏸ stopped")

    if not rows:
        st.info("No quote updates received yet.")
        return

    df_live = _live_dicts_to_df(rows)
    if df_live.empty:
        st.warning("Received live messages, but couldn't parse them into a table.")
        return

    latest = df_live.tail(1)
    if not latest.empty and ("bid_price" in latest.columns) and ("ask_price" in latest.columns):
        lbid = _safe_float(pd.to_numeric(latest["bid_price"], errors="coerce").iloc[0])
        lask = _safe_float(pd.to_numeric(latest["ask_price"], errors="coerce").iloc[0])
        lmid = (lbid + lask) / 2.0 if np.isfinite([lbid, lask]).all() else np.nan
        lsp_bps = ((lask - lbid) / lmid) * 10000.0 if np.isfinite([lbid, lask, lmid]).all() and lmid > 0 else np.nan

        m1, m2, m3 = st.columns(3)
        m1.metric("Bid", f"{lbid:.4f}" if np.isfinite(lbid) else "—")
        m2.metric("Ask", f"{lask:.4f}" if np.isfinite(lask) else "—")
        m3.metric("Spread (bps)", f"{lsp_bps:.1f}" if np.isfinite(lsp_bps) else "—")

    show_cols = [c for c in ["timestamp", "symbol", "bid_price", "ask_price", "mid", "spread_bps", "bid_size", "ask_size", "message"] if c in df_live.columns]
    view = df_live.sort_values("timestamp").tail(160) if "timestamp" in df_live.columns else df_live.tail(160)
    st.dataframe(view[show_cols] if show_cols else view, use_container_width=True, height=520)


# =============================================================================
# Session init
# =============================================================================
for k, v in {
    "df_raw": None,
    "df_chart": None,
    "sanity": None,
    "load_error": None,
    "ind_error": None,
    "last_symbol": None,
    "last_loaded_at": None,
    "bt_results": None,   # dict from backtester
    "bt_trades": None,    # trades_df
    "bt_error": None,
    "bt_params_sig": None,
    "live_stream": None,
    "live_rows": [],
    "live_autorefresh": True,
    "live_last_symbol": None,
    "cfg_export_json": "",
    "cfg_import_json": "",
}.items():
    _ss_setdefault(k, v)


# =============================================================================
# Header
# =============================================================================
st.title("📈 Pro Algo Trader")
st.caption("Signals • Charts • Backtests • Live quotes")


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("Controls")

    symbol = st.text_input("Ticker", value=str(ss_get("symbol", "AAPL"))).upper().strip()

    mode_label = st.selectbox(
        "Strategy Mode",
        options=["Breakout", "Pullback"],
        index=0 if str(ss_get("mode_label", "Breakout")) == "Breakout" else 1,
    )
    mode = "breakout" if mode_label == "Breakout" else "pullback"

    with st.expander("Strategy Settings", expanded=False):
        atr_entry = st.number_input("ATR entry", 0.0, 10.0, float(ss_get("atr_entry", 1.0)), 0.1)
        atr_stop = st.number_input("ATR stop", 0.1, 20.0, float(ss_get("atr_stop", 2.0)), 0.1)
        atr_target = st.number_input("ATR target", 0.1, 50.0, float(ss_get("atr_target", 3.0)), 0.1)

        rsi_min = st.number_input("RSI min", 0.0, 100.0, float(ss_get("rsi_min", 30.0)))
        rsi_max = st.number_input("RSI max", 0.0, 100.0, float(ss_get("rsi_max", 70.0)))
        rvol_min = st.number_input("RVOL min", 0.0, 10.0, float(ss_get("rvol_min", 1.2)))
        vol_max = st.number_input("Max annual vol", 0.0, 5.0, float(ss_get("vol_max", 1.0)))

    with st.expander("Risk & Sizing", expanded=False):
        enable_position_sizing = st.toggle("Enable Position Sizing", value=bool(ss_get("enable_position_sizing", False)))

        account_capital = st.number_input(
            "Account Capital",
            min_value=100.0,
            max_value=100_000_000.0,
            value=float(ss_get("account_capital", 10_000.0)),
            step=100.0,
            format="%.2f",
        )

        if enable_position_sizing:
            risk_per_trade_pct = st.number_input(
                "Risk per Trade (%)",
                min_value=0.1,
                max_value=10.0,
                value=float(ss_get("risk_per_trade_pct", 1.0)),
                step=0.1,
                format="%.1f",
            )
            max_alloc_pct = st.number_input(
                "Max Allocation (%)",
                min_value=1.0,
                max_value=100.0,
                value=float(ss_get("max_alloc_pct", 10.0)),
                step=1.0,
                format="%.0f",
            )
        else:
            risk_per_trade_pct = float(ss_get("risk_per_trade_pct", 1.0))
            max_alloc_pct = float(ss_get("max_alloc_pct", 10.0))

        mark_to_market = st.toggle("Mark-to-market equity (smoother curve)", value=bool(ss_get("mark_to_market", False)))

    with st.expander("Save / Load Configuration", expanded=False):
        if st.button("Export Settings", use_container_width=True):
            cfg = {
                "symbol": symbol,
                "mode_label": mode_label,
                "atr_entry": float(atr_entry),
                "atr_stop": float(atr_stop),
                "atr_target": float(atr_target),
                "rsi_min": float(rsi_min),
                "rsi_max": float(rsi_max),
                "rvol_min": float(rvol_min),
                "vol_max": float(vol_max),
                "enable_position_sizing": bool(enable_position_sizing),
                "account_capital": float(account_capital),
                "risk_per_trade_pct": float(risk_per_trade_pct),
                "max_alloc_pct": float(max_alloc_pct),
                "mark_to_market": bool(mark_to_market),
            }
            st.session_state["cfg_export_json"] = json.dumps(cfg, indent=2)

        if st.session_state.get("cfg_export_json"):
            st.code(st.session_state["cfg_export_json"], language="json")

        st.session_state["cfg_import_json"] = st.text_area(
            "Paste JSON to load",
            value=str(ss_get("cfg_import_json", "")),
            height=140,
        )
        if st.button("Import Settings", use_container_width=True):
            try:
                parsed = json.loads(st.session_state["cfg_import_json"] or "{}")
                if not isinstance(parsed, dict):
                    raise ValueError("JSON must be an object.")
                # whitelist keys
                allowed = {
                    "symbol","mode_label",
                    "atr_entry","atr_stop","atr_target",
                    "rsi_min","rsi_max","rvol_min","vol_max",
                    "enable_position_sizing","account_capital","risk_per_trade_pct","max_alloc_pct",
                    "mark_to_market",
                }
                for k, v in parsed.items():
                    if k in allowed:
                        st.session_state[k] = v
                st.rerun()
            except Exception as e:
                st.error("Invalid JSON.")
                st.caption(f"{type(e).__name__}: {e}")

    st.divider()
    load_btn = st.button("🔄 Load / Refresh", use_container_width=True)

# Persist sidebar inputs
for k, v in {
    "symbol": symbol,
    "mode_label": mode_label,
    "atr_entry": atr_entry,
    "atr_stop": atr_stop,
    "atr_target": atr_target,
    "rsi_min": rsi_min,
    "rsi_max": rsi_max,
    "rvol_min": rvol_min,
    "vol_max": vol_max,
    "enable_position_sizing": enable_position_sizing,
    "account_capital": account_capital,
    "risk_per_trade_pct": risk_per_trade_pct,
    "max_alloc_pct": max_alloc_pct,
    "mark_to_market": mark_to_market,
}.items():
    st.session_state[k] = v


# =============================================================================
# Auto-load logic
# =============================================================================
needs_load = (load_btn or (st.session_state.get("df_raw") is None) or (st.session_state.get("last_symbol") != symbol))

if st.session_state.get("live_stream") is not None and st.session_state.get("live_last_symbol") != symbol:
    _stop_live_stream()

force_refresh = int(time.time()) if load_btn else 0
if needs_load:
    with st.spinner(f"Loading {symbol}…"):
        _load_and_prepare(symbol, force_refresh=force_refresh)

df_raw = st.session_state.get("df_raw")
df_chart = st.session_state.get("df_chart")

if df_raw is None or getattr(df_raw, "empty", True):
    st.error(f"Could not load data for {symbol}.")
    if st.session_state.get("load_error"):
        st.caption(st.session_state["load_error"])
    st.stop()

if df_chart is None or getattr(df_chart, "empty", True):
    st.error("Data loaded but indicator dataframe is empty.")
    if st.session_state.get("ind_error"):
        st.caption(st.session_state["ind_error"])
    st.stop()

df_plot = _tail_for_plot(df_chart, 700)


# =============================================================================
# Tabs
# =============================================================================
tab_signal, tab_charts, tab_backtest, tab_live = st.tabs(["✅ Signal", "📊 Charts", "🧪 Backtest", "📡 Live"])


# =============================================================================
# Signal tab
# =============================================================================
with tab_signal:
    st.subheader(f"{symbol} — Signal")

    san = st.session_state.get("sanity")
    if isinstance(san, dict) and san.get("warnings"):
        with st.expander("Data notes", expanded=False):
            for w in san["warnings"][:5]:
                st.write(f"• {w}")

    score = compute_signal_score(df_chart, float(rsi_min), float(rsi_max), float(rvol_min), float(vol_max))

    hero = st.container(border=True)
    with hero:
        a, b = st.columns([1.2, 1], vertical_alignment="center")
        with a:
            if score.label == "BUY":
                st.success(f"**BUY** • Score {score.score}/100")
            elif score.label == "SELL":
                st.error(f"**SELL** • Score {score.score}/100")
            elif score.label == "WAIT":
                st.warning(f"**WAIT** • Score {score.score}/100")
            else:
                st.info(f"**HOLD** • Score {score.score}/100")
            st.caption(score.summary)
        with b:
            last = df_chart.iloc[-1]
            close = _safe_float(last.get("close", np.nan))
            atr14 = _safe_float(last.get("atr14", np.nan))
            st.metric("Close", f"{close:.2f}" if np.isfinite(close) else "—")
            st.metric("ATR(14)", f"{atr14:.3f}" if np.isfinite(atr14) else "—")

    with st.expander("Why this score?", expanded=False):
        for r in score.reasons[:10]:
            st.write(f"• {r}")


# =============================================================================
# Charts tab (trade markers included if backtest run)
# =============================================================================
with tab_charts:
    st.subheader(f"{symbol} — Charts")
    trades_for_markers = st.session_state.get("bt_trades")
    st.plotly_chart(plot_price(df_plot, symbol, trades=trades_for_markers), use_container_width=True)

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        if "rsi14" in df_plot.columns:
            st.plotly_chart(plot_indicator(df_plot, "rsi14", "RSI(14)", hlines=[30, 70], ymin=0, ymax=100), use_container_width=True)
        else:
            st.info("RSI not available.")
    with c2:
        if "rvol" in df_plot.columns:
            st.plotly_chart(plot_indicator(df_plot, "rvol", "RVOL", hlines=[1.0]), use_container_width=True)
        else:
            st.info("RVOL not available.")
    with c3:
        if "vol_ann" in df_plot.columns:
            st.plotly_chart(plot_indicator(df_plot, "vol_ann", "Annualized Volatility"), use_container_width=True)
        else:
            st.info("Volatility not available.")


# =============================================================================
# Backtest tab (Performance section + empty safe handling)
# =============================================================================
with tab_backtest:
    st.subheader("Backtest")
    run_backtest_btn = st.button("🧪 Run Backtest", use_container_width=True)

    # Fixed: keep simple
    horizon_bars = 20

    bt_params = dict(
        symbol=symbol,
        mode=mode,
        horizon=horizon_bars,
        atr_entry=float(atr_entry),
        atr_stop=float(atr_stop),
        atr_target=float(atr_target),
        require_risk_on=False,
        rsi_min=float(rsi_min),
        rsi_max=float(rsi_max),
        rvol_min=float(rvol_min),
        vol_max=float(vol_max),
        cooldown_bars=0,
        include_spread_penalty=True,
        assumed_spread_bps=5.0,
        start_equity=float(account_capital),
        enable_position_sizing=bool(enable_position_sizing),
        risk_pct=float(risk_per_trade_pct) / 100.0,
        max_alloc_pct=float(max_alloc_pct) / 100.0,
        mark_to_market=bool(mark_to_market),
        # You can expose these later if you want:
        slippage_bps=0.0,
        commission_per_order=0.0,
        spread_mode="taker_only",
        exit_priority="stop_first",
    )
    sig = _bt_params_signature(bt_params)

    if run_backtest_btn:
        st.session_state["bt_error"] = None
        with st.spinner("Running backtest…"):
            try:
                results, trades_df = backtest_strategy(
                    df=df_chart.reset_index().rename(columns={"index": "timestamp"}) if isinstance(df_chart.index, pd.DatetimeIndex) else df_chart,
                    market_df=None,
                    **bt_params,
                )
                st.session_state["bt_results"] = results
                st.session_state["bt_trades"] = trades_df
                st.session_state["bt_params_sig"] = sig
            except Exception as e:
                st.session_state["bt_results"] = None
                st.session_state["bt_trades"] = None
                st.session_state["bt_params_sig"] = None
                st.session_state["bt_error"] = f"{type(e).__name__}: {e}"

    if st.session_state.get("bt_error"):
        st.error("Backtest failed.")
        st.caption(st.session_state["bt_error"])

    results = st.session_state.get("bt_results")
    trades = st.session_state.get("bt_trades")

    if not isinstance(results, dict):
        st.info("No results yet. Click **🧪 Run Backtest** above.")
    else:
        if st.session_state.get("bt_params_sig") != sig:
            st.warning("Showing results from earlier parameters. Run again to update.")

        df_bt = results.get("df_bt", pd.DataFrame())
        ntr = int(results.get("trades", 0) or 0)

        st.markdown("### Performance")
        pwrap = st.container(border=True)
        with pwrap:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Trades", f"{ntr}")
            c2.metric("Win rate", f"{results.get('win_rate', float('nan')):.1%}" if np.isfinite(results.get("win_rate", np.nan)) else "—")
            c3.metric("Total return", f"{results.get('total_return', float('nan')):.1%}" if np.isfinite(results.get("total_return", np.nan)) else "—")
            c4.metric("Max drawdown", f"{results.get('max_drawdown', float('nan')):.1%}" if np.isfinite(results.get("max_drawdown", np.nan)) else "—")
            c5.metric("Sharpe", f"{results.get('sharpe', float('nan')):.2f}" if np.isfinite(results.get("sharpe", np.nan)) else "—")

        # Equity curve + drawdown
        if isinstance(df_bt, pd.DataFrame) and (not df_bt.empty) and ("equity" in df_bt.columns):
            eq = pd.to_numeric(df_bt["equity"], errors="coerce").dropna()
            if len(eq) > 2:
                ts = pd.to_datetime(df_bt.get("timestamp", pd.Series(index=df_bt.index, dtype="datetime64[ns]")), errors="coerce", utc=True)
                if ts.isna().all():
                    x = np.arange(len(eq))
                else:
                    x = ts.iloc[-len(eq):]

                peak = eq.cummax()
                dd = (eq / peak) - 1.0

                st.plotly_chart(
                    go.Figure([go.Scatter(x=x, y=eq.values, mode="lines")]).update_layout(
                        title="Equity Curve",
                        height=320,
                        margin=dict(l=10, r=10, t=45, b=10),
                        hovermode="x unified",
                        showlegend=False,
                    ),
                    use_container_width=True,
                )

                st.plotly_chart(
                    go.Figure([go.Scatter(x=x, y=dd.values, mode="lines")]).update_layout(
                        title="Drawdown",
                        height=260,
                        margin=dict(l=10, r=10, t=45, b=10),
                        hovermode="x unified",
                        showlegend=False,
                    ),
                    use_container_width=True,
                )

        # Trades table (handle empty safely)
        if not isinstance(trades, pd.DataFrame) or trades.empty:
            st.warning("Backtest completed but produced no trades.")
            notes = results.get("notes_for_beginners", [])
            if isinstance(notes, list) and notes:
                with st.expander("Notes", expanded=False):
                    for n in notes[:8]:
                        st.write(f"• {n}")
        else:
            st.download_button(
                "⬇️ Download trades (CSV)",
                data=trades.to_csv(index=False).encode("utf-8"),
                file_name=f"{symbol}_trades.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.dataframe(trades, use_container_width=True, height=560)

        with st.expander("Assumptions", expanded=False):
            st.json(results.get("assumptions", {}))


# =============================================================================
# Live tab (fragment + fallback refresh)
# =============================================================================
with tab_live:
    st.subheader("Live Quotes")

    if not LIVE_AVAILABLE:
        st.info("Live module not available.")
    elif not _has_keys_in_secrets():
        st.info("Live is disabled (missing Alpaca keys in Streamlit secrets).")
    else:
        stream = st.session_state.get("live_stream")
        live_running = _live_running(stream)

        a, b, c, d = st.columns([1, 1, 1, 2], vertical_alignment="center")
        start_clicked = a.button("▶ Start", use_container_width=True, disabled=live_running)
        stop_clicked = b.button("⏹ Stop", use_container_width=True, disabled=not live_running)
        clear_clicked = c.button("🧹 Clear", use_container_width=True)
        st.session_state["live_autorefresh"] = d.toggle("Auto refresh", value=bool(st.session_state.get("live_autorefresh", True)))

        if clear_clicked:
            st.session_state["live_rows"] = []

        if start_clicked and not live_running:
            try:
                api_key = str(st.secrets.get("ALPACA_KEY", "")).strip()
                sec_key = str(st.secrets.get("ALPACA_SECRET", "")).strip()
                stream = RealtimeStream(api_key, sec_key, symbol)
                stream.start()
                st.session_state["live_stream"] = stream
                st.session_state["live_last_symbol"] = symbol
            except Exception as e:
                st.session_state["live_stream"] = None
                st.error("Failed to start live stream.")
                st.caption(f"{type(e).__name__}: {e}")

        if stop_clicked and live_running:
            _stop_live_stream()

        _live_panel(symbol)

        # Fallback periodic refresh (helps even when fragment alone doesn't re-run)
        if _live_running(st.session_state.get("live_stream")) and bool(st.session_state.get("live_autorefresh", True)):
            if st_autorefresh is not None:
                st_autorefresh(interval=2000, key=f"live_refresh_{symbol}")
