# app.py
from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.backtester import backtest_strategy
from utils.data_loader import load_historical, sanity_check_bars
from utils.indicators import add_indicators_inplace

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
# Streamlit config
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


def _keep_cfg_keys(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Whitelist config keys we allow importing/exporting."""
    allowed = {
        "symbol",
        "strategy_type",
        "account_capital",
        "risk_per_trade_pct",
        "atr_entry",
        "atr_stop",
        "atr_target",
        "rsi_min",
        "rsi_max",
        "rvol_min",
        "vol_max",
    }
    return {k: cfg[k] for k in allowed if k in cfg}


# =============================================================================
# Trade marker extraction
# =============================================================================
def _trades_to_markers(trades: pd.DataFrame, df_index: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (entries_df, exits_df) with columns ['ts','px'] for plotting markers.
    Robust to different trade dataframe column names.
    """
    if trades is None or getattr(trades, "empty", True):
        return pd.DataFrame(columns=["ts", "px"]), pd.DataFrame(columns=["ts", "px"])

    t = trades.copy()

    # Candidate columns (time)
    entry_time_cols = ["entry_time", "entry_ts", "entry_timestamp", "entry_date", "open_time", "buy_time"]
    exit_time_cols = ["exit_time", "exit_ts", "exit_timestamp", "exit_date", "close_time", "sell_time"]

    # Candidate columns (price)
    entry_price_cols = ["entry_price", "entry_px", "buy_price", "fill_price", "open_price"]
    exit_price_cols = ["exit_price", "exit_px", "sell_price", "close_price"]

    # Candidate columns (bar index)
    entry_idx_cols = ["entry_idx", "entry_i", "buy_idx", "i_entry"]
    exit_idx_cols = ["exit_idx", "exit_i", "sell_idx", "i_exit"]

    def pick_col(cols: List[str]) -> Optional[str]:
        for c in cols:
            if c in t.columns:
                return c
        return None

    et = pick_col(entry_time_cols)
    xt = pick_col(exit_time_cols)
    ep = pick_col(entry_price_cols)
    xp = pick_col(exit_price_cols)
    ei = pick_col(entry_idx_cols)
    xi = pick_col(exit_idx_cols)

    def to_ts(col_time: Optional[str], col_idx: Optional[str]) -> pd.Series:
        if col_time:
            ts = pd.to_datetime(t[col_time], errors="coerce", utc=True)
            return ts
        if col_idx:
            idx = pd.to_numeric(t[col_idx], errors="coerce")
            out = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns, UTC]")
            for j, v in idx.items():
                if pd.notna(v):
                    k = int(v)
                    if 0 <= k < len(df_index):
                        out.loc[j] = df_index[k]
            return out
        return pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns, UTC]")

    entry_ts = to_ts(et, ei)
    exit_ts = to_ts(xt, xi)

    def to_px(col_price: Optional[str], fallback_col: str) -> pd.Series:
        if col_price:
            return pd.to_numeric(t[col_price], errors="coerce")
        if fallback_col in t.columns:
            return pd.to_numeric(t[fallback_col], errors="coerce")
        return pd.Series(np.nan, index=t.index, dtype="float64")

    entry_px = to_px(ep, "entry")
    exit_px = to_px(xp, "exit")

    entries = pd.DataFrame({"ts": entry_ts, "px": entry_px}).dropna(subset=["ts", "px"])
    exits = pd.DataFrame({"ts": exit_ts, "px": exit_px}).dropna(subset=["ts", "px"])

    return entries, exits


# =============================================================================
# Plotting (with optional trade markers)
# =============================================================================
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

    # ---- Trade markers
    if isinstance(trades, pd.DataFrame) and (not trades.empty):
        entries, exits = _trades_to_markers(trades, df.index)
        if not entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=entries["ts"],
                    y=entries["px"],
                    mode="markers",
                    name="Buy",
                    marker=dict(symbol="triangle-up", size=10),
                )
            )
        if not exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=exits["ts"],
                    y=exits["px"],
                    mode="markers",
                    name="Sell",
                    marker=dict(symbol="triangle-down", size=10),
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
# Strategy / signal helpers
# =============================================================================
def compute_lookback_low_high(df_ind: pd.DataFrame, lookback: int) -> Tuple[float, float]:
    lb = int(max(10, lookback))
    tail = df_ind.tail(lb)
    if tail.empty or ("low" not in tail.columns) or ("high" not in tail.columns):
        return np.nan, np.nan
    lo = float(np.nanmin(pd.to_numeric(tail["low"], errors="coerce").values))
    hi = float(np.nanmax(pd.to_numeric(tail["high"], errors="coerce").values))
    return lo, hi


def detect_big_jump(df: pd.DataFrame, thresh: float = 0.18) -> Optional[Dict[str, Any]]:
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return None
    c = pd.to_numeric(df["close"], errors="coerce")
    r = c.pct_change().abs().dropna()
    if r.empty:
        return None
    ts = r.idxmax()
    mv = _safe_float(r.loc[ts]) if ts in r.index else np.nan
    if np.isfinite(mv) and mv >= float(thresh):
        return {"ts": ts, "abs_move": mv}
    return None


def compute_trade_plan(
    df_ind: pd.DataFrame,
    strategy_type: str,
    atr_entry: float,
    atr_stop: float,
    atr_target: float,
    *,
    assumed_spread_bps: float,
    include_spread_penalty: bool,
) -> Dict[str, Any]:
    """
    Strategy-aware plan:
      - Trend Breakout: entry above close
      - Mean Reversion: entry below close (pullback)
    """
    last = df_ind.iloc[-1]
    close = _safe_float(last.get("close", np.nan))
    atr = _safe_float(last.get("atr14", np.nan))

    if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
        return {"entry": np.nan, "stop": np.nan, "target": np.nan, "rr": np.nan, "atr": atr, "type": "—"}

    stype = (strategy_type or "").lower().strip()
    is_mr = "mean" in stype

    if is_mr:
        entry = close - float(atr_entry) * atr
        plan_type = "Mean Reversion (limit buy)"
    else:
        entry = close + float(atr_entry) * atr
        plan_type = "Trend Breakout (stop trigger)"

    if include_spread_penalty and assumed_spread_bps > 0 and np.isfinite(entry):
        entry *= (1.0 + assumed_spread_bps / 10000.0)

    stop = entry - float(atr_stop) * atr
    target = entry + float(atr_target) * atr
    risk = entry - stop
    rr = ((target - entry) / risk) if risk > 0 else np.nan
    return {"entry": entry, "stop": stop, "target": target, "rr": rr, "atr": atr, "type": plan_type}


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


def _bt_params_signature(**kwargs) -> str:
    return "|".join([f"{k}={v}" for k, v in sorted((k, str(v)) for k, v in kwargs.items())])


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


# =============================================================================
# Cached loaders
# =============================================================================
@st.cache_data(ttl=15 * 60, show_spinner=False)
def _cached_load_alpaca(symbol: str, force_refresh: int) -> pd.DataFrame:
    api_key = str(st.secrets.get("ALPACA_KEY", "")).strip()
    sec_key = str(st.secrets.get("ALPACA_SECRET", "")).strip()
    if not api_key or not sec_key:
        raise RuntimeError("Missing Alpaca keys in Streamlit secrets.")
    df, _ = load_historical(symbol, api_key, sec_key, force_refresh=force_refresh)
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
# Backtest call helper (filters kwargs to match your backtester signature)
# =============================================================================
def _call_backtest(**kwargs):
    sig = inspect.signature(backtest_strategy)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return backtest_strategy(**allowed)


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
    "bt_results": None,
    "bt_trades": None,
    "bt_error": None,
    "bt_params_sig": None,
    "live_stream": None,
    "live_rows": [],
    "live_autorefresh": True,
    "live_last_symbol": None,
}.items():
    _ss_setdefault(k, v)


# =============================================================================
# Header
# =============================================================================
st.title("📈 Pro Algo Trader")
st.caption("Signals • Charts • Backtests • Optional live quotes")


# =============================================================================
# Sidebar
#   - Strategy selection
#   - Risk inputs: Account Capital + Risk per Trade (%)
#   - Save/Load configuration JSON
# =============================================================================
with st.sidebar:
    st.header("Controls")

    symbol = st.text_input("Ticker", value=ss_get("symbol", "AAPL")).upper().strip()

    strategy_type = st.selectbox(
        "Strategy Type",
        options=["Trend Breakout", "Mean Reversion"],
        index=0 if ss_get("strategy_type", "Trend Breakout") == "Trend Breakout" else 1,
    )

    account_capital = st.number_input(
        "Account Capital",
        min_value=100.0,
        max_value=100_000_000.0,
        value=float(ss_get("account_capital", 10_000.0)),
        step=100.0,
        format="%.2f",
    )
    risk_per_trade_pct = st.number_input(
        "Risk per Trade (%)",
        min_value=0.1,
        max_value=10.0,
        value=float(ss_get("risk_per_trade_pct", 1.0)),
        step=0.1,
        format="%.1f",
        help="Percent of capital you’re willing to lose if stop is hit (for position sizing, if supported).",
    )

    with st.expander("Strategy Settings", expanded=False):
        atr_entry = st.number_input("ATR entry", 0.0, 10.0, float(ss_get("atr_entry", 1.0)), 0.1)
        atr_stop = st.number_input("ATR stop", 0.1, 20.0, float(ss_get("atr_stop", 2.0)), 0.1)
        atr_target = st.number_input("ATR target", 0.1, 50.0, float(ss_get("atr_target", 3.0)), 0.1)

        rsi_min = st.number_input("RSI min", 0.0, 100.0, float(ss_get("rsi_min", 30.0)))
        rsi_max = st.number_input("RSI max", 0.0, 100.0, float(ss_get("rsi_max", 70.0)))
        rvol_min = st.number_input("RVOL min", 0.0, 10.0, float(ss_get("rvol_min", 1.2)))
        vol_max = st.number_input("Max annual vol", 0.0, 5.0, float(ss_get("vol_max", 1.0)))

    # Fixed defaults (keep UI simple)
    include_spread_penalty = True
    assumed_spread_bps = 5.0
    sr_lookback = 50
    chart_window = 700

    with st.expander("Save / Load Configuration", expanded=False):
        if st.button("Export Settings", use_container_width=True):
            cfg = _keep_cfg_keys(
                {
                    "symbol": symbol,
                    "strategy_type": strategy_type,
                    "account_capital": float(account_capital),
                    "risk_per_trade_pct": float(risk_per_trade_pct),
                    "atr_entry": float(atr_entry),
                    "atr_stop": float(atr_stop),
                    "atr_target": float(atr_target),
                    "rsi_min": float(rsi_min),
                    "rsi_max": float(rsi_max),
                    "rvol_min": float(rvol_min),
                    "vol_max": float(vol_max),
                }
            )
            st.session_state["cfg_export_json"] = json.dumps(cfg, indent=2)

        export_val = ss_get("cfg_export_json", "")
        if export_val:
            st.code(export_val, language="json")

        import_val = st.text_area("Import JSON", value=ss_get("cfg_import_json", ""), height=140, key="cfg_import_json")
        if st.button("Import Settings", use_container_width=True):
            try:
                parsed = json.loads(import_val or "{}")
                parsed = _keep_cfg_keys(parsed if isinstance(parsed, dict) else {})
                if not parsed:
                    st.warning("No valid settings found in JSON.")
                else:
                    for k, v in parsed.items():
                        st.session_state[k] = v
                    st.rerun()
            except Exception as e:
                st.error("Invalid JSON.")
                st.caption(f"{type(e).__name__}: {e}")

    st.divider()
    load_btn = st.button("🔄 Load / Refresh", use_container_width=True)

# Persist inputs
for k, v in {
    "symbol": symbol,
    "strategy_type": strategy_type,
    "account_capital": float(account_capital),
    "risk_per_trade_pct": float(risk_per_trade_pct),
    "atr_entry": float(atr_entry),
    "atr_stop": float(atr_stop),
    "atr_target": float(atr_target),
    "rsi_min": float(rsi_min),
    "rsi_max": float(rsi_max),
    "rvol_min": float(rvol_min),
    "vol_max": float(vol_max),
}.items():
    st.session_state[k] = v


# =============================================================================
# Load logic
# =============================================================================
def _needs_load(symbol_now: str) -> bool:
    return (st.session_state.get("df_raw") is None) or (st.session_state.get("last_symbol") != symbol_now)


if st.session_state.get("live_stream") is not None and st.session_state.get("live_last_symbol") != symbol:
    _stop_live_stream()

force_refresh = int(time.time()) if load_btn else 0
if load_btn or _needs_load(symbol):
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

df_plot = _tail_for_plot(df_chart, int(chart_window))


# =============================================================================
# Tabs
# =============================================================================
tab_signal, tab_charts, tab_backtest, tab_live = st.tabs(["✅ Signal", "📊 Charts", "🧪 Backtest", "📡 Live"])


# =============================================================================
# Signal tab
# =============================================================================
with tab_signal:
    box = st.container(border=True)
    with box:
        st.subheader(f"{symbol} — Signal")
        ts = st.session_state.get("last_loaded_at")
        if ts is not None:
            st.caption(f"Last updated: {ts}")
        if st.session_state.get("ind_error"):
            st.warning("Some indicators failed (signal may be degraded).")

    jump = detect_big_jump(df_chart, thresh=0.18)
    if jump:
        st.warning("Unusual price jump detected (possible split/corporate action).")
        st.caption(f"Max |close-to-close| move: {jump['abs_move']:.1%} at {jump['ts']}")

    san = st.session_state.get("sanity")
    if isinstance(san, dict) and san.get("warnings"):
        with st.expander("Data notes", expanded=False):
            for w in san["warnings"][:5]:
                st.write(f"• {w}")

    score = compute_signal_score(df_chart, float(rsi_min), float(rsi_max), float(rvol_min), float(vol_max))

    plan = compute_trade_plan(
        df_chart,
        strategy_type=strategy_type,
        atr_entry=float(atr_entry),
        atr_stop=float(atr_stop),
        atr_target=float(atr_target),
        assumed_spread_bps=float(assumed_spread_bps),
        include_spread_penalty=bool(include_spread_penalty),
    )

    lo, hi = compute_lookback_low_high(df_chart, int(sr_lookback))
    last = df_chart.iloc[-1]
    close = _safe_float(last.get("close", np.nan))
    rsi = _safe_float(last.get("rsi14", np.nan))
    rvol = _safe_float(last.get("rvol", np.nan))
    vol_ann = _safe_float(last.get("vol_ann", np.nan))
    atr14 = _safe_float(last.get("atr14", np.nan))

    hero = st.container(border=True)
    with hero:
        a, b, c = st.columns([1.2, 1, 1], vertical_alignment="center")
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
            st.metric("Close", f"{close:.2f}" if np.isfinite(close) else "—")
            st.metric("ATR(14)", f"{atr14:.3f}" if np.isfinite(atr14) else "—")
        with c:
            rr = plan.get("rr", np.nan)
            st.metric("Planned Entry", f"{plan['entry']:.2f}" if np.isfinite(plan.get("entry", np.nan)) else "—")
            st.metric("R:R", f"{rr:.2f}" if np.isfinite(rr) else "—")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Lookback Low", f"{lo:.2f}" if np.isfinite(lo) else "—")
    m2.metric("Lookback High", f"{hi:.2f}" if np.isfinite(hi) else "—")
    m3.metric("RSI", f"{rsi:.1f}" if np.isfinite(rsi) else "—")
    m4.metric("RVOL", f"{rvol:.2f}" if np.isfinite(rvol) else "—")
    m5.metric("Ann. Vol", f"{vol_ann:.2f}" if np.isfinite(vol_ann) else "—")

    st.divider()
    p = st.container(border=True)
    with p:
        st.subheader("Trade Plan (ATR-based)")
        st.caption(plan.get("type", "—"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Entry", f"{plan['entry']:.2f}" if np.isfinite(plan.get("entry", np.nan)) else "—")
        c2.metric("Stop", f"{plan['stop']:.2f}" if np.isfinite(plan.get("stop", np.nan)) else "—")
        c3.metric("Target", f"{plan['target']:.2f}" if np.isfinite(plan.get("target", np.nan)) else "—")

    with st.expander("Why this score?", expanded=False):
        for r in score.reasons[:10]:
            st.write(f"• {r}")


# =============================================================================
# Charts tab (with trade markers)
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

    with st.expander("Data preview", expanded=False):
        st.dataframe(df_plot.tail(160), use_container_width=True, height=460)


# =============================================================================
# Backtest tab
#   - Run button here
#   - Uses Account Capital as start equity
#   - Passes Risk per trade % if your backtester supports it
# =============================================================================
with tab_backtest:
    st.subheader("Backtest")
    run_backtest_btn = st.button("🧪 Run Backtest", use_container_width=True)

    # Keep this simple & stable
    horizon_bars = 20

    # Strategy mode mapping (future-proof)
    mode = "breakout" if strategy_type == "Trend Breakout" else "mean_reversion"

    bt_params = dict(
        symbol=symbol,
        strategy_type=strategy_type,
        mode=mode,
        horizon=horizon_bars,
        atr_entry=float(atr_entry),
        atr_stop=float(atr_stop),
        atr_target=float(atr_target),
        rsi_min=float(rsi_min),
        rsi_max=float(rsi_max),
        rvol_min=float(rvol_min),
        vol_max=float(vol_max),
        include_spread_penalty=bool(include_spread_penalty),
        assumed_spread_bps=float(assumed_spread_bps),
        account_capital=float(account_capital),
        risk_per_trade_pct=float(risk_per_trade_pct),
    )
    sig = _bt_params_signature(**bt_params)

    if run_backtest_btn:
        st.session_state["bt_error"] = None
        with st.spinner("Running backtest…"):
            try:
                df_bt, trades = _call_backtest(
                    df=df_chart,
                    market_df=None,
                    horizon=horizon_bars,
                    mode=mode,  # may be ignored if your backtester only supports breakout
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
                    start_equity=float(account_capital),
                    risk_per_trade_pct=float(risk_per_trade_pct),  # only used if supported by backtester
                )
                st.session_state["bt_results"] = df_bt
                st.session_state["bt_trades"] = trades
                st.session_state["bt_params_sig"] = sig
            except Exception as e:
                st.session_state["bt_results"] = None
                st.session_state["bt_trades"] = None
                st.session_state["bt_params_sig"] = None
                st.session_state["bt_error"] = f"{type(e).__name__}: {e}"

    if st.session_state.get("bt_error"):
        st.error("Backtest failed.")
        st.caption(st.session_state["bt_error"])

    trades = st.session_state.get("bt_trades")
    df_bt = st.session_state.get("bt_results")

    if trades is None or getattr(trades, "empty", True):
        st.info("No results yet. Click **🧪 Run Backtest** above.")
    else:
        if st.session_state.get("bt_params_sig") != sig:
            st.warning("Showing results from earlier parameters. Run again to update.")

        t = trades.copy()

        wrap = st.container(border=True)
        with wrap:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades", f"{len(t)}")

            p = pd.to_numeric(t.get("pnl_per_share", pd.Series(dtype=float)), errors="coerce")
            win_rate = float((p > 0).sum()) / max(1, len(p)) if len(p) else np.nan
            c2.metric("Win rate", f"{win_rate:.1%}" if np.isfinite(win_rate) else "—")

            total_ret = np.nan
            maxdd = np.nan
            if isinstance(df_bt, pd.DataFrame) and (not df_bt.empty) and ("equity" in df_bt.columns):
                eq = pd.to_numeric(df_bt["equity"], errors="coerce").dropna()
                if len(eq) > 2:
                    total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1.0
                    dd = (eq / eq.cummax()) - 1.0
                    maxdd = float(dd.min())

            c3.metric("Total return", f"{total_ret:.1%}" if np.isfinite(total_ret) else "—")
            c4.metric("Max drawdown", f"{maxdd:.1%}" if np.isfinite(maxdd) else "—")

        st.divider()
        st.download_button(
            "⬇️ Download trades (CSV)",
            data=t.to_csv(index=False).encode("utf-8"),
            file_name=f"{symbol}_trades.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(t, use_container_width=True, height=560)


# =============================================================================
# Live tab (st.fragment if available; fallback to autorefresh/manual)
# =============================================================================
def _fragment_decorator():
    # Streamlit versions differ; prefer st.fragment if present
    frag = getattr(st, "fragment", None)
    if callable(frag):
        return frag
    # fallback: identity decorator
    def identity(fn):
        return fn
    return identity


@_fragment_decorator()
def _live_panel(symbol: str) -> None:
    stream = st.session_state.get("live_stream")
    live_running = _live_running(stream)

    rows: List[dict] = st.session_state.get("live_rows", [])
    if stream is not None:
        try:
            new_msgs = stream.get_latest(max_items=250)
        except Exception:
            new_msgs = []
        if new_msgs:
            st.session_state["live_rows"].extend([_msg_to_dict(x) for x in new_msgs])
            st.session_state["live_rows"] = st.session_state["live_rows"][-800:]
            rows = st.session_state["live_rows"]

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

        # Fragment panel (updates section, not whole page)
        _live_panel(symbol)

        # Fallback refresh loop if fragments are not actually running periodically.
        if _live_running(st.session_state.get("live_stream")) and bool(st.session_state.get("live_autorefresh", True)):
            if st_autorefresh is not None:
                st_autorefresh(interval=900, key=f"live_refresh_{symbol}")
