# app.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Your modular project imports (kept)
from utils.data_loader import load_historical, sanity_check_bars
from utils.indicators import add_indicators_inplace
from utils.backtester import backtest_strategy

# Optional: community autorefresh component
try:
    from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
except Exception:
    st_autorefresh = None  # type: ignore

# Optional: live streaming module
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
# UI polish (modern + cleaner)
# ---------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 0.9rem; padding-bottom: 1.1rem; max-width: 1320px; }
[data-testid="stSidebar"] .block-container { padding-top: 0.8rem; }

h1 { margin-bottom: 0.15rem; letter-spacing: -0.3px; }
h2 { margin-top: 0.6rem; margin-bottom: 0.25rem; letter-spacing: -0.2px; }
h3 { margin-top: 0.6rem; margin-bottom: 0.25rem; }

[data-testid="stMetric"] { padding: 0.1rem 0.15rem; border-radius: 10px; }
[data-testid="stMetricLabel"] p { font-size: 0.86rem; opacity: 0.85; }
[data-testid="stMetricValue"] div { font-size: 1.35rem; }

div[data-testid="stVerticalBlockBorderWrapper"]{ border-radius: 16px; }

.small-muted { color: rgba(250,250,250,0.65); font-size: 0.9rem; }

.stButton button { border-radius: 14px; }
.stDownloadButton button { border-radius: 14px; }

code { font-size: 0.92rem; }

footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Session init
# ---------------------------
def _ss_setdefault(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


# Core state
_ss_setdefault("df_raw", None)
_ss_setdefault("df_chart", None)
_ss_setdefault("sanity", None)
_ss_setdefault("load_error", None)
_ss_setdefault("ind_error", None)
_ss_setdefault("last_symbol", None)
_ss_setdefault("last_loaded_at", None)
_ss_setdefault("data_source", None)
_ss_setdefault("alpaca_dbg", None)  # ✅ NEW: store alpaca debug dict

# Backtest persistence
_ss_setdefault("bt_results", None)   # will store df_bt
_ss_setdefault("bt_trades", None)
_ss_setdefault("bt_error", None)
_ss_setdefault("bt_params_sig", None)

# Live
_ss_setdefault("live_stream", None)
_ss_setdefault("live_rows", [])
_ss_setdefault("live_autorefresh", True)
_ss_setdefault("live_last_symbol", None)


# ---------------------------
# Helpers
# ---------------------------
def ss_get(name: str, default: Any) -> Any:
    return st.session_state.get(name, default)


def has_keys(api_key: str, sec_key: str) -> bool:
    return bool(api_key and sec_key and str(api_key).strip() and str(sec_key).strip())


def _safe_float(x: Any) -> float:
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
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
    return out


def _tail_for_plot(df: pd.DataFrame, n: int) -> pd.DataFrame:
    n = int(max(80, n))
    return df.tail(n) if len(df) > n else df


def _ensure_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common naming variants into open/high/low/close/volume."""
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    cols = {str(c).lower(): c for c in out.columns}
    mapping: Dict[str, str] = {}

    for want in ["open", "high", "low", "close", "volume"]:
        if want in out.columns:
            continue
        if want in cols:
            mapping[cols[want]] = want
            continue

        variants = {
            "open": ["o", "opn"],
            "high": ["h"],
            "low": ["l"],
            "close": ["c", "adj close", "adj_close", "adjclose"],
            "volume": ["v", "vol"],
        }.get(want, [])

        for v in variants:
            if v in cols:
                mapping[cols[v]] = want
                break

    if mapping:
        out = out.rename(columns=mapping)
    return out


def _coerce_ohlcv_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Keep plotting/indicators stable by forcing numeric OHLCV."""
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    price_cols = [c for c in ["open", "high", "low", "close"] if c in out.columns]
    if price_cols:
        out = out.dropna(subset=price_cols)
    return out


def plot_price(df: pd.DataFrame, symbol: str) -> go.Figure:
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
        fig.add_trace(go.Scatter(x=x, y=df.get("close", pd.Series(index=x, dtype=float)), mode="lines", name="close"))

    for ma_col, label in [("ma50", "MA50"), ("ma200", "MA200")]:
        if ma_col in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[ma_col], mode="lines", name=label))

    fig.update_layout(
        title=f"{symbol} — Price",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True)
    return fig


def plot_indicator(
    df: pd.DataFrame,
    col: str,
    title: str,
    height: int = 260,
    hlines: Optional[List[float]] = None,
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
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True)
    if (y0 is not None) and (y1 is not None):
        fig.update_yaxes(range=[float(y0), float(y1)])
    return fig


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
    idx = r.idxmax()
    v = _safe_float(r.loc[idx]) if idx in r.index else np.nan
    if np.isfinite(v) and v >= float(thresh):
        return {"ts": idx, "abs_move": v}
    return None


def compute_trade_plan_breakout(
    df_ind: pd.DataFrame,
    atr_entry: float,
    atr_stop: float,
    atr_target: float,
    assumed_spread_bps: float,
    include_spread_penalty: bool,
) -> Dict[str, Any]:
    last = df_ind.iloc[-1]
    close = _safe_float(last.get("close", np.nan))
    atr = _safe_float(last.get("atr14", np.nan))

    if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
        return {"entry": np.nan, "stop": np.nan, "target": np.nan, "rr": np.nan, "entry_type": "—", "atr": atr}

    entry = close + float(atr_entry) * atr
    entry_type = "Breakout (stop trigger)"

    if include_spread_penalty and assumed_spread_bps > 0 and np.isfinite(entry):
        entry *= (1.0 + assumed_spread_bps / 10000.0)

    stop = entry - float(atr_stop) * atr
    target = entry + float(atr_target) * atr

    risk = entry - stop
    reward = target - entry
    rr = (reward / risk) if risk > 0 else np.nan

    return {"entry": entry, "stop": stop, "target": target, "rr": rr, "entry_type": entry_type, "atr": atr}


@dataclass(frozen=True)
class SignalScore:
    label: str
    score: int
    summary: str
    reasons: List[str]


def compute_signal_score(
    df_ind: pd.DataFrame,
    rsi_min: float,
    rsi_max: float,
    rvol_min: float,
    vol_max: float,
) -> SignalScore:
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
        reasons.append("Trend mixed (MA stack not aligned)")

    if rsi < rsi_min:
        score -= 10
        reasons.append(f"RSI below min ({rsi:.1f} < {rsi_min:.0f})")
    elif rsi > rsi_max:
        score -= 10
        reasons.append(f"RSI above max ({rsi:.1f} > {rsi_max:.0f})")
    else:
        score += 6
        reasons.append(f"RSI in range ({rsi:.1f})")

    if rvol < rvol_min:
        score -= 10
        reasons.append(f"RVOL low ({rvol:.2f} < {rvol_min:.2f})")
    else:
        score += 8
        reasons.append(f"RVOL ok ({rvol:.2f})")

    if vol_ann > vol_max:
        score -= 10
        reasons.append(f"Vol high ({vol_ann:.2f} > {vol_max:.2f})")
    else:
        score += 6
        reasons.append(f"Vol ok ({vol_ann:.2f})")

    score = int(np.clip(score, 0, 100))
    if score >= 70 and uptrend:
        return SignalScore("BUY", score, "Favorable trend + filters mostly supportive.", reasons)
    if score <= 30 and downtrend:
        return SignalScore("SELL", score, "Bearish conditions dominate.", reasons)
    return SignalScore("HOLD", score, "Mixed/neutral conditions (or filters not strong enough).", reasons)


def _bt_params_signature(**kwargs) -> str:
    items = sorted((k, str(v)) for k, v in kwargs.items())
    return "|".join([f"{k}={v}" for k, v in items])


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


def _live_msgs_to_df(rows: List[Any]) -> pd.DataFrame:
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
        df["spread"] = spread
        df["spread_bps"] = np.where(mid > 0, (spread / mid) * 10000.0, np.nan)

    return df


# ---------------------------
# Data loading (Alpaca preferred; Yahoo fallback)
# ---------------------------

# ✅ UPDATED: return df + dbg, and accept force_refresh so refresh is real
@st.cache_data(ttl=15 * 60, show_spinner=False)
def _cached_load_alpaca(symbol: str, api_key: str, sec_key: str, force_refresh: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df, dbg = load_historical(symbol, api_key, sec_key, force_refresh=force_refresh)
    return df, dbg


@st.cache_data(ttl=30 * 60, show_spinner=False)
def _cached_load_yahoo(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance not installed. Add `yfinance` to requirements.")
    t = yf.Ticker(symbol)
    hist = t.history(period=period, interval=interval, auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError("Yahoo returned no data.")
    hist = hist.rename(columns={c: c.lower() for c in hist.columns})
    hist = hist.reset_index().rename(columns={"Date": "timestamp", "Datetime": "timestamp"})
    if "timestamp" not in hist.columns:
        hist["timestamp"] = pd.to_datetime(hist.index)
    return hist


def _load_and_prepare(
    symbol: str,
    api_key: str,
    sec_key: str,
    yahoo_fallback: bool,
    yahoo_period: str,
    *,
    force_refresh: int,
) -> None:
    st.session_state["load_error"] = None
    st.session_state["ind_error"] = None
    st.session_state["data_source"] = None
    st.session_state["alpaca_dbg"] = None

    df: Optional[pd.DataFrame] = None
    err_alpaca: Optional[str] = None

    # Alpaca first if keys exist
    if has_keys(api_key, sec_key):
        try:
            df, dbg = _cached_load_alpaca(symbol, api_key, sec_key, force_refresh=force_refresh)
            st.session_state["alpaca_dbg"] = dbg
            feed = dbg.get("feed", "default")
            st.session_state["data_source"] = f"Alpaca ({feed})"
        except Exception as e:
            err_alpaca = f"{type(e).__name__}: {e}"
            df = None

    # Yahoo fallback
    if (df is None or getattr(df, "empty", True)) and yahoo_fallback:
        try:
            df = _cached_load_yahoo(symbol, period=yahoo_period, interval="1d")
            st.session_state["data_source"] = f"Yahoo ({yahoo_period})"
        except Exception as e:
            err_y = f"{type(e).__name__}: {e}"
            msg = f"Yahoo failed: {err_y}"
            if err_alpaca:
                msg = f"Alpaca failed: {err_alpaca} | {msg}"
            st.session_state["load_error"] = msg
            st.session_state["df_raw"] = None
            st.session_state["df_chart"] = None
            st.session_state["last_symbol"] = None
            return

    if df is None or getattr(df, "empty", True):
        st.session_state["df_raw"] = None
        st.session_state["df_chart"] = None
        st.session_state["last_symbol"] = None
        st.session_state["load_error"] = st.session_state.get("load_error") or (f"Alpaca failed: {err_alpaca}" if err_alpaca else "No data.")
        return

    # normalize and coerce numeric
    df = _ensure_ohlcv_cols(df)
    df = _coerce_ohlcv_numeric(df)

    st.session_state["df_raw"] = df
    st.session_state["last_symbol"] = symbol
    st.session_state["last_loaded_at"] = pd.Timestamp.utcnow()

    try:
        st.session_state["sanity"] = sanity_check_bars(df) if df is not None else None
    except Exception:
        st.session_state["sanity"] = None

    try:
        df_chart = _as_utc_dtindex(df)
        add_indicators_inplace(df_chart)
        st.session_state["df_chart"] = df_chart
    except Exception as e:
        st.session_state["df_chart"] = _as_utc_dtindex(df)
        st.session_state["ind_error"] = f"{type(e).__name__}: {e}"


# ---------------------------
# Secrets
# ---------------------------
api_key = st.secrets.get("ALPACA_KEY", "")
sec_key = st.secrets.get("ALPACA_SECRET", "")


# ---------------------------
# Header
# ---------------------------
st.title("📈 Pro Algo Trader")
st.caption("Clean UI • Modular indicators • Persistent backtests • Optional live quotes")


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Controls")

    symbol = st.text_input("Ticker", value=ss_get("symbol", "AAPL")).upper().strip()
    horizon = st.number_input("Max hold (bars)", min_value=1, max_value=200, value=int(ss_get("horizon", 20)))

    with st.expander("Strategy", expanded=False):
        atr_entry = st.number_input("ATR entry (breakout)", 0.0, 10.0, float(ss_get("atr_entry", 1.0)), 0.1)
        atr_stop = st.number_input("ATR stop", 0.1, 20.0, float(ss_get("atr_stop", 2.0)), 0.1)
        atr_target = st.number_input("ATR target", 0.1, 50.0, float(ss_get("atr_target", 3.0)), 0.1)

        st.caption("Signal uses a soft score. Backtest uses hard filters.")
        rsi_min = st.number_input("RSI min", 0.0, 100.0, float(ss_get("rsi_min", 30.0)))
        rsi_max = st.number_input("RSI max", 0.0, 100.0, float(ss_get("rsi_max", 70.0)))
        rvol_min = st.number_input("RVOL min", 0.0, 10.0, float(ss_get("rvol_min", 1.2)))
        vol_max = st.number_input("Max annual vol", 0.0, 5.0, float(ss_get("vol_max", 1.0)))

    with st.expander("Execution realism", expanded=False):
        include_spread_penalty = st.checkbox("Include spread penalty", value=bool(ss_get("include_spread_penalty", True)))
        assumed_spread_bps = st.number_input("Assumed spread (bps)", 0.0, 200.0, float(ss_get("assumed_spread_bps", 5.0)))

    with st.expander("Charts", expanded=False):
        sr_lookback = st.number_input("Lookback low/high (bars)", min_value=10, max_value=300, value=int(ss_get("sr_lookback", 50)), step=5)
        chart_window = st.number_input("Chart window (bars)", min_value=120, max_value=3000, value=int(ss_get("chart_window", 700)), step=50)

    with st.expander("Data source", expanded=False):
        yahoo_fallback = st.toggle("Use Yahoo fallback (recommended)", value=bool(ss_get("yahoo_fallback", True)))
        opts = ["1y", "2y", "5y", "10y", "max"]
        default = str(ss_get("yahoo_period", "5y"))
        idx = opts.index(default) if default in opts else opts.index("5y")
        yahoo_period = st.selectbox("Yahoo history", options=opts, index=idx)
        st.caption("If Alpaca load fails (or no keys), app can still run using Yahoo.")

    st.divider()
    c1, c2 = st.columns(2)
    load_btn = c1.button("🔄 Load / Refresh", use_container_width=True)
    run_backtest_btn = c2.button("🧪 Run Backtest", use_container_width=True)

    st.divider()
    if has_keys(api_key, sec_key):
        st.success("Alpaca keys detected (Live can work).")
    else:
        st.info("No Alpaca keys found in secrets. Live disabled, Yahoo can still load data.")

# Persist inputs
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
    "chart_window": chart_window,
    "yahoo_fallback": yahoo_fallback,
    "yahoo_period": yahoo_period,
}.items():
    st.session_state[k] = v


# ---------------------------
# Auto-load logic
# ---------------------------
needs_load = (
    load_btn
    or (st.session_state.get("df_raw") is None)
    or (st.session_state.get("last_symbol") != symbol)
)

if st.session_state.get("live_stream") is not None and st.session_state.get("live_last_symbol") != symbol:
    _stop_live_stream()

# ✅ cache-bust only on manual refresh
force_refresh = int(time.time()) if load_btn else 0

if needs_load:
    with st.spinner(f"Loading {symbol}…"):
        _load_and_prepare(
            symbol,
            api_key,
            sec_key,
            yahoo_fallback=bool(yahoo_fallback),
            yahoo_period=str(yahoo_period),
            force_refresh=force_refresh,
        )

df_raw = st.session_state.get("df_raw")
df_chart = st.session_state.get("df_chart")
data_source = st.session_state.get("data_source") or "—"

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

df_plot = _tail_for_plot(df_chart, int(chart_window))


# ---------------------------
# Tabs
# ---------------------------
tab_signal, tab_charts, tab_backtest, tab_live = st.tabs(
    ["✅ Signal", "📊 Charts", "🧪 Backtest", "📡 Live"]
)


# ---------------------------
# Signal tab
# ---------------------------
with tab_signal:
    top = st.container(border=True)
    with top:
        left, right = st.columns([2, 1], vertical_alignment="center")
        with left:
            st.subheader(f"{symbol} — Signal")
            st.caption(f"Data source: **{data_source}** • Last load: {st.session_state.get('last_loaded_at')}")
        with right:
            if st.session_state.get("ind_error"):
                st.warning("Some indicators failed; signal may be degraded.")

    jump = detect_big_jump(df_chart, thresh=0.18)
    if jump:
        st.warning("Large price jump detected (possible split/corporate action/data issue).")
        st.caption(f"Max |close-to-close| move: {jump['abs_move']:.1%} at {jump['ts']}")

    # ✅ Minimal beginner-friendly data warnings (not spammy)
    san = st.session_state.get("sanity")
    if isinstance(san, dict) and san.get("warnings"):
        with st.expander("Data quality notes", expanded=False):
            for w in san["warnings"][:8]:
                st.write(f"• {w}")

    # ✅ Optional alpaca debug
    dbg = st.session_state.get("alpaca_dbg")
    if isinstance(dbg, dict):
        with st.expander("Alpaca debug (advanced)", expanded=False):
            st.json(dbg)

    score = compute_signal_score(df_chart, float(rsi_min), float(rsi_max), float(rvol_min), float(vol_max))

    plan = compute_trade_plan_breakout(
        df_chart,
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

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Lookback Low", f"{lo:.2f}" if np.isfinite(lo) else "—")
    m2.metric("Lookback High", f"{hi:.2f}" if np.isfinite(hi) else "—")
    m3.metric("RSI", f"{rsi:.1f}" if np.isfinite(rsi) else "—")
    m4.metric("RVOL", f"{rvol:.2f}" if np.isfinite(rvol) else "—")
    m5.metric("Ann. Vol", f"{vol_ann:.2f}" if np.isfinite(vol_ann) else "—")
    m6.metric("Data", data_source)

    st.divider()
    p = st.container(border=True)
    with p:
        st.subheader("Breakout Plan (ATR-based)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry", f"{plan['entry']:.2f}" if np.isfinite(plan.get("entry", np.nan)) else "—")
        c2.metric("Stop", f"{plan['stop']:.2f}" if np.isfinite(plan.get("stop", np.nan)) else "—")
        c3.metric("Target", f"{plan['target']:.2f}" if np.isfinite(plan.get("target", np.nan)) else "—")
        c4.metric("Assumed spread (bps)", f"{assumed_spread_bps:.1f}" if include_spread_penalty else "0.0")
        st.caption(f"Multiples: entry={atr_entry:.2f}×ATR, stop={atr_stop:.2f}×ATR, target={atr_target:.2f}×ATR")

    with st.expander("Why this score?", expanded=False):
        for r in score.reasons[:10]:
            st.write(f"• {r}")

    st.info("Beginner note: the **Signal score is soft** (penalizes conditions). The **Backtest is strict** (filters must pass to trade).")


# ---------------------------
# Charts tab
# ---------------------------
with tab_charts:
    st.subheader(f"{symbol} — Charts")
    st.caption("Candles + moving averages, plus key indicators.")
    st.plotly_chart(plot_price(df_plot, symbol), use_container_width=True)

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        if "rsi14" in df_plot.columns:
            st.plotly_chart(plot_indicator(df_plot, "rsi14", "RSI(14)", height=260, hlines=[30, 70], y0=0, y1=100), use_container_width=True)
        else:
            st.info("RSI not available.")
    with c2:
        if "rvol" in df_plot.columns:
            st.plotly_chart(plot_indicator(df_plot, "rvol", "RVOL", height=260, hlines=[1.0]), use_container_width=True)
        else:
            st.info("RVOL not available.")
    with c3:
        if "vol_ann" in df_plot.columns:
            st.plotly_chart(plot_indicator(df_plot, "vol_ann", "Annualized Volatility", height=260), use_container_width=True)
        else:
            st.info("Volatility not available.")

    with st.expander("Data preview", expanded=False):
        st.dataframe(df_plot.tail(120), use_container_width=True, height=420)


# ---------------------------
# Backtest tab
# ---------------------------
with tab_backtest:
    st.subheader("Backtest (Breakout-only)")
    st.caption("Runs using `utils.backtester.backtest_strategy()` and persists results across reruns.")

    bt_params = dict(
        symbol=symbol,
        horizon=int(horizon),
        atr_entry=float(atr_entry),
        atr_stop=float(atr_stop),
        atr_target=float(atr_target),
        rsi_min=float(rsi_min),
        rsi_max=float(rsi_max),
        rvol_min=float(rvol_min),
        vol_max=float(vol_max),
        include_spread_penalty=bool(include_spread_penalty),
        assumed_spread_bps=float(assumed_spread_bps),
    )
    sig = _bt_params_signature(**bt_params)

    if run_backtest_btn:
        st.session_state["bt_error"] = None
        with st.spinner("Running backtest…"):
            try:
                df_bt, trades = backtest_strategy(
                    df=df_chart,
                    market_df=None,
                    horizon=int(horizon),
                    mode="breakout",
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
                st.session_state["bt_results"] = df_bt   # ✅ df_bt
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
        st.info("No backtest results yet. Click **🧪 Run Backtest** in the sidebar.")
    else:
        if st.session_state.get("bt_params_sig") != sig:
            st.warning("Showing backtest results from earlier parameters. Run again to update.")

        t = trades.copy()
        wrap = st.container(border=True)
        with wrap:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades", f"{len(t)}")

            p = pd.to_numeric(t.get("pnl_per_share", pd.Series(dtype=float)), errors="coerce")
            win_rate = float((p > 0).sum()) / max(1, len(p)) if len(p) else np.nan
            c2.metric("Win rate", f"{win_rate:.1%}" if np.isfinite(win_rate) else "—")

            # ✅ Derive return/maxDD from equity if available
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

        # ✅ Equity + DD plots if equity exists
        if isinstance(df_bt, pd.DataFrame) and (not df_bt.empty) and ("equity" in df_bt.columns):
            eq = pd.to_numeric(df_bt["equity"], errors="coerce").dropna()
            if len(eq) > 2:
                peak = eq.cummax()
                dd = (eq / peak) - 1.0

                st.plotly_chart(
                    go.Figure([go.Scatter(x=np.arange(len(eq)), y=eq.values, mode="lines")]).update_layout(
                        title="Equity Curve (per bar)",
                        height=320,
                        margin=dict(l=10, r=10, t=45, b=10),
                        hovermode="x unified",
                        showlegend=False,
                    ),
                    use_container_width=True,
                )
                st.plotly_chart(
                    go.Figure([go.Scatter(x=np.arange(len(dd)), y=dd.values, mode="lines")]).update_layout(
                        title="Drawdown (per bar)",
                        height=260,
                        margin=dict(l=10, r=10, t=45, b=10),
                        hovermode="x unified",
                        showlegend=False,
                    ),
                    use_container_width=True,
                )
            else:
                st.info("Equity curve exists but is too short to plot.")
        else:
            st.info("Equity curve not available (enable sizing/equity tracking in backtester to show it).")

        st.divider()
        st.dataframe(t, use_container_width=True, height=560)


# ---------------------------
# Live tab
# ---------------------------
with tab_live:
    st.subheader("Live Quotes")
    st.caption("Requires Alpaca keys and a working `utils.live_stream.RealtimeStream`.")

    if not LIVE_AVAILABLE:
        st.info("Live module not available (or import failed).")
    elif not has_keys(api_key, sec_key):
        st.info("Live is disabled because Alpaca keys are missing in Streamlit secrets.")
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
                stream = RealtimeStream(api_key, sec_key, symbol)
                stream.start()
                st.session_state["live_stream"] = stream
                st.session_state["live_last_symbol"] = symbol
                live_running = True
            except Exception as e:
                st.session_state["live_stream"] = None
                st.error("Failed to start live stream.")
                st.caption(f"{type(e).__name__}: {e}")
                live_running = False

        if stop_clicked and live_running:
            _stop_live_stream()
            live_running = False

        stream = st.session_state.get("live_stream")
        if stream is not None:
            try:
                new_msgs = stream.get_latest(max_items=250)
            except Exception:
                new_msgs = []
            if new_msgs:
                st.session_state["live_rows"].extend(new_msgs)
                st.session_state["live_rows"] = st.session_state["live_rows"][-600:]

        st.caption("Status: ✅ running" if live_running else "Status: ⏸ stopped")

        rows = st.session_state.get("live_rows", [])
        if not rows:
            st.info("No quote updates received yet.")
        else:
            df_live = _live_msgs_to_df(rows)

            if df_live.empty:
                st.warning("Received live messages, but couldn't parse them into a table.")
                st.write("Last raw message:", rows[-1])
            else:
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

                show_cols = [c for c in ["timestamp", "symbol", "bid_price", "ask_price", "mid", "spread_bps", "bid_size", "ask_size", "message"] if c in df_live.columns]
                sort_col = "timestamp" if "timestamp" in df_live.columns else None
                view = df_live.sort_values(sort_col).tail(140) if sort_col else df_live.tail(140)
                st.dataframe(view[show_cols] if show_cols else view, use_container_width=True, height=520)

        if live_running and bool(st.session_state.get("live_autorefresh", True)):
            if st_autorefresh is not None:
                st_autorefresh(interval=800, key=f"live_refresh_{symbol}")
            else:
                st.caption("Tip: install `streamlit-autorefresh` to auto-refresh without manual reruns.")

