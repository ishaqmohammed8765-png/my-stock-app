# app.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_historical, sanity_check_bars
from utils.indicators import add_indicators_inplace
from utils.backtester import backtest_strategy

# Optional: Yahoo fallback + "current-ish" price
YF_AVAILABLE = True
try:
    import yfinance as yf  # pip install yfinance
except Exception:
    YF_AVAILABLE = False


# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


# =============================================================================
# Page config + CSS
# =============================================================================
st.set_page_config(
    page_title="Pro Algo Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
def ss_init(defaults: dict[str, Any]) -> None:
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def ss_get(key: str, default: Any) -> Any:
    return st.session_state.get(key, default)


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def has_alpaca_keys() -> bool:
    k = str(st.secrets.get("ALPACA_KEY", "")).strip()
    s = str(st.secrets.get("ALPACA_SECRET", "")).strip()
    return bool(k and s)


def tail_for_plot(df: pd.DataFrame, n: int) -> pd.DataFrame:
    n = int(max(120, n))
    return df.tail(n) if len(df) > n else df


def bt_params_signature(d: dict[str, Any]) -> str:
    # stable signature so we can warn if UI has changed since last run
    return "|".join([f"{k}={v}" for k, v in sorted((k, str(v)) for k, v in d.items())])


# =============================================================================
# Small formatting helpers
# =============================================================================
def fmt_pct(x: Any) -> str:
    v = safe_float(x)
    return "—" if not np.isfinite(v) else f"{v * 100:.1f}%"


def fmt_num(x: Any, nd: int = 2) -> str:
    v = safe_float(x)
    return "—" if not np.isfinite(v) else f"{v:.{nd}f}"


# =============================================================================
# Backtest interpretation (beginner friendly)
# =============================================================================
def interpret_backtest(results: dict, trades_df: pd.DataFrame) -> dict[str, Any]:
    """
    Turn raw backtest stats into a beginner-friendly verdict.

    We intentionally keep this simple:
      - sample size (trades)
      - avg R (edge proxy)
      - max drawdown (pain)
      - sharpe (risk-adjusted quality)
    """
    n = int(results.get("trades", 0) or 0)
    avg_r = safe_float(results.get("avg_r_multiple", np.nan))
    dd = safe_float(results.get("max_drawdown", np.nan))
    sh = safe_float(results.get("sharpe", np.nan))

    score = 0
    bullets: list[str] = []

    # sample size
    if n < 20:
        bullets.append("❌ Too few trades (<20). Results are mostly noise.")
        score -= 3
    elif n < 50:
        bullets.append("⚠️ Only 20–49 trades. Treat results cautiously.")
        score -= 1
    else:
        bullets.append("✅ 50+ trades. More reliable signal than noise.")
        score += 2

    # edge proxy
    if np.isfinite(avg_r):
        if avg_r <= 0:
            bullets.append("❌ Avg R ≤ 0. This setup is not showing an edge (after your rules).")
            score -= 3
        elif avg_r < 0.15:
            bullets.append("⚠️ Avg R is small (<0.15). Edge may be fragile.")
            score -= 1
        else:
            bullets.append("✅ Avg R suggests a positive edge for this ticker/time period.")
            score += 2
    else:
        bullets.append("⚠️ Avg R unavailable (often happens when there are too few trades).")

    # drawdown
    if np.isfinite(dd):
        if dd <= -0.35:
            bullets.append("❌ Drawdown worse than −35%. Most beginners quit during this.")
            score -= 3
        elif dd <= -0.25:
            bullets.append("⚠️ Drawdown −25% to −35%. Needs discipline and smaller sizing.")
            score -= 1
        else:
            bullets.append("✅ Drawdown is more manageable.")
            score += 1

    # sharpe
    if np.isfinite(sh):
        if sh < 0.5:
            bullets.append("⚠️ Sharpe < 0.5 (weak risk-adjusted performance).")
            score -= 1
        elif sh < 1.0:
            bullets.append("✅ Sharpe 0.5–1.0 (okay).")
            score += 1
        elif sh <= 2.0:
            bullets.append("✅ Sharpe 1.0–2.0 (good).")
            score += 2
        else:
            bullets.append("⚠️ Sharpe > 2.0 can be overfit on one ticker/time window.")
            score += 1

    if score >= 3:
        verdict = "✅ Looks promising (test more tickers / longer history)"
        kind = "success"
    elif score >= 0:
        verdict = "⚠️ Mixed / fragile (tweak + retest)"
        kind = "warning"
    else:
        verdict = "❌ Not convincing (likely noise or negative edge)"
        kind = "error"

    return {"kind": kind, "verdict": verdict, "bullets": bullets}


def render_interpretation(results: dict, trades_df: pd.DataFrame) -> None:
    info = interpret_backtest(results, trades_df)

    if info["kind"] == "success":
        st.success(info["verdict"])
    elif info["kind"] == "warning":
        st.warning(info["verdict"])
    else:
        st.error(info["verdict"])

    with st.expander("How to read these results (beginner guide)", expanded=False):
        st.markdown(
            "- **Trades**: <20 is mostly noise; 50+ is more trustworthy.\n"
            "- **Avg R**: per-trade edge proxy; >0.15 is decent.\n"
            "- **Max drawdown**: worst pain; if you can’t handle it, you won’t follow the system.\n"
            "- **Sharpe**: risk-adjusted quality; >1 is good, >2 can be overfit.\n"
        )
        for b in info["bullets"]:
            st.markdown(f"- {b}")


# =============================================================================
# Data preparation
# =============================================================================
def ensure_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    lower = {str(c).lower(): c for c in out.columns}

    variants = {
        "open": ["open", "o", "opn"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "adj close", "adj_close", "adjclose"],
        "volume": ["volume", "v", "vol"],
    }

    mapping: dict[str, str] = {}
    for want, keys in variants.items():
        if want in out.columns:
            continue
        for k in keys:
            if k in lower:
                mapping[lower[k]] = want
                break

    return out.rename(columns=mapping) if mapping else out


def coerce_ohlcv_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    price_cols = [c for c in ["open", "high", "low", "close"] if c in out.columns]
    return out.dropna(subset=price_cols) if price_cols else out


def as_utc_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index()
        out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
    return out


def df_for_backtest(df_chart: pd.DataFrame) -> pd.DataFrame:
    """Backtester expects a `timestamp` column."""
    if isinstance(df_chart.index, pd.DatetimeIndex):
        idx_name = df_chart.index.name or "timestamp"
        tmp = df_chart.reset_index()
        if idx_name != "timestamp":
            tmp = tmp.rename(columns={idx_name: "timestamp"})
        return tmp
    return df_chart.copy()


# =============================================================================
# Cached loaders
# =============================================================================
@st.cache_data(ttl=15 * 60, show_spinner=False)
def cached_load_alpaca(symbol: str, force_refresh: int) -> pd.DataFrame:
    api_key = str(st.secrets.get("ALPACA_KEY", "")).strip()
    sec_key = str(st.secrets.get("ALPACA_SECRET", "")).strip()
    if not api_key or not sec_key:
        raise RuntimeError("Missing Alpaca keys in Streamlit secrets.")
    df, _dbg = load_historical(symbol, api_key, sec_key, force_refresh=force_refresh)
    return df


@st.cache_data(ttl=30 * 60, show_spinner=False)
def cached_load_yahoo(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
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


@st.cache_data(ttl=30, show_spinner=False)
def cached_current_price_yahoo(symbol: str) -> tuple[float, str]:
    """Best-effort 'current-ish' price from Yahoo (may be delayed)."""
    if not YF_AVAILABLE:
        return np.nan, "Unavailable (yfinance not installed)"

    try:
        t = yf.Ticker(symbol)

        fi = getattr(t, "fast_info", None)
        if fi and isinstance(fi, dict):
            px = fi.get("last_price") or fi.get("lastPrice") or fi.get("regularMarketPrice")
            px = safe_float(px)
            if np.isfinite(px) and px > 0:
                return float(px), "Yahoo (fast_info)"

        info = getattr(t, "info", None)
        if isinstance(info, dict):
            px = safe_float(info.get("regularMarketPrice"))
            if np.isfinite(px) and px > 0:
                return float(px), "Yahoo (info)"

        intr = t.history(period="1d", interval="1m")
        if intr is not None and not intr.empty:
            last = safe_float(intr["Close"].iloc[-1])
            if np.isfinite(last) and last > 0:
                return float(last), "Yahoo (1m)"
    except Exception:
        pass

    return np.nan, "Unavailable"


def load_and_prepare(symbol: str, *, force_refresh: int) -> None:
    st.session_state["load_error"] = None
    st.session_state["ind_error"] = None
    st.session_state["data_source"] = None

    df: pd.DataFrame | None = None
    err_primary: str | None = None

    # Primary: Alpaca (if keys exist)
    if has_alpaca_keys():
        try:
            df = cached_load_alpaca(symbol, force_refresh=force_refresh)
            st.session_state["data_source"] = "Alpaca"
        except Exception as e:
            err_primary = f"{type(e).__name__}: {e}"
            df = None

    # Fallback: Yahoo
    if df is None or getattr(df, "empty", True):
        try:
            df = cached_load_yahoo(symbol, period="5y", interval="1d")
            st.session_state["data_source"] = "Yahoo"
        except Exception as e:
            err_y = f"{type(e).__name__}: {e}"
            st.session_state["load_error"] = f"{err_primary} | {err_y}" if err_primary else err_y
            st.session_state["df_raw"] = None
            st.session_state["df_chart"] = None
            st.session_state["last_symbol"] = None
            st.session_state["data_source"] = None
            return

    df = coerce_ohlcv_numeric(ensure_ohlcv_cols(df))
    st.session_state["df_raw"] = df
    st.session_state["last_symbol"] = symbol
    st.session_state["last_loaded_at"] = pd.Timestamp.utcnow()

    # Keep sanity checks internal (no debug-heavy UI)
    try:
        st.session_state["sanity"] = sanity_check_bars(df)
    except Exception:
        st.session_state["sanity"] = None

    try:
        df_chart = as_utc_dtindex(df)
        add_indicators_inplace(df_chart)
        st.session_state["df_chart"] = df_chart
    except Exception as e:
        st.session_state["df_chart"] = as_utc_dtindex(df)
        st.session_state["ind_error"] = f"{type(e).__name__}: {e}"


# =============================================================================
# Support/Resistance + Trade Plan
# =============================================================================
def compute_support_resistance(df_ind: pd.DataFrame, lookback: int) -> tuple[float, float]:
    """
    Beginner-friendly, robust S/R:
    Uses percentiles instead of raw min/max so a single wick doesn't dominate.
    """
    lb = int(max(20, lookback))
    tail = df_ind.tail(lb)
    if tail.empty or ("low" not in tail.columns) or ("high" not in tail.columns):
        return np.nan, np.nan

    lows = pd.to_numeric(tail["low"], errors="coerce").dropna()
    highs = pd.to_numeric(tail["high"], errors="coerce").dropna()
    if lows.empty or highs.empty:
        return np.nan, np.nan

    support = float(np.nanpercentile(lows.values, 10))
    resistance = float(np.nanpercentile(highs.values, 90))
    return support, resistance


def compute_trade_plan(
    df_ind: pd.DataFrame,
    *,
    mode: str,
    atr_entry: float,
    atr_stop: float,
    atr_target: float,
) -> dict[str, float]:
    """Simple long-only plan (educational)."""
    last = df_ind.iloc[-1]
    close = safe_float(last.get("close", np.nan))
    atr = safe_float(last.get("atr14", np.nan))

    if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
        return {"entry": np.nan, "stop": np.nan, "target": np.nan, "rr": np.nan}

    mode_l = str(mode).lower().strip()
    entry = close + float(atr_entry) * atr if mode_l == "breakout" else close - float(atr_entry) * atr
    stop = entry - float(atr_stop) * atr
    target = entry + float(atr_target) * atr

    risk = entry - stop
    reward = target - entry
    rr = (reward / risk) if risk > 0 else np.nan
    return {"entry": float(entry), "stop": float(stop), "target": float(target), "rr": float(rr) if np.isfinite(rr) else np.nan}


# =============================================================================
# Plots
# =============================================================================
def extract_markers(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def plot_price(
    df: pd.DataFrame,
    symbol: str,
    trades: pd.DataFrame | None = None,
    *,
    support: float | None = None,
    resistance: float | None = None,
) -> go.Figure:
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

    if support is not None and np.isfinite(support):
        fig.add_hline(y=float(support), line_width=1, line_dash="dot")
    if resistance is not None and np.isfinite(resistance):
        fig.add_hline(y=float(resistance), line_width=1, line_dash="dot")

    if isinstance(trades, pd.DataFrame) and not trades.empty:
        entries, exits = extract_markers(trades)
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
    hlines: list[float] | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
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
# Signals
# =============================================================================
@dataclass(frozen=True)
class SignalScore:
    label: str
    score: int
    summary: str
    reasons: list[str]


def compute_signal_score(
    df_ind: pd.DataFrame, rsi_min: float, rsi_max: float, rvol_min: float, vol_max: float
) -> SignalScore:
    need = ["close", "ma50", "ma200", "rsi14", "rvol", "vol_ann", "atr14"]
    missing = [c for c in need if c not in df_ind.columns]
    if missing:
        return SignalScore("WAIT", 0, "Indicators not ready.", [f"Missing: {', '.join(missing[:6])}"])

    last = df_ind.iloc[-1]
    close = safe_float(last.get("close"))
    ma50 = safe_float(last.get("ma50"))
    ma200 = safe_float(last.get("ma200"))
    rsi = safe_float(last.get("rsi14"))
    rvol = safe_float(last.get("rvol"))
    vol_ann = safe_float(last.get("vol_ann"))
    atr = safe_float(last.get("atr14"))

    vals = np.array([close, ma50, ma200, rsi, rvol, vol_ann, atr], dtype="float64")
    if not np.isfinite(vals).all():
        return SignalScore("WAIT", 0, "Waiting for enough history.", ["Non-finite indicator values"])

    reasons: list[str] = []
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

    # Long-only labels
    if score >= 70 and uptrend:
        return SignalScore("BUY", score, "Favorable trend + filters supportive.", reasons)
    if score <= 30 and downtrend:
        return SignalScore("AVOID", score, "Bearish conditions dominate (avoid long).", reasons)
    return SignalScore("HOLD", score, "Mixed/neutral conditions.", reasons)


def get_current_price(symbol: str, df_chart: pd.DataFrame) -> tuple[float, str]:
    """Best-effort current price: Yahoo quick price (cached) -> last historical close."""
    px, src = cached_current_price_yahoo(symbol)
    if np.isfinite(px) and px > 0:
        return float(px), src

    if isinstance(df_chart, pd.DataFrame) and (not df_chart.empty) and ("close" in df_chart.columns):
        last_close = safe_float(df_chart["close"].iloc[-1])
        if np.isfinite(last_close):
            return float(last_close), "Last close (historical)"

    return np.nan, "Unavailable"


# =============================================================================
# Session init
# =============================================================================
ss_init(
    {
        "df_raw": None,
        "df_chart": None,
        "sanity": None,
        "load_error": None,
        "ind_error": None,
        "last_symbol": None,
        "last_loaded_at": None,
        "data_source": None,
        "bt_results": None,
        "bt_trades": None,
        "bt_error": None,
        "bt_params_sig": None,
        "cfg_export_json": "",
        "cfg_import_json": "",
        # UI defaults
        "symbol": "AAPL",
        "mode_label": "Breakout",
        "atr_entry": 1.0,
        "atr_stop": 2.0,
        "atr_target": 3.0,
        "rsi_min": 30.0,
        "rsi_max": 70.0,
        "rvol_min": 1.2,
        "vol_max": 1.0,
        "sr_lookback": 80,
        "enable_position_sizing": False,
        "account_capital": 100.0,
        "risk_per_trade_pct": 1.0,
        "max_alloc_pct": 10.0,
        "mark_to_market": False,
        "horizon_bars": 20,
    }
)


# =============================================================================
# Header
# =============================================================================
st.title("📈 Pro Algo Trader")
st.caption("Beginner-friendly signals • charts • backtests (educational, not financial advice)")


# =============================================================================
# Sidebar (simple + friendly)
# =============================================================================
with st.sidebar:
    st.header("Controls")

    symbol = st.text_input("Ticker", value=str(ss_get("symbol", "AAPL"))).upper().strip()

    mode_label = st.selectbox(
        "Strategy Style",
        options=["Breakout", "Pullback"],
        index=0 if str(ss_get("mode_label", "Breakout")) == "Breakout" else 1,
        help="Breakout = enter above price. Pullback = enter below price.",
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

        sr_lookback = st.number_input("Support/Resistance lookback (bars)", 20, 300, int(ss_get("sr_lookback", 80)), 5)

    with st.expander("Risk & Sizing", expanded=False):
        enable_position_sizing = st.toggle("Enable Position Sizing", value=bool(ss_get("enable_position_sizing", False)))

        # ✅ allow < £100
        account_capital = st.number_input(
            "Account Capital (£)",
            min_value=1.0,
            max_value=100_000_000.0,
            value=float(ss_get("account_capital", 100.0)),
            step=1.0,
            format="%.2f",
            help="Used for sizing in the backtest. Does not place real trades.",
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
                "sr_lookback": int(sr_lookback),
                "enable_position_sizing": bool(enable_position_sizing),
                "account_capital": float(account_capital),
                "risk_per_trade_pct": float(risk_per_trade_pct),
                "max_alloc_pct": float(max_alloc_pct),
                "mark_to_market": bool(mark_to_market),
                "horizon_bars": int(ss_get("horizon_bars", 20)),
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
                allowed = {
                    "symbol",
                    "mode_label",
                    "atr_entry",
                    "atr_stop",
                    "atr_target",
                    "rsi_min",
                    "rsi_max",
                    "rvol_min",
                    "vol_max",
                    "sr_lookback",
                    "enable_position_sizing",
                    "account_capital",
                    "risk_per_trade_pct",
                    "max_alloc_pct",
                    "mark_to_market",
                    "horizon_bars",
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

# Persist inputs
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
    "sr_lookback": sr_lookback,
    "enable_position_sizing": enable_position_sizing,
    "account_capital": account_capital,
    "risk_per_trade_pct": risk_per_trade_pct,
    "max_alloc_pct": max_alloc_pct,
    "mark_to_market": mark_to_market,
}.items():
    st.session_state[k] = v


# =============================================================================
# Auto-load
# =============================================================================
needs_load = load_btn or (st.session_state.get("df_raw") is None) or (st.session_state.get("last_symbol") != symbol)
force_refresh = int(time.time()) if load_btn else 0

if needs_load:
    with st.spinner(f"Loading {symbol}…"):
        load_and_prepare(symbol, force_refresh=force_refresh)

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

df_plot = tail_for_plot(df_chart, 700)

src = st.session_state.get("data_source")
if src:
    st.caption(f"Data source: **{src}**")


# =============================================================================
# Tabs (Live tab removed)
# =============================================================================
tab_signal, tab_charts, tab_backtest = st.tabs(["✅ Signal", "📊 Charts", "🧪 Backtest"])


# =============================================================================
# Signal tab
# =============================================================================
with tab_signal:
    st.subheader(f"{symbol} — Signal")

    current_px, current_src = get_current_price(symbol, df_chart)

    score = compute_signal_score(df_chart, float(rsi_min), float(rsi_max), float(rvol_min), float(vol_max))
    support, resistance = compute_support_resistance(df_chart, int(sr_lookback))

    plan = compute_trade_plan(
        df_chart,
        mode=mode,
        atr_entry=float(atr_entry),
        atr_stop=float(atr_stop),
        atr_target=float(atr_target),
    )

    hero = st.container(border=True)
    with hero:
        a, b, c = st.columns([1.2, 1, 1], vertical_alignment="center")
        with a:
            if score.label == "BUY":
                st.success(f"**BUY** • Score {score.score}/100")
            elif score.label == "AVOID":
                st.error(f"**AVOID** • Score {score.score}/100")
            elif score.label == "WAIT":
                st.warning(f"**WAIT** • Score {score.score}/100")
            else:
                st.info(f"**HOLD** • Score {score.score}/100")
            st.caption(score.summary)

        with b:
            st.metric("Current price", f"{current_px:.2f}" if np.isfinite(current_px) else "—")
            st.caption(f"Source: {current_src}")

        with c:
            st.metric("Support", f"{support:.2f}" if np.isfinite(support) else "—")
            st.metric("Resistance", f"{resistance:.2f}" if np.isfinite(resistance) else "—")

    st.markdown("### Where to place support / resistance (simple guidance)")
    box = st.container(border=True)
    with box:
        if np.isfinite(support) and np.isfinite(resistance):
            st.write(f"• **Support zone** ~ **{support:.2f}** → many traders place a stop a little *below* this.")
            st.write(f"• **Resistance zone** ~ **{resistance:.2f}** → many traders take profit or get cautious near this.")
            st.caption("These are percentile-based levels from the last lookback window (less sensitive to wicks).")
        else:
            st.info("Not enough data to compute stable support/resistance yet.")

    st.markdown("### ATR Plan (optional)")
    p = st.container(border=True)
    with p:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry", f"{plan['entry']:.2f}" if np.isfinite(plan["entry"]) else "—")
        c2.metric("Stop", f"{plan['stop']:.2f}" if np.isfinite(plan["stop"]) else "—")
        c3.metric("Target", f"{plan['target']:.2f}" if np.isfinite(plan["target"]) else "—")
        c4.metric("R:R", f"{plan['rr']:.2f}" if np.isfinite(plan["rr"]) else "—")
        st.caption(f"Mode: {mode_label} • Long-only plan (for beginners)")

    with st.expander("Why this score?", expanded=False):
        for r in score.reasons[:12]:
            st.write(f"• {r}")


# =============================================================================
# Charts tab
# =============================================================================
with tab_charts:
    st.subheader(f"{symbol} — Charts")

    trades_for_markers = st.session_state.get("bt_trades")
    support, resistance = compute_support_resistance(df_chart, int(sr_lookback))

    st.plotly_chart(
        plot_price(df_plot, symbol, trades=trades_for_markers, support=support, resistance=resistance),
        use_container_width=True,
    )

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
# Backtest tab (IMPROVED: aligned button + interpretation + cash ledger)
# =============================================================================
with tab_backtest:
    st.subheader("Backtest")

    # ✅ Fix alignment: put inputs + button in one form row
    with st.form("bt_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1.25, 1.25, 1.2, 1.6])
        horizon_bars = c1.number_input("Max hold (bars)", 1, 200, int(ss_get("horizon_bars", 20)), 1)
        use_cash_ledger = c2.toggle(
            "Realistic cash (recommended)",
            value=True,
            help="When ON, the backtest won't buy shares you can't afford. Best for small accounts.",
        )
        gate_mode = c3.selectbox(
            "Gating",
            ["soft", "hard", "off"],
            index=0,
            help="Soft = size down weak setups. Hard = skip weak setups. Off = no gating.",
        )
        run_backtest_btn = c4.form_submit_button("🧪 Run Backtest", use_container_width=True)

    st.session_state["horizon_bars"] = int(horizon_bars)

    prob_gating = (gate_mode != "off")
    gate_mode_val = "soft" if gate_mode == "soft" else "hard"

    # Keep parameter names aligned with your backtester signature.
    bt_params = dict(
        mode=mode,
        horizon=int(horizon_bars),
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

        # execution realism (keep simple defaults)
        slippage_bps=5.0,
        commission_per_order=0.0,
        spread_mode="taker_only",
        exit_priority="stop_first",
        time_exit_price="open",

        # sizing
        enable_position_sizing=bool(enable_position_sizing),
        risk_pct=float(risk_per_trade_pct) / 100.0,
        max_alloc_pct=float(max_alloc_pct) / 100.0,
        min_risk_per_share=1e-6,

        # accounting
        use_cash_ledger=bool(use_cash_ledger),
        allow_margin=False,

        # display
        mark_to_market=bool(mark_to_market),

        # gating
        prob_gating=bool(prob_gating),
        prob_is_frac=0.85,
        prob_min=0.50,
        min_bucket_trades=6,
        min_avg_r=-0.05,
        gate_mode=gate_mode_val,
    )
    sig = bt_params_signature(bt_params)

    if run_backtest_btn:
        st.session_state["bt_error"] = None
        with st.spinner("Running backtest…"):
            try:
                df_in = df_for_backtest(df_chart)
                results, trades_df = backtest_strategy(df=df_in, market_df=None, **bt_params)
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

        # ✅ Interpretation panel
        render_interpretation(results, trades if isinstance(trades, pd.DataFrame) else pd.DataFrame())

        st.markdown("### Performance")
        pwrap = st.container(border=True)
        with pwrap:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Trades", f"{ntr}")
            c2.metric("Win rate", fmt_pct(results.get("win_rate", np.nan)))
            c3.metric("Total return", fmt_pct(results.get("total_return", np.nan)))
            c4.metric("Max drawdown", fmt_pct(results.get("max_drawdown", np.nan)))
            c5.metric("Sharpe", fmt_num(results.get("sharpe", np.nan), 2))
            c6.metric("Avg R", fmt_num(results.get("avg_r_multiple", np.nan), 2))

        # Equity / Drawdown charts
        if isinstance(df_bt, pd.DataFrame) and (not df_bt.empty) and ("equity" in df_bt.columns):
            eq = pd.to_numeric(df_bt["equity"], errors="coerce").dropna()
            if len(eq) > 2:
                ts = pd.to_datetime(
                    df_bt.get("timestamp", pd.Series(index=df_bt.index, dtype="datetime64[ns]")),
                    errors="coerce",
                    utc=True,
                )
                x = np.arange(len(eq)) if (not isinstance(ts, pd.Series) or ts.isna().all()) else ts.iloc[-len(eq):]

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

            # Optional cash curve if enabled and present
            if "cash" in df_bt.columns and df_bt["cash"].notna().any():
                cash_s = pd.to_numeric(df_bt["cash"], errors="coerce").dropna()
                if len(cash_s) > 2:
                    st.plotly_chart(
                        go.Figure([go.Scatter(x=df_bt.index, y=cash_s.values, mode="lines")]).update_layout(
                            title="Cash (ledger)",
                            height=240,
                            margin=dict(l=10, r=10, t=45, b=10),
                            hovermode="x unified",
                            showlegend=False,
                        ),
                        use_container_width=True,
                    )

        # Trades table + download
        if not isinstance(trades, pd.DataFrame) or trades.empty:
            st.warning("Backtest completed but produced no trades.")
            st.caption("Tip: switch gating to **soft** or **off**, widen RSI/RVOL filters, or try a different ticker.")
        else:
            st.download_button(
                "⬇️ Download trades (CSV)",
                data=trades.to_csv(index=False).encode("utf-8"),
                file_name=f"{symbol}_trades.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.dataframe(trades, use_container_width=True, height=560)

        with st.expander("Assumptions (advanced)", expanded=False):
            st.json(results.get("assumptions", {}))
