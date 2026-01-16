# app.py
from __future__ import annotations

import html
import json
import logging
import random
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional, Tuple

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
# Constants (centralize "magic numbers")
# =============================================================================
APP_TITLE = "📈 Pro Algo Trader"
APP_CAPTION = "Beginner-friendly signals • charts • backtests (educational, not financial advice)"

CACHE_TTL_ALPACA_SEC = 15 * 60
CACHE_TTL_YAHOO_HIST_SEC = 30 * 60
CACHE_TTL_YAHOO_PRICE_SEC = 30
CACHE_TTL_YAHOO_NEWS_SEC = 10 * 60
CACHE_TTL_YAHOO_META_SEC = 6 * 60 * 60

PLOT_TAIL_MIN_BARS = 120
PLOT_TAIL_DEFAULT_BARS = 700

SR_MIN_LOOKBACK = 20
SR_SUPPORT_PCTL = 10
SR_RESIST_PCTL = 90

# Signal score weights
SCORE_BASE = 50
SCORE_TREND_BONUS = 22
SCORE_RSI_GOOD = 6
SCORE_RSI_BAD = 10
SCORE_RVOL_GOOD = 8
SCORE_RVOL_BAD = 10
SCORE_VOL_GOOD = 6
SCORE_VOL_BAD = 10

SCORE_BUY_MIN = 70
SCORE_AVOID_MAX = 30

# Backtest defaults (execution realism)
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_SPREAD_BPS = 5.0
DEFAULT_COMMISSION = 0.0

# Gating defaults (simple)
DEFAULT_PROB_IS_FRAC = 0.85
DEFAULT_PROB_MIN = 0.50
DEFAULT_MIN_BUCKET_TRADES = 6
DEFAULT_MIN_AVG_R = -0.05

DEFAULT_OP_MAX_MCAP_B = 5.0
DEFAULT_OP_SR_LOOKBACK = 60

SMALL_CAP_POOL = [
    "SOFI",
    "PLTR",
    "IONQ",
    "RKLB",
    "SOUN",
    "BBAI",
    "RUN",
    "BLNK",
    "FUBO",
    "UPST",
    "ASTS",
    "MARA",
    "RIOT",
    "LCID",
    "FSLY",
    "HUT",
    "TLRY",
    "JOBY",
    "DNMR",
    "CLSK",
]


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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --bg: #0f1117;
  --surface: #151923;
  --surface-2: #1b2130;
  --border: rgba(255, 255, 255, 0.06);
  --text: #f5f7ff;
  --muted: #a6b0c3;
  --accent: #6ea8fe;
  --radius: 16px;
}
html, body, [class*="stApp"] { font-family: "Inter", sans-serif; background: var(--bg); color: var(--text); }
.block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; max-width: 1320px; }
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
h1, h2, h3, h4 { letter-spacing: -0.3px; color: var(--text); }
h1 { margin-bottom: 0.15rem; font-weight: 700; }
p, label, span { color: var(--muted); }
[data-testid="stMetric"] {
  background: var(--surface-2);
  padding: 0.75rem 0.9rem;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  box-shadow: 0 8px 20px rgba(10, 14, 23, 0.35);
}
div[data-testid="stVerticalBlockBorderWrapper"]{
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: var(--surface);
  padding: 0.65rem;
}
.stButton button, .stDownloadButton button {
  border-radius: 14px;
  background: linear-gradient(135deg, #5b8dff, #7b5bff);
  color: #fff;
  border: none;
  font-weight: 600;
  padding: 0.5rem 1.1rem;
  box-shadow: 0 10px 24px rgba(80, 112, 255, 0.25);
}
.stButton button:hover, .stDownloadButton button:hover {
  transform: translateY(-1px);
  box-shadow: 0 14px 28px rgba(80, 112, 255, 0.3);
}
.stSelectbox, .stNumberInput, .stTextInput, .stDateInput, .stSlider {
  background: transparent;
}
input, textarea, select {
  background: var(--surface-2) !important;
  color: var(--text) !important;
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] { border-radius: var(--radius); border: 1px solid var(--border); }
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


def parse_watchlist(raw: str) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[,\s]+", raw.upper().strip())
    return [p for p in dict.fromkeys(parts) if p]


def has_alpaca_keys() -> bool:
    try:
        k = str(st.secrets.get("ALPACA_KEY", "")).strip()
        s = str(st.secrets.get("ALPACA_SECRET", "")).strip()
    except Exception:
        return False
    return bool(k and s)


def tail_for_plot(df: pd.DataFrame, n: int) -> pd.DataFrame:
    n = int(max(PLOT_TAIL_MIN_BARS, n))
    return df.tail(n) if len(df) > n else df


def bt_params_signature(d: dict[str, Any]) -> str:
    def _val(v: Any) -> str:
        if isinstance(v, float):
            if np.isnan(v):
                return "nan"
            return f"{v:.8g}"
        return str(v)

    return "|".join([f"{k}={_val(v)}" for k, v in sorted(d.items(), key=lambda kv: kv[0])])


# =============================================================================
# Data preparation
# =============================================================================
def ensure_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common OHLCV column name variants to: open, high, low, close, volume.
    Keeps other columns intact.
    """
    if df is None or getattr(df, "empty", True):
        return df

    out = df.copy()
    cols_lower = {str(c).lower(): str(c) for c in out.columns}

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
        found_original: Optional[str] = None
        for k in keys:
            if k in cols_lower:
                found_original = cols_lower[k]
                break
        if found_original is not None:
            mapping[found_original] = want

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

    # If we have a timestamp column, use it.
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
        return out

    # If already datetime index, normalize to UTC
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index()
        out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
        return out

    # Otherwise leave unchanged
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
@st.cache_data(ttl=CACHE_TTL_ALPACA_SEC, show_spinner=False)
def cached_load_alpaca(symbol: str, force_refresh: int) -> pd.DataFrame:
    try:
        api_key = str(st.secrets.get("ALPACA_KEY", "")).strip()
        sec_key = str(st.secrets.get("ALPACA_SECRET", "")).strip()
    except Exception as exc:
        raise RuntimeError("Missing Alpaca keys in Streamlit secrets.") from exc
    if not api_key or not sec_key:
        raise RuntimeError("Missing Alpaca keys in Streamlit secrets.")
    df, _dbg = load_historical(symbol, api_key, sec_key, force_refresh=force_refresh)
    return df


def _normalize_yahoo_history_to_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance .history() returns a DataFrame indexed by Date/Datetime.
    This function reliably produces a 'timestamp' column without the common reset_index pitfalls.
    """
    if df is None or df.empty:
        raise RuntimeError("Yahoo returned no data.")

    hist = df.copy()

    # yfinance typically uses DatetimeIndex; keep it, then reset to a proper column.
    if not isinstance(hist.index, pd.DatetimeIndex):
        # Try to coerce index to datetime if possible
        try:
            hist.index = pd.to_datetime(hist.index)
        except Exception:
            pass

    # Reset index into a column and discover its name
    hist = hist.reset_index()
    # Common names: 'Date', 'Datetime', or 'index'
    idx_col = None
    for cand in ["Datetime", "Date", "index"]:
        if cand in hist.columns:
            idx_col = cand
            break
    # If none found, try the first column if it looks datetime-like
    if idx_col is None and len(hist.columns) > 0:
        first = hist.columns[0]
        try:
            pd.to_datetime(hist[first], errors="raise")
            idx_col = first
        except Exception:
            idx_col = None

    if idx_col is None:
        raise RuntimeError("Could not determine timestamp column from Yahoo history.")

    hist = hist.rename(columns={idx_col: "timestamp"})

    # Lowercase OHLCV columns (but keep 'timestamp' as-is)
    ren = {}
    for c in hist.columns:
        if c != "timestamp":
            ren[c] = str(c).lower()
    hist = hist.rename(columns=ren)

    # Coerce timestamp
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True, errors="coerce")
    hist = hist.dropna(subset=["timestamp"]).sort_values("timestamp")

    return hist


@st.cache_data(ttl=CACHE_TTL_YAHOO_HIST_SEC, show_spinner=False)
def cached_load_yahoo(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance not installed.")
    t = yf.Ticker(symbol)
    raw = t.history(period=period, interval=interval, auto_adjust=False)
    return _normalize_yahoo_history_to_timestamp(raw)


@st.cache_data(ttl=CACHE_TTL_YAHOO_PRICE_SEC, show_spinner=False)
def cached_current_price_yahoo(symbol: str) -> Tuple[float, str]:
    """Best-effort last price from Yahoo (often delayed)."""
    if not YF_AVAILABLE:
        return np.nan, "Unavailable (yfinance not installed)"

    try:
        t = yf.Ticker(symbol)

        fi = getattr(t, "fast_info", None)
        if fi and isinstance(fi, dict):
            px = fi.get("last_price") or fi.get("lastPrice") or fi.get("regularMarketPrice")
            px = safe_float(px)
            if np.isfinite(px) and px > 0:
                return float(px), "Yahoo (fast_info, may be delayed)"

        info = getattr(t, "info", None)
        if isinstance(info, dict):
            px = safe_float(info.get("regularMarketPrice"))
            if np.isfinite(px) and px > 0:
                return float(px), "Yahoo (info, may be delayed)"

        intr = t.history(period="1d", interval="1m")
        if intr is not None and not intr.empty:
            last = safe_float(intr["Close"].iloc[-1] if "Close" in intr.columns else np.nan)
            if np.isfinite(last) and last > 0:
                return float(last), "Yahoo (1m, may be delayed)"
    except Exception:
        pass

    return np.nan, "Unavailable"


@st.cache_data(ttl=CACHE_TTL_YAHOO_META_SEC, show_spinner=False)
def cached_market_cap_yahoo(symbol: str) -> float:
    if not YF_AVAILABLE:
        return np.nan

    try:
        t = yf.Ticker(symbol)

        fi = getattr(t, "fast_info", None)
        if fi and isinstance(fi, dict):
            mc = safe_float(fi.get("market_cap") or fi.get("marketCap"))
            if np.isfinite(mc) and mc > 0:
                return float(mc)

        info = getattr(t, "info", None)
        if isinstance(info, dict):
            mc = safe_float(info.get("marketCap"))
            if np.isfinite(mc) and mc > 0:
                return float(mc)
    except Exception:
        return np.nan

    return np.nan


def _strip_html(text: str) -> str:
    cleaned = html.unescape(text or "")
    while "<" in cleaned and ">" in cleaned:
        start = cleaned.find("<")
        end = cleaned.find(">", start + 1)
        if start == -1 or end == -1:
            break
        cleaned = cleaned[:start] + cleaned[end + 1 :]
    return " ".join(cleaned.split())


def fetch_yahoo_rss_news(symbol: str) -> list[dict[str, Any]]:
    if not symbol:
        return []

    base = "https://feeds.finance.yahoo.com/rss/2.0/headline"
    query = urllib.parse.urlencode({"s": symbol, "region": "US", "lang": "en-US"})
    url = f"{base}?{query}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
    except Exception:
        return []

    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(raw)
    except Exception:
        return []

    items = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or "Untitled"
        link = item.findtext("link")
        pub = item.findtext("pubDate")
        summary = item.findtext("description") or ""

        ts_val: Optional[int] = None
        if pub:
            try:
                dt = parsedate_to_datetime(pub)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ts_val = int(dt.timestamp())
            except Exception:
                ts_val = None

        items.append(
            {
                "title": title,
                "link": link,
                "publisher": "Yahoo Finance",
                "providerPublishTime": ts_val,
                "summary": _strip_html(summary),
            }
        )

    return items


@st.cache_data(ttl=CACHE_TTL_YAHOO_NEWS_SEC, show_spinner=False)
def cached_news_yahoo(symbol: str) -> list[dict[str, Any]]:
    if not YF_AVAILABLE:
        return fetch_yahoo_rss_news(symbol)

    try:
        t = yf.Ticker(symbol)
        items = getattr(t, "news", None)
        if isinstance(items, list):
            if items:
                return items
    except Exception:
        return fetch_yahoo_rss_news(symbol)

    return fetch_yahoo_rss_news(symbol)


def format_news_timestamp(ts: Any) -> str:
    try:
        dt = pd.to_datetime(ts, unit="s", utc=True)
    except Exception:
        try:
            dt = pd.to_datetime(ts, utc=True)
        except Exception:
            return "Unknown time"
    if pd.isna(dt):
        return "Unknown time"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def load_and_prepare(symbol: str, *, force_refresh: int) -> None:
    st.session_state["load_error"] = None
    st.session_state["ind_error"] = None
    st.session_state["data_source"] = None

    df: Optional[pd.DataFrame] = None
    err_primary: Optional[str] = None

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

    # Normalize columns
    df = ensure_ohlcv_cols(df)
    df = coerce_ohlcv_numeric(df)

    st.session_state["df_raw"] = df
    st.session_state["last_symbol"] = symbol
    st.session_state["last_loaded_at"] = pd.Timestamp.utcnow()

    # Internal sanity checks (kept out of UI)
    try:
        st.session_state["sanity"] = sanity_check_bars(df)
    except Exception:
        st.session_state["sanity"] = None

    # Indicators
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
def compute_support_resistance(df_ind: pd.DataFrame, lookback: int) -> Tuple[float, float]:
    """Robust S/R: percentiles so a single wick doesn't dominate."""
    lb = int(max(SR_MIN_LOOKBACK, lookback))
    tail = df_ind.tail(lb)
    if tail.empty or ("low" not in tail.columns) or ("high" not in tail.columns):
        return np.nan, np.nan

    lows = pd.to_numeric(tail["low"], errors="coerce").dropna()
    highs = pd.to_numeric(tail["high"], errors="coerce").dropna()
    if lows.empty or highs.empty:
        return np.nan, np.nan

    support = float(np.nanpercentile(lows.values, SR_SUPPORT_PCTL))
    resistance = float(np.nanpercentile(highs.values, SR_RESIST_PCTL))
    return support, resistance


def compute_trade_plan(
    df_ind: pd.DataFrame,
    *,
    mode: str,
    atr_entry: float,
    atr_stop: float,
    atr_target: float,
) -> dict[str, float]:
    """Simple long-only plan (beginner-friendly)."""
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

    return {
        "entry": float(entry),
        "stop": float(stop),
        "target": float(target),
        "rr": float(rr) if np.isfinite(rr) else np.nan,
    }


def score_breakout_candidate(
    df_ind: pd.DataFrame,
    *,
    lookback: int,
) -> dict[str, Any]:
    if df_ind is None or df_ind.empty:
        return {
            "score": 0,
            "label": "Low",
            "close": np.nan,
            "rsi": np.nan,
            "rvol": np.nan,
            "adx": np.nan,
            "trend": "Unknown",
            "dist_to_res_pct": np.nan,
            "support": np.nan,
            "resistance": np.nan,
        }

    last = df_ind.iloc[-1]
    close = safe_float(last.get("close", np.nan))
    rsi = safe_float(last.get("rsi14", np.nan))
    rvol = safe_float(last.get("rvol", np.nan))
    adx = safe_float(last.get("adx14", np.nan))
    trend = str(last.get("trend_state", "Unknown"))

    support, resistance = compute_support_resistance(df_ind, lookback)
    dist_to_res = (resistance - close) / close if np.isfinite(resistance) and np.isfinite(close) and close > 0 else np.nan

    score = 0
    if trend == "Up":
        score += 35
    elif trend == "Mixed":
        score += 15

    if np.isfinite(dist_to_res):
        if dist_to_res <= 0.03:
            score += 25
        elif dist_to_res <= 0.07:
            score += 15

    if np.isfinite(rvol) and rvol >= 1.2:
        score += 15

    if np.isfinite(rsi) and 45 <= rsi <= 70:
        score += 15

    if np.isfinite(adx) and adx >= 20:
        score += 10

    if score >= 70:
        label = "Breakout"
    elif score >= 50:
        label = "Watch"
    else:
        label = "Low"

    return {
        "score": int(score),
        "label": label,
        "close": close,
        "rsi": rsi,
        "rvol": rvol,
        "adx": adx,
        "trend": trend,
        "dist_to_res_pct": dist_to_res * 100 if np.isfinite(dist_to_res) else np.nan,
        "support": support,
        "resistance": resistance,
    }


@st.cache_data(ttl=15 * 60, show_spinner=False)
def screen_breakout_watchlist(
    symbols: tuple[str, ...],
    *,
    max_market_cap_b: float,
    lookback: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not symbols or not YF_AVAILABLE:
        return pd.DataFrame(rows)

    max_market_cap = float(max_market_cap_b) * 1e9

    for sym in symbols:
        try:
            df = cached_load_yahoo(sym, period="1y", interval="1d")
        except Exception:
            continue

        if df is None or df.empty:
            continue

        df = ensure_ohlcv_cols(df)
        df = coerce_ohlcv_numeric(df)
        try:
            add_indicators_inplace(df)
        except Exception:
            continue

        mc = cached_market_cap_yahoo(sym)
        if np.isfinite(mc) and mc > max_market_cap:
            continue

        metrics = score_breakout_candidate(df, lookback=lookback)
        rows.append(
            {
                "Symbol": sym,
                "Market Cap (B)": (mc / 1e9) if np.isfinite(mc) else np.nan,
                "Close": metrics["close"],
                "To Resistance (%)": metrics["dist_to_res_pct"],
                "RSI(14)": metrics["rsi"],
                "RVOL": metrics["rvol"],
                "ADX(14)": metrics["adx"],
                "Trend": metrics["trend"],
                "Score": metrics["score"],
                "Label": metrics["label"],
            }
        )

    if not rows:
        return pd.DataFrame(rows)

    df_out = pd.DataFrame(rows)
    return df_out.sort_values(by=["Score", "Symbol"], ascending=[False, True]).reset_index(drop=True)


def pick_random_breakout_idea(
    symbols: list[str],
    *,
    max_market_cap_b: float,
    lookback: int,
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    results = screen_breakout_watchlist(tuple(symbols), max_market_cap_b=max_market_cap_b, lookback=lookback)
    if results is None or results.empty:
        return None, None

    candidates = results[results["Label"].isin(["Breakout", "Watch"])]
    source = candidates if not candidates.empty else results
    pick_symbol = random.choice(source["Symbol"].tolist())
    pick_row = source.loc[source["Symbol"] == pick_symbol].iloc[0].to_dict()
    return pick_symbol, pick_row


# =============================================================================
# Plots
# =============================================================================
def extract_markers(trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades is None or getattr(trades, "empty", True):
        return pd.DataFrame(columns=["ts", "px"]), pd.DataFrame(columns=["ts", "px"])

    t = trades.copy()
    if "entry_ts" in t.columns:
        t["entry_ts"] = pd.to_datetime(t["entry_ts"], errors="coerce", utc=True)
    if "exit_ts" in t.columns:
        t["exit_ts"] = pd.to_datetime(t["exit_ts"], errors="coerce", utc=True)

    entries = pd.DataFrame(
        {
            "ts": t.get("entry_ts", pd.Series(dtype="datetime64[ns, UTC]")),
            "px": pd.to_numeric(t.get("entry_px", pd.Series(dtype=float)), errors="coerce"),
        }
    ).dropna(subset=["ts", "px"])

    exits = pd.DataFrame(
        {
            "ts": t.get("exit_ts", pd.Series(dtype="datetime64[ns, UTC]")),
            "px": pd.to_numeric(t.get("exit_px", pd.Series(dtype=float)), errors="coerce"),
        }
    ).dropna(subset=["ts", "px"])

    return entries, exits


def plot_price(
    df: pd.DataFrame,
    symbol: str,
    trades: Optional[pd.DataFrame] = None,
    *,
    support: Optional[float] = None,
    resistance: Optional[float] = None,
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
    hlines: Optional[list[float]] = None,
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
# Signals
# =============================================================================
@dataclass(frozen=True)
class SignalScore:
    label: str
    score: int
    summary: str
    reasons: list[str]


@dataclass(frozen=True)
class OpportunityScore:
    label: str
    score: int
    timeframe: str
    summary: str
    reasons: list[str]


def compute_signal_score(df_ind: pd.DataFrame, rsi_min: float, rsi_max: float, rvol_min: float, vol_max: float) -> SignalScore:
    need = ["close", "ma50", "ma200", "rsi14", "rvol", "vol_ann", "atr14"]
    missing = [c for c in need if c not in df_ind.columns]
    if missing:
        return SignalScore("WAIT", 0, "Indicators not ready.", [f"Missing: {', '.join(missing[:6])}"])

    if "ind_ready" in df_ind.columns and not bool(df_ind["ind_ready"].iloc[-1]):
        return SignalScore(
            "WAIT",
            0,
            "Waiting for enough history.",
            ["Indicators not ready (need more bars for stable values)."],
        )

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
    score = SCORE_BASE

    uptrend = (close > ma50 > ma200)
    downtrend = (close < ma50 < ma200)
    if uptrend:
        score += SCORE_TREND_BONUS
        reasons.append("Uptrend (close > MA50 > MA200)")
    elif downtrend:
        score -= SCORE_TREND_BONUS
        reasons.append("Downtrend (close < MA50 < MA200)")
    else:
        reasons.append("Trend mixed")

    if rsi < rsi_min:
        score -= SCORE_RSI_BAD
        reasons.append(f"RSI low ({rsi:.1f})")
    elif rsi > rsi_max:
        score -= SCORE_RSI_BAD
        reasons.append(f"RSI high ({rsi:.1f})")
    else:
        score += SCORE_RSI_GOOD
        reasons.append(f"RSI ok ({rsi:.1f})")

    if rvol < rvol_min:
        score -= SCORE_RVOL_BAD
        reasons.append(f"RVOL low ({rvol:.2f})")
    else:
        score += SCORE_RVOL_GOOD
        reasons.append(f"RVOL ok ({rvol:.2f})")

    if vol_ann > vol_max:
        score -= SCORE_VOL_BAD
        reasons.append(f"Vol high ({vol_ann:.2f})")
    else:
        score += SCORE_VOL_GOOD
        reasons.append(f"Vol ok ({vol_ann:.2f})")

    score = int(np.clip(score, 0, 100))

    # Clear beginner-friendly label logic:
    # - BUY requires both a high score and an uptrend
    # - AVOID requires low score and downtrend
    # - Otherwise HOLD (neutral/mixed)
    if score >= SCORE_BUY_MIN and uptrend:
        return SignalScore("BUY", score, "Favorable trend + filters supportive.", reasons)
    if score <= SCORE_AVOID_MAX and downtrend:
        return SignalScore("AVOID", score, "Bearish conditions dominate (avoid long).", reasons)
    return SignalScore("HOLD", score, "Mixed/neutral conditions.", reasons)


def compute_opportunity_score(
    df_ind: pd.DataFrame,
    *,
    news_score: int,
    market_cap_m: float,
    float_m: float,
    short_interest: float,
    catalyst_days: int,
) -> OpportunityScore:
    need = ["close", "ma50", "ma200", "rvol", "vol_ann", "atr14"]
    missing = [c for c in need if c not in df_ind.columns]
    if missing:
        return OpportunityScore("WAIT", 0, "Unknown", "Indicators not ready.", [f"Missing: {', '.join(missing[:6])}"])

    if "ind_ready" in df_ind.columns and not bool(df_ind["ind_ready"].iloc[-1]):
        return OpportunityScore(
            "WAIT",
            0,
            "Unknown",
            "Waiting for enough history.",
            ["Indicators not ready (need more bars for stable values)."],
        )

    last = df_ind.iloc[-1]
    close = safe_float(last.get("close"))
    ma50 = safe_float(last.get("ma50"))
    ma200 = safe_float(last.get("ma200"))
    rvol = safe_float(last.get("rvol"))
    vol_ann = safe_float(last.get("vol_ann"))
    atr = safe_float(last.get("atr14"))

    vals = np.array([close, ma50, ma200, rvol, vol_ann, atr], dtype="float64")
    if not np.isfinite(vals).all():
        return OpportunityScore("WAIT", 0, "Unknown", "Waiting for enough history.", ["Non-finite indicator values"])

    reasons: list[str] = []
    score = 45

    uptrend = close > ma50 > ma200
    if uptrend:
        score += 12
        reasons.append("Price in uptrend (close > MA50 > MA200)")
    else:
        reasons.append("Trend not fully aligned")

    if rvol >= 1.5:
        score += 10
        reasons.append(f"Unusual volume (RVOL {rvol:.2f})")
    elif rvol >= 1.1:
        score += 5
        reasons.append(f"Above-average volume (RVOL {rvol:.2f})")
    else:
        reasons.append(f"Quiet volume (RVOL {rvol:.2f})")

    if vol_ann >= 0.6:
        score += 8
        reasons.append(f"High volatility (ann. vol {vol_ann:.2f})")
    elif vol_ann >= 0.35:
        score += 4
        reasons.append(f"Moderate volatility (ann. vol {vol_ann:.2f})")
    else:
        reasons.append(f"Low volatility (ann. vol {vol_ann:.2f})")

    if market_cap_m <= 300:
        score += 12
        reasons.append("Micro/small cap (market cap ≤ $300M)")
    elif market_cap_m <= 1000:
        score += 6
        reasons.append("Small cap (market cap ≤ $1B)")
    else:
        reasons.append("Larger cap (market cap > $1B)")

    if float_m <= 50:
        score += 8
        reasons.append("Low float (≤ 50M shares)")
    elif float_m <= 150:
        score += 4
        reasons.append("Moderate float (≤ 150M shares)")
    else:
        reasons.append("Higher float (> 150M shares)")

    if short_interest >= 15:
        score += 8
        reasons.append(f"Elevated short interest ({short_interest:.1f}%)")
    elif short_interest >= 7:
        score += 4
        reasons.append(f"Moderate short interest ({short_interest:.1f}%)")
    else:
        reasons.append(f"Low short interest ({short_interest:.1f}%)")

    if news_score >= 30:
        score += 12
        reasons.append("Strong positive news tone")
    elif news_score >= 10:
        score += 6
        reasons.append("Mild positive news tone")
    elif news_score <= -20:
        score -= 8
        reasons.append("Negative news tone")
    else:
        reasons.append("Neutral news tone")

    score = int(np.clip(score, 0, 100))

    timeframe = "3-6 months"
    if catalyst_days <= 7:
        timeframe = "1-3 weeks"
    elif catalyst_days <= 30:
        timeframe = "1-2 months"
    elif catalyst_days <= 90:
        timeframe = "1-3 months"

    if news_score >= 40 and rvol >= 2.0:
        timeframe = "Days to 2 weeks"

    if score >= 75:
        label = "HIGH"
        summary = "Speculative opportunity setup with multiple catalysts."
    elif score >= 55:
        label = "WATCH"
        summary = "Some ingredients are present; needs confirmation."
    else:
        label = "LOW"
        summary = "Limited near-term opportunity signals."

    return OpportunityScore(label, score, timeframe, summary, reasons)


def get_latest_price(symbol: str, df_chart: pd.DataFrame) -> Tuple[float, str]:
    """Best-effort latest available price: Yahoo (cached) else last close."""
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
        # Backtest UI state
        "use_cash_ledger": True,
        "gate_ui": "soft",
        "sizing_ui": "Fixed £ per trade",
        "invest_amount": 25.0,
        "horizon_bars": 20,
        "show_news": True,
        "include_news_tone": True,
        "op_watchlist": "SOFI, PLTR, IONQ, RKLB, LCID",
        "op_max_mcap_b": 2.0,
        "op_sr_lookback": 60,
        "op_scan_results": None,
    }
)


# =============================================================================
# Header
# =============================================================================
st.title(APP_TITLE)
st.caption(APP_CAPTION)


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

    with st.expander("Account", expanded=False):
        account_capital = st.number_input(
            "Starting account (£)",
            min_value=1.0,
            max_value=100_000_000.0,
            value=float(ss_get("account_capital", 100.0)),
            step=1.0,
            format="%.2f",
            help="Used for backtest starting equity. Does not place real trades.",
        )
        mark_to_market = st.toggle("Mark-to-market equity (smoother curve)", value=bool(ss_get("mark_to_market", False)))

    with st.expander("News", expanded=False):
        show_news = st.toggle("Show news panel", value=bool(ss_get("show_news", True)))

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
                "account_capital": float(account_capital),
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
                    "account_capital",
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
    "sr_lookback": sr_lookback,
    "account_capital": account_capital,
    "mark_to_market": mark_to_market,
    "show_news": show_news,
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

df_plot = tail_for_plot(df_chart, PLOT_TAIL_DEFAULT_BARS)

src = st.session_state.get("data_source")
if src:
    st.caption(f"Data source: **{src}**")

sanity = st.session_state.get("sanity")
if isinstance(sanity, dict) and sanity.get("warnings"):
    with st.expander("Data quality checks", expanded=False):
        for warning in sanity.get("warnings", []):
            st.write(f"• {warning}")
        stats = sanity.get("stats", {})
        if stats:
            st.caption(f"Rows: {stats.get('rows')} | Start: {stats.get('start')} | End: {stats.get('end')}")

# Precompute S/R once per rerun (used in multiple tabs)
support_level, resistance_level = compute_support_resistance(df_chart, int(sr_lookback))


# =============================================================================
# Tabs
# =============================================================================
tab_labels = ["✅ Signal", "🚀 Opportunity", "📊 Charts", "🧪 Backtest"]
show_news_tab = bool(ss_get("show_news", True))
if show_news_tab:
    tab_labels.append("🗞️ News")
tabs = st.tabs(tab_labels)
tab_signal, tab_opportunity, tab_charts, tab_backtest = tabs[:4]
tab_news = tabs[4] if show_news_tab else None


# =============================================================================
# Signal tab
# =============================================================================
with tab_signal:
    st.subheader(f"{symbol} — Signal")
    st.caption("Signals require enough history for indicators (early bars may show WAIT).")

    latest_px, latest_src = get_latest_price(symbol, df_chart)
    score = compute_signal_score(df_chart, float(rsi_min), float(rsi_max), float(rvol_min), float(vol_max))

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
            st.caption("Note: This is a heuristic score, not a guaranteed prediction.")

        with b:
            st.metric("Latest available price", f"{latest_px:.2f}" if np.isfinite(latest_px) else "—")
            st.caption(f"Source: {latest_src}")

        with c:
            st.metric("Support", f"{support_level:.2f}" if np.isfinite(support_level) else "—")
            st.metric("Resistance", f"{resistance_level:.2f}" if np.isfinite(resistance_level) else "—")

    st.markdown("### Where to place support / resistance (simple guidance)")
    box = st.container(border=True)
    with box:
        if np.isfinite(support_level) and np.isfinite(resistance_level):
            st.write(f"• **Support zone** ~ **{support_level:.2f}** → many traders place a stop a little *below* this.")
            st.write(f"• **Resistance zone** ~ **{resistance_level:.2f}** → many traders take profit or get cautious near this.")
            st.caption("These are percentile-based levels from the lookback window (less sensitive to single wicks).")
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
        st.caption(f"Mode: {mode_label} • Long-only plan")
        st.caption("Important: Backtests can differ from this snapshot plan because fills and gaps vary day-to-day.")

    with st.expander("Why this score?", expanded=False):
        for r in score.reasons[:12]:
            st.write(f"• {r}")
        st.caption("Label rules: BUY requires both a high score and an uptrend; AVOID requires low score and downtrend.")


# =============================================================================
# Opportunity tab
# =============================================================================
with tab_opportunity:
    st.subheader(f"{symbol} — Opportunity radar")
    st.caption("One random small-cap breakout idea plus a quick opportunity score.")

    st.markdown("### Random breakout idea")
    pick_btn = st.button("New random idea", use_container_width=True)
    if not YF_AVAILABLE:
        st.info("Random breakout ideas need the yfinance package.")
    else:
        if pick_btn or not st.session_state.get("op_pick_symbol"):
            pick_symbol, pick_row = pick_random_breakout_idea(
                SMALL_CAP_POOL,
                max_market_cap_b=DEFAULT_OP_MAX_MCAP_B,
                lookback=DEFAULT_OP_SR_LOOKBACK,
            )
            st.session_state["op_pick_symbol"] = pick_symbol
            st.session_state["op_pick_row"] = pick_row

        pick_symbol = st.session_state.get("op_pick_symbol")
        pick_row = st.session_state.get("op_pick_row") or {}
        if not pick_symbol or not pick_row:
            st.warning("No eligible breakout ideas found right now. Try again later.")
        else:
            label = str(pick_row.get("Label", "Low"))
            score = int(pick_row.get("Score", 0))
            badge = st.container(border=True)
            with badge:
                if label == "Breakout":
                    st.success(f"**{pick_symbol}** • Breakout candidate • Score {score}/100")
                elif label == "Watch":
                    st.warning(f"**{pick_symbol}** • Watchlist candidate • Score {score}/100")
                else:
                    st.info(f"**{pick_symbol}** • Low momentum • Score {score}/100")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Trend", pick_row.get("Trend", "—"))
                dist = safe_float(pick_row.get("To Resistance (%)"))
                c2.metric("To resistance", f"{dist:.1f}%" if np.isfinite(dist) else "—")
                close = safe_float(pick_row.get("Close"))
                c3.metric("Last close", f"{close:.2f}" if np.isfinite(close) else "—")
                rsi = safe_float(pick_row.get("RSI(14)"))
                c4.metric("RSI", f"{rsi:.0f}" if np.isfinite(rsi) else "—")

                rvol = safe_float(pick_row.get("RVOL"))
                adx = safe_float(pick_row.get("ADX(14)"))
                st.caption(
                    "Quick read: "
                    f"RVOL {rvol:.2f} • ADX {adx:.0f}"
                    if np.isfinite(rvol) and np.isfinite(adx)
                    else "Quick read: RVOL — • ADX —"
                )

    st.divider()

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("### Quick inputs")
        market_cap_default = float(ss_get("op_market_cap_m", 300.0))
        if YF_AVAILABLE:
            mc_est = cached_market_cap_yahoo(symbol)
            if np.isfinite(mc_est):
                market_cap_default = float(mc_est / 1e6)

        st.metric("Market cap (est.)", f"{market_cap_default:,.0f}M" if np.isfinite(market_cap_default) else "—")
        short_interest = st.slider(
            "Short interest (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(ss_get("op_short_interest", 8.0)),
            step=0.5,
        )
        catalyst_days = st.number_input(
            "Next catalyst (days)",
            min_value=1,
            max_value=365,
            value=int(ss_get("op_catalyst_days", 30)),
            step=1,
        )
        with st.expander("Optional inputs", expanded=False):
            market_cap_m = st.number_input(
                "Market cap (USD, millions)",
                min_value=10.0,
                max_value=20000.0,
                value=market_cap_default,
                step=10.0,
            )
            float_m = st.number_input(
                "Float (millions of shares)",
                min_value=1.0,
                max_value=2000.0,
                value=float(ss_get("op_float_m", 60.0)),
                step=5.0,
            )
            include_news_tone = st.toggle("Include news tone", value=bool(ss_get("include_news_tone", True)))
            st.session_state["include_news_tone"] = bool(include_news_tone)
            if include_news_tone:
                news_score = st.slider(
                    "News tone (subjective)",
                    min_value=-100,
                    max_value=100,
                    value=int(ss_get("op_news_score", 15)),
                    step=5,
                )
            else:
                news_score = 0
                st.caption("News tone set to neutral.")

        if "market_cap_m" not in locals():
            market_cap_m = market_cap_default
        if "float_m" not in locals():
            float_m = float(ss_get("op_float_m", 60.0))
        if "news_score" not in locals():
            news_score = int(ss_get("op_news_score", 15))

        st.session_state["op_market_cap_m"] = float(market_cap_m)
        st.session_state["op_float_m"] = float(float_m)
        st.session_state["op_short_interest"] = float(short_interest)
        st.session_state["op_catalyst_days"] = int(catalyst_days)
        st.session_state["op_news_score"] = int(news_score)

    with right:
        st.markdown("### Opportunity score")
        opp = compute_opportunity_score(
            df_chart,
            news_score=int(news_score),
            market_cap_m=float(market_cap_m),
            float_m=float(float_m),
            short_interest=float(short_interest),
            catalyst_days=int(catalyst_days),
        )

        badge = st.container(border=True)
        with badge:
            if opp.label == "HIGH":
                st.success(f"**HIGH** • Score {opp.score}/100")
            elif opp.label == "WATCH":
                st.warning(f"**WATCH** • Score {opp.score}/100")
            else:
                st.info(f"**LOW** • Score {opp.score}/100")
            st.caption(opp.summary)

        st.metric("Estimated timeframe", opp.timeframe)
        st.caption("Shorter timeframes imply higher volatility and higher risk.")

        with st.expander("Why this score?", expanded=False):
            for r in opp.reasons[:12]:
                st.write(f"• {r}")
            st.caption("Opportunity scores are heuristic and do not guarantee outcomes.")


# =============================================================================
# Charts tab
# =============================================================================
with tab_charts:
    st.subheader(f"{symbol} — Charts")

    trades_for_markers = st.session_state.get("bt_trades")

    st.plotly_chart(
        plot_price(df_plot, symbol, trades=trades_for_markers, support=support_level, resistance=resistance_level),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        if "rsi14" in df_plot.columns:
            st.plotly_chart(
                plot_indicator(df_plot, "rsi14", "RSI(14)", hlines=[30, 70], ymin=0, ymax=100),
                use_container_width=True,
            )
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
# Backtest tab
# =============================================================================
with tab_backtest:
    st.subheader("Backtest")

    # Clean, aligned "controls row" using a form
    with st.form("bt_form", clear_on_submit=False):
        c1, c2, c3, c4, c5 = st.columns(
            [1.25, 1.15, 1.25, 1.35, 1.6],
            vertical_alignment="bottom",
        )

        horizon_bars = c1.number_input("Max hold (bars)", 1, 200, int(ss_get("horizon_bars", 20)), 1)
        st.session_state["horizon_bars"] = int(horizon_bars)

        use_cash_ledger = c2.toggle(
            "Realistic cash",
            value=bool(ss_get("use_cash_ledger", True)),
            help="If ON: you can’t buy shares you can’t afford (recommended for small accounts).",
        )
        st.session_state["use_cash_ledger"] = bool(use_cash_ledger)

        gate_ui = c3.selectbox(
            "Probability gating",
            ["soft", "hard", "off"],
            index={"soft": 0, "hard": 1, "off": 2}.get(str(ss_get("gate_ui", "soft")), 0),
            help="Soft = take weaker setups smaller. Hard = skip weak setups. Off = no gating.",
        )
        st.session_state["gate_ui"] = gate_ui

        sizing_ui = c4.selectbox(
            "Position sizing",
            ["Fixed £ per trade", "% of account"],
            index=0 if str(ss_get("sizing_ui", "Fixed £ per trade")) == "Fixed £ per trade" else 1,
            help="Fixed £ is easiest: each trade uses about the same money.",
        )
        st.session_state["sizing_ui"] = sizing_ui

        run_backtest_btn = c5.form_submit_button("🧪 Run Backtest", use_container_width=True)

    # Extra controls (outside the form so the row stays aligned)
    if sizing_ui == "Fixed £ per trade":
        invest_amount = st.number_input(
            "Amount invested per trade (£)",
            min_value=1.0,
            value=float(ss_get("invest_amount", 25.0)),
            step=1.0,
            format="%.2f",
            help="If this is too small to buy 1 share, trades may be skipped.",
        )
        st.session_state["invest_amount"] = float(invest_amount)
        sizing_mode = "fixed_amount"

        # Conservative default: keep risk_pct moderate (still applied in your backtester logic)
        risk_pct = 0.02
        max_alloc_pct_frac = 1.0  # not used in fixed_amount mode

        # Helpful affordability hint for beginners (approx)
        # Use latest available price as a rough estimate, but remind it's approximate.
        latest_px, _latest_src = get_latest_price(symbol, df_chart)
        if np.isfinite(latest_px) and latest_px > 0:
            min_needed = float(latest_px)
            if invest_amount < min_needed:
                st.info(
                    f"At ~{latest_px:.2f} per share, you may need about **£{min_needed:.2f}** to buy 1 share. "
                    "If your per-trade amount is smaller, some trades can be skipped."
                )
    else:
        invest_amount = float(ss_get("invest_amount", 25.0))
        sizing_mode = "percent"

        risk_per_trade_pct = st.number_input(
            "Risk per trade (%)",
            min_value=0.1,
            max_value=10.0,
            value=float(ss_get("risk_per_trade_pct", 1.0)),
            step=0.1,
            format="%.1f",
            help="How much of your account you’re willing to lose if the stop is hit.",
        )
        st.session_state["risk_per_trade_pct"] = float(risk_per_trade_pct)

        max_alloc_pct_ui = st.number_input(
            "Max allocation (%)",
            min_value=1.0,
            max_value=100.0,
            value=float(ss_get("max_alloc_pct", 10.0)),
            step=1.0,
            format="%.0f",
            help="Caps the position size as a % of account.",
        )
        st.session_state["max_alloc_pct"] = float(max_alloc_pct_ui)

        risk_pct = float(risk_per_trade_pct) / 100.0
        max_alloc_pct_frac = float(max_alloc_pct_ui) / 100.0

    prob_gating = (gate_ui != "off")
    gate_mode = "soft" if gate_ui == "soft" else "hard"

    # Small visible execution box (so assumptions aren’t “hidden”)
    exec_box = st.container(border=True)
    with exec_box:
        st.markdown("**Execution assumptions (affects results)**")
        a, b, c = st.columns(3)
        a.metric("Slippage", f"{DEFAULT_SLIPPAGE_BPS:.1f} bps")
        b.metric("Spread", f"{DEFAULT_SPREAD_BPS:.1f} bps")
        c.metric("Commission", f"£{DEFAULT_COMMISSION:.2f} / order")
        st.caption("These are simplified costs. Real fills can be better or worse depending on liquidity and volatility.")
        st.caption("Probability gating is applied only after an initial in-sample window to reduce look-ahead bias.")

    # Build params for your backtester
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
        assumed_spread_bps=float(DEFAULT_SPREAD_BPS),
        start_equity=float(account_capital),

        # execution realism
        slippage_bps=float(DEFAULT_SLIPPAGE_BPS),
        commission_per_order=float(DEFAULT_COMMISSION),
        spread_mode="taker_only",
        exit_priority="stop_first",
        time_exit_price="open",

        # sizing
        enable_position_sizing=True,
        sizing_mode=str(sizing_mode),
        invest_amount=float(invest_amount),
        risk_pct=float(risk_pct),
        max_alloc_pct=float(max_alloc_pct_frac),
        min_risk_per_share=1e-6,

        # realism for small accounts
        use_cash_ledger=bool(use_cash_ledger),
        allow_margin=False,

        # display
        mark_to_market=bool(mark_to_market),

        # gating
        prob_gating=bool(prob_gating),
        prob_is_frac=float(DEFAULT_PROB_IS_FRAC),
        prob_min=float(DEFAULT_PROB_MIN),
        min_bucket_trades=int(DEFAULT_MIN_BUCKET_TRADES),
        min_avg_r=float(DEFAULT_MIN_AVG_R),
        gate_mode=str(gate_mode),
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

        st.markdown("### Performance")
        pwrap = st.container(border=True)
        with pwrap:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Trades", f"{ntr}")
            m2.metric("Win rate", f"{results.get('win_rate', float('nan')):.1%}" if np.isfinite(results.get("win_rate", np.nan)) else "—")
            m3.metric("Total return", f"{results.get('total_return', float('nan')):.1%}" if np.isfinite(results.get("total_return", np.nan)) else "—")
            m4.metric("Max drawdown", f"{results.get('max_drawdown', float('nan')):.1%}" if np.isfinite(results.get("max_drawdown", np.nan)) else "—")
            m5.metric("Sharpe", f"{results.get('sharpe', float('nan')):.2f}" if np.isfinite(results.get("sharpe", np.nan)) else "—")

        with st.expander("How to interpret this", expanded=False):
            st.write(
                "- **Trades**: too few trades = stats can be noisy.\n"
                "- **Win rate**: you can still lose money with a high win rate if losses are larger than wins.\n"
                "- **Total return**: result over this historical period with these settings.\n"
                "- **Max drawdown**: worst peak-to-trough drop. Lower is generally better.\n"
                "- **Sharpe**: risk-adjusted return. Higher is better, but not very meaningful with few trades.\n"
                "- **Fixed £ per trade**: if £ is too small to buy 1 share, trades may be skipped.\n"
                "- **Realistic cash**: ON means the backtest won’t magically buy what you can’t afford.\n"
                "- **Execution assumptions**: slippage/spread/commission can materially change results."
            )

        warnings = results.get("warnings", [])
        if warnings:
            with st.expander("Backtest warnings", expanded=False):
                for w in warnings:
                    st.write(f"• {w}")

        if isinstance(df_bt, pd.DataFrame) and (not df_bt.empty) and ("equity" in df_bt.columns):
            eq = pd.to_numeric(df_bt["equity"], errors="coerce").dropna()
            if len(eq) > 2:
                ts_series = df_bt.get("timestamp", None)
                ts = pd.to_datetime(ts_series, errors="coerce", utc=True) if ts_series is not None else None
                x = np.arange(len(eq)) if (ts is None or (isinstance(ts, pd.Series) and ts.isna().all())) else ts.iloc[-len(eq):]

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

        if not isinstance(trades, pd.DataFrame) or trades.empty:
            st.warning("Backtest completed but produced no trades.")
            st.caption("Tip: try a longer lookback period (data), adjust filters (RSI/RVOL/Vol), or use a different ticker.")
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
# News tab
# =============================================================================
if tab_news is not None:
    with tab_news:
        st.subheader(f"{symbol} — News")
        if not YF_AVAILABLE:
            st.info("Yahoo news requires the yfinance package.")
        else:
            with st.spinner("Loading news..."):
                items = cached_news_yahoo(symbol)

            if not items:
                st.info("No news items available right now.")
            else:
                for item in items[:15]:
                    title = str(item.get("title", "Untitled"))
                    link = item.get("link") or item.get("url")
                    publisher = item.get("publisher") or item.get("source")
                    ts = format_news_timestamp(item.get("providerPublishTime"))
                    summary = str(item.get("summary", "")).strip()

                    header = f"**{title}**"
                    if link:
                        header = f"**[{title}]({link})**"
                    st.markdown(header)
                    meta_parts = [p for p in [publisher, ts] if p]
                    if meta_parts:
                        st.caption(" • ".join(meta_parts))
                    if summary:
                        st.write(summary)
                    st.divider()
