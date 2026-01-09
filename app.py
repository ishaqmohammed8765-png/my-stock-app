"""
Trading Dashboard (Simple) + RSI + Market Regime + RVOL
Educational only. Not financial advice.

Added "best" improvements (low-UI, high value):
A) Data Integrity Engine (practical + lightweight)
   1) Data sanity checks (OHLC validity, monotonic timestamps, missing/duplicate days)
   2) Split-like jump detection + "Possible Corporate Action" warning banner
   3) Optional multi-source daily-close validator via yfinance (if installed)
      - Flags if |close discrepancy| > threshold (default 0.75%)

B) Execution realism improvements
   4) Dynamic slippage model in backtest based on liquidity (ADV $) and volatility proxy
      - still includes base bps; increases slippage for illiquid names / large position vs ADV
   5) Spread-aware live warning when live quotes exist (wide spread = execution risk)

C) Concentration / correlated risk (optional, minimal UI)
   6) Optional portfolio tickers box (expander): computes average correlation to current ticker
      - If highly correlated, it soft-reduces position sizing (kelly multiplier) + shows warning

Important: Still keeps your UI minimal (only one optional expander).
"""

from __future__ import annotations

import atexit
import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# Optional best UX auto-refresh (install: streamlit-autorefresh)
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    HAS_ST_AUTOR = True
except Exception:
    HAS_ST_AUTOR = False

# Optional secondary-source validator (install: yfinance)
try:
    import yfinance as yf  # type: ignore
    HAS_YF = True
except Exception:
    HAS_YF = False

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

try:
    from alpaca.data.enums import DataFeed  # newer alpaca-py
    HAS_DATAFEED = True
except Exception:
    DataFeed = None  # type: ignore
    HAS_DATAFEED = False


# =========================
# CONFIG
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dash_simple")

TRADING_DAYS = 252
RISK_FREE = 0.045

VOL_FLOOR, VOL_CAP, VOL_DEFAULT = 0.10, 1.50, 0.30
KELLY_MIN, KELLY_MAX = 0.01, 0.15
STOP_MULT = 0.50
MC_SIMS = 1000

QUEUE_MAX = 1500
MAX_TRADES = 200
MAX_QUOTES = 200
MAX_RECONNECTS = 8

MIN_HIST_DAYS = 240  # need room for MA200 + regime
MAX_BT_ITERS = 250

AUTO_REFRESH_MS = 1500

SCORE_STRONG_BUY = 80
SCORE_BUY = 65
SCORE_CONDITIONAL = 50
SCORE_HOLD = 35

# Background defaults (no UI controls)
MC_METHOD = "student_t"
BARS_FEED = "IEX"
WS_FEED = "iex"

MAX_ALLOC_PCT = 0.10  # 10% of equity max position
MAX_RISK_PCT = 0.02   # 2% of equity max risk per trade
BASE_SLIPPAGE_BPS = 5
COMMISSION = 0.0

# Market regime reference ticker (simple default)
REGIME_TICKER_PRIMARY = "QQQ"
REGIME_TICKER_FALLBACK = "SPY"

# Backtest strategy threshold (aligns with your score meanings)
BT_MIN_SCORE = SCORE_CONDITIONAL  # trade only when >= 50

# Data validator thresholds
YF_CLOSE_DIFF_PCT = 0.0075  # 0.75%
WIDE_SPREAD_BPS_WARN = 25   # warn if spread > 25 bps

# Concentration risk
CORR_LOOKBACK_DAYS = 120
CORR_WARN = 0.70            # avg corr above this => warning
KELLY_CORR_PENALTY_MAX = 0.45  # up to 45% size reduction if very correlated


# =========================
# MODELS
# =========================
@dataclass(frozen=True)
class TradeMetrics:
    price: float
    vol: float
    atr: float
    ma50: float
    ma200: float
    buy: float
    sell: float
    stop: float

    prob_touch_buy: float
    prob_touch_sell: float

    rrr: float
    kelly: float
    qty: int
    expected_move: float
    risk_per_share: float
    cap_alloc_qty: int
    cap_risk_qty: int

    rsi14: float
    rvol: float

    # Data / execution diagnostics
    adv_dollar: float
    est_slippage_bps: float


@dataclass(frozen=True)
class BacktestResults:
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    total_pnl: float


# =========================
# HELPERS
# =========================
def validate_keys(k: str, s: str) -> bool:
    return bool(k and s and len(k) >= 10 and len(s) >= 10)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(ts: Any) -> datetime:
    try:
        if ts is None:
            return utc_now()
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        if hasattr(ts, "to_pydatetime"):
            dt = ts.to_pydatetime()
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return utc_now()


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def mid_from_quote(q: Optional[Dict[str, Any]]) -> Optional[float]:
    if not q:
        return None
    try:
        bid, ask = float(q.get("bid", 0)), float(q.get("ask", 0))
    except Exception:
        return None
    if bid <= 0 or ask <= 0:
        return None
    return (bid + ask) / 2.0


def drain(q: queue.Queue, dest: List[Dict[str, Any]], maxlen: int) -> None:
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        dest.append(item)
        if len(dest) > maxlen:
            del dest[: len(dest) - maxlen]


def is_rate_limit_error(e: Exception) -> bool:
    s = str(e).lower()
    return ("429" in s) or ("too many requests" in s) or ("rate limit" in s)


def backoff_seconds(attempt: int) -> float:
    return float(min(2 ** attempt, 30))


def live_autorefresh(interval_ms: int, key: str = "live_refresh") -> None:
    """Best-effort auto-refresh without sleep-based rerun loops."""
    if HAS_ST_AUTOR:
        st_autorefresh(interval=interval_ms, key=key)
    else:
        st.caption("Tip: install `streamlit-autorefresh` for smoother live updates. Manual refresh is fine.")


def last_completed_index(df: pd.DataFrame) -> int:
    """
    Alpaca daily bars can include today's partial bar.
    If last bar date is today (UTC), use previous bar for RSI/RVOL stability.
    """
    if df is None or df.empty:
        return -1
    try:
        ts = df["timestamp"].iloc[-1]
        dt = parse_ts(ts)
        if dt.date() >= datetime.utcnow().date() and len(df) >= 2:
            return -2
    except Exception:
        pass
    return -1


def align_to_timestamp(df: pd.DataFrame, ts: Any) -> int:
    """Return the last index in df with timestamp <= ts. If none, return -1."""
    if df is None or df.empty:
        return -1
    t = parse_ts(ts)
    try:
        s = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        idxs = np.nonzero((s <= pd.Timestamp(t)).to_numpy())[0]
        return int(idxs.max()) if len(idxs) else -1
    except Exception:
        return -1


# =========================
# DATA INTEGRITY ENGINE
# =========================
def sanity_check_bars(df: pd.DataFrame) -> List[str]:
    """Lightweight OHLCV sanity checks; returns list of warning strings."""
    warns: List[str] = []
    if df is None or df.empty:
        return ["No historical data loaded."]

    # Timestamps monotonic?
    try:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if ts.isna().any():
            warns.append("Some timestamps could not be parsed.")
        if not ts.is_monotonic_increasing:
            warns.append("Timestamps are not strictly increasing (data ordering issue).")
        if ts.duplicated().any():
            warns.append("Duplicate timestamps detected (possible duplicate bars).")
    except Exception:
        warns.append("Could not validate timestamp integrity.")

    # OHLC validity
    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")

    if (o <= 0).any() or (h <= 0).any() or (l <= 0).any() or (c <= 0).any():
        warns.append("Non-positive OHLC values detected (bad ticks or corporate action mismatch).")
    if (v < 0).any():
        warns.append("Negative volume detected (data error).")

    # High/Low consistency
    bad_h = (h < np.maximum(o, c)).sum()
    bad_l = (l > np.minimum(o, c)).sum()
    if bad_h > 0 or bad_l > 0:
        warns.append("OHLC inconsistency detected (high/low not enclosing open/close).")

    # Missing trading days heuristic (doesn't know exchange holidays; uses median gap)
    try:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
        if len(ts) > 10:
            gaps = ts.diff().dt.days.dropna()
            if gaps.median() > 2:
                warns.append("Large gaps between bars detected (missing days / sparse feed).")
    except Exception:
        pass

    return warns


def detect_split_like_events(df: pd.DataFrame, threshold_low: float = 0.45, threshold_high: float = 0.55) -> List[str]:
    """
    Detect split-like jumps where close ratio is near 0.5, 2.0, 3.0, 1/3, 1/4.
    This is heuristic; returns human-readable warnings.
    """
    warns: List[str] = []
    if df is None or df.empty or len(df) < 5:
        return warns

    c = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(c) < 5:
        return warns

    r = (c / c.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return warns

    # check for ratios near common split factors
    targets = [0.5, 2.0, 3.0, 0.3333, 4.0, 0.25]
    for t in targets:
        near = (r - t).abs() <= 0.03  # within ~3%
        if near.any():
            idxs = list(r.index[near][-3:])  # last few
            for ix in idxs:
                dt = parse_ts(df.loc[ix, "timestamp"]).date()
                warns.append(f"Possible corporate action near {dt}: close ratio ≈ {r.loc[ix]:.2f} (split/dividend adjustment mismatch).")
            break

    return warns


@st.cache_data(ttl=3600, show_spinner=False)
def yf_daily_close(ticker: str, period_days: int = 400) -> Optional[pd.Series]:
    """Fetch daily closes from yfinance (if available). Cached."""
    if not HAS_YF:
        return None
    t = (ticker or "").upper().strip()
    if not t:
        return None
    try:
        df = yf.download(t, period=f"{period_days}d", interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index, utc=True)
        return s.dropna()
    except Exception:
        return None


def validate_price_vs_yf(ticker: str, our_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Compare last completed daily close vs yfinance last daily close.
    Returns: (our_close, yf_close, warning_str)
    """
    if not HAS_YF or our_df is None or our_df.empty:
        return None, None, None

    i = last_completed_index(our_df)
    if i == -1:
        our_close = float(pd.to_numeric(our_df["close"], errors="coerce").iloc[-1])
        our_ts = parse_ts(our_df["timestamp"].iloc[-1])
    else:
        our_close = float(pd.to_numeric(our_df["close"], errors="coerce").iloc[i])
        our_ts = parse_ts(our_df["timestamp"].iloc[i])

    yf_close_series = yf_daily_close(ticker)
    if yf_close_series is None or yf_close_series.empty:
        return our_close, None, None

    # Align by date (UTC date)
    our_date = pd.Timestamp(our_ts).date()
    yf_same = yf_close_series[yf_close_series.index.date == our_date]
    if yf_same.empty:
        # fallback to last available yf close
        yf_close = float(yf_close_series.iloc[-1])
    else:
        yf_close = float(yf_same.iloc[-1])

    if our_close <= 0 or yf_close <= 0:
        return our_close, yf_close, None

    diff_pct = abs(our_close - yf_close) / ((our_close + yf_close) / 2)
    if diff_pct > YF_CLOSE_DIFF_PCT:
        warn = f"Data Warning: Daily close differs vs Yahoo by {diff_pct*100:.2f}% (our {our_close:.2f} vs YF {yf_close:.2f}). Check adjustments/feed."
        return our_close, yf_close, warn

    return our_close, yf_close, None


# =========================
# LIVE STREAM
# =========================
class RealtimeStream:
    def __init__(self, api_key: str, secret_key: str, ticker: str,
                 trade_q: queue.Queue, quote_q: queue.Queue, feed: str = WS_FEED):
        self.api_key = api_key
        self.secret_key = secret_key
        self.ticker = ticker.upper().strip()
        self.trade_q = trade_q
        self.quote_q = quote_q
        self.feed = feed

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._stream: Optional[StockDataStream] = None
        self.reconnects = 0

    async def on_trade(self, data: Any) -> None:
        try:
            px = float(getattr(data, "price", 0.0))
            sz = int(getattr(data, "size", 0))
            if px <= 0 or sz <= 0:
                return
            payload = {
                "type": "trade",
                "ts": parse_ts(getattr(data, "timestamp", None)),
                "symbol": getattr(data, "symbol", self.ticker),
                "price": px,
                "size": sz,
            }
            try:
                self.trade_q.put_nowait(payload)
            except queue.Full:
                try:
                    self.trade_q.get_nowait()
                    self.trade_q.put_nowait(payload)
                except Exception:
                    pass
        except Exception:
            log.exception("trade handler error")

    async def on_quote(self, data: Any) -> None:
        try:
            bid = float(getattr(data, "bid_price", 0.0))
            ask = float(getattr(data, "ask_price", 0.0))
            if bid <= 0 or ask <= 0:
                return
            payload = {
                "type": "quote",
                "ts": parse_ts(getattr(data, "timestamp", None)),
                "symbol": getattr(data, "symbol", self.ticker),
                "bid": bid,
                "ask": ask,
                "bid_size": int(getattr(data, "bid_size", 0)),
                "ask_size": int(getattr(data, "ask_size", 0)),
                "spread": float(ask - bid),
            }
            try:
                self.quote_q.put_nowait(payload)
            except queue.Full:
                try:
                    self.quote_q.get_nowait()
                    self.quote_q.put_nowait(payload)
                except Exception:
                    pass
        except Exception:
            log.exception("quote handler error")

    def run(self) -> None:
        while not self._stop.is_set() and self.reconnects < MAX_RECONNECTS:
            try:
                self._stream = StockDataStream(self.api_key, self.secret_key, feed=self.feed)
                self._stream.subscribe_trades(self.on_trade, self.ticker)
                self._stream.subscribe_quotes(self.on_quote, self.ticker)
                self._stream.run()
                self.reconnects = 0
            except Exception:
                self.reconnects += 1
                log.exception("websocket error (reconnect %s)", self.reconnects)
                time.sleep(min(2 ** self.reconnects, 60))

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            if self._stream is not None:
                try:
                    self._stream.stop()
                except Exception:
                    try:
                        self._stream.stop_ws()
                    except Exception:
                        pass
                self._stream = None


# =========================
# ALPACA LOADERS
# =========================
def _bars_to_df(bars_obj: Any) -> Optional[pd.DataFrame]:
    df = getattr(bars_obj, "df", None)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    data = getattr(bars_obj, "data", None)
    if not data:
        return None
    rows = []
    for sym, bars in data.items():
        for b in bars:
            rows.append({
                "symbol": sym,
                "timestamp": getattr(b, "timestamp", None) or getattr(b, "t", None),
                "open": getattr(b, "open", None) or getattr(b, "o", None),
                "high": getattr(b, "high", None) or getattr(b, "h", None),
                "low": getattr(b, "low", None) or getattr(b, "l", None),
                "close": getattr(b, "close", None) or getattr(b, "c", None),
                "volume": getattr(b, "volume", None) or getattr(b, "v", None),
            })
    out = pd.DataFrame(rows)
    return out if not out.empty else None


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_historical(ticker: str, api_key: str, secret_key: str,
                    days_back: int = 720) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    dbg: Dict[str, Any] = {"ticker": ticker, "feed": BARS_FEED, "steps": []}

    ticker = (ticker or "").upper().strip()
    if not validate_keys(api_key, secret_key) or not ticker:
        dbg["error"] = "Invalid keys or ticker"
        return None, dbg

    client = StockHistoricalDataClient(api_key, secret_key)
    base = dict(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame.Day,
        start=datetime.utcnow() - timedelta(days=days_back),
        end=datetime.utcnow(),
    )

    feed_val = None
    if HAS_DATAFEED:
        feed_val = DataFeed.SIP if BARS_FEED.upper() == "SIP" else DataFeed.IEX

    try:
        if feed_val is not None:
            req = StockBarsRequest(**base, feed=feed_val)
            dbg["steps"].append("bars request WITH feed")
        else:
            req = StockBarsRequest(**base)
            dbg["steps"].append("bars request WITHOUT feed")
    except TypeError as e:
        dbg["steps"].append(f"feed not supported building request -> {e}")
        req = StockBarsRequest(**base)

    bars = None
    for attempt in range(6):
        try:
            bars = client.get_stock_bars(req)
            break
        except Exception as e:
            if is_rate_limit_error(e):
                wait = backoff_seconds(attempt)
                dbg["steps"].append(f"429 rate limit -> sleep {wait:.0f}s")
                time.sleep(wait)
                continue
            dbg["error"] = f"Exception: {e}"
            return None, dbg

    if bars is None:
        dbg["error"] = "Failed to fetch bars"
        return None, dbg

    raw = _bars_to_df(bars)
    if raw is None or raw.empty:
        dbg["error"] = "Empty bars response"
        return None, dbg

    df = raw
    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels >= 2:
        try:
            df = df.xs(ticker, level=0).reset_index()
            dbg["steps"].append("MultiIndex xs(symbol)")
        except Exception:
            df = df.reset_index()
            dbg["steps"].append("MultiIndex reset fallback")

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df = df[df["symbol"] == ticker].copy()

    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        dbg["error"] = "Missing required columns"
        dbg["columns"] = list(df.columns)
        return None, dbg

    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) < MIN_HIST_DAYS:
        dbg["error"] = f"Too few rows ({len(df)})"
        return None, dbg

    dbg["rows"] = int(len(df))
    return df, dbg


@st.cache_data(ttl=10, show_spinner=False)
def load_latest_quote(ticker: str, api_key: str, secret_key: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    dbg: Dict[str, Any] = {"ticker": ticker}
    ticker = (ticker or "").upper().strip()
    if not validate_keys(api_key, secret_key) or not ticker:
        dbg["error"] = "Invalid keys or ticker"
        return None, dbg
    try:
        client = StockHistoricalDataClient(api_key, secret_key)
        req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        resp = client.get_stock_latest_quote(req)
        quotes = getattr(resp, "data", resp)
        q = quotes.get(ticker) if hasattr(quotes, "get") else quotes[ticker]
        bid = float(getattr(q, "bid_price", 0.0))
        ask = float(getattr(q, "ask_price", 0.0))
        if bid <= 0 or ask <= 0:
            dbg["error"] = "Invalid bid/ask"
            return None, dbg
        return {
            "type": "quote",
            "ts": utc_now(),
            "symbol": ticker,
            "bid": bid,
            "ask": ask,
            "bid_size": int(getattr(q, "bid_size", 0)),
            "ask_size": int(getattr(q, "ask_size", 0)),
            "spread": float(ask - bid),
        }, {"ok": True, **dbg}
    except Exception as e:
        dbg["error"] = str(e)
        return None, dbg


# =========================
# ANALYTICS
# =========================
def annual_vol(df: pd.DataFrame, span: int = 20) -> float:
    if df is None or df.empty or len(df) < 12:
        return VOL_DEFAULT
    c = pd.to_numeric(df["close"], errors="coerce")
    r = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 12:
        return VOL_DEFAULT
    v = r.ewm(span=span).std().iloc[-1] * np.sqrt(TRADING_DAYS)
    if not np.isfinite(v):
        return VOL_DEFAULT
    return clamp(float(v), VOL_FLOOR, VOL_CAP)


def atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty or len(df) < period + 1:
        return 0.0
    x = df.copy()
    x["h_l"] = x["high"] - x["low"]
    x["h_pc"] = (x["high"] - x["close"].shift(1)).abs()
    x["l_pc"] = (x["low"] - x["close"].shift(1)).abs()
    x["tr"] = x[["h_l", "h_pc", "l_pc"]].max(axis=1)
    v = x["tr"].rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else 0.0


def expected_move(price: float, vol: float, days: int) -> float:
    if price <= 0 or vol <= 0 or days <= 0:
        return 0.0
    return float(price * vol * np.sqrt(days / TRADING_DAYS))


def prob_hit_mc(S: float, K: float, vol: float, days: int, sims: int, method: str) -> float:
    """Probability a path ever touches K within 'days' steps."""
    if S <= 0 or K <= 0 or vol <= 0 or days <= 0 or sims <= 0:
        return 0.0

    dt = 1 / TRADING_DAYS
    try:
        if method == "student_t":
            dof = 5
            shocks = stats.t.rvs(df=dof, size=(sims, days)) / np.sqrt(dof / (dof - 2))
        else:
            shocks = np.random.standard_normal((sims, days))

        drift = -0.5 * (vol ** 2) * dt
        diffusion = vol * np.sqrt(dt) * shocks
        paths = S * np.exp(np.cumsum(drift + diffusion, axis=1))

        hit = np.any(paths >= K, axis=1) if K >= S else np.any(paths <= K, axis=1)
        return float(hit.mean())
    except Exception:
        return 0.0


def sharpe(returns: np.ndarray) -> float:
    if returns is None or len(returns) == 0:
        return 0.0
    sd = returns.std()
    if sd == 0:
        return 0.0
    excess = returns - (RISK_FREE / TRADING_DAYS)
    s = (excess.mean() / sd) * np.sqrt(TRADING_DAYS)
    return float(s) if np.isfinite(s) else 0.0


def max_drawdown(cumret: np.ndarray) -> float:
    if cumret is None or len(cumret) == 0:
        return 0.0
    peak = np.maximum.accumulate(cumret)
    dd = (cumret - peak) / (peak + 1e-12)
    m = float(dd.min())
    return m if np.isfinite(m) else 0.0


def rsi14_from_close(df: pd.DataFrame) -> float:
    """RSI(14) using Wilder-like smoothing."""
    if df is None or df.empty or len(df) < 20:
        return 50.0
    closes = pd.to_numeric(df["close"], errors="coerce").dropna().values
    if len(closes) < 20:
        return 50.0
    if last_completed_index(df) == -2 and len(closes) >= 2:
        closes = closes[:-1]
    if len(closes) < 20:
        return 50.0

    delta = np.diff(closes)
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)

    period = 14
    roll_up = pd.Series(up).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down).ewm(alpha=1/period, adjust=False).mean()

    rs = roll_up.iloc[-1] / (roll_down.iloc[-1] + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    if not np.isfinite(rsi):
        return 50.0
    return float(clamp(rsi, 0.0, 100.0))


def rvol_20(df: pd.DataFrame) -> float:
    """Relative volume: last completed day volume / 20-day avg volume."""
    if df is None or df.empty or len(df) < 25:
        return 1.0
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    if last_completed_index(df) == -2:
        last_vol = float(vol.iloc[-2])
        avg = float(vol.iloc[-22:-2].mean()) if len(vol) >= 22 else float(vol.mean())
    else:
        last_vol = float(vol.iloc[-1])
        avg = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float(vol.mean())
    if avg <= 0:
        return 1.0
    return float(max(0.0, last_vol / avg))


def market_regime_at(index_df: Optional[pd.DataFrame], ts: Any) -> Tuple[bool, float, float]:
    """(risk_on, close, ma200) as of timestamp ts."""
    if index_df is None or index_df.empty or len(index_df) < 210:
        return True, 0.0, 0.0
    idx = align_to_timestamp(index_df, ts)
    if idx < 209:
        return True, 0.0, 0.0
    closes = pd.to_numeric(index_df["close"], errors="coerce").ffill()
    close = float(closes.iloc[idx])
    ma200 = float(closes.rolling(200).mean().iloc[idx])
    if not np.isfinite(ma200) or ma200 <= 0:
        return True, close, ma200
    return (close >= ma200), close, ma200


def bt_stats(bt: pd.DataFrame) -> BacktestResults:
    if bt is None or bt.empty:
        return BacktestResults(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    wins = int((bt["PnL"] > 0).sum())
    losses = int((bt["PnL"] < 0).sum())
    total = int(len(bt))

    win_rate = (wins / total * 100.0) if total else 0.0
    gross_win = float(bt.loc[bt["PnL"] > 0, "PnL"].sum()) if wins else 0.0
    gross_loss = float(abs(bt.loc[bt["PnL"] < 0, "PnL"].sum())) if losses else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else 0.0

    avg_win = float(bt.loc[bt["PnL"] > 0, "PnL"].mean()) if wins else 0.0
    avg_loss = float(bt.loc[bt["PnL"] < 0, "PnL"].mean()) if losses else 0.0

    mdd = max_drawdown(bt["CumReturn"].values)
    sh = sharpe(bt["Return"].values)
    total_pnl = float(bt["PnL"].sum())

    return BacktestResults(win_rate, pf, mdd, sh, total, wins, losses, avg_win, avg_loss, total_pnl)


def kelly_from_bt(bt: pd.DataFrame, min_trades: int = 30) -> float:
    """Half-Kelly from strategy trade distribution; clamped."""
    if bt is None or bt.empty or len(bt) < min_trades:
        return KELLY_MIN
    wins = bt[bt["PnL"] > 0]
    losses = bt[bt["PnL"] < 0]
    if len(wins) < max(8, min_trades // 4) or len(losses) < max(8, min_trades // 4):
        return KELLY_MIN
    wr = len(wins) / len(bt)
    aw = float(wins["Return"].mean())
    al = float(abs(losses["Return"].mean()))
    if al <= 0 or aw <= 0:
        return KELLY_MIN
    R = aw / al
    f = (wr - ((1 - wr) / R)) * 0.5
    return clamp(float(f), KELLY_MIN, KELLY_MAX)


def adv_dollar_volume(df: pd.DataFrame, lookback: int = 20) -> float:
    """Approximate average daily dollar volume over lookback (close * volume)."""
    if df is None or df.empty or len(df) < lookback + 2:
        return 0.0
    x = df.copy()
    i = last_completed_index(x)
    if i == -2:
        x = x.iloc[:-1]
    x = x.tail(lookback)
    c = pd.to_numeric(x["close"], errors="coerce")
    v = pd.to_numeric(x["volume"], errors="coerce")
    dv = (c * v).replace([np.inf, -np.inf], np.nan).dropna()
    if dv.empty:
        return 0.0
    return float(dv.mean())


def estimate_slippage_bps(base_bps: float, order_value: float, adv_dollar: float, vol: float) -> float:
    """
    Simple impact model:
    - base + impact * (order_value / adv$) + volatility component
    """
    if order_value <= 0:
        return float(base_bps)
    adv = max(adv_dollar, 1.0)
    participation = order_value / adv  # fraction of daily dollar volume
    # tuned to be conservative but not insane
    impact = 8_000.0 * participation   # e.g., 0.1% participation => 8 bps
    vol_component = 15.0 * max(0.0, min(vol, 1.0))  # up to ~15 bps for high vol
    est = base_bps + impact + vol_component
    return float(clamp(est, base_bps, 150.0))


# =========================
# OPTIONAL CONCENTRATION / CORRELATION
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def load_multi_historical(tickers: Tuple[str, ...], api_key: str, secret_key: str) -> Dict[str, pd.DataFrame]:
    """Load historical data for multiple tickers; cached; best-effort."""
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df, _ = load_historical(t, api_key, secret_key)
        if df is not None and not df.empty:
            out[t] = df
    return out


def parse_portfolio_tickers(text: str) -> List[str]:
    """Parse comma/space/newline separated tickers; uppercase; unique."""
    if not text:
        return []
    raw = (
        text.replace("\n", ",")
            .replace(" ", ",")
            .replace(";", ",")
            .split(",")
    )
    toks = []
    for r in raw:
        t = r.strip().upper()
        if t and t.isalnum():
            toks.append(t)
    # unique preserve order
    seen = set()
    uniq = []
    for t in toks:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq[:25]


def avg_corr_to_portfolio(cur: str, port: List[str], data: Dict[str, pd.DataFrame]) -> Optional[float]:
    """Compute avg correlation of cur returns vs portfolio tickers over lookback window."""
    if cur not in data or not port:
        return None
    cur_df = data[cur]
    cur_r = pd.to_numeric(cur_df["close"], errors="coerce").pct_change().dropna().tail(CORR_LOOKBACK_DAYS)
    if cur_r.empty:
        return None
    cors = []
    for t in port:
        if t == cur:
            continue
        if t not in data:
            continue
        r = pd.to_numeric(data[t]["close"], errors="coerce").pct_change().dropna().tail(CORR_LOOKBACK_DAYS)
        if r.empty:
            continue
        # align
        a = pd.concat([cur_r, r], axis=1).dropna()
        if len(a) < 40:
            continue
        c = float(a.iloc[:, 0].corr(a.iloc[:, 1]))
        if np.isfinite(c):
            cors.append(c)
    if not cors:
        return None
    return float(np.mean(cors))


def corr_kelly_multiplier(avg_corr: Optional[float]) -> float:
    """
    Soft position size penalty based on correlation.
    If avg_corr <= 0.3 => no penalty.
    If avg_corr >= 0.9 => max penalty.
    """
    if avg_corr is None or not np.isfinite(avg_corr):
        return 1.0
    c = clamp(avg_corr, -1.0, 1.0)
    if c <= 0.30:
        return 1.0
    # scale [0.3..0.9] -> [0..1]
    x = (c - 0.30) / (0.90 - 0.30)
    penalty = KELLY_CORR_PENALTY_MAX * clamp(x, 0.0, 1.0)
    return float(1.0 - penalty)


# =========================
# SCORING
# =========================
def score_decision(
    price: float,
    ma50: float,
    ma200: float,
    rrr: float,
    sh: float,
    mdd: float,
    wr: float,
    pf: float,
    rsi14: float,
    rvol: float,
    risk_on: bool,
    regime_ticker: str
) -> Tuple[int, str, str, List[str]]:
    score = 0
    reasons: List[str] = []

    # Trend (15)
    if price > ma200 and price > ma50:
        score += 15; reasons.append("Uptrend: above MA50 & MA200")
    elif price > ma200:
        score += 10; reasons.append("Trend: above MA200")
    elif price > ma50:
        score += 5; reasons.append("Trend: above MA50 only")
    else:
        reasons.append("Trend: below MA50 & MA200")

    # RRR (25)
    if rrr >= 3.0:
        score += 25; reasons.append("Risk/Reward: excellent")
    elif rrr >= 2.0:
        score += 20; reasons.append("Risk/Reward: strong")
    elif rrr >= 1.5:
        score += 12; reasons.append("Risk/Reward: acceptable")
    else:
        reasons.append("Risk/Reward: poor")

    # Sharpe (25)
    if sh >= 1.5:
        score += 25; reasons.append("Backtest Sharpe: excellent")
    elif sh >= 1.0:
        score += 20; reasons.append("Backtest Sharpe: good")
    elif sh >= 0.5:
        score += 10; reasons.append("Backtest Sharpe: moderate")
    else:
        reasons.append("Backtest Sharpe: poor")

    # Drawdown (15)
    if abs(mdd) <= 0.10:
        score += 15; reasons.append("Drawdown: low")
    elif abs(mdd) <= 0.20:
        score += 10; reasons.append("Drawdown: moderate")
    elif abs(mdd) <= 0.30:
        score += 5; reasons.append("Drawdown: high")
    else:
        reasons.append("Drawdown: very high")

    # Win rate (10)
    if wr >= 60:
        score += 10; reasons.append("Win rate: high")
    elif wr >= 50:
        score += 7; reasons.append("Win rate: positive")
    elif wr >= 40:
        score += 3; reasons.append("Win rate: below avg")
    else:
        reasons.append("Win rate: low")

    # Profit factor (10)
    if pf >= 2.0:
        score += 10; reasons.append("Profit factor: strong")
    elif pf >= 1.5:
        score += 7; reasons.append("Profit factor: good")
    elif pf >= 1.0:
        score += 3; reasons.append("Profit factor: marginal")
    else:
        reasons.append("Profit factor: weak")

    # Market regime
    if risk_on:
        reasons.append(f"Market regime: {regime_ticker} risk-on (>= MA200)")
        score += 5
    else:
        reasons.append(f"Market regime: {regime_ticker} risk-off (< MA200) — downgraded")
        score -= 15

    # RSI (soft)
    if rsi14 >= 80:
        score -= 15; reasons.append(f"RSI {rsi14:.0f}: very overbought (penalty)")
    elif rsi14 >= 70:
        score -= 8; reasons.append(f"RSI {rsi14:.0f}: overbought (penalty)")
    elif rsi14 <= 20:
        score += 6; reasons.append(f"RSI {rsi14:.0f}: very oversold (small boost)")
    elif rsi14 <= 30:
        score += 3; reasons.append(f"RSI {rsi14:.0f}: oversold (small boost)")
    else:
        reasons.append(f"RSI {rsi14:.0f}: neutral")

    # RVOL
    if rvol >= 1.5:
        score += 6; reasons.append(f"RVOL {rvol:.2f}: strong volume confirmation")
    elif rvol >= 1.0:
        score -= 2; reasons.append(f"RVOL {rvol:.2f}: average volume")
    else:
        score -= 6; reasons.append(f"RVOL {rvol:.2f}: weak volume (penalty)")

    score = int(max(0, min(100, score)))

    if score >= SCORE_STRONG_BUY:
        return score, "STRONG BUY", "🟢", reasons
    if score >= SCORE_BUY:
        return score, "BUY", "🟢", reasons
    if score >= SCORE_CONDITIONAL:
        reasons.append("Timing matters — consider confirmation")
        return score, "CONDITIONAL", "🟡", reasons
    if score >= SCORE_HOLD:
        reasons.append("Wait for a cleaner setup")
        return score, "HOLD", "🟡", reasons

    reasons.append("Risk too high vs reward")
    return score, "AVOID", "🔴", reasons


def score_gauge(score: int, label: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"Score — {label}"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, SCORE_HOLD], "color": "rgba(255,0,0,0.20)"},
                {"range": [SCORE_HOLD, SCORE_CONDITIONAL], "color": "rgba(255,165,0,0.20)"},
                {"range": [SCORE_CONDITIONAL, SCORE_BUY], "color": "rgba(255,255,0,0.20)"},
                {"range": [SCORE_BUY, 100], "color": "rgba(0,255,0,0.20)"},
            ],
            "threshold": {"line": {"width": 3}, "value": score},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=10))
    return fig


# =========================
# STRATEGY BACKTEST (aligned)
# =========================
def _gap_aware_fill(open_px: float, level_px: float, is_stop: bool) -> float:
    if is_stop:
        return float(open_px) if open_px <= level_px else float(level_px)
    return float(open_px) if open_px >= level_px else float(level_px)


def entry_score_proxy(px: float, ma50: float, ma200: float, rrr: float, rsi14: float, rvol: float, risk_on: bool) -> int:
    """
    Non-circular score proxy for entry gating (does not use backtest stats).
    Mirrors your scoring weights for trend/rrr/regime/rsi/rvol.
    """
    s = 0
    if px > ma200 and px > ma50:
        s += 15
    elif px > ma200:
        s += 10
    elif px > ma50:
        s += 5

    if rrr >= 3.0:
        s += 25
    elif rrr >= 2.0:
        s += 20
    elif rrr >= 1.5:
        s += 12

    s += 5 if risk_on else -15

    if rsi14 >= 80:
        s -= 15
    elif rsi14 >= 70:
        s -= 8
    elif rsi14 <= 20:
        s += 6
    elif rsi14 <= 30:
        s += 3

    if rvol >= 1.5:
        s += 6
    elif rvol >= 1.0:
        s -= 2
    else:
        s -= 6

    return int(max(0, min(100, s)))


def backtest_strategy(
    df: pd.DataFrame,
    market_df: Optional[pd.DataFrame],
    horizon: int,
    min_score: int = BT_MIN_SCORE,
) -> pd.DataFrame:
    """
    Enters next-day open when entry_score_proxy >= min_score.
    Exits via stop or TP with OHLC + gap-aware fills, else horizon close.
    Uses dynamic slippage estimate based on simulated position size vs ADV$ and volatility.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if len(df) < max(MIN_HIST_DAYS, 220) + horizon + 5:
        return pd.DataFrame()

    # Use only fully completed bars if today's bar is partial
    df_bt = df.iloc[:-1].copy() if last_completed_index(df) == -2 else df.copy()

    iters = min(len(df_bt) - horizon - 3, MAX_BT_ITERS)
    start = max(210, len(df_bt) - horizon - iters - 2)

    trades: List[Dict[str, Any]] = []

    for i in range(iters):
        idx = start + i
        hist = df_bt.iloc[: idx + 1].copy()
        if idx + 1 >= len(df_bt):
            break

        px = float(pd.to_numeric(hist["close"], errors="coerce").iloc[-1])
        if px <= 0:
            continue

        v = annual_vol(hist)
        move = expected_move(px, v, horizon)
        buy = px - move
        sell = px + move
        stop = buy - (move * STOP_MULT)
        risk_per_share = max(buy - stop, 0.01)
        rrr = (sell - buy) / risk_per_share

        rsi14 = rsi14_from_close(hist)
        rvol = rvol_20(hist)
        ma50 = float(pd.to_numeric(hist["close"], errors="coerce").rolling(50).mean().iloc[-1]) if len(hist) >= 50 else px
        ma200 = float(pd.to_numeric(hist["close"], errors="coerce").rolling(200).mean().iloc[-1]) if len(hist) >= 200 else ma50
        risk_on, _, _ = market_regime_at(market_df, hist["timestamp"].iloc[-1])

        s = entry_score_proxy(px, ma50, ma200, rrr, rsi14, rvol, risk_on)
        if s < min_score:
            continue

        entry_open = float(pd.to_numeric(df_bt["open"], errors="coerce").iloc[idx + 1])
        if entry_open <= 0:
            continue

        # Simulate a representative order size for slippage:
        # assume "unit" account value of $10k to estimate slippage impact fairly across time
        acct_sim = 10_000.0
        adv = adv_dollar_volume(hist, lookback=20)
        # assume we'd allocate up to MAX_ALLOC_PCT; order value ~ alloc
        order_value = acct_sim * MAX_ALLOC_PCT
        slip_bps = estimate_slippage_bps(BASE_SLIPPAGE_BPS, order_value, adv, v)
        slip = slip_bps / 10000.0

        entry_eff = entry_open * (1 + slip)

        exit_eff = None
        exit_reason = "TIME"
        window = df_bt.iloc[idx + 1: idx + 1 + horizon]

        for _, bar in window.iterrows():
            o = float(pd.to_numeric(bar["open"], errors="coerce"))
            h = float(pd.to_numeric(bar["high"], errors="coerce"))
            l = float(pd.to_numeric(bar["low"], errors="coerce"))
            c = float(pd.to_numeric(bar["close"], errors="coerce"))
            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                continue

            # Stop first (conservative)
            if l <= stop:
                fill = _gap_aware_fill(o, stop, is_stop=True)
                exit_eff = fill * (1 - slip)
                exit_reason = "STOP"
                break

            if h >= sell:
                fill = _gap_aware_fill(o, sell, is_stop=False)
                exit_eff = fill * (1 - slip)
                exit_reason = "TP"
                break

        if exit_eff is None:
            last_close = float(pd.to_numeric(window["close"], errors="coerce").iloc[-1])
            exit_eff = last_close * (1 - slip)

        pnl = (exit_eff - entry_eff) - COMMISSION
        ret = pnl / entry_eff if entry_eff > 0 else 0.0

        trades.append({
            "EntryTS": parse_ts(df_bt["timestamp"].iloc[idx + 1]),
            "ExitTS": parse_ts(window["timestamp"].iloc[-1]),
            "Entry": entry_eff,
            "Exit": exit_eff,
            "Reason": exit_reason,
            "PnL": pnl,
            "Return": ret,
            "SlipBps": slip_bps,
        })

    if not trades:
        return pd.DataFrame()

    bt = pd.DataFrame(trades)
    bt["CumPnL"] = bt["PnL"].cumsum()
    bt["CumReturn"] = (1 + bt["Return"]).cumprod() - 1
    return bt


def compute_trade_metrics(
    df: pd.DataFrame,
    price: float,
    horizon: int,
    acct: float,
    kelly_f: float,
    corr_mult: float,
) -> TradeMetrics:
    v = annual_vol(df)
    a = atr(df)

    ma50 = float(pd.to_numeric(df["close"], errors="coerce").rolling(50).mean().iloc[-1]) if len(df) >= 50 else price
    ma200 = float(pd.to_numeric(df["close"], errors="coerce").rolling(200).mean().iloc[-1]) if len(df) >= 200 else ma50

    move = expected_move(price, v, horizon)
    buy = price - move
    sell = price + move
    stop = buy - (move * STOP_MULT)

    prob_touch_buy = prob_hit_mc(price, buy, v, horizon, MC_SIMS, MC_METHOD)
    prob_touch_sell = prob_hit_mc(price, sell, v, horizon, MC_SIMS, MC_METHOD)

    risk_per_share = max(buy - stop, 0.01)
    rrr = (sell - buy) / risk_per_share if risk_per_share > 0 else 0.0

    # Concentration penalty applied as a multiplier to Kelly fraction
    k_eff = float(clamp(kelly_f * corr_mult, KELLY_MIN, KELLY_MAX))

    # caps
    cap_alloc_qty = int((acct * MAX_ALLOC_PCT) / max(price, 0.01))
    cap_risk_qty = int((acct * MAX_RISK_PCT) / risk_per_share) if risk_per_share > 0 else 0

    # dynamic slippage estimate (uses expected order value = min(alloc cap, kelly cap))
    adv = adv_dollar_volume(df, lookback=20)
    order_value = min(acct * MAX_ALLOC_PCT, acct * k_eff)
    est_slip_bps = estimate_slippage_bps(BASE_SLIPPAGE_BPS, order_value, adv, v)

    raw_qty = int((acct * k_eff) / risk_per_share) if risk_per_share > 0 else 0
    qty = max(0, min(raw_qty, cap_alloc_qty, cap_risk_qty))

    rsi14 = rsi14_from_close(df)
    rvol = rvol_20(df)

    return TradeMetrics(
        price=price, vol=v, atr=a, ma50=ma50, ma200=ma200,
        buy=buy, sell=sell, stop=stop,
        prob_touch_buy=float(prob_touch_buy),
        prob_touch_sell=float(prob_touch_sell),
        rrr=float(rrr),
        kelly=float(k_eff),
        qty=int(qty),
        expected_move=float(move),
        risk_per_share=float(risk_per_share),
        cap_alloc_qty=int(cap_alloc_qty),
        cap_risk_qty=int(cap_risk_qty),
        rsi14=float(rsi14),
        rvol=float(rvol),
        adv_dollar=float(adv),
        est_slippage_bps=float(est_slip_bps),
    )


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Trading Dashboard (Simple)", page_icon="📈", layout="wide")

def init_state() -> None:
    defaults = {
        "api_ok": False,
        "api_key": "",
        "secret_key": "",
        "ticker": "NVDA",
        "historical": None,
        "market_df": None,
        "market_ticker": REGIME_TICKER_PRIMARY,
        "latest_quote": None,
        "latest_price": None,
        "trade_q": queue.Queue(maxsize=QUEUE_MAX),
        "quote_q": queue.Queue(maxsize=QUEUE_MAX),
        "trade_history": [],
        "quote_history": [],
        "live_active": False,
        "ws_handler": None,
        "ws_thread": None,
        "last_error": "",
        "debug": {},
        "portfolio_text": "",
        "portfolio_tickers": [],
        "avg_corr": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def stop_live() -> None:
    handler = st.session_state.ws_handler
    thread = st.session_state.ws_thread
    if handler:
        try:
            handler.stop()
        except Exception:
            pass
    if thread and thread.is_alive():
        thread.join(timeout=2)
    st.session_state.ws_handler = None
    st.session_state.ws_thread = None
    st.session_state.live_active = False

atexit.register(lambda: stop_live())

st.title("📈 Trading Dashboard")
st.caption("Simple controls. Safe defaults. Added integrity checks + dynamic slippage + optional correlation sizing. Educational only.")

# ---- Secrets
try:
    s_api = st.secrets.get("ALPACA_KEY", "")
    s_sec = st.secrets.get("ALPACA_SECRET", "")
except Exception:
    s_api = ""
    s_sec = ""

if (s_api and s_sec) and not st.session_state.api_ok:
    st.session_state.api_key = s_api
    st.session_state.secret_key = s_sec
    st.session_state.api_ok = True

# ---- API input
if not st.session_state.api_ok:
    st.info("Enter your Alpaca API keys (paper keys are fine).")
    a1, a2 = st.columns(2)
    with a1:
        api_in = st.text_input("ALPACA_KEY", type="password")
    with a2:
        sec_in = st.text_input("ALPACA_SECRET", type="password")
    if st.button("Save Keys", type="primary", use_container_width=True):
        if validate_keys(api_in, sec_in):
            st.session_state.api_key = api_in
            st.session_state.secret_key = sec_in
            st.session_state.api_ok = True
            st.rerun()
        else:
            st.error("Keys look invalid (empty/too short).")
    st.stop()

# ---- Minimal controls
top = st.container(border=True)
with top:
    c1, c2, c3, c4 = st.columns([1.4, 1.1, 1.1, 1.1])
    with c1:
        ticker = st.text_input("Ticker", value=st.session_state.ticker).upper().strip()
    with c2:
        account_size = st.number_input("Account ($)", min_value=1000, max_value=5_000_000, value=10_000, step=1000)
    with c3:
        horizon = st.select_slider("Horizon", options=[7, 14, 30, 60, 90], value=30)
    with c4:
        st.write("")
        load_btn = st.button("Load / Refresh", type="primary", use_container_width=True)

# ---- Optional portfolio expander (correlation sizing)
with st.expander("Optional: Portfolio tickers (for correlation sizing)", expanded=False):
    st.caption("Paste tickers separated by commas/spaces/newlines (e.g., NVDA AMD TSM). Equal-weight assumed. Leave empty to ignore.")
    st.session_state.portfolio_text = st.text_area("Portfolio tickers", value=st.session_state.portfolio_text, height=70)
    st.caption("If current ticker is highly correlated to your portfolio, position sizing is reduced automatically.")

# ---- Load data
if load_btn:
    stop_live()
    st.session_state.last_error = ""
    st.session_state.debug = {}
    st.session_state.avg_corr = None

    # Clear live buffers so old ticker data doesn't linger
    st.session_state.trade_history = []
    st.session_state.quote_history = []
    while not st.session_state.trade_q.empty():
        try:
            st.session_state.trade_q.get_nowait()
        except Exception:
            break
    while not st.session_state.quote_q.empty():
        try:
            st.session_state.quote_q.get_nowait()
        except Exception:
            break

    with st.spinner(f"Loading {ticker} + market regime data..."):
        df, dbg = load_historical(ticker, st.session_state.api_key, st.session_state.secret_key)
        q, qdbg = load_latest_quote(ticker, st.session_state.api_key, st.session_state.secret_key)

        # Market regime data (try QQQ then SPY)
        mkt_ticker = REGIME_TICKER_PRIMARY
        mdf, mdbg = load_historical(mkt_ticker, st.session_state.api_key, st.session_state.secret_key)
        if mdf is None:
            mkt_ticker = REGIME_TICKER_FALLBACK
            mdf, mdbg = load_historical(mkt_ticker, st.session_state.api_key, st.session_state.secret_key)

        st.session_state.debug = {"historical": dbg, "quote": qdbg, "market": {"ticker": mkt_ticker, **mdbg}}

        st.session_state.ticker = ticker
        st.session_state.latest_quote = q
        st.session_state.latest_price = mid_from_quote(q)

        st.session_state.market_df = mdf
        st.session_state.market_ticker = mkt_ticker

        if df is None:
            st.session_state.historical = None
            st.session_state.last_error = str(dbg.get("error") or "Failed to load historical data.")
        else:
            st.session_state.historical = df
            if st.session_state.latest_price is None:
                st.session_state.latest_price = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])

        # Portfolio parse (stored)
        st.session_state.portfolio_tickers = parse_portfolio_tickers(st.session_state.portfolio_text)

    st.rerun()

# ---- Error banner (friendly)
if st.session_state.last_error:
    msg = st.session_state.last_error
    if "429" in msg or "rate" in msg.lower():
        st.warning("Rate limit hit. Wait ~30–60 seconds and press Load / Refresh again.")
    st.error(msg)

df = st.session_state.historical
if df is None:
    st.info("Press **Load / Refresh** to start.")
    st.stop()

# ---- Data integrity warnings
sanity_warns = sanity_check_bars(df)
split_warns = detect_split_like_events(df)
if sanity_warns:
    st.warning("Data Integrity: " + " | ".join(sanity_warns[:3]) + (" ..." if len(sanity_warns) > 3 else ""))
if split_warns:
    st.warning("Corporate Action Check: " + " | ".join(split_warns[:2]) + (" ..." if len(split_warns) > 2 else ""))

# ---- Optional Yahoo validation
our_close, yf_close, yf_warn = validate_price_vs_yf(st.session_state.ticker, df)
if yf_warn:
    st.warning(yf_warn)
elif HAS_YF:
    st.caption("Price validator: Yahoo check OK (within threshold).")
else:
    st.caption("Price validator: install `yfinance` to enable multi-source daily-close validation.")

# ---- Live toggle (simple)
live_box = st.container(border=True)
with live_box:
    l1, l2, l3 = st.columns([1.0, 1.0, 3.0])
    with l1:
        live_btn = st.button("Stop Live" if st.session_state.live_active else "Start Live", use_container_width=True)
    with l2:
        st.caption(f"Feed: {WS_FEED.upper()}")
    with l3:
        st.caption("Live is optional — analysis works without it.")

if live_btn:
    if st.session_state.live_active:
        stop_live()
    else:
        stop_live()
        handler = RealtimeStream(
            st.session_state.api_key,
            st.session_state.secret_key,
            st.session_state.ticker,
            st.session_state.trade_q,
            st.session_state.quote_q,
            feed=WS_FEED,
        )
        t = threading.Thread(target=handler.run, daemon=True)
        t.start()
        st.session_state.ws_handler = handler
        st.session_state.ws_thread = t
        st.session_state.live_active = True
    st.rerun()

# ---- Live heartbeat + auto-refresh
if st.session_state.live_active:
    thr = st.session_state.ws_thread
    if thr is None or (hasattr(thr, "is_alive") and not thr.is_alive()):
        stop_live()
        st.warning("Live feed stopped (connection closed).")
    else:
        live_autorefresh(AUTO_REFRESH_MS, key="live_refresh")

# ---- Drain live data
drain(st.session_state.trade_q, st.session_state.trade_history, MAX_TRADES)
drain(st.session_state.quote_q, st.session_state.quote_history, MAX_QUOTES)

if st.session_state.quote_history:
    st.session_state.latest_quote = st.session_state.quote_history[-1]
    mid = mid_from_quote(st.session_state.latest_quote)
    if mid is not None:
        st.session_state.latest_price = mid

current_price = float(st.session_state.latest_price or float(pd.to_numeric(df["close"], errors="coerce").iloc[-1]))

# ---- Spread warning (execution)
spread_bps = None
if st.session_state.latest_quote:
    q = st.session_state.latest_quote
    bid, ask = float(q.get("bid", 0)), float(q.get("ask", 0))
    if bid > 0 and ask > 0:
        spread = ask - bid
        mid = (ask + bid) / 2
        if mid > 0:
            spread_bps = 10000.0 * spread / mid
            if spread_bps > WIDE_SPREAD_BPS_WARN:
                st.warning(f"Execution Warning: Spread is wide ({spread_bps:.1f} bps). Signals may be unreliable / slippage higher.")

# ---- Backtest aligned to strategy + dynamic slippage
bt = backtest_strategy(
    df=df,
    market_df=st.session_state.market_df,
    horizon=horizon,
    min_score=BT_MIN_SCORE,
)
btr = bt_stats(bt)

# ---- Kelly from backtest trades
kelly_f = kelly_from_bt(bt)

# ---- Correlation sizing (optional)
corr_mult = 1.0
avg_corr = None
port = st.session_state.portfolio_tickers or []
if port:
    tickers = tuple([st.session_state.ticker] + [t for t in port if t != st.session_state.ticker])
    data = load_multi_historical(tickers, st.session_state.api_key, st.session_state.secret_key)
    avg_corr = avg_corr_to_portfolio(st.session_state.ticker, port, data)
    corr_mult = corr_kelly_multiplier(avg_corr)

# ---- Compute trade metrics (uses kelly * corr penalty)
tm = compute_trade_metrics(
    df=df,
    price=current_price,
    horizon=horizon,
    acct=float(account_size),
    kelly_f=kelly_f,
    corr_mult=corr_mult,
)

# ---- Market regime (as of latest completed day)
i_comp = last_completed_index(df)
ts_latest = df["timestamp"].iloc[i_comp] if i_comp in (-1, -2) else df["timestamp"].iloc[-1]
regime_ticker = st.session_state.market_ticker or REGIME_TICKER_PRIMARY
risk_on, idx_close, idx_ma200 = market_regime_at(st.session_state.market_df, ts_latest)

score, decision, emoji, reasons = score_decision(
    price=current_price,
    ma50=tm.ma50,
    ma200=tm.ma200,
    rrr=tm.rrr,
    sh=btr.sharpe_ratio,
    mdd=btr.max_drawdown,
    wr=btr.win_rate,
    pf=btr.profit_factor,
    rsi14=tm.rsi14,
    rvol=tm.rvol,
    risk_on=risk_on,
    regime_ticker=regime_ticker
)

# ---- Dashboard layout
row = st.container()
left, right = row.columns([1.4, 1.0], vertical_alignment="top")

with left:
    st.subheader(f"{emoji} {st.session_state.ticker} — {decision}")
    st.caption(
        f"Price: ${current_price:,.2f} • Horizon: {horizon}d • Live: {'ON' if st.session_state.live_active else 'OFF'} • "
        f"Regime: {regime_ticker} {'RISK-ON' if risk_on else 'RISK-OFF'}"
    )

    # Correlation note
    if avg_corr is not None:
        st.caption(f"Concentration check: avg corr vs your portfolio ≈ {avg_corr:.2f} → sizing multiplier {corr_mult:.2f}")
        if avg_corr >= CORR_WARN:
            st.warning("Concentration Risk: Current ticker is highly correlated to your portfolio; position sizing reduced.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Buy (±1σ move)", f"${tm.buy:,.2f}", f"{(tm.buy/current_price - 1)*100:.1f}%")
    k2.metric("Sell (±1σ move)", f"${tm.sell:,.2f}", f"{(tm.sell/current_price - 1)*100:.1f}%")
    k3.metric("Stop", f"${tm.stop:,.2f}", f"{(tm.stop/current_price - 1)*100:.1f}%")
    k4.metric("Position", f"{tm.qty} shares")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("RSI(14)", f"{tm.rsi14:.1f}")
    s2.metric("RVOL(20)", f"{tm.rvol:.2f}")
    s3.metric("RRR", f"{tm.rrr:.2f}")
    s4.metric("Kelly (half, adj)", f"{tm.kelly:.3f}")

    p1, p2, p3 = st.columns(3)
    p1.metric("Prob touch Buy", f"{tm.prob_touch_buy*100:.1f}%")
    p2.metric("Prob touch Sell", f"{tm.prob_touch_sell*100:.1f}%")
    p3.metric("Est. Slippage", f"{tm.est_slippage_bps:.1f} bps")

    with st.expander("Why this decision"):
        for r in reasons:
            st.write(f"- {r}")
        st.caption(
            f"Caps: max alloc {int(MAX_ALLOC_PCT*100)}% (cap {tm.cap_alloc_qty} sh), "
            f"max risk {int(MAX_RISK_PCT*100)}% (cap {tm.cap_risk_qty} sh). "
            f"Backtest: strategy-aligned, gap-aware, dynamic slippage (ADV$ + vol)."
        )

with right:
    st.plotly_chart(score_gauge(score, decision), use_container_width=True)

# ---- Tabs
tab1, tab2, tab3 = st.tabs(["Chart", "Backtest", "Live"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close"))
    if len(df) >= 50:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=pd.to_numeric(df["close"], errors="coerce").rolling(50).mean(), name="MA50"))
    if len(df) >= 200:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=pd.to_numeric(df["close"], errors="coerce").rolling(200).mean(), name="MA200"))

    fig.add_hline(y=tm.buy, line_dash="dot", annotation_text="Buy")
    fig.add_hline(y=tm.sell, line_dash="dot", annotation_text="Sell")
    fig.add_hline(y=tm.stop, line_dash="dot", annotation_text="Stop")

    fig.update_layout(height=520, hovermode="x unified", title=f"{st.session_state.ticker} Price")
    st.plotly_chart(fig, use_container_width=True)

    # Regime mini-panel
    mdf = st.session_state.market_df
    if mdf is not None and not mdf.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=mdf["timestamp"], y=mdf["close"], name=f"{regime_ticker} Close"))
        if len(mdf) >= 200:
            fig2.add_trace(go.Scatter(x=mdf["timestamp"], y=pd.to_numeric(mdf["close"], errors="coerce").rolling(200).mean(), name=f"{regime_ticker} MA200"))
        fig2.update_layout(height=260, hovermode="x unified", title=f"Market Regime: {regime_ticker}")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", btr.total_trades)
    c2.metric("Win Rate", f"{btr.win_rate:.1f}%")
    c3.metric("Profit Factor", f"{btr.profit_factor:.2f}")
    c4.metric("Sharpe", f"{btr.sharpe_ratio:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Max Drawdown", f"{btr.max_drawdown*100:.1f}%")
    c6.metric("Total PnL", f"${btr.total_pnl:,.2f}")
    c7.metric("Avg Win", f"{btr.avg_win:,.2f}")
    c8.metric("Avg Loss", f"{btr.avg_loss:,.2f}")

    if not bt.empty:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(y=bt["CumPnL"], name="Cumulative PnL"))
        fig_bt.update_layout(height=360, title="Equity Curve (Strategy Backtest)")
        st.plotly_chart(fig_bt, use_container_width=True)

        with st.expander("Recent trades"):
            show = bt.tail(15).copy()
            show["EntryTS"] = show["EntryTS"].astype(str)
            show["ExitTS"] = show["ExitTS"].astype(str)
            st.dataframe(show[["EntryTS", "ExitTS", "Reason", "Entry", "Exit", "PnL", "Return", "SlipBps"]], use_container_width=True)

with tab3:
    if not st.session_state.live_active:
        st.info("Live is OFF. Click **Start Live** above if you want streaming trades/quotes.")
    else:
        l1, l2 = st.columns(2)
        with l1:
            st.markdown("**Recent Trades**")
            for t in reversed(st.session_state.trade_history[-12:]):
                ts = parse_ts(t.get("ts")).strftime("%H:%M:%S")
                st.text(f"{ts} | ${t.get('price', 0):.2f} | {t.get('size', 0)}")
        with l2:
            st.markdown("**Latest Quote**")
            q = st.session_state.latest_quote or {}
            bid, ask = q.get("bid"), q.get("ask")
            if bid and ask:
                bid = float(bid); ask = float(ask)
                mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0
                spr = (ask - bid)
                spr_bps = (spr / mid * 10000.0) if mid > 0 else 0.0
                st.text(f"Bid: ${bid:.2f} ({q.get('bid_size','?')})")
                st.text(f"Ask: ${ask:.2f} ({q.get('ask_size','?')})")
                st.text(f"Spread: ${spr:.4f} ({spr_bps:.1f} bps)")
            else:
                st.text("No quote yet...")

st.markdown("---")
st.caption("⚠️ Educational only. Not financial advice.")
