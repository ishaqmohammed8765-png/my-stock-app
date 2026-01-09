"""
Trading Dashboard (Simple) + RSI + Market Regime + RVOL
Educational only. Not financial advice.

Major upgrades implemented:
1) Fix "RRR constant" problem:
   - Stop/Target based on ATR distances (variable per ticker/time).
2) Make entry match labels:
   - Backtest enters using a LIMIT at "Buy" (pullback) OR breakout mode.
3) Levels computed from entry (not prior close), so backtest is coherent.
4) Backtest uses realistic sizing:
   - allocation cap + risk cap + (optional) rolling half-kelly based on PRIOR trades only (no lookahead).
5) Out-of-sample split + benchmark vs Buy&Hold.
6) Trade costs:
   - dynamic slippage model + optional spread penalty + per-trade commission.

UI remains minimal; one "Advanced" expander for strategy params.
"""

from __future__ import annotations

import atexit
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

VOL_FLOOR, VOL_CAP, VOL_DEFAULT = 0.10, 1.80, 0.30

# sizing caps
KELLY_MIN, KELLY_MAX = 0.01, 0.15
MAX_ALLOC_PCT = 0.10   # 10% of equity max position
MAX_RISK_PCT = 0.02    # 2% of equity max risk per trade

# execution costs
BASE_SLIPPAGE_BPS = 5.0
COMMISSION_PER_TRADE = 0.00  # dollars per round trip (entry+exit) in backtest
WIDE_SPREAD_BPS_WARN = 25

# strategy/backtest
MIN_HIST_DAYS = 240
MAX_BT_ITERS = 800          # allow more history for OOS split
BT_OOS_FRAC = 0.30          # last 30% = out-of-sample
BT_MIN_TRADES_FOR_KELLY = 40

# regime
REGIME_TICKER_PRIMARY = "QQQ"
REGIME_TICKER_FALLBACK = "SPY"

# data validator
YF_CLOSE_DIFF_PCT = 0.0075  # 0.75%

# live
QUEUE_MAX = 1500
MAX_TRADES = 200
MAX_QUOTES = 200
MAX_RECONNECTS = 8
AUTO_REFRESH_MS = 1500

# score thresholds
SCORE_STRONG_BUY = 80
SCORE_BUY = 65
SCORE_CONDITIONAL = 50
SCORE_HOLD = 35


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
    risk_per_share: float
    cap_alloc_qty: int
    cap_risk_qty: int

    rsi14: float
    rvol: float

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
    if HAS_ST_AUTOR:
        st_autorefresh(interval=interval_ms, key=key)
    else:
        st.caption("Tip: install `streamlit-autorefresh` for smoother live updates. Manual refresh is fine.")


def last_completed_index(df: pd.DataFrame) -> int:
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
# DATA INTEGRITY
# =========================
def sanity_check_bars(df: pd.DataFrame) -> List[str]:
    warns: List[str] = []
    if df is None or df.empty:
        return ["No historical data loaded."]

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

    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")

    if (o <= 0).any() or (h <= 0).any() or (l <= 0).any() or (c <= 0).any():
        warns.append("Non-positive OHLC values detected (bad ticks or corporate action mismatch).")
    if (v < 0).any():
        warns.append("Negative volume detected (data error).")

    bad_h = (h < np.maximum(o, c)).sum()
    bad_l = (l > np.minimum(o, c)).sum()
    if bad_h > 0 or bad_l > 0:
        warns.append("OHLC inconsistency detected (high/low not enclosing open/close).")

    return warns


def detect_split_like_events(df: pd.DataFrame) -> List[str]:
    warns: List[str] = []
    if df is None or df.empty or len(df) < 5:
        return warns
    c = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(c) < 5:
        return warns
    r = (c / c.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return warns
    targets = [0.5, 2.0, 3.0, 0.3333, 4.0, 0.25]
    for t in targets:
        near = (r - t).abs() <= 0.03
        if near.any():
            idxs = list(r.index[near][-3:])
            for ix in idxs:
                dt = parse_ts(df.loc[ix, "timestamp"]).date()
                warns.append(f"Possible corporate action near {dt}: close ratio ≈ {r.loc[ix]:.2f} (split/dividend mismatch).")
            break
    return warns


@st.cache_data(ttl=3600, show_spinner=False)
def yf_daily_close(ticker: str, period_days: int = 400) -> Optional[pd.Series]:
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

    our_date = pd.Timestamp(our_ts).date()
    yf_same = yf_close_series[yf_close_series.index.date == our_date]
    yf_close = float(yf_same.iloc[-1]) if not yf_same.empty else float(yf_close_series.iloc[-1])

    if our_close <= 0 or yf_close <= 0:
        return our_close, yf_close, None

    diff_pct = abs(our_close - yf_close) / ((our_close + yf_close) / 2)
    if diff_pct > YF_CLOSE_DIFF_PCT:
        warn = f"Data Warning: close differs vs Yahoo by {diff_pct*100:.2f}% (our {our_close:.2f} vs YF {yf_close:.2f})."
        return our_close, yf_close, warn
    return our_close, yf_close, None


# =========================
# LIVE STREAM
# =========================
class RealtimeStream:
    def __init__(self, api_key: str, secret_key: str, ticker: str,
                 trade_q: queue.Queue, quote_q: queue.Queue, feed: str = "iex"):
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
                    days_back: int = 1200) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    dbg: Dict[str, Any] = {"ticker": ticker, "steps": []}

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
        feed_val = DataFeed.SIP if "IEX" == "SIP" else DataFeed.IEX

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
                dbg["steps"].append(f"429 -> sleep {wait:.0f}s")
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


def atr_value(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty or len(df) < period + 2:
        return 0.0
    x = df.copy()
    h = pd.to_numeric(x["high"], errors="coerce")
    l = pd.to_numeric(x["low"], errors="coerce")
    c = pd.to_numeric(x["close"], errors="coerce")
    tr1 = (h - l).abs()
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else 0.0


def rsi14_from_close(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 20:
        return 50.0
    closes = pd.to_numeric(df["close"], errors="coerce").dropna().values
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
    return float(clamp(rsi, 0.0, 100.0)) if np.isfinite(rsi) else 50.0


def rvol_20(df: pd.DataFrame) -> float:
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


def sharpe(returns: np.ndarray) -> float:
    if returns is None or len(returns) == 0:
        return 0.0
    sd = returns.std()
    if sd == 0:
        return 0.0
    excess = returns - (RISK_FREE / TRADING_DAYS)
    s = (excess.mean() / sd) * np.sqrt(TRADING_DAYS)
    return float(s) if np.isfinite(s) else 0.0


def max_drawdown(cum_equity: np.ndarray) -> float:
    if cum_equity is None or len(cum_equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(cum_equity)
    dd = (cum_equity - peak) / (peak + 1e-12)
    m = float(dd.min())
    return m if np.isfinite(m) else 0.0


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

    sh = sharpe(bt["Return"].values)
    total_pnl = float(bt["PnL"].sum())

    # equity curve mdd
    eq = bt["Equity"].values if "Equity" in bt.columns else (1 + bt["Return"]).cumprod()
    mdd = max_drawdown(eq)

    return BacktestResults(win_rate, pf, mdd, sh, total, wins, losses, avg_win, avg_loss, total_pnl)


def adv_dollar_volume(df: pd.DataFrame, lookback: int = 20) -> float:
    if df is None or df.empty or len(df) < lookback + 2:
        return 0.0
    x = df.copy()
    if last_completed_index(x) == -2:
        x = x.iloc[:-1]
    x = x.tail(lookback)
    c = pd.to_numeric(x["close"], errors="coerce")
    v = pd.to_numeric(x["volume"], errors="coerce")
    dv = (c * v).replace([np.inf, -np.inf], np.nan).dropna()
    return float(dv.mean()) if not dv.empty else 0.0


def estimate_slippage_bps(base_bps: float, order_value: float, adv_dollar: float, vol: float) -> float:
    if order_value <= 0:
        return float(base_bps)
    adv = max(adv_dollar, 1.0)
    participation = order_value / adv
    impact = 8_000.0 * participation
    vol_component = 15.0 * max(0.0, min(vol, 1.0))
    est = base_bps + impact + vol_component
    return float(clamp(est, base_bps, 200.0))


def prob_hit_mc(S: float, K: float, vol: float, days: int, sims: int = 800) -> float:
    if S <= 0 or K <= 0 or vol <= 0 or days <= 0 or sims <= 0:
        return 0.0
    dt = 1 / TRADING_DAYS
    try:
        dof = 5
        shocks = stats.t.rvs(df=dof, size=(sims, days)) / np.sqrt(dof / (dof - 2))
        drift = -0.5 * (vol ** 2) * dt
        diffusion = vol * np.sqrt(dt) * shocks
        paths = S * np.exp(np.cumsum(drift + diffusion, axis=1))
        hit = np.any(paths >= K, axis=1) if K >= S else np.any(paths <= K, axis=1)
        return float(hit.mean())
    except Exception:
        return 0.0


# =========================
# STRATEGY LEVELS (ATR BASED)
# =========================
def compute_levels(
    df: pd.DataFrame,
    price: float,
    mode: str,
    atr_mult_entry: float,
    atr_mult_stop: float,
    atr_mult_target: float,
) -> Tuple[float, float, float, float]:
    """
    Returns (buy, stop, sell, risk_per_share)

    Pullback mode:
      buy = price - atr_mult_entry*ATR
      stop = buy - atr_mult_stop*ATR
      sell = buy + atr_mult_target*ATR

    Breakout mode:
      buy = price + atr_mult_entry*ATR
      stop = buy - atr_mult_stop*ATR
      sell = buy + atr_mult_target*ATR
    """
    a = atr_value(df, 14)
    a = max(a, 0.01)

    if mode == "BREAKOUT":
        buy = price + atr_mult_entry * a
    else:
        buy = price - atr_mult_entry * a

    stop = buy - atr_mult_stop * a
    sell = buy + atr_mult_target * a
    risk_per_share = max(buy - stop, 0.01)
    return float(buy), float(stop), float(sell), float(risk_per_share)


def ma(df: pd.DataFrame, n: int) -> float:
    if df is None or df.empty or len(df) < n:
        return float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
    return float(pd.to_numeric(df["close"], errors="coerce").rolling(n).mean().iloc[-1])


# =========================
# SIZING (USED BOTH LIVE METRICS + BACKTEST)
# =========================
def size_position(
    acct: float,
    entry: float,
    stop: float,
    kelly_f: float,
) -> Tuple[int, int, int, float]:
    """
    Returns (qty, cap_alloc_qty, cap_risk_qty, risk_per_share).
    Uses same caps as dashboard.
    """
    risk_per_share = max(entry - stop, 0.01)

    cap_alloc_qty = int((acct * MAX_ALLOC_PCT) / max(entry, 0.01))
    cap_risk_qty = int((acct * MAX_RISK_PCT) / risk_per_share) if risk_per_share > 0 else 0

    # Kelly here is used as a fraction of equity you'd be willing to "risk-scale" with.
    # Convert to shares via risk-based sizing (still capped by alloc and risk).
    raw_qty = int((acct * kelly_f) / risk_per_share) if risk_per_share > 0 else 0
    qty = max(0, min(raw_qty, cap_alloc_qty, cap_risk_qty))
    return qty, cap_alloc_qty, cap_risk_qty, float(risk_per_share)


def rolling_half_kelly_from_trades(trades: List[Dict[str, Any]]) -> float:
    """
    Half-Kelly computed from prior trade returns only.
    Kept conservative and clamped.
    """
    if len(trades) < BT_MIN_TRADES_FOR_KELLY:
        return KELLY_MIN

    rets = np.array([t["Return"] for t in trades if np.isfinite(t.get("Return", np.nan))], dtype=float)
    if len(rets) < BT_MIN_TRADES_FOR_KELLY:
        return KELLY_MIN

    wins = rets[rets > 0]
    losses = rets[rets < 0]
    if len(wins) < 10 or len(losses) < 10:
        return KELLY_MIN

    wr = len(wins) / len(rets)
    aw = float(np.mean(wins))
    al = float(abs(np.mean(losses)))
    if al <= 0 or aw <= 0:
        return KELLY_MIN

    R = aw / al
    f = (wr - ((1 - wr) / R)) * 0.5  # half-kelly
    return clamp(float(f), KELLY_MIN, KELLY_MAX)


# =========================
# BACKTEST (LIMIT ENTRY, ATR LEVELS, REAL SIZING, OOS)
# =========================
def gap_aware_fill(open_px: float, level_px: float, is_stop: bool) -> float:
    # Conservative: stops slip against you, targets slip against you on gaps
    if is_stop:
        return float(open_px) if open_px <= level_px else float(level_px)
    return float(open_px) if open_px >= level_px else float(level_px)


def backtest_strategy(
    df: pd.DataFrame,
    market_df: Optional[pd.DataFrame],
    horizon: int,
    mode: str,
    atr_entry: float,
    atr_stop: float,
    atr_target: float,
    require_risk_on: bool,
    rsi_max: float,
    rsi_min: float,
    rvol_min: float,
    vol_max: float,
    cooldown_bars: int,
    include_spread_penalty: bool,
    assumed_spread_bps: float,
    start_equity: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (trades_df, daily_equity_df)
    - enters with a limit at buy
    - uses ATR-based stop/target from entry
    - uses equity-based sizing with caps
    - dynamic slippage based on ADV$ and vol
    - optional cooldown after STOP
    - OOS split handled outside
    """

    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if len(df) < MIN_HIST_DAYS + horizon + 30:
        return pd.DataFrame(), pd.DataFrame()

    df_bt = df.iloc[:-1].copy() if last_completed_index(df) == -2 else df.copy()
    df_bt = df_bt.reset_index(drop=True)

    # prepare indicators
    close = pd.to_numeric(df_bt["close"], errors="coerce")
    df_bt["ma50"] = close.rolling(50).mean()
    df_bt["ma200"] = close.rolling(200).mean()
    df_bt["rsi14"] = np.nan
    df_bt["rvol"] = np.nan
    df_bt["vol_ann"] = np.nan
    df_bt["atr14"] = np.nan

    # compute rolling stuff efficiently
    # RSI & ATR from helper functions (slower) – keep simple but bounded.
    for i in range(220, len(df_bt)):
        hist = df_bt.iloc[:i+1]
        df_bt.loc[i, "rsi14"] = rsi14_from_close(hist)
        df_bt.loc[i, "rvol"] = rvol_20(hist)
        df_bt.loc[i, "vol_ann"] = annual_vol(hist)
        df_bt.loc[i, "atr14"] = atr_value(hist, 14)

    trades: List[Dict[str, Any]] = []
    equity = float(start_equity)
    equity_curve: List[Dict[str, Any]] = []

    last_stop_i = -10_000

    # Iterate across potential entry days
    # We use i as "signal day" (levels based on price at i), and attempt fill from i+1 forward.
    for i in range(220, min(len(df_bt) - horizon - 2, MAX_BT_ITERS + 220)):
        ts_i = df_bt["timestamp"].iloc[i]
        px = float(pd.to_numeric(df_bt["close"], errors="coerce").iloc[i])
        if px <= 0:
            continue

        # cooldown after stopouts
        if (i - last_stop_i) < cooldown_bars:
            continue

        v = float(df_bt["vol_ann"].iloc[i]) if np.isfinite(df_bt["vol_ann"].iloc[i]) else VOL_DEFAULT
        if v > vol_max:
            continue

        ma50_i = float(df_bt["ma50"].iloc[i]) if np.isfinite(df_bt["ma50"].iloc[i]) else px
        ma200_i = float(df_bt["ma200"].iloc[i]) if np.isfinite(df_bt["ma200"].iloc[i]) else ma50_i
        rsi_i = float(df_bt["rsi14"].iloc[i]) if np.isfinite(df_bt["rsi14"].iloc[i]) else 50.0
        rvol_i = float(df_bt["rvol"].iloc[i]) if np.isfinite(df_bt["rvol"].iloc[i]) else 1.0

        # regime
        risk_on, _, _ = market_regime_at(market_df, ts_i)
        if require_risk_on and not risk_on:
            continue

        # basic trend filter (kept light)
        if px < ma200_i:
            continue

        # RSI & RVOL gates (simple)
        if not (rsi_min <= rsi_i <= rsi_max):
            continue
        if rvol_i < rvol_min:
            continue

        # compute ATR-based levels from signal-day context
        hist = df_bt.iloc[:i+1]
        buy, stop, sell, risk_per_share = compute_levels(
            hist, px, mode=mode,
            atr_mult_entry=atr_entry,
            atr_mult_stop=atr_stop,
            atr_mult_target=atr_target,
        )

        # if buy is silly
        if stop <= 0 or buy <= 0 or sell <= 0 or sell <= buy:
            continue

        # estimate slippage and spread
        adv = adv_dollar_volume(hist, 20)
        # rolling half-kelly from prior trades only
        kelly_f = rolling_half_kelly_from_trades(trades)
        qty, cap_alloc_qty, cap_risk_qty, risk_ps = size_position(equity, buy, stop, kelly_f)
        if qty <= 0:
            continue

        order_value = qty * buy
        slip_bps = estimate_slippage_bps(BASE_SLIPPAGE_BPS, order_value, adv, v)
        if include_spread_penalty:
            slip_bps += max(0.0, assumed_spread_bps / 2.0)  # half-spread penalty per side
        slip = slip_bps / 10000.0

        # attempt fill starting next day: limit order at buy
        entry_eff = None
        entry_day = None

        for j in range(i + 1, min(i + 1 + 5, len(df_bt) - horizon - 1)):  # give it up to 5 days to fill
            o = float(pd.to_numeric(df_bt["open"], errors="coerce").iloc[j])
            h = float(pd.to_numeric(df_bt["high"], errors="coerce").iloc[j])
            l = float(pd.to_numeric(df_bt["low"], errors="coerce").iloc[j])
            if o <= 0 or h <= 0 or l <= 0:
                continue

            if mode == "BREAKOUT":
                # breakout: require price trade up through buy
                if h >= buy:
                    fill = gap_aware_fill(o, buy, is_stop=False)
                    entry_eff = fill * (1 + slip)
                    entry_day = j
                    break
            else:
                # pullback: require price trade down to buy
                if l <= buy:
                    fill = gap_aware_fill(o, buy, is_stop=False)
                    entry_eff = fill * (1 + slip)
                    entry_day = j
                    break

        if entry_eff is None or entry_day is None:
            continue

        # Recompute levels from ENTRY price (coherent)
        # (We keep stop/target distances based on ATR from signal-day; could also use ATR at entry day.)
        entry = float(entry_eff)
        stop_eff = float(stop)
        sell_eff = float(sell)

        # walk forward for exits
        exit_eff = None
        reason = "TIME"
        exit_day = entry_day + horizon

        for j in range(entry_day, min(entry_day + horizon, len(df_bt))):
            o = float(pd.to_numeric(df_bt["open"], errors="coerce").iloc[j])
            h = float(pd.to_numeric(df_bt["high"], errors="coerce").iloc[j])
            l = float(pd.to_numeric(df_bt["low"], errors="coerce").iloc[j])
            c = float(pd.to_numeric(df_bt["close"], errors="coerce").iloc[j])
            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                continue

            # Stop first (conservative)
            if l <= stop_eff:
                fill = gap_aware_fill(o, stop_eff, is_stop=True)
                exit_eff = fill * (1 - slip)
                reason = "STOP"
                exit_day = j
                last_stop_i = i
                break

            if h >= sell_eff:
                fill = gap_aware_fill(o, sell_eff, is_stop=False)
                exit_eff = fill * (1 - slip)
                reason = "TP"
                exit_day = j
                break

        if exit_eff is None:
            c = float(pd.to_numeric(df_bt["close"], errors="coerce").iloc[min(entry_day + horizon - 1, len(df_bt)-1)]))
            exit_eff = c * (1 - slip)
            reason = "TIME"
            exit_day = min(entry_day + horizon - 1, len(df_bt)-1)

        pnl = (exit_eff - entry_eff) * qty - COMMISSION_PER_TRADE
        ret = pnl / max(equity, 1e-9)

        equity = max(1.0, equity + pnl)

        trades.append({
            "SignalTS": parse_ts(ts_i),
            "EntryTS": parse_ts(df_bt["timestamp"].iloc[entry_day]),
            "ExitTS": parse_ts(df_bt["timestamp"].iloc[exit_day]),
            "Mode": mode,
            "Entry": entry_eff,
            "Stop": stop_eff,
            "Target": sell_eff,
            "Exit": exit_eff,
            "Reason": reason,
            "Qty": qty,
            "SlipBps": slip_bps,
            "PnL": pnl,
            "Return": ret,
            "Equity": equity,
            "RSI": rsi_i,
            "RVOL": rvol_i,
        })

        equity_curve.append({
            "timestamp": parse_ts(df_bt["timestamp"].iloc[exit_day]),
            "Equity": equity
        })

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_curve).sort_values("timestamp") if equity_curve else pd.DataFrame()

    return trades_df, eq_df


# =========================
# SCORING (now uses REAL RRR)
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

        # strategy params
        "mode": "PULLBACK",
        "atr_entry": 0.6,
        "atr_stop": 1.6,
        "atr_target": 2.8,
        "require_risk_on": True,
        "rsi_min": 25.0,
        "rsi_max": 75.0,
        "rvol_min": 0.9,
        "vol_max": 1.35,
        "cooldown_bars": 5,
        "spread_penalty": False,
        "assumed_spread_bps": 10.0,
        "start_equity_bt": 10_000.0,
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
st.caption("Minimal UI. Coherent backtest. ATR-based levels. Out-of-sample evaluation. Educational only.")

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

# ---- Advanced expander (still minimal)
with st.expander("Advanced (strategy + realism)", expanded=False):
    colA, colB, colC = st.columns(3)
    with colA:
        st.session_state.mode = st.selectbox("Mode", ["PULLBACK", "BREAKOUT"], index=0 if st.session_state.mode == "PULLBACK" else 1)
        st.session_state.require_risk_on = st.checkbox("Require risk-on regime", value=st.session_state.require_risk_on)
        st.session_state.spread_penalty = st.checkbox("Include spread penalty", value=st.session_state.spread_penalty)
    with colB:
        st.session_state.atr_entry = st.slider("ATR entry mult", 0.1, 2.0, float(st.session_state.atr_entry), 0.1)
        st.session_state.atr_stop = st.slider("ATR stop mult", 0.5, 3.0, float(st.session_state.atr_stop), 0.1)
        st.session_state.atr_target = st.slider("ATR target mult", 0.5, 5.0, float(st.session_state.atr_target), 0.1)
    with colC:
        st.session_state.rsi_min = st.slider("RSI min", 0.0, 60.0, float(st.session_state.rsi_min), 1.0)
        st.session_state.rsi_max = st.slider("RSI max", 40.0, 100.0, float(st.session_state.rsi_max), 1.0)
        st.session_state.rvol_min = st.slider("RVOL min", 0.2, 2.0, float(st.session_state.rvol_min), 0.1)

    colD, colE, colF = st.columns(3)
    with colD:
        st.session_state.vol_max = st.slider("Max annual vol filter", 0.3, 2.0, float(st.session_state.vol_max), 0.05)
        st.session_state.cooldown_bars = st.slider("Cooldown bars after STOP", 0, 20, int(st.session_state.cooldown_bars), 1)
    with colE:
        st.session_state.assumed_spread_bps = st.slider("Assumed spread (bps)", 0.0, 60.0, float(st.session_state.assumed_spread_bps), 1.0)
    with colF:
        st.session_state.start_equity_bt = st.number_input("Backtest starting equity ($)", min_value=1000, max_value=1_000_000, value=int(st.session_state.start_equity_bt), step=1000)

# ---- Load data
if load_btn:
    stop_live()
    st.session_state.last_error = ""
    st.session_state.debug = {}

    # Clear live buffers
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

    st.rerun()

# ---- Error banner
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

our_close, yf_close, yf_warn = validate_price_vs_yf(st.session_state.ticker, df)
if yf_warn:
    st.warning(yf_warn)
elif HAS_YF:
    st.caption("Price validator: Yahoo check OK (within threshold).")
else:
    st.caption("Price validator: install `yfinance` to enable multi-source daily-close validation.")

# ---- Live toggle
live_box = st.container(border=True)
with live_box:
    l1, l2, l3 = st.columns([1.0, 1.0, 3.0])
    with l1:
        live_btn = st.button("Stop Live" if st.session_state.live_active else "Start Live", use_container_width=True)
    with l2:
        st.caption("Feed: IEX")
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
            feed="iex",
        )
        t = threading.Thread(target=handler.run, daemon=True)
        t.start()
        st.session_state.ws_handler = handler
        st.session_state.ws_thread = t
        st.session_state.live_active = True
    st.rerun()

if st.session_state.live_active:
    thr = st.session_state.ws_thread
    if thr is None or (hasattr(thr, "is_alive") and not thr.is_alive()):
        stop_live()
        st.warning("Live feed stopped (connection closed).")
    else:
        live_autorefresh(AUTO_REFRESH_MS, key="live_refresh")

drain(st.session_state.trade_q, st.session_state.trade_history, MAX_TRADES)
drain(st.session_state.quote_q, st.session_state.quote_history, MAX_QUOTES)

if st.session_state.quote_history:
    st.session_state.latest_quote = st.session_state.quote_history[-1]
    mid = mid_from_quote(st.session_state.latest_quote)
    if mid is not None:
        st.session_state.latest_price = mid

current_price = float(st.session_state.latest_price or float(pd.to_numeric(df["close"], errors="coerce").iloc[-1]))

# ---- Spread warning
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
                st.warning(f"Execution Warning: Spread is wide ({spread_bps:.1f} bps). Slippage risk is higher.")

# ---- Backtest with OOS split
df_bt = df.iloc[:-1].copy() if last_completed_index(df) == -2 else df.copy()
df_bt = df_bt.reset_index(drop=True)

split_idx = int(len(df_bt) * (1.0 - BT_OOS_FRAC))
split_idx = max(split_idx, 260)

df_is = df_bt.iloc[:split_idx].copy()
df_oos = df_bt.iloc[split_idx:].copy()

bt_is, eq_is = backtest_strategy(
    df=df_is,
    market_df=st.session_state.market_df,
    horizon=horizon,
    mode=st.session_state.mode,
    atr_entry=float(st.session_state.atr_entry),
    atr_stop=float(st.session_state.atr_stop),
    atr_target=float(st.session_state.atr_target),
    require_risk_on=bool(st.session_state.require_risk_on),
    rsi_max=float(st.session_state.rsi_max),
    rsi_min=float(st.session_state.rsi_min),
    rvol_min=float(st.session_state.rvol_min),
    vol_max=float(st.session_state.vol_max),
    cooldown_bars=int(st.session_state.cooldown_bars),
    include_spread_penalty=bool(st.session_state.spread_penalty),
    assumed_spread_bps=float(st.session_state.assumed_spread_bps),
    start_equity=float(st.session_state.start_equity_bt),
)

bt_oos, eq_oos = backtest_strategy(
    df=df_oos,
    market_df=st.session_state.market_df,
    horizon=horizon,
    mode=st.session_state.mode,
    atr_entry=float(st.session_state.atr_entry),
    atr_stop=float(st.session_state.atr_stop),
    atr_target=float(st.session_state.atr_target),
    require_risk_on=bool(st.session_state.require_risk_on),
    rsi_max=float(st.session_state.rsi_max),
    rsi_min=float(st.session_state.rsi_min),
    rvol_min=float(st.session_state.rvol_min),
    vol_max=float(st.session_state.vol_max),
    cooldown_bars=int(st.session_state.cooldown_bars),
    include_spread_penalty=bool(st.session_state.spread_penalty),
    assumed_spread_bps=float(st.session_state.assumed_spread_bps),
    start_equity=float(st.session_state.start_equity_bt),
)

btr_is = bt_stats(bt_is)
btr_oos = bt_stats(bt_oos)

# ---- Buy & Hold benchmark (OOS window)
bench = pd.DataFrame()
try:
    if not df_oos.empty:
        c = pd.to_numeric(df_oos["close"], errors="coerce").dropna()
        if len(c) >= 2:
            ret = c.pct_change().fillna(0.0)
            eq = float(st.session_state.start_equity_bt) * (1 + ret).cumprod()
            bench = pd.DataFrame({"timestamp": df_oos.loc[c.index, "timestamp"].values, "Equity": eq.values})
except Exception:
    bench = pd.DataFrame()

# ---- Compute LIVE trade metrics with same ATR logic
hist_full = df_bt.copy()
v_live = annual_vol(hist_full)
a_live = atr_value(hist_full, 14)
ma50_live = ma(hist_full, 50)
ma200_live = ma(hist_full, 200)
rsi_live = rsi14_from_close(hist_full)
rvol_live = rvol_20(hist_full)
risk_on_live, _, _ = market_regime_at(st.session_state.market_df, hist_full["timestamp"].iloc[-1])

buy, stop, sell, risk_ps = compute_levels(
    hist_full, current_price,
    mode=st.session_state.mode,
    atr_mult_entry=float(st.session_state.atr_entry),
    atr_mult_stop=float(st.session_state.atr_stop),
    atr_mult_target=float(st.session_state.atr_target),
)
rrr = (sell - buy) / max(buy - stop, 0.01)

# kelly estimate from in-sample trades distribution (for display only)
kelly_disp = rolling_half_kelly_from_trades(bt_is.to_dict("records")) if not bt_is.empty else KELLY_MIN

qty, cap_alloc_qty, cap_risk_qty, risk_per_share = size_position(float(account_size), buy, stop, float(kelly_disp))

adv = adv_dollar_volume(hist_full, 20)
order_value = qty * buy
est_slip_bps = estimate_slippage_bps(BASE_SLIPPAGE_BPS, order_value, adv, v_live)

prob_touch_buy = prob_hit_mc(current_price, buy, v_live, horizon, sims=700)
prob_touch_sell = prob_hit_mc(current_price, sell, v_live, horizon, sims=700)

score, decision, emoji, reasons = score_decision(
    price=current_price,
    ma50=ma50_live,
    ma200=ma200_live,
    rrr=rrr,
    sh=btr_oos.sharpe_ratio if btr_oos.total_trades > 0 else btr_is.sharpe_ratio,
    mdd=btr_oos.max_drawdown if btr_oos.total_trades > 0 else btr_is.max_drawdown,
    wr=btr_oos.win_rate if btr_oos.total_trades > 0 else btr_is.win_rate,
    pf=btr_oos.profit_factor if btr_oos.total_trades > 0 else btr_is.profit_factor,
    rsi14=rsi_live,
    rvol=rvol_live,
    risk_on=risk_on_live,
    regime_ticker=st.session_state.market_ticker or REGIME_TICKER_PRIMARY
)

# ---- Dashboard layout
row = st.container()
left, right = row.columns([1.4, 1.0], vertical_alignment="top")

with left:
    st.subheader(f"{emoji} {st.session_state.ticker} — {decision}")
    st.caption(
        f"Price: ${current_price:,.2f} • Horizon: {horizon}d • Mode: {st.session_state.mode} • "
        f"Regime: {st.session_state.market_ticker} {'RISK-ON' if risk_on_live else 'RISK-OFF'} • "
        f"OOS window: {int(BT_OOS_FRAC*100)}%"
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Buy (limit)", f"${buy:,.2f}", f"{(buy/current_price - 1)*100:.1f}%")
    k2.metric("Target", f"${sell:,.2f}", f"{(sell/current_price - 1)*100:.1f}%")
    k3.metric("Stop", f"${stop:,.2f}", f"{(stop/current_price - 1)*100:.1f}%")
    k4.metric("Position", f"{qty} shares")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("RSI(14)", f"{rsi_live:.1f}")
    s2.metric("RVOL(20)", f"{rvol_live:.2f}")
    s3.metric("RRR", f"{rrr:.2f}")
    s4.metric("Kelly (rolling, IS)", f"{kelly_disp:.3f}")

    p1, p2, p3 = st.columns(3)
    p1.metric("Prob touch Buy (MC)", f"{prob_touch_buy*100:.1f}%")
    p2.metric("Prob touch Target (MC)", f"{prob_touch_sell*100:.1f}%")
    p3.metric("Est. Slippage", f"{est_slip_bps:.1f} bps")

    with st.expander("Why this decision"):
        for r in reasons:
            st.write(f"- {r}")
        st.caption(
            f"Caps: max alloc {int(MAX_ALLOC_PCT*100)}% (cap {cap_alloc_qty} sh), "
            f"max risk {int(MAX_RISK_PCT*100)}% (cap {cap_risk_qty} sh). "
            f"Backtest: LIMIT entry at Buy, ATR stop/target, dynamic slippage, out-of-sample."
        )

with right:
    st.plotly_chart(score_gauge(score, decision), use_container_width=True)

# ---- Tabs
tab1, tab2, tab3 = st.tabs(["Chart", "Backtest", "Live"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["close"], name="Close"))
    fig.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["ma50"], name="MA50"))
    fig.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["ma200"], name="MA200"))
    fig.add_hline(y=buy, line_dash="dot", annotation_text="Buy (limit)")
    fig.add_hline(y=sell, line_dash="dot", annotation_text="Target")
    fig.add_hline(y=stop, line_dash="dot", annotation_text="Stop")
    fig.update_layout(height=520, hovermode="x unified", title=f"{st.session_state.ticker} Price")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### In-sample vs Out-of-sample")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IS Trades", btr_is.total_trades)
    c2.metric("IS Win Rate", f"{btr_is.win_rate:.1f}%")
    c3.metric("IS Profit Factor", f"{btr_is.profit_factor:.2f}")
    c4.metric("IS Sharpe", f"{btr_is.sharpe_ratio:.2f}")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("OOS Trades", btr_oos.total_trades)
    d2.metric("OOS Win Rate", f"{btr_oos.win_rate:.1f}%")
    d3.metric("OOS Profit Factor", f"{btr_oos.profit_factor:.2f}")
    d4.metric("OOS Sharpe", f"{btr_oos.sharpe_ratio:.2f}")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("OOS Max DD", f"{btr_oos.max_drawdown*100:.1f}%")
    e2.metric("OOS Total PnL", f"${btr_oos.total_pnl:,.2f}")
    e3.metric("OOS Avg Win", f"{btr_oos.avg_win:,.2f}")
    e4.metric("OOS Avg Loss", f"{btr_oos.avg_loss:,.2f}")

    # Equity curves
    fig_eq = go.Figure()
    if not eq_oos.empty:
        fig_eq.add_trace(go.Scatter(x=eq_oos["timestamp"], y=eq_oos["Equity"], name="Strategy (OOS)"))
    if not bench.empty:
        fig_eq.add_trace(go.Scatter(x=bench["timestamp"], y=bench["Equity"], name="Buy & Hold (OOS)"))
    fig_eq.update_layout(height=360, title="Out-of-sample Equity Curve")
    st.plotly_chart(fig_eq, use_container_width=True)

    if not bt_oos.empty:
        with st.expander("Recent OOS trades"):
            show = bt_oos.tail(15).copy()
            show["EntryTS"] = show["EntryTS"].astype(str)
            show["ExitTS"] = show["ExitTS"].astype(str)
            st.dataframe(show[["EntryTS", "ExitTS", "Reason", "Entry", "Exit", "Qty", "PnL", "Return", "SlipBps"]], use_container_width=True)

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
