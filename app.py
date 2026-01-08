"""
Trading Dashboard Pro — Simple UI (No Advanced)
Educational only. Not financial advice.

Minimal controls:
- Ticker
- Account size
- Horizon
- Load/Refresh
- Start/Stop Live

Background defaults (safe):
- Monte Carlo: student_t
- Historical feed: IEX
- Live feed: iex
- Kelly sizing: half-kelly + caps (10% max allocation, 2% max risk/trade)
- Backtest: entry next-day open, slippage 5 bps, commission $0
- Rate limit 429 backoff
- Websocket cleanup via atexit
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

MIN_HIST_DAYS = 40
MAX_BT_ITERS = 80

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
SLIPPAGE_BPS = 5
COMMISSION = 0.0


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
    buy_prob: float
    sell_prob: float
    rrr: float
    kelly: float
    qty: int
    expected_move: float
    risk_per_share: float
    cap_alloc_qty: int
    cap_risk_qty: int


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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

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

        try:
            loop.stop()
            loop.close()
        except Exception:
            pass

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


@st.cache_data(ttl=3600, show_spinner=False)
def load_historical(ticker: str, api_key: str, secret_key: str,
                    days_back: int = 260) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
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
    if len(df) < 30:
        dbg["error"] = f"Too few rows ({len(df)})"
        return None, dbg

    dbg["rows"] = int(len(df))
    return df, dbg


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
    r = np.log(df["close"] / df["close"].shift(1)).dropna()
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

        if K >= S:
            hit = np.any(paths >= K, axis=1)
        else:
            hit = np.any(paths <= K, axis=1)

        return float(hit.mean())
    except Exception:
        return 0.0


def kelly(df: pd.DataFrame, min_trades: int = 60) -> float:
    if df is None or df.empty or len(df) < min_trades:
        return KELLY_MIN
    r = df["close"].pct_change().dropna()
    if len(r) < min_trades:
        return KELLY_MIN
    wins = r[r > 0]
    losses = r[r < 0]
    if len(wins) < 15 or len(losses) < 15:
        return KELLY_MIN
    wr = len(wins) / len(r)
    aw = float(wins.mean())
    al = float(abs(losses.mean()))
    if al <= 0 or aw <= 0:
        return KELLY_MIN
    R = aw / al
    f = (wr - ((1 - wr) / R)) * 0.5
    return clamp(float(f), KELLY_MIN, KELLY_MAX)


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


def backtest(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if len(df) < horizon + MIN_HIST_DAYS + 2:
        return pd.DataFrame()

    iters = min(len(df) - horizon - 1, MAX_BT_ITERS)
    start = len(df) - horizon - iters - 1

    out = []
    slip = SLIPPAGE_BPS / 10000.0

    for i in range(iters):
        idx = start + i
        entry = float(df["open"].iloc[idx + 1])  # next day open
        if entry <= 0:
            continue

        v = annual_vol(df.iloc[: idx + 1])
        move = expected_move(entry, v, horizon)
        stop = entry - (move * STOP_MULT)

        window = df["close"].iloc[idx + 1: idx + 1 + horizon]
        if window.empty:
            continue

        stop_hit = window[window <= stop]
        exit_px = float(stop_hit.iloc[0]) if not stop_hit.empty else float(window.iloc[-1])

        entry_eff = entry * (1 + slip)
        exit_eff = exit_px * (1 - slip)

        pnl = (exit_eff - entry_eff) - COMMISSION
        ret = pnl / entry_eff
        out.append({"PnL": pnl, "Return": ret})

    if not out:
        return pd.DataFrame()

    bt = pd.DataFrame(out)
    bt["CumPnL"] = bt["PnL"].cumsum()
    bt["CumReturn"] = (1 + bt["Return"]).cumprod() - 1
    return bt


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


def compute_trade_metrics(df: pd.DataFrame, price: float, horizon: int, acct: float) -> TradeMetrics:
    v = annual_vol(df)
    a = atr(df)

    ma50 = float(df["close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else price
    ma200 = float(df["close"].rolling(200).mean().iloc[-1]) if len(df) >= 200 else float(df["close"].rolling(max(2, len(df)//2)).mean().iloc[-1])

    move = expected_move(price, v, horizon)
    buy = price - move
    sell = price + move
    stop = buy - (move * STOP_MULT)

    buy_prob = 1.0 - prob_hit_mc(price, buy, v, horizon, MC_SIMS, MC_METHOD)
    sell_prob = prob_hit_mc(price, sell, v, horizon, MC_SIMS, MC_METHOD)

    risk_per_share = max(buy - stop, 0.01)
    rrr = (sell - buy) / risk_per_share

    k = kelly(df)
    raw_qty = int((acct * k) / risk_per_share)

    cap_alloc_qty = int((acct * MAX_ALLOC_PCT) / max(price, 0.01))
    cap_risk_qty = int((acct * MAX_RISK_PCT) / risk_per_share)

    qty = max(0, min(raw_qty, cap_alloc_qty, cap_risk_qty))

    return TradeMetrics(
        price=price, vol=v, atr=a, ma50=ma50, ma200=ma200,
        buy=buy, sell=sell, stop=stop,
        buy_prob=float(buy_prob), sell_prob=float(sell_prob),
        rrr=float(rrr), kelly=float(k), qty=int(qty),
        expected_move=float(move),
        risk_per_share=float(risk_per_share),
        cap_alloc_qty=int(cap_alloc_qty),
        cap_risk_qty=int(cap_risk_qty),
    )


def score_decision(price: float, ma50: float, ma200: float, rrr: float,
                   sh: float, mdd: float, wr: float, pf: float) -> Tuple[int, str, str, List[str]]:
    score = 0
    reasons: List[str] = []

    if price > ma200 and price > ma50:
        score += 15; reasons.append("Uptrend: above MA50 & MA200")
    elif price > ma200:
        score += 10; reasons.append("Trend: above MA200")
    elif price > ma50:
        score += 5; reasons.append("Trend: above MA50 only")
    else:
        reasons.append("Trend: below MA50 & MA200")

    if rrr >= 3.0:
        score += 25; reasons.append("Risk/Reward: excellent")
    elif rrr >= 2.0:
        score += 20; reasons.append("Risk/Reward: strong")
    elif rrr >= 1.5:
        score += 12; reasons.append("Risk/Reward: acceptable")
    else:
        reasons.append("Risk/Reward: poor")

    if sh >= 1.5:
        score += 25; reasons.append("Backtest Sharpe: excellent")
    elif sh >= 1.0:
        score += 20; reasons.append("Backtest Sharpe: good")
    elif sh >= 0.5:
        score += 10; reasons.append("Backtest Sharpe: moderate")
    else:
        reasons.append("Backtest Sharpe: poor")

    if abs(mdd) <= 0.10:
        score += 15; reasons.append("Drawdown: low")
    elif abs(mdd) <= 0.20:
        score += 10; reasons.append("Drawdown: moderate")
    elif abs(mdd) <= 0.30:
        score += 5; reasons.append("Drawdown: high")
    else:
        reasons.append("Drawdown: very high")

    if wr >= 60:
        score += 10; reasons.append("Win rate: high")
    elif wr >= 50:
        score += 7; reasons.append("Win rate: positive")
    elif wr >= 40:
        score += 3; reasons.append("Win rate: below avg")
    else:
        reasons.append("Win rate: low")

    if pf >= 2.0:
        score += 10; reasons.append("Profit factor: strong")
    elif pf >= 1.5:
        score += 7; reasons.append("Profit factor: good")
    elif pf >= 1.0:
        score += 3; reasons.append("Profit factor: marginal")
    else:
        reasons.append("Profit factor: weak")

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
st.caption("Simple controls. Safe defaults. Educational only — not financial advice.")

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

# ---- Load data
if load_btn:
    stop_live()
    st.session_state.last_error = ""
    st.session_state.debug = {}

    with st.spinner(f"Loading {ticker}..."):
        df, dbg = load_historical(ticker, st.session_state.api_key, st.session_state.secret_key)
        q, qdbg = load_latest_quote(ticker, st.session_state.api_key, st.session_state.secret_key)
        st.session_state.debug = {"historical": dbg, "quote": qdbg}

        st.session_state.ticker = ticker
        st.session_state.latest_quote = q
        st.session_state.latest_price = mid_from_quote(q)

        if df is None:
            st.session_state.historical = None
            st.session_state.last_error = str(dbg.get("error") or "Failed to load historical data.")
        else:
            st.session_state.historical = df
            if st.session_state.latest_price is None:
                st.session_state.latest_price = float(df["close"].iloc[-1])

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

if st.session_state.live_active:
    st.autorefresh(interval=AUTO_REFRESH_MS, key="live_refresh")

# ---- Drain live data
drain(st.session_state.trade_q, st.session_state.trade_history, MAX_TRADES)
drain(st.session_state.quote_q, st.session_state.quote_history, MAX_QUOTES)

if st.session_state.quote_history:
    st.session_state.latest_quote = st.session_state.quote_history[-1]
    mid = mid_from_quote(st.session_state.latest_quote)
    if mid is not None:
        st.session_state.latest_price = mid

current_price = float(st.session_state.latest_price or df["close"].iloc[-1])

# ---- Compute
bt = backtest(df, horizon)
btr = bt_stats(bt)

tm = compute_trade_metrics(df=df, price=current_price, horizon=horizon, acct=float(account_size))
score, decision, emoji, reasons = score_decision(
    current_price, tm.ma50, tm.ma200, tm.rrr,
    btr.sharpe_ratio, btr.max_drawdown, btr.win_rate, btr.profit_factor
)

# ---- Dashboard layout
row = st.container()
left, right = row.columns([1.4, 1.0], vertical_alignment="top")

with left:
    st.subheader(f"{emoji} {st.session_state.ticker} — {decision}")
    st.caption(f"Price: ${current_price:,.2f} • Horizon: {horizon}d • Live: {'ON' if st.session_state.live_active else 'OFF'}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Buy", f"${tm.buy:,.2f}", f"{(tm.buy/current_price - 1)*100:.1f}%")
    k2.metric("Sell", f"${tm.sell:,.2f}", f"{(tm.sell/current_price - 1)*100:.1f}%")
    k3.metric("Stop", f"${tm.stop:,.2f}", f"{(tm.stop/current_price - 1)*100:.1f}%")
    k4.metric("Position", f"{tm.qty} shares")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Vol (ann.)", f"{tm.vol:.2f}")
    s2.metric("ATR", f"{tm.atr:.2f}")
    s3.metric("RRR", f"{tm.rrr:.2f}")
    s4.metric("Kelly (half)", f"{tm.kelly:.3f}")

    with st.expander("Why this decision"):
        for r in reasons:
            st.write(f"- {r}")
        st.caption(
            f"Safety caps: max alloc {int(MAX_ALLOC_PCT*100)}% (cap {tm.cap_alloc_qty} shares), "
            f"max risk {int(MAX_RISK_PCT*100)}% (cap {tm.cap_risk_qty} shares). "
            f"Backtest: entry next-day open, slippage {SLIPPAGE_BPS} bps."
        )

with right:
    st.plotly_chart(score_gauge(score, decision), use_container_width=True)

# ---- Tabs
tab1, tab2, tab3 = st.tabs(["Chart", "Backtest", "Live"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close"))
    if len(df) >= 50:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"].rolling(50).mean(), name="MA50"))
    if len(df) >= 200:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"].rolling(200).mean(), name="MA200"))

    fig.add_hline(y=tm.buy, line_dash="dot", annotation_text="Buy")
    fig.add_hline(y=tm.sell, line_dash="dot", annotation_text="Sell")
    fig.add_hline(y=tm.stop, line_dash="dot", annotation_text="Stop")

    fig.update_layout(height=520, hovermode="x unified", title=f"{st.session_state.ticker} Price")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", btr.total_trades)
    c2.metric("Win Rate", f"{btr.win_rate:.1f}%")
    c3.metric("Profit Factor", f"{btr.profit_factor:.2f}")
    c4.metric("Sharpe", f"{btr.sharpe_ratio:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Max Drawdown", f"{btr.max_drawdown*100:.1f}%")
    c6.metric("Total PnL", f"${btr.total_pnl:,.2f}")
    c7.metric("Avg Win", f"${btr.avg_win:,.2f}")
    c8.metric("Avg Loss", f"${btr.avg_loss:,.2f}")

    if not bt.empty:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(y=bt["CumPnL"], name="Cumulative PnL"))
        fig_bt.update_layout(height=360, title="Equity Curve")
        st.plotly_chart(fig_bt, use_container_width=True)

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
                st.text(f"Bid: ${bid:.2f} ({q.get('bid_size','?')})")
                st.text(f"Ask: ${ask:.2f} ({q.get('ask_size','?')})")
                st.text(f"Spread: ${q.get('spread', 0):.4f}")
            else:
                st.text("No quote yet...")

st.markdown("---")
st.caption("⚠️ Educational only. Not financial advice.")
