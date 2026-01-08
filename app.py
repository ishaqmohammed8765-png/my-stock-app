"""
Trading Dashboard Pro (Clean Edition)
- Alpaca historical + latest quote
- Optional live trades/quotes (websocket in thread w/ asyncio loop)
- Volatility, ATR, expected move, Monte Carlo hit probabilities
- Kelly sizing, simple backtest, scoring recommendation
Educational only. Not financial advice.
"""

from __future__ import annotations

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
log = logging.getLogger("dash")

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

DEFAULT_WS_FEED = "iex"


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


# =========================
# LIVE STREAM
# =========================
class RealtimeStream:
    def __init__(self, api_key: str, secret_key: str, ticker: str,
                 trade_q: queue.Queue, quote_q: queue.Queue, feed: str = DEFAULT_WS_FEED):
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
            payload = {"type": "trade", "ts": parse_ts(getattr(data, "timestamp", None)),
                       "symbol": getattr(data, "symbol", self.ticker), "price": px, "size": sz}
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
            payload = {"type": "quote", "ts": parse_ts(getattr(data, "timestamp", None)),
                       "symbol": getattr(data, "symbol", self.ticker),
                       "bid": bid, "ask": ask,
                       "bid_size": int(getattr(data, "bid_size", 0)),
                       "ask_size": int(getattr(data, "ask_size", 0)),
                       "spread": float(ask - bid)}
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
# ALPACA LOADERS (ROBUST)
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
                    days_back: int = 260, feed_str: str = "IEX") -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    dbg: Dict[str, Any] = {"ticker": ticker, "feed": feed_str, "steps": []}

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
        feed_val = DataFeed.SIP if feed_str.upper() == "SIP" else DataFeed.IEX

    # try with feed, fallback without feed for older alpaca-py
    try:
        if feed_val is not None:
            req = StockBarsRequest(**base, feed=feed_val)
            dbg["steps"].append("bars request WITH feed")
        else:
            req = StockBarsRequest(**base)
            dbg["steps"].append("bars request WITHOUT feed (no DataFeed)")
        bars = client.get_stock_bars(req)
    except TypeError as e:
        dbg["steps"].append(f"feed not supported -> retry: {e}")
        req = StockBarsRequest(**base)
        bars = client.get_stock_bars(req)

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
            dbg["steps"].append("MultiIndex reset_index fallback")

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df = df[df["symbol"] == ticker].copy()

    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        dbg["error"] = f"Missing columns: {sorted(list(required - set(df.columns)))}"
        dbg["columns"] = list(df.columns)
        return None, dbg

    df = df.sort_values("timestamp").reset_index(drop=True)
    dbg["rows"] = int(len(df))
    return (df if len(df) >= 30 else None), dbg


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
            shocks = np.random.normal(size=(sims, days))
        drift = -0.5 * (vol ** 2) * dt
        diffusion = vol * np.sqrt(dt) * shocks
        paths = S * np.exp(np.cumsum(drift + diffusion, axis=1))
        return float((paths >= K).any(axis=1).mean())
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
    f = (wr - ((1 - wr) / R)) * 0.5  # half-kelly
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
    if len(df) < horizon + MIN_HIST_DAYS:
        return pd.DataFrame()

    iters = min(len(df) - horizon, MAX_BT_ITERS)
    start = len(df) - horizon - iters

    out = []
    for i in range(iters):
        idx = start + i
        entry = float(df["close"].iloc[idx])
        if entry <= 0:
            continue

        v = annual_vol(df.iloc[: idx + 1])
        move = expected_move(entry, v, horizon)
        stop = entry - (move * STOP_MULT)

        window = df["close"].iloc[idx : idx + horizon]
        if window.empty:
            continue

        stop_hit = window[window <= stop]
        exit_px = float(stop_hit.iloc[0]) if not stop_hit.empty else float(window.iloc[-1])

        pnl = exit_px - entry
        ret = pnl / entry
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


def compute_trade_metrics(df: pd.DataFrame, price: float, horizon: int, method: str, acct: float) -> TradeMetrics:
    v = annual_vol(df)
    a = atr(df)

    ma50 = float(df["close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else price
    ma200 = float(df["close"].rolling(200).mean().iloc[-1]) if len(df) >= 200 else float(df["close"].rolling(max(2, len(df)//2)).mean().iloc[-1])

    move = expected_move(price, v, horizon)
    buy = price - move
    sell = price + move
    stop = buy - (move * STOP_MULT)

    buy_prob = 1.0 - prob_hit_mc(price, buy, v, horizon, MC_SIMS, method)
    sell_prob = prob_hit_mc(price, sell, v, horizon, MC_SIMS, method)

    risk_per_share = max(buy - stop, 0.01)
    rrr = (sell - buy) / risk_per_share

    k = kelly(df)
    qty = int((acct * k) / risk_per_share)

    return TradeMetrics(price, v, a, ma50, ma200, buy, sell, stop, buy_prob, sell_prob, rrr, k, max(qty, 0), move)


def score_decision(price: float, ma50: float, ma200: float, rrr: float, sh: float, mdd: float, wr: float, pf: float):
    score = 0
    reasons: List[str] = []

    # Trend (15)
    if price > ma200 and price > ma50:
        score += 15; reasons.append("✅ Strong uptrend (above MA50 & MA200)")
    elif price > ma200:
        score += 10; reasons.append("✅ Above MA200")
    elif price > ma50:
        score += 5; reasons.append("⚠️ Above MA50 only")
    else:
        reasons.append("❌ Below MA50 & MA200")

    # RRR (25)
    if rrr >= 3.0:
        score += 25; reasons.append("✅ Excellent risk/reward (RRR ≥ 3)")
    elif rrr >= 2.0:
        score += 20; reasons.append("✅ Strong risk/reward (RRR ≥ 2)")
    elif rrr >= 1.5:
        score += 12; reasons.append("⚠️ Acceptable risk/reward (RRR ≥ 1.5)")
    else:
        reasons.append("❌ Poor risk/reward")

    # Sharpe (25)
    if sh >= 1.5:
        score += 25; reasons.append("✅ Excellent Sharpe (≥ 1.5)")
    elif sh >= 1.0:
        score += 20; reasons.append("✅ Good Sharpe (≥ 1.0)")
    elif sh >= 0.5:
        score += 10; reasons.append("⚠️ Moderate Sharpe (≥ 0.5)")
    else:
        reasons.append("❌ Poor Sharpe")

    # Drawdown (15)
    if abs(mdd) <= 0.10:
        score += 15; reasons.append("✅ Low drawdown (≤ 10%)")
    elif abs(mdd) <= 0.20:
        score += 10; reasons.append("✅ Moderate drawdown (≤ 20%)")
    elif abs(mdd) <= 0.30:
        score += 5; reasons.append("⚠️ High drawdown (≤ 30%)")
    else:
        reasons.append("❌ Very high drawdown (> 30%)")

    # Winrate (10)
    if wr >= 60:
        score += 10; reasons.append("✅ High win rate (≥ 60%)")
    elif wr >= 50:
        score += 7; reasons.append("✅ Positive win rate (≥ 50%)")
    elif wr >= 40:
        score += 3; reasons.append("⚠️ Below-average win rate (≥ 40%)")
    else:
        reasons.append("❌ Low win rate (< 40%)")

    # Profit factor (10)
    if pf >= 2.0:
        score += 10; reasons.append("✅ Strong profit factor (≥ 2)")
    elif pf >= 1.5:
        score += 7; reasons.append("✅ Good profit factor (≥ 1.5)")
    elif pf >= 1.0:
        score += 3; reasons.append("⚠️ Marginal profit factor (≥ 1)")
    else:
        reasons.append("❌ Weak profit factor (< 1)")

    if score >= SCORE_STRONG_BUY:
        return score, "STRONG BUY", "🟢", reasons
    if score >= SCORE_BUY:
        return score, "BUY", "🟢", reasons
    if score >= SCORE_CONDITIONAL:
        reasons.append("💡 Timing matters — consider entry confirmation")
        return score, "CONDITIONAL BUY", "🟡", reasons
    if score >= SCORE_HOLD:
        reasons.append("⏸️ Wait for a better setup")
        return score, "HOLD", "🟡", reasons

    reasons.append("🛑 Risk too high vs reward")
    return score, "AVOID", "🔴", reasons


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Trading Dashboard Pro (Clean)", page_icon="📈", layout="wide")
st.title("📈 Trading Dashboard Pro (Clean)")
st.caption("Educational only — not financial advice.")

# Session state
if "api_ok" not in st.session_state:
    st.session_state.api_ok = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "secret_key" not in st.session_state:
    st.session_state.secret_key = ""
if "ticker" not in st.session_state:
    st.session_state.ticker = "NVDA"
if "historical" not in st.session_state:
    st.session_state.historical = None
if "latest_quote" not in st.session_state:
    st.session_state.latest_quote = None
if "latest_price" not in st.session_state:
    st.session_state.latest_price = None
if "trade_q" not in st.session_state:
    st.session_state.trade_q = queue.Queue(maxsize=QUEUE_MAX)
if "quote_q" not in st.session_state:
    st.session_state.quote_q = queue.Queue(maxsize=QUEUE_MAX)
if "trade_history" not in st.session_state:
    st.session_state.trade_history = []
if "quote_history" not in st.session_state:
    st.session_state.quote_history = []
if "live_active" not in st.session_state:
    st.session_state.live_active = False
if "ws_handler" not in st.session_state:
    st.session_state.ws_handler = None
if "ws_thread" not in st.session_state:
    st.session_state.ws_thread = None
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "debug" not in st.session_state:
    st.session_state.debug = {}


def stop_live():
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


# Secrets (optional)
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

# API input
if not st.session_state.api_ok:
    st.info("Enter your Alpaca API keys.")
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

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    account_size = st.number_input("Account Size ($)", 1000, 5_000_000, 10_000, 1000)
    horizon = st.slider("Time Horizon (days)", 5, 120, 30, 5)
    mc_method = st.selectbox("Monte Carlo shocks", ["student_t", "normal"], index=0)

    st.markdown("---")
    bars_feed = st.selectbox("Historical feed", ["IEX", "SIP"], index=0)
    ws_feed = st.selectbox("Live WS feed", ["iex", "sip"], index=0)

    st.markdown("---")
    if st.session_state.live_active:
        if st.button("🔴 Stop Live Feed", use_container_width=True):
            stop_live()
            st.rerun()
    else:
        if st.button("🟢 Start Live Feed", use_container_width=True):
            if st.session_state.historical is None:
                st.warning("Load a ticker first.")
            else:
                stop_live()
                handler = RealtimeStream(
                    st.session_state.api_key,
                    st.session_state.secret_key,
                    st.session_state.ticker,
                    st.session_state.trade_q,
                    st.session_state.quote_q,
                    feed=ws_feed,
                )
                t = threading.Thread(target=handler.run, daemon=True)
                t.start()
                st.session_state.ws_handler = handler
                st.session_state.ws_thread = t
                st.session_state.live_active = True
                st.rerun()

# Main: ticker load
c1, c2, c3 = st.columns([2.0, 1.0, 1.0])
with c1:
    ticker = st.text_input("Ticker", value=st.session_state.ticker).upper().strip()

with c2:
    if st.button("📥 Load Data", type="primary", use_container_width=True):
        stop_live()
        st.session_state.last_error = ""
        st.session_state.debug = {}
        with st.spinner(f"Loading {ticker}..."):
            try:
                df, dbg = load_historical(ticker, st.session_state.api_key, st.session_state.secret_key, feed_str=bars_feed)
                q, qdbg = load_latest_quote(ticker, st.session_state.api_key, st.session_state.secret_key)
                st.session_state.debug = {"historical": dbg, "quote": qdbg}

                if df is None:
                    st.session_state.historical = None
                    st.session_state.ticker = ticker
                    st.session_state.latest_quote = q
                    st.session_state.latest_price = mid_from_quote(q)
                    st.session_state.last_error = f"Failed to load historical for {ticker} (open Debug)."
                else:
                    st.session_state.historical = df
                    st.session_state.ticker = ticker
                    st.session_state.latest_quote = q
                    st.session_state.latest_price = mid_from_quote(q) if q else float(df["close"].iloc[-1])
            except Exception as e:
                st.session_state.last_error = f"Load error: {e}"
        st.rerun()

with c3:
    if st.button("🧹 Reset", use_container_width=True):
        stop_live()
        st.session_state.historical = None
        st.session_state.latest_quote = None
        st.session_state.latest_price = None
        st.session_state.trade_history = []
        st.session_state.quote_history = []
        st.session_state.last_error = ""
        st.session_state.debug = {}
        st.rerun()

if st.session_state.last_error:
    st.error(st.session_state.last_error)

if st.session_state.debug:
    with st.expander("🛠 Debug"):
        st.json(st.session_state.debug)

df = st.session_state.historical
if df is None:
    st.info("Load a ticker to begin.")
    st.stop()

# Auto refresh on live
if st.session_state.live_active:
    st.autorefresh(interval=AUTO_REFRESH_MS, key="live_refresh")

# Drain queues -> history
drain(st.session_state.trade_q, st.session_state.trade_history, MAX_TRADES)
drain(st.session_state.quote_q, st.session_state.quote_history, MAX_QUOTES)

# Update quote/price from live
if st.session_state.quote_history:
    st.session_state.latest_quote = st.session_state.quote_history[-1]
    mid = mid_from_quote(st.session_state.latest_quote)
    if mid is not None:
        st.session_state.latest_price = mid

current_price = float(st.session_state.latest_price or df["close"].iloc[-1])

# Compute analytics
bt = backtest(df, horizon)
btr = bt_stats(bt)
tm = compute_trade_metrics(df, current_price, horizon, mc_method, account_size)
score, decision, emoji, reasons = score_decision(
    current_price, tm.ma50, tm.ma200, tm.rrr, btr.sharpe_ratio, btr.max_drawdown, btr.win_rate, btr.profit_factor
)

st.markdown(f"## {emoji} Recommendation: **{decision}**")
st.markdown(f"**Score: {score}/100**")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Price", f"${current_price:,.2f}")
m1.metric("Volatility (ann.)", f"{tm.vol:.2f}")

m2.metric("Buy Level", f"${tm.buy:,.2f}", f"{(tm.buy/current_price - 1)*100:.1f}%")
m2.metric("Stop Loss", f"${tm.stop:,.2f}", f"{(tm.stop/current_price - 1)*100:.1f}%")

m3.metric("Sell Target", f"${tm.sell:,.2f}", f"{(tm.sell/current_price - 1)*100:.1f}%")
m3.metric("RRR", f"{tm.rrr:.2f}")

m4.metric("Kelly Fraction", f"{tm.kelly:.3f}")
m4.metric("Position Size", f"{tm.qty} shares")

st.markdown("### 📋 Summary")
for r in reasons:
    st.write(f"- {r}")

# Chart
st.markdown("### 📈 Price + Levels")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close"))

if len(df) >= 50:
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"].rolling(50).mean(), name="MA50"))
if len(df) >= 200:
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"].rolling(200).mean(), name="MA200"))

fig.add_hline(y=tm.buy, line_dash="dot", annotation_text="Buy")
fig.add_hline(y=tm.sell, line_dash="dot", annotation_text="Sell")
fig.add_hline(y=tm.stop, line_dash="dot", annotation_text="Stop")

fig.update_layout(height=520, hovermode="x unified", title=f"{st.session_state.ticker} Analysis")
st.plotly_chart(fig, use_container_width=True)

# Backtest
st.markdown("### 📊 Backtest")
b1, b2, b3, b4 = st.columns(4)
b1.metric("Trades", btr.total_trades)
b2.metric("Win Rate", f"{btr.win_rate:.1f}%")
b3.metric("Profit Factor", f"{btr.profit_factor:.2f}")
b4.metric("Sharpe", f"{btr.sharpe_ratio:.2f}")

if not bt.empty:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=bt["CumPnL"], name="Cum PnL"))
    fig2.update_layout(height=350, title="Equity Curve")
    st.plotly_chart(fig2, use_container_width=True)

# Live
if st.session_state.live_active:
    st.markdown("### 🔴 Live Market Feed")
    l1, l2 = st.columns(2)

    with l1:
        st.markdown("**Recent Trades**")
        for t in reversed(st.session_state.trade_history[-10:]):
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
