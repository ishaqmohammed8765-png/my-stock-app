"""
Professional Real-Time Trading Dashboard
Combines Alpaca WebSocket streaming with advanced quantitative analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
import threading
import queue
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, Any, List
import logging
from scipy import stats
from functools import wraps

# =====================================================
# LOGGING CONFIGURATION
# =====================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================
# CONSTANTS
# =====================================================
VOL_FLOOR = 0.10
VOL_CAP = 1.5
KELLY_MAX = 0.15
KELLY_MIN = 0.01
MONTE_CARLO_SIMS = 1000
RISK_FREE_RATE = 0.045  # 4.5% default

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Real-Time Trading Dashboard Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# WEBSOCKET HANDLER CLASS
# =====================================================
class AlpacaRealtimeHandler:
    """Enhanced WebSocket handler with trade and quote streaming"""
    
    def __init__(self, api_key: str, secret_key: str, ticker: str, 
                 trade_queue: queue.Queue, quote_queue: queue.Queue):
        self.api_key = api_key
        self.secret_key = secret_key
        self.ticker = ticker.upper()
        self.trade_queue = trade_queue
        self.quote_queue = quote_queue
        self.stream = None
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_price = None
        self.trade_count = 0
        
    async def trade_handler(self, data):
        """Handle incoming trades"""
        try:
            trade_data = {
                'type': 'trade',
                'timestamp': data.timestamp,
                'symbol': data.symbol,
                'price': data.price,
                'size': data.size,
                'exchange': getattr(data, 'exchange', 'N/A')
            }
            self.last_price = data.price
            self.trade_count += 1
            self.trade_queue.put(trade_data)
            logger.info(f"Trade #{self.trade_count}: {data.symbol} @ ${data.price:.2f}")
        except Exception as e:
            logger.error(f"Trade handler error: {e}")
    
    async def quote_handler(self, data):
        """Handle incoming quotes (bid/ask)"""
        try:
            quote_data = {
                'type': 'quote',
                'timestamp': data.timestamp,
                'symbol': data.symbol,
                'bid': data.bid_price,
                'ask': data.ask_price,
                'bid_size': data.bid_size,
                'ask_size': data.ask_size,
                'spread': data.ask_price - data.bid_price
            }
            self.quote_queue.put(quote_data)
        except Exception as e:
            logger.error(f"Quote handler error: {e}")
    
    def start_stream(self):
        """Start WebSocket with auto-reconnection"""
        self.is_running = True
        
        while self.is_running and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                logger.info(f"🔌 Connecting to Alpaca for {self.ticker}...")
                
                self.stream = StockDataStream(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    feed='iex'
                )
                
                self.stream.subscribe_trades(self.trade_handler, self.ticker)
                self.stream.subscribe_quotes(self.quote_handler, self.ticker)
                
                self.stream.run()
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.reconnect_attempts += 1
                
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    wait_time = min(2 ** self.reconnect_attempts, 60)
                    logger.info(f"Reconnecting in {wait_time}s... ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
                    time.sleep(wait_time)
                else:
                    error_msg = {'type': 'error', 'message': 'Connection failed after maximum retries'}
                    self.trade_queue.put(error_msg)
                    break
    
    def stop_stream(self):
        """Stop WebSocket gracefully"""
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop()
                logger.info("✅ WebSocket stopped")
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")


# =====================================================
# HISTORICAL DATA FUNCTIONS
# =====================================================
@st.cache_data(ttl=3600)
def fetch_historical_data(ticker: str, api_key: str, secret_key: str, 
                         days_back: int = 120) -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV data from Alpaca"""
    try:
        client = StockHistoricalDataClient(api_key, secret_key)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=days_back),
            end=datetime.now()
        )
        
        bars = client.get_stock_bars(request_params)
        df = bars.df
        
        if ticker in df.index.get_level_values(0):
            df = df.xs(ticker, level=0)
            df = df.reset_index()
            return df
        
        return None
        
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return None


def get_latest_quote(ticker: str, api_key: str, secret_key: str) -> Optional[Dict]:
    """Get latest bid/ask quote"""
    try:
        client = StockHistoricalDataClient(api_key, secret_key)
        request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        quotes = client.get_stock_latest_quote(request)
        
        if ticker in quotes:
            quote = quotes[ticker]
            return {
                'bid': quote.bid_price,
                'ask': quote.ask_price,
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'spread': quote.ask_price - quote.bid_price
            }
    except Exception as e:
        logger.error(f"Error fetching quote: {e}")
    return None


# =====================================================
# QUANTITATIVE ANALYSIS FUNCTIONS
# =====================================================
def annual_volatility(df: pd.DataFrame) -> float:
    """Calculate annualized volatility using EWMA"""
    if df.empty or len(df) < 10:
        return 0.30
    
    returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    if len(returns) < 10:
        return 0.30
    
    vol = returns.ewm(span=20).std().iloc[-1] * np.sqrt(252)
    return float(np.clip(vol, VOL_FLOOR, VOL_CAP))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range"""
    if len(df) < period:
        return 0.0
    
    df = df.copy()
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    
    atr = df['tr'].rolling(window=period).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else 0.0


def expected_move(price: float, vol: float, days: int = 30) -> float:
    """Expected price move over given period"""
    return price * vol * np.sqrt(days / 252)


def prob_hit_mc_advanced(S: float, K: float, vol: float, days: int = 30, 
                        sims: int = MONTE_CARLO_SIMS, method: str = "student_t",
                        df_hist: Optional[pd.DataFrame] = None) -> float:
    """
    Advanced Monte Carlo simulation with fat-tail distributions
    
    Args:
        S: Current price
        K: Target price
        vol: Annual volatility
        days: Time horizon
        sims: Number of simulations
        method: "student_t", "bootstrap", or "normal"
        df_hist: Historical data for bootstrap
    """
    if vol <= 0 or S <= 0 or K <= 0:
        return 0.0
    
    dt = 1/252
    
    if method == "student_t":
        dof = 5
        random_shocks = stats.t.rvs(df=dof, size=(sims, days)) / np.sqrt(dof / (dof - 2))
        
    elif method == "bootstrap" and df_hist is not None:
        returns = np.log(df_hist["close"] / df_hist["close"].shift(1)).dropna()
        if len(returns) < 30:
            dof = 5
            random_shocks = stats.t.rvs(df=dof, size=(sims, days)) / np.sqrt(dof / (dof - 2))
        else:
            hist_returns = returns.values
            random_indices = np.random.randint(0, len(hist_returns), size=(sims, days))
            random_shocks = hist_returns[random_indices]
            current_vol = np.std(hist_returns) * np.sqrt(252)
            if current_vol > 0:
                random_shocks = random_shocks * (vol / current_vol)
    else:
        random_shocks = np.random.normal(size=(sims, days))
    
    price_paths = S * np.exp(np.cumsum((-0.5*vol**2)*dt + vol*np.sqrt(dt)*random_shocks, axis=1))
    hits = (price_paths >= K).any(axis=1)
    
    return float(hits.mean())


def kelly_fraction(df: pd.DataFrame) -> float:
    """Calculate Kelly Criterion for position sizing"""
    if df.empty or len(df) < 30:
        return 0.02
    
    returns = df["close"].pct_change().dropna()
    
    if len(returns) < 30:
        return 0.02
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(wins) < 10 or len(losses) < 10:
        return 0.02
    
    win_rate = len(wins) / len(returns)
    loss_rate = 1 - win_rate
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    
    if avg_loss == 0:
        return 0.02
    
    R = avg_win / avg_loss
    fraction = win_rate - (loss_rate / R)
    
    return float(np.clip(fraction * 0.5, KELLY_MIN, KELLY_MAX))


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)
    return float(sharpe)


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    if len(cumulative_returns) == 0:
        return 0.0
    
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / (running_max + 1e-10)
    return float(drawdown.min())


def backtest_vectorized(df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """Vectorized backtest with no look-ahead bias"""
    if df.empty or len(df) < days + 30:
        return pd.DataFrame({"PnL": [], "Return": []})
    
    max_iterations = min(len(df) - days, 45)
    start_idx = len(df) - days - max_iterations
    
    pnl = np.zeros(max_iterations)
    returns = np.zeros(max_iterations)
    
    for i in range(max_iterations):
        actual_idx = start_idx + i
        entry_price = df["close"].iloc[actual_idx]
        
        historical_data = df.iloc[:actual_idx+1]
        if len(historical_data) > 20:
            log_returns = np.log(historical_data["close"] / historical_data["close"].shift(1)).dropna()
            vol = log_returns.ewm(span=20).std().iloc[-1] * np.sqrt(252)
            vol = float(np.clip(vol, VOL_FLOOR, VOL_CAP))
        else:
            vol = 0.30
        
        move = entry_price * vol * np.sqrt(days / 252)
        stop_price = entry_price - move * 0.5
        
        window = df["close"].iloc[actual_idx:actual_idx+days]
        
        stop_hit_mask = window <= stop_price
        if stop_hit_mask.any():
            exit_price = window[stop_hit_mask].iloc[0]
        else:
            exit_price = window.iloc[-1]
        
        pnl[i] = exit_price - entry_price
        returns[i] = (exit_price - entry_price) / entry_price
    
    bt = pd.DataFrame({"PnL": pnl, "Return": returns})
    bt["CumPnL"] = bt["PnL"].cumsum()
    bt["CumReturn"] = (1 + bt["Return"]).cumprod() - 1
    
    return bt


def evaluate_trade_decision(RRR: float, sharpe: float, max_dd: float, win_rate: float,
                           profit_factor: float, buy_prob: float, sell_prob: float,
                           price: float, ma_50: float, ma_200: float) -> tuple:
    """
    Comprehensive trade evaluation with trend filter
    Returns: (score, decision, color, emoji, reasons)
    """
    score = 0
    reasons = []
    
    # Trend Filter (0-15 points)
    if price > ma_200 and price > ma_50:
        score += 15
        reasons.append("✅ Strong uptrend (above 50 & 200 MA)")
    elif price > ma_200:
        score += 10
        reasons.append("✅ Above 200 MA")
    elif price > ma_50:
        score += 5
        reasons.append("⚠️ Above 50 MA only")
    else:
        score += 0
        reasons.append("❌ Below both MAs (falling knife risk)")
    
    # Risk-Reward Ratio (0-25 points)
    if RRR >= 3.0:
        score += 25
        reasons.append("✅ Excellent RRR (≥3.0)")
    elif RRR >= 2.0:
        score += 20
        reasons.append("✅ Strong RRR (≥2.0)")
    elif RRR >= 1.5:
        score += 12
        reasons.append("⚠️ Acceptable RRR (≥1.5)")
    else:
        score += 0
        reasons.append("❌ Poor RRR (<1.5)")
    
    # Sharpe Ratio (0-25 points)
    if sharpe >= 1.5:
        score += 25
        reasons.append("✅ Excellent Sharpe (≥1.5)")
    elif sharpe >= 1.0:
        score += 20
        reasons.append("✅ Good Sharpe (≥1.0)")
    elif sharpe >= 0.5:
        score += 10
        reasons.append("⚠️ Moderate Sharpe (≥0.5)")
    else:
        score += 0
        reasons.append("❌ Poor Sharpe (<0.5)")
    
    # Max Drawdown (0-15 points)
    if abs(max_dd) <= 0.10:
        score += 15
        reasons.append("✅ Low drawdown (≤10%)")
    elif abs(max_dd) <= 0.20:
        score += 10
        reasons.append("✅ Moderate drawdown (≤20%)")
    elif abs(max_dd) <= 0.30:
        score += 5
        reasons.append("⚠️ High drawdown (≤30%)")
    else:
        score += 0
        reasons.append("❌ Very high drawdown (>30%)")
    
    # Win Rate (0-10 points)
    if win_rate >= 60:
        score += 10
        reasons.append("✅ High win rate (≥60%)")
    elif win_rate >= 50:
        score += 7
        reasons.append("✅ Positive win rate (≥50%)")
    elif win_rate >= 40:
        score += 3
        reasons.append("⚠️ Below-average win rate (≥40%)")
    else:
        score += 0
        reasons.append("❌ Low win rate (<40%)")
    
    # Profit Factor (0-10 points)
    if profit_factor >= 2.0:
        score += 10
        reasons.append("✅ Strong profit factor (≥2.0)")
    elif profit_factor >= 1.5:
        score += 7
        reasons.append("✅ Good profit factor (≥1.5)")
    elif profit_factor >= 1.0:
        score += 3
        reasons.append("⚠️ Marginal profit factor (≥1.0)")
    else:
        score += 0
        reasons.append("❌ Negative profit factor (<1.0)")
    
    # Determine decision
    if score >= 80:
        decision = "STRONG BUY"
        color = "success"
        emoji = "🟢"
    elif score >= 65:
        decision = "BUY"
        color = "success"
        emoji = "🟢"
    elif score >= 50:
        decision = "CONDITIONAL BUY"
        color = "warning"
        emoji = "🟡"
        reasons.append("💡 Review timing carefully")
    elif score >= 35:
        decision = "HOLD"
        color = "warning"
        emoji = "🟡"
        reasons.append("⏸️ Wait for better setup")
    else:
        decision = "AVOID"
        color = "error"
        emoji = "🔴"
        reasons.append("🛑 Risk too high")
    
    return score, decision, color, emoji, reasons


# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if 'trade_queue' not in st.session_state:
    st.session_state.trade_queue = queue.Queue()
if 'quote_queue' not in st.session_state:
    st.session_state.quote_queue = queue.Queue()
if 'websocket_thread' not in st.session_state:
    st.session_state.websocket_thread = None
if 'websocket_handler' not in st.session_state:
    st.session_state.websocket_handler = None
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'quote_history' not in st.session_state:
    st.session_state.quote_history = []
if 'latest_price' not in st.session_state:
    st.session_state.latest_price = None
if 'latest_quote' not in st.session_state:
    st.session_state.latest_quote = None
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = ""
if 'calculations_cache' not in st.session_state:
    st.session_state.calculations_cache = {}


# =====================================================
# SIDEBAR CONFIGURATION
# =====================================================
with st.sidebar:
    st.markdown("### 📈 Real-Time Trading Dashboard Pro")
    st.markdown("---")
    
    # API Configuration
    st.subheader("🔐 API Configuration")
    try:
        api_key = st.secrets["ALPACA_KEY"]
        secret_key = st.secrets["ALPACA_SECRET"]
        st.success("✅ API Keys loaded")
    except:
        st.warning("⚠️ Using manual input")
        api_key = st.text_input("API Key", type="password", key="api_key")
        secret_key = st.text_input("Secret Key", type="password", key="secret_key")
    
    st.markdown("---")
    
    # Stock Selection
    st.subheader("📊 Stock Selection")
    ticker_input = st.text_input("Ticker Symbol", value="NVDA", key="ticker_input").upper()
    
    col1, col2 = st.columns(2)
    with col1:
        load_hist_btn = st.button("📥 Load Data", type="primary", use_container_width=True)
    with col2:
        start_stream_btn = st.button("🚀 Go Live", type="secondary", use_container_width=True)
    
    stop_stream_btn = st.button("⏹️ Stop Stream", use_container_width=True)
    
    st.markdown("---")
    
    # Configuration
    st.subheader("⚙️ Configuration")
    account_size = st.number_input("Account Size ($)", min_value=1000, value=100000, step=1000)
    sim_days = st.slider("Simulation Days", 7, 90, 30)
    mc_method = st.selectbox(
        "Monte Carlo Method",
        ["student_t", "bootstrap", "normal"],
        format_func=lambda x: {
            "student_t": "Student's t (Fat Tails)",
            "bootstrap": "Historical Bootstrap",
            "normal": "Standard Normal"
        }[x]
    )
    
    st.markdown("---")
    st.caption("💡 **Real-time data** from Alpaca IEX feed")
    st.caption("🔬 **Advanced analytics** with fat-tail modeling")
    st.caption("🎯 **Institutional risk management**")


# =====================================================
# LOAD HISTORICAL DATA
# =====================================================
if load_hist_btn and api_key and secret_key:
    with st.spinner(f"Loading historical data for {ticker_input}..."):
        hist_data = fetch_historical_data(ticker_input, api_key, secret_key)
        if hist_data is not None and not hist_data.empty:
            st.session_state.historical_data = hist_data
            st.session_state.current_ticker = ticker_input
            st.session_state.calculations_cache = {}
            
            # Get latest quote
            quote = get_latest_quote(ticker_input, api_key, secret_key)
            if quote:
                st.session_state.latest_quote = quote
                st.session_state.latest_price = (quote['bid'] + quote['ask']) / 2
            
            st.sidebar.success(f"✅ Loaded {len(hist_data)} days")
            st.rerun()
        else:
            st.sidebar.error(f"❌ Could not load {ticker_input}")


# =====================================================
# START/STOP WEBSOCKET STREAM
# =====================================================
if start_stream_btn and api_key and secret_key and st.session_state.current_ticker:
    if st.session_state.websocket_thread and st.session_state.websocket_thread.is_alive():
        st.sidebar.warning("Stream already active")
    else:
        st.session_state.websocket_handler = AlpacaRealtimeHandler(
            api_key=api_key,
            secret_key=secret_key,
            ticker=st.session_state.current_ticker,
            trade_queue=st.session_state.trade_queue,
            quote_queue=st.session_state.quote_queue
        )
        
        st.session_state.websocket_thread = threading.Thread(
            target=st.session_state.websocket_handler.start_stream,
            daemon=True
        )
        st.session_state.websocket_thread.start()
        st.sidebar.success(f"🟢 Live stream started")
        st.rerun()

if stop_stream_btn:
    if st.session_state.websocket_handler:
        st.session_state.websocket_handler.stop_stream()
        st.session_state.websocket_thread = None
        st.session_state.websocket_handler = None
        st.sidebar.info("🛑 Stream stopped")
        st.rerun()


# =====================================================
# PROCESS QUEUE DATA
# =====================================================
is_streaming = (st.session_state.websocket_thread and 
               st.session_state.websocket_thread.is_alive())

# Process trade queue
try:
    while not st.session_state.trade_queue.empty():
        data = st.session_state.trade_queue.get_nowait()
        
        if data.get('type') == 'error':
            st.error(f"❌ {data.get('message')}")
        elif data.get('type') == 'trade':
            st.session_state.trade_history.append(data)
            st.session_state.latest_price = data['price']
            
            if len(st.session_state.trade_history) > 100:
                st.session_state.trade_history = st.session_state.trade_history[-100:]
except queue.Empty:
    pass

# Process quote queue
try:
    while not st.session_state.quote_queue.empty():
        data = st.session_state.quote_queue.get_nowait()
        
        if data.get('type') == 'quote':
            st.session_state.quote_history.append(data)
            st.session_state.latest_quote = {
                'bid': data['bid'],
                'ask': data['ask'],
                'bid_size': data['bid_size'],
                'ask_size': data['ask_size'],
                'spread': data['spread']
            }
            
            if len(st.session_state.quote_history) > 50:
                st.session_state.quote_history = st.session_state.quote_history[-50:]
except queue.Empty:
    pass


# =====================================================
# MAIN DASHBOARD
# =====================================================
st.title("📈 Real-Time Trading Dashboard Pro")

# Stream status indicator
if is_streaming:
    st.success(f"🟢 **LIVE** - Streaming {st.session_state.current_ticker} | Trades: {len(st.session_state.trade_history)}")
else:
    st.info("⚪ **OFFLINE** - Load historical data and click 'Go Live' to start")

if not st.session_state.historical_data or st.session_state.historical_data.empty:
    st.warning("👈 Load historical data from the sidebar to begin analysis")
    st.stop()

# =====================================================
# CALCULATIONS
# =====================================================
df = st.session_state.historical_data
ticker = st.session_state.current_ticker

cache_key = f"{ticker}_{sim_days}_{mc_method}_{account_size}"

if cache_key not in st.session_state.calculations_cache:
    # Use real-time price if available, else historical
    if st.session_state.latest_price:
        price = float(st.session_state.latest_price)
    else:
        price = float(df["close"].iloc[-1])
    
    vol = annual_volatility(df)
    atr = calculate_atr(df)
    
    ma_50 = df["close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else price
    ma_200 = df["close"].rolling(200).mean().iloc[-1] if len(df) >= 200 else df["close"].rolling(len(df)//2).mean().iloc[-1]
    
    move = expected_move(price, vol, sim_days)
    buy = price - move
    sell = price + move
    stop = buy - move * 0.5
    
    buy_prob = 1 - prob_hit_mc_advanced(price, buy, vol, sim_days, MONTE_CARLO_SIMS, mc_method, df)
    sell_prob = prob_hit_mc_advanced(price, sell, vol, sim_days, MONTE_CARLO_SIMS, mc_method, df)
    
    RRR = (sell - buy) / max(buy - stop, 0.01)
    kelly = kelly_fraction(df)
    qty = int((account_size * kelly) / max(buy - stop, 0.01))
    
    st.session_state.calculations_cache[cache_key] = {
        "price": price, "vol": vol, "atr": atr, "ma_50": ma_50, "ma_200": ma_200,
        "move": move, "buy": buy, "sell": sell, "stop": stop,
        "buy_prob": buy_prob, "sell_prob": sell_prob,
        "RRR": RRR, "kelly": kelly, "qty": qty
    }

calc = st.session_state.calculations_cache[cache_key]
price = calc["price"]
vol = calc["vol"]
atr = calc["atr"]
ma_50 = calc["ma_50"]
ma_200 = calc["ma_200"]
buy = calc["buy"]
sell = calc["sell"]
stop = calc["stop"]
buy_prob = calc["buy_prob"]
sell_prob = calc["sell_prob"]
RRR = calc["RRR"]
kelly = calc["kelly"]
qty = calc["qty"]


# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💰 Trade Setup", "📊 Live Chart", "📈 Order Book", 
    "📜 Backtest", "📁 Portfolio", "🚀 Scanner"
])

# =====================================================
# TAB 1: TRADE SETUP
# =====================================================
with tab1:
    # Real-time metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.metric(
        "Current Price",
        f"${price:.2f}",
        delta=f"{((price/df['close'].iloc[-2] - 1)*100):.2f}%" if len(df) > 1 else None
    )
    col2.metric("50-Day MA", f"${ma_50:.2f}", 
                delta=f"{((price/ma_50 - 1)*100):.1f}%")
    col3.metric("200-Day MA", f"${ma_200:.2f}",
                delta=f"{((price/ma_200 - 1)*100):.1f}%")
    col4.metric("Volatility", f"{vol*100:.1f}%")
    col5.metric("ATR (14)", f"${atr:.2f}")
    col6.metric("Kelly %", f"{kelly*100:.1f}%")
    
    # Live order book metrics
    if st.session_state.latest_quote:
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        quote = st.session_state.latest_quote
        mid_price = (quote['bid'] + quote['ask']) / 2
        
        col1.metric("💵 Bid", f"${quote['bid']:.2f}", 
                   delta=f"{quote['bid_size']:,.0f} shares",
                   delta_color="off")
        col2.metric("💰 Ask", f"${quote['ask']:.2f}",
                   delta=f"{quote['ask_size']:,.0f} shares",
                   delta_color="off")
        col3.metric("📊 Mid", f"${mid_price:.2f}")
        col4.metric("📏 Spread", f"${quote['spread']:.4f}")
        col5.metric("📐 Spread %", f"{(quote['spread']/mid_price)*100:.3f}%")
    
    # Trend status
    st.markdown("---")
    if price > ma_50 and price > ma_200:
        st.success("🟢 **Strong Uptrend** - Above 50 & 200-day MAs")
    elif price > ma_50:
        st.info("🟡 **Moderate Trend** - Above 50-day MA")
    elif price > ma_200:
        st.warning("🟡 **Mixed Signals** - Above 200-day MA only")
    else:
        st.error("🔴 **Downtrend** - Below both MAs (falling knife risk)")
    
    st.markdown("---")
    
    # Trade levels
    st.subheader("📍 Quantitative Trade Levels")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🟢 Buy Target", f"${buy:.2f}", f"{buy_prob*100:.1f}% prob")
        st.caption(f"Entry if price drops {((price-buy)/price)*100:.1f}%")
    
    with col2:
        st.metric("🔵 Sell Target", f"${sell:.2f}", f"{sell_prob*100:.1f}% prob")
        st.caption(f"Profit target at +{((sell-buy)/buy)*100:.1f}%")
    
    with col3:
        st.metric("🔴 Stop-Loss", f"${stop:.2f}")
        st.caption(f"Max risk: ${buy - stop:.2f}/share")
    
    st.markdown("---")
    
    # Backtest for decision
    bt = backtest_vectorized(df, sim_days)
    
    if not bt.empty and len(bt) > 0:
        win_rate = (bt["PnL"] > 0).mean() * 100
        profit_factor = bt[bt["PnL"]>0]["PnL"].sum() / max(abs(bt[bt["PnL"]<0]["PnL"].sum()), 0.01)
        max_dd = calculate_max_drawdown(bt["CumReturn"].values)
        sharpe = calculate_sharpe_ratio(bt["Return"].values)
    else:
        win_rate = profit_factor = max_dd = sharpe = 0
    
    # Trade decision
    score, decision, color, emoji, reasons = evaluate_trade_decision(
        RRR, sharpe, max_dd, win_rate, profit_factor, buy_prob, sell_prob, 
        price, ma_50, ma_200
    )
    
    st.subheader("🎯 AI Trade Recommendation")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Decision Score", f"{score}/100")
        if color == "success":
            st.success(f"{emoji} **{decision}**")
        elif color == "warning":
            st.warning(f"{emoji} **{decision}**")
        else:
            st.error(f"{emoji} **{decision}**")
    
    with col2:
        st.markdown("**Analysis Factors:**")
        for reason in reasons:
            st.markdown(f"- {reason}")
    
    st.markdown("---")
    
    # Position sizing
    st.subheader("⚖️ Institutional Position Sizing")
    col1, col2, col3, col4 = st.columns(4)
    
    expected_profit = (sell - buy) * qty
    max_loss = (buy - stop) * qty
    
    col1.metric("Shares", f"{qty:,}")
    col2.metric("Risk/Reward", f"{RRR:.2f}x")
    col3.metric("Expected Profit", f"${expected_profit:,.2f}")
    col4.metric("Maximum Risk", f"${max_loss:,.2f}")
    
    st.info(f"💡 Position value: **${qty * buy:,.2f}** ({(qty*buy/account_size)*100:.1f}% of account)")
    
    # Performance metrics
    st.subheader("📊 Historical Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Win Rate", f"{win_rate:.1f}%")
    col2.metric("Profit Factor", f"{profit_factor:.2f}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col4.metric("Max Drawdown", f"{max_dd*100:.1f}%")
    
    st.markdown("---")
    
    # Add to portfolio
    if st.button("➕ Add to Portfolio", type="primary", use_container_width=True):
        trade = {
            "Ticker": ticker,
            "Buy": round(buy, 2),
            "Sell": round(sell, 2),
            "Stop": round(stop, 2),
            "Quantity": qty,
            "Expected PnL": round(expected_profit, 2),
            "Max Loss": round(max_loss, 2),
            "RRR": round(RRR, 2),
            "Decision": decision,
            "Score": score,
            "Entry Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.portfolio.append(trade)
        st.success(f"✅ Added {ticker} to portfolio!")
        st.balloons()


# =====================================================
# TAB 2: LIVE CHART
# =====================================================
with tab2:
    st.subheader(f"{ticker} - Real-Time Price Chart")
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(f'{ticker} Price with Trade Levels', 'Volume'),
        shared_xaxes=True
    )
    
    # Historical price
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["close"], name="Price",
                  line=dict(color="blue", width=2)),
        row=1, col=1
    )
    
    # Moving averages
    if len(df) >= 50:
        ma50 = df["close"].rolling(50).mean()
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=ma50, name="50 MA",
                      line=dict(color="orange", width=1, dash="dot")),
            row=1, col=1
        )
    
    if len(df) >= 200:
        ma200 = df["close"].rolling(200).mean()
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=ma200, name="200 MA",
                      line=dict(color="purple", width=1, dash="dot")),
            row=1, col=1
        )
    
    # Real-time trades overlay
    if st.session_state.trade_history:
        trade_df = pd.DataFrame(st.session_state.trade_history)
        fig.add_trace(
            go.Scatter(x=trade_df["timestamp"], y=trade_df["price"],
                      mode='markers', name="Live Trades",
                      marker=dict(color='red', size=8, symbol='diamond')),
            row=1, col=1
        )
    
    # Trade levels
    colors = {"Buy": "green", "Sell": "red", "Stop": "orange"}
    for y_val, name in [(buy, "Buy"), (sell, "Sell"), (stop, "Stop")]:
        fig.add_hline(
            y=y_val, line_dash="dash", line_color=colors[name],
            annotation_text=f"{name}: ${y_val:.2f}",
            annotation_position="right",
            row=1, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(x=df["timestamp"], y=df["volume"], name="Volume",
              marker_color="lightblue"),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_layout(height=700, showlegend=True, hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# TAB 3: ORDER BOOK
# =====================================================
with tab3:
    st.subheader("📈 Live Order Book & Market Microstructure")
    
    if not is_streaming:
        st.info("📡 Start live stream to see real-time order book data")
    elif not st.session_state.quote_history:
        st.info("⏳ Waiting for quote data...")
    else:
        # Latest quote display
        quote = st.session_state.latest_quote
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 💵 BID")
            st.metric("Price", f"${quote['bid']:.2f}", delta=None, delta_color="off")
            st.metric("Size", f"{quote['bid_size']:,} shares")
            st.caption("Buyers willing to pay")
        
        with col2:
            st.markdown("### 📊 SPREAD")
            mid = (quote['bid'] + quote['ask']) / 2
            st.metric("Spread", f"${quote['spread']:.4f}")
            st.metric("Spread %", f"{(quote['spread']/mid)*100:.3f}%")
            st.metric("Mid Price", f"${mid:.2f}")
        
        with col3:
            st.markdown("### 💰 ASK")
            st.metric("Price", f"${quote['ask']:.2f}", delta=None, delta_color="off")
            st.metric("Size", f"{quote['ask_size']:,} shares")
            st.caption("Sellers asking price")
        
        st.markdown("---")
        
        # Quote history chart
        if len(st.session_state.quote_history) > 1:
            quote_df = pd.DataFrame(st.session_state.quote_history[-50:])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=quote_df["timestamp"], y=quote_df["bid"],
                name="Bid", line=dict(color="green", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=quote_df["timestamp"], y=quote_df["ask"],
                name="Ask", line=dict(color="red", width=2),
                fill='tonexty', fillcolor='rgba(255,0,0,0.1)'
            ))
            
            fig.update_layout(
                title="Real-Time Bid-Ask Spread Evolution",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Spread analysis
            st.subheader("📊 Spread Analysis")
            avg_spread = quote_df["spread"].mean()
            max_spread = quote_df["spread"].max()
            min_spread = quote_df["spread"].min()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Spread", f"${avg_spread:.4f}")
            col2.metric("Max Spread", f"${max_spread:.4f}")
            col3.metric("Min Spread", f"${min_spread:.4f}")
            
            # Liquidity assessment
            if avg_spread < 0.05:
                st.success("✅ **High Liquidity** - Tight spreads indicate active market")
            elif avg_spread < 0.10:
                st.info("ℹ️ **Moderate Liquidity** - Reasonable trading conditions")
            else:
                st.warning("⚠️ **Low Liquidity** - Wide spreads may impact execution")


# =====================================================
# TAB 4: BACKTEST
# =====================================================
with tab4:
    st.subheader("📜 Advanced Backtesting & Risk Analysis")
    st.info(f"📘 Vectorized backtest: {sim_days}-day hold with 50% stop-loss")
    st.caption(f"🏦 Risk-free rate: {RISK_FREE_RATE*100:.2f}% | Monte Carlo: {mc_method}")
    
    bt = backtest_vectorized(df, sim_days)
    
    if not bt.empty and len(bt) > 0:
        # Metrics
        win_rate = (bt["PnL"] > 0).mean() * 100
        total_trades = len(bt)
        winning_trades = (bt["PnL"] > 0).sum()
        losing_trades = (bt["PnL"] < 0).sum()
        profit_factor = bt[bt["PnL"]>0]["PnL"].sum() / max(abs(bt[bt["PnL"]<0]["PnL"].sum()), 0.01)
        total_pnl = bt["PnL"].sum()
        avg_win = bt[bt["PnL"]>0]["PnL"].mean() if winning_trades > 0 else 0
        avg_loss = bt[bt["PnL"]<0]["PnL"].mean() if losing_trades > 0 else 0
        max_dd = calculate_max_drawdown(bt["CumReturn"].values)
        sharpe = calculate_sharpe_ratio(bt["Return"].values)
        
        # Display
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", total_trades)
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Profit Factor", f"{profit_factor:.2f}")
        col4.metric("Total P&L", f"${total_pnl:,.2f}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Win", f"${avg_win:.2f}")
        col2.metric("Avg Loss", f"${avg_loss:.2f}")
        col3.metric("Max Drawdown", f"{max_dd*100:.2f}%")
        col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        st.markdown("---")
        
        # Charts
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.1,
            subplot_titles=('Cumulative P&L', 'Trade Distribution')
        )
        
        fig.add_trace(
            go.Scatter(x=list(range(len(bt))), y=bt["CumPnL"],
                      name="Cumulative P&L", fill='tozeroy',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=bt["PnL"], name="P&L Distribution",
                        marker_color='lightblue', nbinsx=30),
            row=2, col=1
        )
        
        fig.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk interpretation
        st.markdown("### 📊 Risk Assessment")
        
        if sharpe > 1.5:
            st.success(f"✅ Excellent Sharpe ({sharpe:.2f}) - Strong risk-adjusted returns")
        elif sharpe > 1.0:
            st.info(f"ℹ️ Good Sharpe ({sharpe:.2f}) - Reasonable returns")
        else:
            st.warning(f"⚠️ Low Sharpe ({sharpe:.2f}) - Review strategy")
        
        if abs(max_dd) < 0.10:
            st.success(f"✅ Low Drawdown ({max_dd*100:.1f}%) - Capital preservation excellent")
        elif abs(max_dd) < 0.20:
            st.warning(f"⚠️ Moderate Drawdown ({max_dd*100:.1f}%)")
        else:
            st.error(f"❌ High Drawdown ({max_dd*100:.1f}%) - Significant risk")
    else:
        st.warning("❌ Insufficient data for backtesting")


# =====================================================
# TAB 5: PORTFOLIO
# =====================================================
with tab5:
    st.subheader("📁 Portfolio Management")
    
    if not st.session_state.portfolio:
        st.info("📭 No positions. Add trades from the **Trade Setup** tab.")
    else:
        pf = pd.DataFrame(st.session_state.portfolio)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Positions", len(pf))
        col2.metric("Total Expected P&L", f"${pf['Expected PnL'].sum():,.2f}")
        col3.metric("Total Risk", f"${pf['Max Loss'].sum():,.2f}")
        col4.metric("Avg RRR", f"{pf['RRR'].mean():.2f}x")
        
        st.markdown("---")
        st.dataframe(pf, use_container_width=True, height=400)
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pf['Ticker'], y=pf['Expected PnL'],
                            name='Expected P&L', marker_color='lightgreen'))
        fig.add_trace(go.Bar(x=pf['Ticker'], y=pf['Max Loss'],
                            name='Max Loss', marker_color='lightcoral'))
        fig.update_layout(title="Portfolio Risk/Reward", barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🗑️ Clear Portfolio", use_container_width=True):
            st.session_state.portfolio = []
            st.rerun()


# =====================================================
# TAB 6: MARKET SCANNER
# =====================================================
with tab6:
    st.subheader("🚀 Real-Time Market Scanner")
    st.caption("🔍 Scanning top tech stocks with trend filter")
    
    universe = ["AAPL","MSFT","NVDA","META","AMZN","GOOGL","TSLA","AMD","NFLX","AVGO"]
    
    if st.button("🔍 Scan Market", type="primary", use_container_width=True):
        if not api_key or not secret_key:
            st.error("❌ API keys required")
        else:
            candidates = []
            progress = st.progress(0)
            status = st.empty()
            
            for idx, sym in enumerate(universe):
                status.text(f"Scanning {sym}... ({idx+1}/{len(universe)})")
                progress.progress((idx + 1) / len(universe))
                
                hist = fetch_historical_data(sym, api_key, secret_key, 120)
                if hist is None or hist.empty or len(hist) < 60:
                    continue
                
                try:
                    p = hist["close"].iloc[-1]
                    ma50 = hist["close"].rolling(50).mean().iloc[-1] if len(hist) >= 50 else p
                    ma200 = hist["close"].rolling(min(200, len(hist))).mean().iloc[-1]
                    
                    # Trend filter
                    if p < ma50:
                        continue
                    
                    v = annual_volatility(hist)
                    m = expected_move(p, v, sim_days)
                    b = p - m
                    s = p + m
                    st = b - m * 0.5
                    
                    prob = 1 - prob_hit_mc_advanced(p, b, v, sim_days, MONTE_CARLO_SIMS, mc_method, hist)
                    rrr = (s - b) / max(b - st, 0.01)
                    ev = prob * (s - b)
                    
                    trend = 2 if (p > ma200 and p > ma50) else 1
                    
                    if prob >= 0.55 and rrr >= 1.5:
                        candidates.append({
                            'EV': ev, 'Ticker': sym, 'Price': p, 'Buy': b,
                            'Sell': s, 'Stop': st, 'RRR': rrr, 'Prob': prob,
                            'Vol': v, 'Trend': '🟢 Strong' if trend == 2 else '🟡 Moderate'
                        })
                except:
                    continue
            
            progress.empty()
            status.empty()
            
            if not candidates:
                st.warning("❌ No opportunities found")
            else:
                cdf = pd.DataFrame(candidates).sort_values('EV', ascending=False)
                st.success(f"✅ Found {len(cdf)} opportunities!")
                
                st.markdown("### 🏆 Top Opportunities")
                for i, row in cdf.head(3).iterrows():
                    with st.expander(f"#{cdf.index.get_loc(i)+1} - {row['Ticker']} {row['Trend']}", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Price", f"${row['Price']:.2f}")
                        col2.metric("Buy", f"${row['Buy']:.2f}")
                        col3.metric("Sell", f"${row['Sell']:.2f}")
                        col4.metric("RRR", f"{row['RRR']:.2f}x")
                
                st.markdown("---")
                st.dataframe(cdf, use_container_width=True)


# =====================================================
# AUTO-REFRESH FOR LIVE DATA
# =====================================================
if is_streaming:
    time.sleep(0.1)
    st.rerun()

