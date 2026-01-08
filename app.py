"""
Real-Time Trading Dashboard - Complete Fixed Edition
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
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import logging
from scipy import stats
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================================
# CONFIGURATION & CONSTANTS
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Risk Management Constants
VOL_FLOOR = 0.10
VOL_CAP = 1.5
KELLY_MAX = 0.15
KELLY_MIN = 0.01
DEFAULT_VOLATILITY = 0.30
STOP_MULTIPLIER = 0.5

# Simulation Constants
MONTE_CARLO_SIMS = 1000
RISK_FREE_RATE = 0.045
TRADING_DAYS_PER_YEAR = 252

# WebSocket Constants
MAX_RECONNECT_ATTEMPTS = 5
MAX_TRADE_HISTORY = 100
MAX_QUOTE_HISTORY = 50
QUEUE_MAX_SIZE = 1000

# Backtest Constants
MAX_BACKTEST_ITERATIONS = 45
MIN_HISTORICAL_DAYS = 30

# UI Constants
AUTO_REFRESH_INTERVAL = 2.0

# Scoring Thresholds
class ScoringThresholds:
    RRR_EXCELLENT = 3.0
    RRR_STRONG = 2.0
    RRR_ACCEPTABLE = 1.5
    
    SHARPE_EXCELLENT = 1.5
    SHARPE_GOOD = 1.0
    SHARPE_MODERATE = 0.5
    
    DRAWDOWN_LOW = 0.10
    DRAWDOWN_MODERATE = 0.20
    DRAWDOWN_HIGH = 0.30
    
    WINRATE_HIGH = 60
    WINRATE_POSITIVE = 50
    WINRATE_BELOW_AVG = 40
    
    PROFIT_FACTOR_STRONG = 2.0
    PROFIT_FACTOR_GOOD = 1.5
    PROFIT_FACTOR_MARGINAL = 1.0

# Decision Scores
SCORE_STRONG_BUY = 80
SCORE_BUY = 65
SCORE_CONDITIONAL = 50
SCORE_HOLD = 35

st.set_page_config(
    page_title="Trading Dashboard Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# DATA CLASSES
# =====================================================
@dataclass
class TradeMetrics:
    """Container for trade analysis metrics"""
    price: float
    vol: float
    atr: float
    ma_50: float
    ma_200: float
    buy: float
    sell: float
    stop: float
    buy_prob: float
    sell_prob: float
    rrr: float
    kelly: float
    qty: int
    expected_move: float


@dataclass
class BacktestResults:
    """Container for backtest results"""
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


# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def validate_api_keys(api_key: str, secret_key: str) -> bool:
    """Validate API key format"""
    if not api_key or not secret_key:
        return False
    if len(api_key) < 10 or len(secret_key) < 10:
        return False
    return True


def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'trade_queue': queue.Queue(maxsize=QUEUE_MAX_SIZE),
        'quote_queue': queue.Queue(maxsize=QUEUE_MAX_SIZE),
        'websocket_thread': None,
        'websocket_handler': None,
        'trade_history': [],
        'quote_history': [],
        'latest_price': None,
        'latest_quote': None,
        'portfolio': [],
        'historical_data': None,
        'current_ticker': "",
        'calculations_cache': {},
        'api_configured': False,
        'last_refresh': time.time(),
        'live_feed_active': False,
        'api_key': "",
        'secret_key': "",
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def cleanup_websocket():
    """Safely cleanup websocket connections"""
    if st.session_state.websocket_handler:
        try:
            st.session_state.websocket_handler.stop_stream()
        except:
            pass
        st.session_state.websocket_handler = None
    
    if st.session_state.websocket_thread and st.session_state.websocket_thread.is_alive():
        st.session_state.websocket_thread.join(timeout=2)
    
    st.session_state.websocket_thread = None
    st.session_state.live_feed_active = False


# =====================================================
# WEBSOCKET HANDLER
# =====================================================
class AlpacaRealtimeHandler:
    """WebSocket handler with improved error handling"""
    
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
        self.max_reconnect_attempts = MAX_RECONNECT_ATTEMPTS
        self._lock = threading.Lock()
        
    async def trade_handler(self, data):
        """Handle incoming trade data"""
        try:
            trade_data = {
                'type': 'trade',
                'timestamp': data.timestamp,
                'symbol': data.symbol,
                'price': float(data.price),
                'size': int(data.size),
                'exchange': getattr(data, 'exchange', 'N/A')
            }
            
            if trade_data['price'] > 0 and trade_data['size'] > 0:
                try:
                    self.trade_queue.put_nowait(trade_data)
                except queue.Full:
                    try:
                        self.trade_queue.get_nowait()
                        self.trade_queue.put_nowait(trade_data)
                    except:
                        pass
        except Exception as e:
            logger.error(f"Trade handler error: {e}")
    
    async def quote_handler(self, data):
        """Handle incoming quote data"""
        try:
            if data.bid_price <= 0 or data.ask_price <= 0:
                return
                
            quote_data = {
                'type': 'quote',
                'timestamp': data.timestamp,
                'symbol': data.symbol,
                'bid': float(data.bid_price),
                'ask': float(data.ask_price),
                'bid_size': int(data.bid_size),
                'ask_size': int(data.ask_size),
                'spread': float(data.ask_price - data.bid_price)
            }
            
            try:
                self.quote_queue.put_nowait(quote_data)
            except queue.Full:
                try:
                    self.quote_queue.get_nowait()
                    self.quote_queue.put_nowait(quote_data)
                except:
                    pass
        except Exception as e:
            logger.error(f"Quote handler error: {e}")
    
    def start_stream(self):
        """Start WebSocket stream"""
        self.is_running = True
        
        while self.is_running and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self.stream = StockDataStream(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    feed='iex'
                )
                
                self.stream.subscribe_trades(self.trade_handler, self.ticker)
                self.stream.subscribe_quotes(self.quote_handler, self.ticker)
                
                logger.info(f"Starting stream for {self.ticker}")
                self.stream.run()
                
                self.reconnect_attempts = 0
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.reconnect_attempts += 1
                
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    wait_time = min(2 ** self.reconnect_attempts, 60)
                    time.sleep(wait_time)
                else:
                    break
    
    def stop_stream(self):
        """Safely stop the stream"""
        with self._lock:
            self.is_running = False
            if self.stream:
                try:
                    self.stream.stop()
                except Exception as e:
                    logger.error(f"Error stopping stream: {e}")
                finally:
                    self.stream = None


# =====================================================
# DATA FUNCTIONS
# =====================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_data(ticker: str, api_key: str, secret_key: str, 
                         days_back: int = 120) -> Optional[pd.DataFrame]:
    """Fetch historical data from Alpaca"""
    if not validate_api_keys(api_key, secret_key):
        return None
        
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
        
        if df.empty:
            return None
        
        if ticker in df.index.get_level_values(0):
            df = df.xs(ticker, level=0)
            df = df.reset_index()
            
            if len(df) < 10:
                return None
                
            required_cols = ['close', 'high', 'low', 'volume', 'timestamp']
            if not all(col in df.columns for col in required_cols):
                return None
                
            return df
        
        return None
        
    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {e}")
        return None


def get_latest_quote(ticker: str, api_key: str, secret_key: str) -> Optional[Dict]:
    """Get latest quote"""
    if not validate_api_keys(api_key, secret_key):
        return None
        
    try:
        client = StockHistoricalDataClient(api_key, secret_key)
        request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        quotes = client.get_stock_latest_quote(request)
        
        if ticker in quotes:
            quote = quotes[ticker]
            
            if quote.bid_price <= 0 or quote.ask_price <= 0:
                return None
                
            return {
                'bid': float(quote.bid_price),
                'ask': float(quote.ask_price),
                'bid_size': int(quote.bid_size),
                'ask_size': int(quote.ask_size),
                'spread': float(quote.ask_price - quote.bid_price)
            }
    except Exception as e:
        logger.error(f"Error getting quote for {ticker}: {e}")
        
    return None


# =====================================================
# ANALYSIS FUNCTIONS
# =====================================================
def annual_volatility(df: pd.DataFrame, span: int = 20) -> float:
    """Calculate annualized volatility"""
    if df.empty or len(df) < 10:
        return DEFAULT_VOLATILITY
        
    try:
        returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        
        if len(returns) < 10:
            return DEFAULT_VOLATILITY
            
        vol = returns.ewm(span=span).std().iloc[-1] * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        if not np.isfinite(vol):
            return DEFAULT_VOLATILITY
            
        return float(np.clip(vol, VOL_FLOOR, VOL_CAP))
        
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        return DEFAULT_VOLATILITY


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range"""
    if len(df) < period:
        return 0.0
        
    try:
        df = df.copy()
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        
        atr = df['tr'].rolling(window=period).mean().iloc[-1]
        
        return float(atr) if np.isfinite(atr) else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return 0.0


def expected_move(price: float, vol: float, days: int = 30) -> float:
    """Calculate expected price movement"""
    if price <= 0 or vol <= 0:
        return 0.0
    return price * vol * np.sqrt(days / TRADING_DAYS_PER_YEAR)


def prob_hit_mc_advanced(S: float, K: float, vol: float, days: int = 30, 
                        sims: int = MONTE_CARLO_SIMS, method: str = "student_t",
                        df_hist: Optional[pd.DataFrame] = None) -> float:
    """Monte Carlo simulation for probability"""
    if vol <= 0 or S <= 0 or K <= 0 or days <= 0:
        return 0.0
    
    try:
        dt = 1 / TRADING_DAYS_PER_YEAR
        
        if method == "student_t":
            dof = 5
            random_shocks = stats.t.rvs(df=dof, size=(sims, days)) / np.sqrt(dof / (dof - 2))
        else:
            random_shocks = np.random.normal(size=(sims, days))
        
        drift = -0.5 * vol ** 2 * dt
        diffusion = vol * np.sqrt(dt) * random_shocks
        
        price_paths = S * np.exp(np.cumsum(drift + diffusion, axis=1))
        
        hits = (price_paths >= K).any(axis=1)
        
        return float(hits.mean())
        
    except Exception as e:
        logger.error(f"Error in Monte Carlo: {e}")
        return 0.0


def kelly_fraction(df: pd.DataFrame, min_trades: int = 30) -> float:
    """Calculate Kelly Criterion"""
    if df.empty or len(df) < min_trades:
        return KELLY_MIN
        
    try:
        returns = df["close"].pct_change().dropna()
        
        if len(returns) < min_trades:
            return KELLY_MIN
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) < 10 or len(losses) < 10:
            return KELLY_MIN
        
        win_rate = len(wins) / len(returns)
        loss_rate = 1 - win_rate
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        if avg_loss == 0 or avg_win == 0:
            return KELLY_MIN
        
        R = avg_win / avg_loss
        fraction = win_rate - (loss_rate / R)
        fraction = fraction * 0.5
        
        return float(np.clip(fraction, KELLY_MIN, KELLY_MAX))
        
    except Exception as e:
        logger.error(f"Error calculating Kelly: {e}")
        return KELLY_MIN


def calculate_sharpe_ratio(returns: np.ndarray, 
                          risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) == 0:
        return 0.0
        
    try:
        returns_std = returns.std()
        if returns_std == 0:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
        sharpe = (excess_returns.mean() / returns_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        return float(sharpe) if np.isfinite(sharpe) else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating Sharpe: {e}")
        return 0.0


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    if len(cumulative_returns) == 0:
        return 0.0
        
    try:
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (running_max + 1e-10)
        
        max_dd = float(drawdown.min())
        return max_dd if np.isfinite(max_dd) else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating drawdown: {e}")
        return 0.0


def backtest_vectorized(df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """Vectorized backtest"""
    min_required = days + MIN_HISTORICAL_DAYS
    if df.empty or len(df) < min_required:
        return pd.DataFrame({"PnL": [], "Return": []})
    
    try:
        max_iterations = min(len(df) - days, MAX_BACKTEST_ITERATIONS)
        start_idx = len(df) - days - max_iterations
        
        results = []
        
        for i in range(max_iterations):
            actual_idx = start_idx + i
            entry_price = df["close"].iloc[actual_idx]
            
            if entry_price <= 0:
                continue
            
            historical_data = df.iloc[:actual_idx+1]
            
            if len(historical_data) > 20:
                log_returns = np.log(historical_data["close"] / historical_data["close"].shift(1)).dropna()
                vol = log_returns.ewm(span=20).std().iloc[-1] * np.sqrt(TRADING_DAYS_PER_YEAR)
                vol = float(np.clip(vol, VOL_FLOOR, VOL_CAP))
            else:
                vol = DEFAULT_VOLATILITY
            
            move = entry_price * vol * np.sqrt(days / TRADING_DAYS_PER_YEAR)
            stop_price = entry_price - move * STOP_MULTIPLIER
            
            window = df["close"].iloc[actual_idx:actual_idx+days]
            
            if len(window) == 0:
                continue
            
            stop_hit_mask = window <= stop_price
            if stop_hit_mask.any():
                exit_price = window[stop_hit_mask].iloc[0]
            else:
                exit_price = window.iloc[-1]
            
            if exit_price > 0:
                pnl = exit_price - entry_price
                ret = (exit_price - entry_price) / entry_price
                
                results.append({
                    'PnL': pnl,
                    'Return': ret
                })
        
        if not results:
            return pd.DataFrame({"PnL": [], "Return": []})
        
        bt = pd.DataFrame(results)
        bt["CumPnL"] = bt["PnL"].cumsum()
        bt["CumReturn"] = (1 + bt["Return"]).cumprod() - 1
        
        return bt
        
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        return pd.DataFrame({"PnL": [], "Return": []})


def calculate_backtest_metrics(bt: pd.DataFrame) -> BacktestResults:
    """Extract backtest metrics"""
    if bt.empty or len(bt) == 0:
        return BacktestResults(
            win_rate=0, profit_factor=0, max_drawdown=0, sharpe_ratio=0,
            total_trades=0, winning_trades=0, losing_trades=0,
            avg_win=0, avg_loss=0, total_pnl=0
        )
    
    try:
        winning_trades = (bt["PnL"] > 0).sum()
        losing_trades = (bt["PnL"] < 0).sum()
        total_trades = len(bt)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_wins = bt[bt["PnL"] > 0]["PnL"].sum() if winning_trades > 0 else 0
        total_losses = abs(bt[bt["PnL"] < 0]["PnL"].sum()) if losing_trades > 0 else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
        
        avg_win = bt[bt["PnL"] > 0]["PnL"].mean() if winning_trades > 0 else 0
        avg_loss = bt[bt["PnL"] < 0]["PnL"].mean() if losing_trades > 0 else 0
        
        max_dd = calculate_max_drawdown(bt["CumReturn"].values)
        sharpe = calculate_sharpe_ratio(bt["Return"].values)
        total_pnl = bt["PnL"].sum()
        
        return BacktestResults(
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            max_drawdown=float(max_dd),
            sharpe_ratio=float(sharpe),
            total_trades=int(total_trades),
            winning_trades=int(winning_trades),
            losing_trades=int(losing_trades),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            total_pnl=float(total_pnl)
        )
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return BacktestResults(
            win_rate=0, profit_factor=0, max_drawdown=0, sharpe_ratio=0,
            total_trades=0, winning_trades=0, losing_trades=0,
            avg_win=0, avg_loss=0, total_pnl=0
        )


def evaluate_trade_decision(rrr: float, sharpe: float, max_dd: float, 
                           win_rate: float, profit_factor: float, 
                           buy_prob: float, sell_prob: float,
                           price: float, ma_50: float, ma_200: float) -> Tuple[int, str, str, str, List[str]]:
    """Evaluate trade opportunity"""
    score = 0
    reasons = []
    
    # Trend Analysis (15 points)
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
        reasons.append("❌ Below both MAs")
    
    # Risk/Reward (25 points)
    if rrr >= ScoringThresholds.RRR_EXCELLENT:
        score += 25
        reasons.append(f"✅ Excellent RRR (≥{ScoringThresholds.RRR_EXCELLENT})")
    elif rrr >= ScoringThresholds.RRR_STRONG:
        score += 20
        reasons.append(f"✅ Strong RRR (≥{ScoringThresholds.RRR_STRONG})")
    elif rrr >= ScoringThresholds.RRR_ACCEPTABLE:
        score += 12
        reasons.append(f"⚠️ Acceptable RRR (≥{ScoringThresholds.RRR_ACCEPTABLE})")
    else:
        reasons.append(f"❌ Poor RRR (<{ScoringThresholds.RRR_ACCEPTABLE})")
    
    # Sharpe Ratio (25 points)
    if sharpe >= ScoringThresholds.SHARPE_EXCELLENT:
        score += 25
        reasons.append(f"✅ Excellent Sharpe (≥{ScoringThresholds.SHARPE_EXCELLENT})")
    elif sharpe >= ScoringThresholds.SHARPE_GOOD:
        score += 20
        reasons.append(f"✅ Good Sharpe (≥{ScoringThresholds.SHARPE_GOOD})")
    elif sharpe >= ScoringThresholds.SHARPE_MODERATE:
        score += 10
        reasons.append(f"⚠️ Moderate Sharpe (≥{ScoringThresholds.SHARPE_MODERATE})")
    else:
        reasons.append(f"❌ Poor Sharpe (<{ScoringThresholds.SHARPE_MODERATE})")
    
    # Max Drawdown (15 points)
    if abs(max_dd) <= ScoringThresholds.DRAWDOWN_LOW:
        score += 15
        reasons.append(f"✅ Low drawdown (≤{ScoringThresholds.DRAWDOWN_LOW*100:.0f}%)")
    elif abs(max_dd) <= ScoringThresholds.DRAWDOWN_MODERATE:
        score += 10
        reasons.append(f"✅ Moderate drawdown (≤{ScoringThresholds.DRAWDOWN_MODERATE*100:.0f}%)")
    elif abs(max_dd) <= ScoringThresholds.DRAWDOWN_HIGH:
        score += 5
        reasons.append(f"⚠️ High drawdown (≤{ScoringThresholds.DRAWDOWN_HIGH*100:.0f}%)")
    else:
        reasons.append(f"❌ Very high drawdown (>{ScoringThresholds.DRAWDOWN_HIGH*100:.0f}%)")
    
    # Win Rate (10 points)
    if win_rate >= ScoringThresholds.WINRATE_HIGH:
        score += 10
        reasons.append(f"✅ High win rate (≥{ScoringThresholds.WINRATE_HIGH}%)")
    elif win_rate >= ScoringThresholds.WINRATE_POSITIVE:
        score += 7
        reasons.append(f"✅ Positive win rate (≥{ScoringThresholds.WINRATE_POSITIVE}%)")
    elif win_rate >= ScoringThresholds.WINRATE_BELOW_AVG:
        score += 3
        reasons.append(f"⚠️ Below-average win rate (≥{ScoringThresholds.WINRATE_BELOW_AVG}%)")
    else:
        reasons.append(f"❌ Low win rate (<{ScoringThresholds.WINRATE_BELOW_AVG}%)")
    
    # Profit Factor (10 points)
    if profit_factor >= ScoringThresholds.PROFIT_FACTOR_STRONG:
        score += 10
        reasons.append(f"✅ Strong profit factor (≥{ScoringThresholds.PROFIT_FACTOR_STRONG})")
    elif profit_factor >= ScoringThresholds.PROFIT_FACTOR_GOOD:
        score += 7
        reasons.append(f"✅ Good profit factor (≥{ScoringThresholds.PROFIT_FACTOR_GOOD})")
    elif profit_factor >= ScoringThresholds.PROFIT_FACTOR_MARGINAL:
        score += 3
        reasons.append(f"⚠️ Marginal profit factor (≥{ScoringThresholds.PROFIT_FACTOR_MARGINAL})")
    else:
        reasons.append(f"❌ Negative profit factor (<{ScoringThresholds.PROFIT_FACTOR_MARGINAL})")
    
    # Determine decision
    if score >= SCORE_STRONG_BUY:
        decision = "STRONG BUY"
        color = "success"
        emoji = "🟢"
    elif score >= SCORE_BUY:
        decision = "BUY"
        color = "success"
        emoji = "🟢"
    elif score >= SCORE_CONDITIONAL:
        decision = "CONDITIONAL BUY"
        color = "warning"
        emoji = "🟡"
        reasons.append("💡 Review timing carefully")
    elif score >= SCORE_HOLD:
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


def calculate_trade_metrics(df: pd.DataFrame, price: float, 
                           sim_days: int, mc_method: str, 
                           account_size: float) -> TradeMetrics:
    """Calculate all trade metrics"""
    try:
        vol = annual_volatility(df)
        atr = calculate_atr(df)
        
        ma_50 = df["close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else price
        ma_200 = (df["close"].rolling(200).mean().iloc[-1] if len(df) >= 200 
                 else df["close"].rolling(len(df)//2).mean().iloc[-1])
        
        move = expected_move(price, vol, sim_days)
        buy = price - move
        sell = price + move
        stop = buy - move * STOP_MULTIPLIER
        
        buy_prob = 1 - prob_hit_mc_advanced(price, buy, vol, sim_days, MONTE_CARLO_SIMS, mc_method, df)
        sell_prob = prob_hit_mc_advanced(price, sell, vol, sim_days, MONTE_CARLO_SIMS, mc_method, df)
        
        rrr = (sell - buy) / max(buy - stop, 0.01)
        kelly = kelly_fraction(df)
        qty = int((account_size * kelly) / max(buy - stop, 0.01))
        
        return TradeMetrics(
            price=price, vol=vol, atr=atr, ma_50=ma_50, ma_200=ma_200,
            buy=buy, sell=sell, stop=stop, buy_prob=buy_prob, sell_prob=sell_prob,
            rrr=rrr, kelly=kelly, qty=qty, expected_move=move
        )
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise


# =====================================================
# INITIALIZE SESSION STATE
# =====================================================
initialize_session_state()

# =====================================================
# MAIN APP
# =====================================================
st.title("📈 Trading Dashboard Pro")
st.caption("Real-time stock analysis with AI recommendations")

# Get API keys
try:
    default_api = st.secrets.get("ALPACA_KEY", "")
    default_secret = st.secrets.get("ALPACA_SECRET", "")
except:
    default_api = ""
    default_secret = ""

# Set API keys if available from secrets
if default_api and default_secret and not st.session_state.api_configured:
    st.session_state.api_key = default_api
    st.session_state.secret_key = default_secret
    st.session_state.api_configured = True

# =====================================================
# STEP 1: API CONFIGURATION (ONLY SHOW IF NOT CONFIGURED)
# =====================================================
if not st.session_state.api_configured:
    st.markdown("---")
    with st.container():
        st.markdown("### 📝 STEP 1: Configure API Keys")
        st.markdown("""
        #### Get Your FREE API Keys (Takes 2 minutes)
        
        1. Go to **[alpaca.markets](https://alpaca.markets)** and click "Sign Up"
        2. Choose **"Paper Trading"** (NO REAL MONEY - It's free practice!)
        3. After signup, go to **"API Keys"** in the menu
        4. Click **"Generate New Keys"** and copy both keys
        5. Paste them below 👇
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_key_input = st.text_input(
                "🔑 API Key",
                type="password",
                placeholder="PKXXXXXXXXXXXXXXXX",
                help="Your Alpaca API Key (starts with 'PK')"
            )
        
        with col2:
            secret_key_input = st.text_input(
                "🔐 Secret Key",
                type="password",
                placeholder="Your secret key here",
                help="Your Alpaca Secret Key"
            )
        
        if st.button("✅ Confirm API Keys", type="primary", use_container_width=True):
            if validate_api_keys(api_key_input, secret_key_input):
                st.session_state.api_key = api_key_input
                st.session_state.secret_key = secret_key_input
                st.session_state.api_configured = True
                st.success("✅ API Keys Configured! Proceeding to stock selection...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid API keys. Please check and try again.")
    
    st.stop()

# =====================================================
# STEP 2: STOCK SELECTION
# =====================================================
st.markdown("---")
st.markdown("### 📊 Select a Stock")

col1, col2 = st.columns([2, 1])

with col1:
    ticker_input = st.text_input(
        "Stock Ticker Symbol",
        value=st.session_state.current_ticker or "NVDA",
        placeholder="Enter ticker (e.g., AAPL, TSLA)"
    ).upper()

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    load_button = st.button(
        "📥 **LOAD DATA**",
        type="primary",
        use_container_width=True
    )

if load_button and ticker_input:
    with st.spinner(f"⏳ Loading {ticker_input} data..."):
        # Clean up old websocket
        cleanup_websocket()
        
        hist_data = fetch_historical_data(
            ticker_input, 
            st.session_state.api_key, 
            st.session_state.secret_key
        )
        
        if hist_data is not None:
            st.session_state.historical_data = hist_data
            st.session_state.current_ticker = ticker_input
            st.session_state.calculations_cache = {}
            
            quote = get_latest_quote(
                ticker_input, 
                st.session_state.api_key, 
                st.session_state.secret_key
            )
            if quote:
                st.session_state.latest_quote = quote
                st.session_state.latest_price = (quote['bid'] + quote['ask']) / 2
            
            st.success(f"✅ Loaded {len(hist_data)} days for {ticker_input}")
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"❌ Failed to load {ticker_input}. Check ticker and try again.")

# =====================================================
# MAIN ANALYSIS (ONLY SHOW IF DATA LOADED)
# =====================================================
if st.session_state.historical_data is not None:
    df = st.session_state.historical_data
    ticker = st.session_state.current_ticker
    
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        account_size = st.number_input(
            "💰 Account Size ($)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )
        
        sim_days = st.slider(
            "📅 Time Horizon (Days)",
            min_value=5,
            max_value=90,
            value=30,
            step=5
        )
        
        mc_method = st.selectbox(
            "🎲 Simulation Method",
            ["student_t", "normal"],
            help="Student-t accounts for fat tails"
        )
        
        st.markdown("---")
        
        # Live Feed Toggle
        if st.button(
            "🔴 Stop Live Feed" if st.session_state.live_feed_active else "🟢 Start Live Feed",
            use_container_width=True
        ):
            if st.session_state.live_feed_active:
                cleanup_websocket()
                st.success("Live feed stopped")
            else:
                # Start websocket
                handler = AlpacaRealtimeHandler(
                    st.session_state.api_key,
                    st.session_state.secret_key,
                    ticker,
                    st.session_state.trade_queue,
                    st.session_state.quote_queue
                )
                
                thread = threading.Thread(target=handler.start_stream, daemon=True)
                thread.start()
                
                st.session_state.websocket_handler = handler
                st.session_state.websocket_thread = thread
                st.session_state.live_feed_active = True
                st.success("Live feed started!")
            
            time.sleep(1)
            st.rerun()
    
    # Get current price
    if st.session_state.latest_price:
        current_price = st.session_state.latest_price
    else:
        current_price = df["close"].iloc[-1]
    
    # Calculate metrics
    try:
        metrics = calculate_trade_metrics(
            df, current_price, sim_days, mc_method, account_size
        )
        
        # Run backtest
        bt = backtest_vectorized(df, sim_days)
        bt_metrics = calculate_backtest_metrics(bt)
        
        # Evaluate decision
        score, decision, color, emoji, reasons = evaluate_trade_decision(
            metrics.rrr,
            bt_metrics.sharpe_ratio,
            bt_metrics.max_drawdown,
            bt_metrics.win_rate,
            bt_metrics.profit_factor,
            metrics.buy_prob,
            metrics.sell_prob,
            current_price,
            metrics.ma_50,
            metrics.ma_200
        )
        
        # Display recommendation
        st.markdown(f"## {emoji} Trade Recommendation: **{decision}**")
        st.markdown(f"**Score: {score}/100**")
        
        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
            st.metric("Buy Level", f"${metrics.buy:.2f}", f"{((metrics.buy/current_price-1)*100):.1f}%")
        
        with col2:
            st.metric("Sell Target", f"${metrics.sell:.2f}", f"{((metrics.sell/current_price-1)*100):.1f}%")
            st.metric("Stop Loss", f"${metrics.stop:.2f}", f"{((metrics.stop/current_price-1)*100):.1f}%")
        
        with col3:
            st.metric("Risk/Reward Ratio", f"{metrics.rrr:.2f}")
            st.metric("Position Size", f"{metrics.qty} shares")
        
        with col4:
            st.metric("Win Rate", f"{bt_metrics.win_rate:.1f}%")
            st.metric("Sharpe Ratio", f"{bt_metrics.sharpe_ratio:.2f}")
        
        # Reasons
        st.markdown("### 📋 Analysis Summary")
        for reason in reasons:
            st.markdown(f"- {reason}")
        
        # Chart
        st.markdown("### 📈 Price Chart with Levels")
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['close'],
            name='Price',
            line=dict(color='#2196F3', width=2)
        ))
        
        # Moving averages
        if len(df) >= 50:
            ma50 = df['close'].rolling(50).mean()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=ma50,
                name='MA50',
                line=dict(color='#FF9800', width=1, dash='dash')
            ))
        
        if len(df) >= 200:
            ma200 = df['close'].rolling(200).mean()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=ma200,
                name='MA200',
                line=dict(color='#9C27B0', width=1, dash='dash')
            ))
        
        # Trade levels
        fig.add_hline(y=metrics.buy, line_dash="dot", line_color="green", 
                     annotation_text="Buy Level")
        fig.add_hline(y=metrics.sell, line_dash="dot", line_color="blue", 
                     annotation_text="Sell Target")
        fig.add_hline(y=metrics.stop, line_dash="dot", line_color="red", 
                     annotation_text="Stop Loss")
        
        fig.update_layout(
            title=f"{ticker} Price Analysis",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Backtest results
        if not bt.empty:
            st.markdown("### 📊 Backtest Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Trades", bt_metrics.total_trades)
                st.metric("Winning Trades", bt_metrics.winning_trades)
            
            with col2:
                st.metric("Profit Factor", f"{bt_metrics.profit_factor:.2f}")
                st.metric("Max Drawdown", f"{bt_metrics.max_drawdown*100:.1f}%")
            
            with col3:
                st.metric("Avg Win", f"${bt_metrics.avg_win:.2f}")
                st.metric("Avg Loss", f"${bt_metrics.avg_loss:.2f}")
            
            # Equity curve
            fig_bt = go.Figure()
            
            fig_bt.add_trace(go.Scatter(
                y=bt['CumPnL'],
                name='Cumulative P&L',
                line=dict(color='#4CAF50', width=2)
            ))
            
            fig_bt.update_layout(
                title="Backtest Equity Curve",
                xaxis_title="Trade Number",
                yaxis_title="Cumulative P&L ($)",
                height=400
            )
            
            st.plotly_chart(fig_bt, use_container_width=True)
        
        # Live feed display
        if st.session_state.live_feed_active:
            st.markdown("### 🔴 Live Market Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Recent Trades**")
                trade_placeholder = st.empty()
                
                # Get recent trades from queue
                trades = []
                while not st.session_state.trade_queue.empty():
                    try:
                        trade = st.session_state.trade_queue.get_nowait()
                        st.session_state.trade_history.append(trade)
                        if len(st.session_state.trade_history) > MAX_TRADE_HISTORY:
                            st.session_state.trade_history.pop(0)
                    except:
                        break
                
                if st.session_state.trade_history:
                    recent_trades = st.session_state.trade_history[-5:]
                    for trade in reversed(recent_trades):
                        st.text(f"{trade['timestamp'].strftime('%H:%M:%S')} | "
                               f"${trade['price']:.2f} | {trade['size']} shares")
            
            with col2:
                st.markdown("**Live Quote**")
                
                # Get latest quote
                while not st.session_state.quote_queue.empty():
                    try:
                        quote = st.session_state.quote_queue.get_nowait()
                        st.session_state.latest_quote = quote
                    except:
                        break
                
                if st.session_state.latest_quote:
                    q = st.session_state.latest_quote
                    st.text(f"Bid: ${q['bid']:.2f} ({q['bid_size']})")
                    st.text(f"Ask: ${q['ask']:.2f} ({q['ask_size']})")
                    st.text(f"Spread: ${q['spread']:.4f}")
        
    except Exception as e:
        st.error(f"Error in analysis: {e}")
        logger.error(f"Analysis error: {e}", exc_info=True)

else:
    st.info("👆 Please load stock data to begin analysis")

# Footer
st.markdown("---")
st.caption("⚠️ This is for educational purposes only. Not financial advice.")
