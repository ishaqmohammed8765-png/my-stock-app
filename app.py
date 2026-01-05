import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Professional Day Trading Dashboard", page_icon="📊", layout="wide")

# ================== API KEY ==================
api_key = st.sidebar.text_input(
    "Finnhub API Key",
    value=st.secrets.get("FINNHUB_API_KEY", ""),
    type="password"
)
if not api_key:
    st.info("Please enter your Finnhub API key in the sidebar.")
    st.stop()

# ================== RSI & SMA Thresholds ==================
RSI_STRONG_BUY = st.secrets.get("RSI_STRONG_BUY", 35)
RSI_BUY_MAX = st.secrets.get("RSI_BUY_MAX", 50)
RSI_SELL_MIN = st.secrets.get("RSI_SELL_MIN", 50)
RSI_STRONG_SELL = st.secrets.get("RSI_STRONG_SELL", 65)

# ================== HELPER FUNCTIONS ==================

def calculate_rsi(data, period=14):
    """Wilder's RSI using EMA smoothing"""
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_obv(data):
    """Vectorized OBV"""
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

def calculate_vwap(data):
    """Volume Weighted Average Price"""
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

def calculate_sma(data, period):
    return data['Close'].rolling(period).mean()

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD & Signal"""
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_rvol(current_volume, historical_data):
    """Relative Volume with timezone-aware intraday data"""
    if historical_data is None or historical_data.empty or current_volume is None:
        return None
    
    # Convert historical data to New York timezone
    if historical_data.index.tzinfo is None:
        historical_data = historical_data.tz_localize("UTC").tz_convert("America/New_York")
    
    avg_volume = historical_data['Volume'].mean()
    return current_volume / avg_volume if avg_volume > 0 else None

def calculate_pivot_points(data, current_price):
    high = float(data['High'].iloc[-1])
    low = float(data['Low'].iloc[-1])
    close = current_price
    pivot = (high + low + close) / 3
    support = (2 * pivot) - high
    resistance = (2 * pivot) - low
    return pivot, support, resistance

def calculate_probability(data, current_price, target_price):
    daily_data = data['Close'].resample('D').last().dropna()
    daily_returns = daily_data.pct_change().dropna() if len(daily_data) >= 20 else daily_data.tail(20).pct_change().dropna()
    volatility = daily_returns.std()
    expected_return = (target_price - current_price) / current_price
    z_score = expected_return / volatility if volatility > 0 else 0
    probability = stats.norm.cdf(abs(z_score))*100 if z_score>0 else (1-stats.norm.cdf(abs(z_score)))*100
    return min(probability,95.0), z_score, volatility

def calculate_investment_risk(current_price, sell_target, investment_amount, stop_loss_pct, slippage_pct):
    shares = investment_amount / current_price if current_price>0 else 0
    stop_loss_price = current_price*(1-stop_loss_pct/100)
    risk_per_share = current_price - stop_loss_price
    total_risk = risk_per_share*shares
    profit_per_share = sell_target - current_price
    gross_profit = profit_per_share*shares
    slippage_cost = gross_profit*(slippage_pct/100)
    net_profit = gross_profit - slippage_cost
    risk_percentage = (total_risk/investment_amount*100) if investment_amount>0 else 0
    return {
        'shares': shares,
        'stop_loss_price': stop_loss_price,
        'total_risk': total_risk,
        'risk_percentage': risk_percentage,
        'gross_profit': gross_profit,
        'slippage_cost': slippage_cost,
        'net_profit': net_profit
    }

def get_verdict(ema_8, ema_20, rsi):
    ema_bullish = ema_8 > ema_20
    ema_bearish = ema_8 < ema_20
    if ema_bullish and rsi < RSI_STRONG_BUY:
        return "🚀 STRONG BUY", "success"
    elif ema_bullish and RSI_STRONG_BUY <= rsi <= RSI_BUY_MAX:
        return "✅ BUY", "success"
    elif ema_bearish and rsi > RSI_STRONG_SELL:
        return "🔻 STRONG SELL", "error"
    elif ema_bearish and RSI_SELL_MIN <= rsi <= RSI_STRONG_SELL:
        return "⚠️ SELL", "error"
    else:
        return "⏸️ WAIT", "info"

# ================== DATA FETCHING ==================

@st.cache_data(ttl=5, show_spinner=False)
def fetch_finnhub_price(ticker):
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={api_key}"
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None, None, None, None, f"API Error: {r.status_code}"
        data = r.json()
        if data.get('c',0)==0 and data.get('pc',0)==0:
            return None,None,None,None,f"Ticker '{ticker}' not found"
        return data.get('c'), data.get('d'), data.get('dp'), data.get('v'), None
    except Exception as e:
        return None,None,None,None,f"Error: {str(e)}"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_historical(ticker):
    try:
        stock = yf.Ticker(ticker)
        data_1y = stock.history(period="1y", interval="1d")
        data_15m = stock.history(period="5d", interval="15m")
        if data_15m.empty:
            return None, None, "No 15-min data available"
        data_15m['EMA_8'] = data_15m['Close'].ewm(span=8, adjust=False).mean()
        data_15m['EMA_20'] = data_15m['Close'].ewm(span=20, adjust=False).mean()
        data_15m['RSI'] = calculate_rsi(data_15m)
        data_15m['OBV'] = calculate_obv(data_15m)
        data_15m['VWAP'] = calculate_vwap(data_15m)
        if not data_1y.empty:
            data_1y['SMA_50'] = calculate_sma(data_1y,50)
            data_1y['SMA_200'] = calculate_sma(data_1y,200)
        return data_15m, data_1y, None
    except Exception as e:
        return None, None, f"Error: {str(e)}"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_finnhub_news(ticker):
    try:
        today = datetime.now(ZoneInfo("America/New_York"))
        week_ago = today - timedelta(days=7)
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={week_ago.date()}&to={today.date()}&token={api_key}"
        r = requests.get(url, timeout=10)
        if r.status_code!=200: return [], f"Error fetching news: {r.status_code}"
        news = r.json()
        return news[:5] if news else [], None
    except Exception as e:
        return [], f"Error: {str(e)}"

# ================== SIDEBAR ==================

st.sidebar.header("⚙️ Trading Settings")
with st.sidebar.form("trading_form", clear_on_submit=False):
    ticker = st.text_input("Ticker","NVDA",help="US stock symbols")
    investment_amount = st.number_input("💵 Investment ($)",1.0,1000000.0,100.0,10.0)
    stop_loss_pct = st.number_input("🛡️ Stop Loss (%)",1.0,20.0,5.0,0.5)
    slippage_pct = st.slider("📉 Slippage (%)",0.0,10.0,2.0,0.5)
    calculate_button = st.form_submit_button("⚡ Analyze",type="primary")

# ================== MAIN LOGIC ==================

if calculate_button and ticker:
    with st.spinner(f"Fetching data for {ticker.upper()}..."):
        current_price, change, percent_change, current_volume, price_error = fetch_finnhub_price(ticker.upper())
        data_15m, data_1y, hist_error = fetch_yfinance_historical(ticker.upper())
        news_items, news_error = fetch_finnhub_news(ticker.upper())
    
    if price_error:
        st.warning(price_error)
    elif data_15m is None or data_15m.empty:
        st.error(f"No intraday data available for {ticker.upper()}.")
        st.stop()
    else:
        ema_8 = data_15m['EMA_8'].iloc[-1]
        ema_20 = data_15m['EMA_20'].iloc[-1]
        rsi = data_15m['RSI'].iloc[-1]
        obv = data_15m['OBV'].iloc[-1]
        vwap = data_15m['VWAP'].iloc[-1]
        pivot, support, resistance = calculate_pivot_points(data_15m,current_price)
        probability, z_score, volatility = calculate_probability(data_15m,current_price,resistance)
        calc = calculate_investment_risk(current_price,resistance,investment_amount,stop_loss_pct,slippage_pct)
        verdict, verdict_type = get_verdict(ema_8,ema_20,rsi)
        
        st.session_state.update({
            'ticker':ticker.upper(),
            'current_price':current_price,
            'change':change,
            'percent_change':percent_change,
            'current_volume':current_volume,
            'ema_8':ema_8,
            'ema_20':ema_20,
            'rsi':rsi,
            'obv':obv,
            'vwap':vwap,
            'pivot':pivot,
            'support':support,
            'resistance':resistance,
            'probability':probability,
            'z_score':z_score,
            'volatility':volatility,
            'calc':calc,
            'verdict':verdict,
            'verdict_type':verdict_type,
            'data_15m':data_15m,
            'data_1y':data_1y,
            'news_items':news_items,
            'news_error':news_error
        })

# ================== DISPLAY ==================
# (All previous display code remains, safe to use st.session_state indicators)
