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
from streamlit_autorefresh import st_autorefresh

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Day Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# ============ API KEY HANDLING ============
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "")

if not FINNHUB_API_KEY:
    FINNHUB_API_KEY = st.sidebar.text_input("Enter Finnhub API Key", type="password")
    if not FINNHUB_API_KEY:
        st.warning("Please provide Finnhub API Key to fetch real-time data.")
        st.stop()

# ============ TRADING PARAMETERS ============
st.sidebar.header("⚙️ Trading Settings")
ticker = st.sidebar.text_input("Stock Symbol", value="NVDA")
investment_amount = st.sidebar.number_input("Investment ($)", value=100.0, step=10.0)
stop_loss_pct = st.sidebar.number_input("Stop Loss (%)", value=5.0, min_value=1.0, max_value=20.0, step=0.5)
slippage_pct = st.sidebar.slider("Slippage (%)", value=2.0, min_value=0.0, max_value=10.0, step=0.5)

# Auto-refresh every 15s for live price
st_autorefresh(interval=15000, key="price_refresh")

# ============ HELPER FUNCTIONS ============

def calculate_wilder_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder smoothing (EMA style)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_obv(data):
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

def calculate_vwap(data):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

def calculate_rvol(current_volume, historical_data):
    if historical_data is None or historical_data.empty or not current_volume:
        return None
    avg_volume = historical_data['Volume'].mean()
    return current_volume / avg_volume if avg_volume > 0 else None

def calculate_sma(data, period):
    return data['Close'].rolling(window=period).mean()

def calculate_macd(data):
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_pivot_points(data, current_price):
    high = float(data['High'].iloc[-1])
    low = float(data['Low'].iloc[-1])
    close = current_price
    pivot = (high + low + close)/3
    support = (2*pivot) - high
    resistance = (2*pivot) - low
    return pivot, support, resistance

def calculate_probability(data, current_price, target_price):
    daily_data = data['Close'].resample('D').last().dropna()
    returns = daily_data.pct_change().tail(20).dropna()
    vol = returns.std()
    expected = (target_price - current_price)/current_price
    z_score = expected / vol if vol>0 else 0
    prob = stats.norm.cdf(abs(z_score))*100 if z_score>0 else (1-stats.norm.cdf(abs(z_score)))*100
    prob = min(prob, 95)
    return prob, z_score, vol

def calculate_investment_risk(current_price, sell_target, investment_amount, stop_loss_pct, slippage_pct):
    shares = investment_amount / current_price if current_price > 0 else 0
    stop_loss_price = current_price*(1 - stop_loss_pct/100)
    risk_per_share = current_price - stop_loss_price
    total_risk = risk_per_share*shares
    profit_per_share = sell_target - current_price
    gross_profit = profit_per_share*shares
    slippage_cost = gross_profit*(slippage_pct/100)
    net_profit = gross_profit - slippage_cost
    rr_ratio = net_profit/total_risk if total_risk>0 else 0
    return {'shares':shares, 'stop_loss_price':stop_loss_price, 'total_risk':total_risk,
            'risk_percentage':total_risk/investment_amount*100 if investment_amount>0 else 0,
            'gross_profit':gross_profit, 'slippage_cost':slippage_cost, 'net_profit':net_profit,
            'rr_ratio':rr_ratio}

def get_verdict(ema_8, ema_20, rsi):
    if ema_8>ema_20 and rsi<35:
        return "🚀 STRONG BUY","success"
    elif ema_8>ema_20 and 35<=rsi<=50:
        return "✅ BUY","success"
    elif ema_8<ema_20 and rsi>65:
        return "🔻 STRONG SELL","error"
    elif ema_8<ema_20 and 50<=rsi<=65:
        return "⚠️ SELL","error"
    else:
        return "⏸️ WAIT","info"

# ============ DATA FETCHING ============

@st.cache_data(ttl=10)
def fetch_finnhub_price(ticker):
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=5)
        data = r.json()
        if data.get('c',0)==0:
            return None,None,None,None,f"Ticker '{ticker}' not found"
        return data.get('c'), data.get('d'), data.get('dp'), data.get('v'), None
    except Exception as e:
        return None,None,None,None,str(e)

@st.cache_data(ttl=3600)
def fetch_yfinance_historical(ticker):
    try:
        stock = yf.Ticker(ticker)
        data_15m = stock.history(period="5d", interval="15m", actions=False)
        data_1y = stock.history(period="1y", interval="1d", actions=False)
        if data_15m.empty:
            return None,None,"No 15-min historical data"
        
        # Vectorized indicators
        data_15m['EMA_8'] = data_15m['Close'].ewm(span=8, adjust=False).mean()
        data_15m['EMA_20'] = data_15m['Close'].ewm(span=20, adjust=False).mean()
        data_15m['RSI'] = calculate_wilder_rsi(data_15m)
        data_15m['OBV'] = calculate_obv(data_15m)
        data_15m['VWAP'] = calculate_vwap(data_15m)
        data_15m = data_15m.tz_localize('UTC').tz_convert('America/New_York')
        
        if not data_1y.empty:
            data_1y['SMA_50'] = calculate_sma(data_1y,50)
            data_1y['SMA_200'] = calculate_sma(data_1y,200)
            data_1y['RSI_14'] = calculate_wilder_rsi(data_1y)
        
        return data_15m, data_1y, None
    except Exception as e:
        return None,None,str(e)

@st.cache_data(ttl=300)
def fetch_finnhub_news(ticker):
    try:
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={week_ago.date()}&to={today.date()}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=10)
        news = r.json()
        return news[:5] if news else [], None
    except:
        return [], "News fetch error"

# ============ MAIN EXECUTION ============

if ticker:
    with st.spinner(f"Fetching data for {ticker}..."):
        price, change, pct_change, volume, price_error = fetch_finnhub_price(ticker.upper())
        data_15m, data_1y, hist_error = fetch_yfinance_historical(ticker.upper())
        news, news_error = fetch_finnhub_news(ticker.upper())

    if price_error:
        st.warning(price_error)
    else:
        rvol = calculate_rvol(volume, data_15m) if volume else None
        ema_8 = data_15m['EMA_8'].iloc[-1]
        ema_20 = data_15m['EMA_20'].iloc[-1]
        rsi = data_15m['RSI'].iloc[-1]
        macd, macd_signal, macd_hist = calculate_macd(data_15m)
        pivot, support, resistance = calculate_pivot_points(data_15m, price)
        probability, z_score, volatility = calculate_probability(data_15m, price, resistance)
        calc = calculate_investment_risk(price,resistance,investment_amount,stop_loss_pct,slippage_pct)
        verdict, verdict_type = get_verdict(ema_8, ema_20, rsi)
        
        st.session_state.update({
            'ticker':ticker.upper(), 'current_price':price, 'change':change,
            'percent_change':pct_change,'volume':volume,'rvol':rvol,
            'ema_8':ema_8,'ema_20':ema_20,'rsi':rsi,'pivot':pivot,
            'support':support,'resistance':resistance,'probability':probability,
            'z_score':z_score,'volatility':volatility,'calc':calc,'verdict':verdict,
            'verdict_type':verdict_type,'data_15m':data_15m,'data_1y':data_1y,
            'news':news,'news_error':news_error
        })

    st.markdown(f"## {ticker.upper()} - Real-Time Price: ${price:.2f} ({pct_change:+.2f}%)")

# ============ TABS: Calculator / Analysis / News ============
tab1,tab2,tab3 = st.tabs(["Calculator","Analysis","News"])

with tab1:
    if 'calc' in st.session_state:
        st.markdown(f"### Trading Verdict: {st.session_state['verdict']}")
        st.metric("Stop Loss Price", f"${calc['stop_loss_price']:.2f}")
        st.metric("Net Profit", f"${calc['net_profit']:.2f}")
        st.metric("R/R Ratio", f"{calc['rr_ratio']:.2f}:1")

with tab2:
    if 'data_15m' in st.session_state:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                            row_heights=[0.7,0.3])
        fig.add_trace(go.Candlestick(x=data_15m.index,
                                     open=data_15m['Open'], high=data_15m['High'],
                                     low=data_15m['Low'], close=data_15m['Close'],
                                     name="Price"), row=1,col=1)
        fig.add_trace(go.Scatter(x=data_15m.index, y=data_15m['EMA_8'], name='EMA8'),row=1,col=1)
        fig.add_trace(go.Scatter(x=data_15m.index, y=data_15m['EMA_20'], name='EMA20'),row=1,col=1)
        fig.add_trace(go.Bar(x=data_15m.index, y=macd_hist, name='MACD Hist', marker_color='orange'), row=2,col=1)
        st.plotly_chart(fig,use_container_width=True)

with tab3:
    if 'news' in st.session_state:
        for n in st.session_state['news']:
            st.markdown(f"### {n.get('headline')}")
            st.markdown(n.get('summary')[:200]+"...")
            st.markdown(f"[Read More]({n.get('url')})")
