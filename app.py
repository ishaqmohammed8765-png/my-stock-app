import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from scipy import stats

# ============ PAGE CONFIG ============
st.set_page_config(page_title="Day Trading Dashboard", page_icon="📊", layout="wide")

# ============ API KEY ============
api_key = st.secrets.get("FINNHUB_API_KEY", None)
if not api_key:
    api_key = st.sidebar.text_input("Finnhub API Key", type="password")
if not api_key:
    st.warning("Enter your Finnhub API key to proceed.")
    st.stop()

# ============ SIDEBAR SETTINGS ============
st.sidebar.header("⚙️ Trading Settings")
ticker = st.sidebar.text_input("Ticker", value="NVDA").upper()
investment_amount = st.sidebar.number_input("💵 Investment ($)", value=100.0, min_value=1.0, step=10.0)
stop_loss_pct = st.sidebar.number_input("🛡️ Stop Loss (%)", value=5.0, min_value=1.0, max_value=20.0, step=0.5)
slippage_pct = st.sidebar.slider("📉 Slippage (%)", 0.0, 10.0, 2.0, 0.5)
analyze = st.sidebar.button("⚡ Analyze")

# ============ HELPER FUNCTIONS ============

def fetch_finnhub_quote(ticker, api_key):
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={api_key}"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if not data.get("c"):
            return None
        return data
    except:
        return None

def fetch_yfinance_data(ticker):
    try:
        df_15m = yf.Ticker(ticker).history(period="5d", interval="15m")
        df_1y = yf.Ticker(ticker).history(period="1y", interval="1d")
        if df_15m.empty:
            return None, None
        return df_15m, df_1y
    except:
        return None, None

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_obv(df):
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

def calculate_vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * df['Volume']).cumsum() / df['Volume'].cumsum()

def calculate_macd(df):
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_pivot(df, current_price):
    high, low = df['High'].iloc[-1], df['Low'].iloc[-1]
    pivot = (high + low + current_price)/3
    support = 2*pivot - high
    resistance = 2*pivot - low
    return pivot, support, resistance

def calculate_rvol(current_volume, df_15m):
    df_today = df_15m.tz_convert('America/New_York').between_time("09:30", "16:00")
    avg_volume = df_today['Volume'].mean()
    if avg_volume and current_volume:
        return current_volume / avg_volume
    return None

def get_trading_verdict(ema_8, ema_20, rsi):
    RSI_STRONG_BUY, RSI_BUY_MAX, RSI_SELL_MIN, RSI_STRONG_SELL = 35, 50, 50, 65
    if ema_8 > ema_20 and rsi < RSI_STRONG_BUY:
        return "🚀 STRONG BUY", "success"
    elif ema_8 > ema_20 and RSI_STRONG_BUY <= rsi <= RSI_BUY_MAX:
        return "✅ BUY", "success"
    elif ema_8 < ema_20 and rsi > RSI_STRONG_SELL:
        return "🔻 STRONG SELL", "error"
    elif ema_8 < ema_20 and RSI_SELL_MIN <= rsi <= RSI_STRONG_SELL:
        return "⚠️ SELL", "error"
    else:
        return "⏸️ WAIT", "info"

# ============ MAIN ANALYSIS ============
if analyze and ticker:
    with st.spinner(f"Fetching data for {ticker}..."):
        quote = fetch_finnhub_quote(ticker, api_key)
        if not quote:
            st.error("Failed to fetch real-time quote.")
            st.stop()

        df_15m, df_1y = fetch_yfinance_data(ticker)
        if df_15m is None:
            st.error("Failed to fetch historical data.")
            st.stop()

        df_15m['EMA_8'] = df_15m['Close'].ewm(span=8, adjust=False).mean()
        df_15m['EMA_20'] = df_15m['Close'].ewm(span=20, adjust=False).mean()
        df_15m['RSI'] = calculate_rsi(df_15m)
        df_15m['OBV'] = calculate_obv(df_15m)
        df_15m['VWAP'] = calculate_vwap(df_15m)
        df_15m['MACD'], df_15m['MACD_SIGNAL'], df_15m['MACD_HIST'] = calculate_macd(df_15m)

        current_price = quote['c']
        current_volume = quote['v']
        ema_8, ema_20, rsi = df_15m['EMA_8'].iloc[-1], df_15m['EMA_20'].iloc[-1], df_15m['RSI'].iloc[-1]

        pivot, support, resistance = calculate_pivot(df_15m, current_price)
        rvol = calculate_rvol(current_volume, df_15m)
        verdict, verdict_type = get_trading_verdict(ema_8, ema_20, rsi)

    # ============ DISPLAY ============
    st.subheader(f"{ticker} Real-Time Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${current_price:.2f}")
    col2.metric("EMA 8", f"${ema_8:.2f}")
    col3.metric("EMA 20", f"${ema_20:.2f}")
    col4.metric("RSI", f"{rsi:.2f}")

    st.markdown(f"## Trading Signal: {verdict}")

    st.subheader("📊 Charts")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_15m.index,
                                 open=df_15m['Open'],
                                 high=df_15m['High'],
                                 low=df_15m['Low'],
                                 close=df_15m['Close'],
                                 name='Price'))
    fig.add_trace(go.Scatter(x=df_15m.index, y=df_15m['EMA_8'], name='EMA 8', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=df_15m.index, y=df_15m['EMA_20'], name='EMA 20', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_15m.index, y=df_15m['VWAP'], name='VWAP', line=dict(color='green', dash='dot')))
    st.plotly_chart(fig, use_container_width=True)

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df_15m.index, y=df_15m['RSI'], name='RSI', line=dict(color='cyan')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    st.plotly_chart(fig_rsi, use_container_width=True)

