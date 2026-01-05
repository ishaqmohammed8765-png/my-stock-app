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

# ============ TITLE ============
st.title("📊 Day Trading Dashboard")
st.markdown("---")

# ============ API KEY ============
api_key = st.secrets.get("FINNHUB_API_KEY", None) if hasattr(st, 'secrets') else None
if not api_key:
    api_key = st.sidebar.text_input("Finnhub API Key", type="password", help="Enter your Finnhub API key")

if not api_key:
    st.info("👈 Please enter your Finnhub API key in the sidebar to get started.")
    st.markdown("""
    ### How to get a Finnhub API Key:
    1. Visit [Finnhub.io](https://finnhub.io)
    2. Sign up for a free account
    3. Get your API key from the dashboard
    4. Enter it in the sidebar
    """)
    st.stop()

# ============ SIDEBAR SETTINGS ============
st.sidebar.header("⚙️ Trading Settings")
ticker = st.sidebar.text_input("Ticker", value="NVDA").upper()
investment_amount = st.sidebar.number_input("💵 Investment ($)", value=100.0, min_value=1.0, step=10.0)
stop_loss_pct = st.sidebar.number_input("🛡️ Stop Loss (%)", value=5.0, min_value=1.0, max_value=20.0, step=0.5)
slippage_pct = st.sidebar.slider("📉 Slippage (%)", 0.0, 10.0, 2.0, 0.5)
analyze = st.sidebar.button("⚡ Analyze", type="primary")

# ============ HELPER FUNCTIONS ============

def fetch_finnhub_quote(ticker, api_key):
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={api_key}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Check if API returned an error
        if 'error' in data:
            st.error(f"API Error: {data['error']}")
            return None
            
        if not data.get("c") or data.get("c") == 0:
            st.error(f"No data returned for ticker {ticker}. Please check if the ticker is valid.")
            return None
        return data
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error fetching quote: {str(e)}")
        return None

def fetch_yfinance_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        df_15m = ticker_obj.history(period="5d", interval="15m")
        df_1y = ticker_obj.history(period="1y", interval="1d")
        
        if df_15m.empty:
            st.error(f"No historical data available for {ticker}")
            return None, None
            
        return df_15m, df_1y
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
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
    try:
        # Handle timezone-aware and timezone-naive data
        if df_15m.index.tz is None:
            # If data is timezone-naive, localize it to UTC first, then convert
            df_tz = df_15m.tz_localize('UTC').tz_convert('America/New_York')
        else:
            # If already timezone-aware, just convert
            df_tz = df_15m.tz_convert('America/New_York')
        
        df_today = df_tz.between_time("09:30", "16:00")
        
        if df_today.empty:
            return None
            
        avg_volume = df_today['Volume'].mean()
        if avg_volume and current_volume and avg_volume > 0:
            return current_volume / avg_volume
        return None
    except Exception as e:
        st.warning(f"Could not calculate relative volume: {str(e)}")
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

# ============ INITIAL STATE MESSAGE ============
if not analyze:
    st.info(f"👈 Click '⚡ Analyze' in the sidebar to analyze **{ticker}**")
    st.markdown("""
    ### Features:
    - Real-time price data from Finnhub
    - Technical indicators (EMA, RSI, MACD, VWAP)
    - Trading signals based on trend analysis
    - Interactive charts with Plotly
    """)

# ============ MAIN ANALYSIS ============
if analyze and ticker:
    with st.spinner(f"🔄 Fetching data for {ticker}..."):
        quote = fetch_finnhub_quote(ticker, api_key)
        if not quote:
            st.stop()

        df_15m, df_1y = fetch_yfinance_data(ticker)
        if df_15m is None:
            st.stop()

        # Calculate indicators
        df_15m['EMA_8'] = df_15m['Close'].ewm(span=8, adjust=False).mean()
        df_15m['EMA_20'] = df_15m['Close'].ewm(span=20, adjust=False).mean()
        df_15m['RSI'] = calculate_rsi(df_15m)
        df_15m['OBV'] = calculate_obv(df_15m)
        df_15m['VWAP'] = calculate_vwap(df_15m)
        df_15m['MACD'], df_15m['MACD_SIGNAL'], df_15m['MACD_HIST'] = calculate_macd(df_15m)

        current_price = quote['c']
        current_volume = quote.get('v', 0)
        ema_8 = df_15m['EMA_8'].iloc[-1]
        ema_20 = df_15m['EMA_20'].iloc[-1]
        rsi = df_15m['RSI'].iloc[-1]

        pivot, support, resistance = calculate_pivot(df_15m, current_price)
        rvol = calculate_rvol(current_volume, df_15m)
        verdict, verdict_type = get_trading_verdict(ema_8, ema_20, rsi)

    # ============ DISPLAY ============
    st.success(f"✅ Data fetched successfully for {ticker}")
    
    st.subheader(f"{ticker} Real-Time Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${current_price:.2f}")
    col2.metric("EMA 8", f"${ema_8:.2f}")
    col3.metric("EMA 20", f"${ema_20:.2f}")
    col4.metric("RSI", f"{rsi:.2f}")

    # Trading signal with colored container
    if verdict_type == "success":
        st.success(f"## Trading Signal: {verdict}")
    elif verdict_type == "error":
        st.error(f"## Trading Signal: {verdict}")
    else:
        st.info(f"## Trading Signal: {verdict}")

    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Support", f"${support:.2f}")
    col6.metric("Pivot", f"${pivot:.2f}")
    col7.metric("Resistance", f"${resistance:.2f}")
    if rvol:
        col8.metric("Rel. Volume", f"{rvol:.2f}x")
    else:
        col8.metric("Rel. Volume", "N/A")

    st.markdown("---")
    st.subheader("📊 Price Chart")
    
    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_15m.index,
        open=df_15m['Open'],
        high=df_15m['High'],
        low=df_15m['Low'],
        close=df_15m['Close'],
        name='Price'
    ))
    fig.add_trace(go.Scatter(x=df_15m.index, y=df_15m['EMA_8'], name='EMA 8', line=dict(color='gold', width=2)))
    fig.add_trace(go.Scatter(x=df_15m.index, y=df_15m['EMA_20'], name='EMA 20', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=df_15m.index, y=df_15m['VWAP'], name='VWAP', line=dict(color='green', dash='dot', width=2)))
    
    fig.update_layout(
        title=f"{ticker} Price Chart (15min intervals)",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # RSI chart
    st.subheader("📈 RSI Indicator")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df_15m.index, y=df_15m['RSI'], name='RSI', line=dict(color='cyan', width=2)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    fig_rsi.update_layout(
        title="Relative Strength Index (RSI)",
        xaxis_title="Time",
        yaxis_title="RSI",
        height=300,
        hovermode='x unified'
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD chart
    st.subheader("📉 MACD Indicator")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df_15m.index, y=df_15m['MACD'], name='MACD', line=dict(color='blue', width=2)))
    fig_macd.add_trace(go.Scatter(x=df_15m.index, y=df_15m['MACD_SIGNAL'], name='Signal', line=dict(color='orange', width=2)))
    fig_macd.add_trace(go.Bar(x=df_15m.index, y=df_15m['MACD_HIST'], name='Histogram', marker_color='gray'))
    
    fig_macd.update_layout(
        title="MACD (Moving Average Convergence Divergence)",
        xaxis_title="Time",
        yaxis_title="MACD",
        height=300,
        hovermode='x unified'
    )
    st.plotly_chart(fig_macd, use_container_width=True)

