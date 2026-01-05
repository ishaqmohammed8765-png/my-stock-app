import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Professional Day Trading Dashboard", page_icon="📊", layout="wide")
st.title("📊 Professional Day Trading Dashboard")
st.markdown("Real-time data • Technical analysis • Market news")

# ================== API KEY HANDLING ==================
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", None)
if not FINNHUB_API_KEY:
    FINNHUB_API_KEY = st.sidebar.text_input(
        "Enter Finnhub API Key", type="password", help="Required for real-time quotes/news."
    )
if not FINNHUB_API_KEY:
    st.warning("Finnhub API key is required. Enter it in the sidebar.")
    st.stop()

# ================== HELPER FUNCTIONS ==================
def calculate_rsi_wilder(data, period=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_obv_vectorized(data):
    return (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

def calculate_vwap(data):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    return (tp * data['Volume']).cumsum() / data['Volume'].cumsum()

def calculate_pivot_points(data, current_price):
    high = float(data['High'].iloc[-1])
    low = float(data['Low'].iloc[-1])
    close = current_price
    pivot = (high + low + close) / 3
    support = (2 * pivot) - high
    resistance = (2 * pivot) - low
    return pivot, support, resistance

def calculate_probability(data, current_price, target_price):
    daily = data['Close'].resample('D').last().dropna()
    daily_returns = daily.pct_change().dropna() if len(daily) >= 20 else daily.tail(20).pct_change().dropna()
    vol = daily_returns.std()
    expected_return = (target_price - current_price) / current_price
    z_score = expected_return / vol if vol > 0 else 0
    prob = stats.norm.cdf(abs(z_score)) * 100 if z_score > 0 else (1 - stats.norm.cdf(abs(z_score))) * 100
    prob = min(prob, 95.0)
    return prob, z_score, vol

def calculate_investment_risk(current_price, sell_target, investment_amount, stop_loss_pct, slippage_pct):
    shares = investment_amount / current_price if current_price > 0 else 0
    stop_loss_price = current_price * (1 - stop_loss_pct / 100)
    risk_per_share = current_price - stop_loss_price
    total_risk = risk_per_share * shares
    profit_per_share = sell_target - current_price
    gross_profit = profit_per_share * shares
    slippage_cost = gross_profit * (slippage_pct / 100)
    net_profit = gross_profit - slippage_cost
    risk_pct = (total_risk / investment_amount * 100) if investment_amount > 0 else 0
    return {
        'shares': shares,
        'stop_loss_price': stop_loss_price,
        'total_risk': total_risk,
        'risk_percentage': risk_pct,
        'gross_profit': gross_profit,
        'slippage_cost': slippage_cost,
        'net_profit': net_profit
    }

def get_verdict(ema_8, ema_20, rsi, rsi_buy=35, rsi_sell=65):
    if ema_8 > ema_20 and rsi < rsi_buy:
        return "🚀 STRONG BUY", "success"
    elif ema_8 > ema_20 and rsi < 50:
        return "✅ BUY", "success"
    elif ema_8 < ema_20 and rsi > rsi_sell:
        return "🔻 STRONG SELL", "error"
    elif ema_8 < ema_20 and rsi > 50:
        return "⚠️ SELL", "error"
    else:
        return "⏸️ WAIT", "info"

# ================== FETCH FUNCTIONS ==================
@st.cache_data(ttl=5)
def fetch_finnhub_price(ticker: str):
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return None, f"API error: {resp.status_code}"
        data = resp.json()
        if not data.get('c'):
            return None, f"Ticker '{ticker}' not found"
        return data, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600)
def fetch_yfinance_15m(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5d", interval="15m")
        if df.empty:
            return None, "No historical data returned from yfinance."
        df = df.tz_localize('UTC').tz_convert('America/New_York')
        df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['RSI'] = calculate_rsi_wilder(df)
        df['OBV'] = calculate_obv_vectorized(df)
        df['VWAP'] = calculate_vwap(df)
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=300)
def fetch_finnhub_news(ticker: str):
    try:
        today = datetime.now(tz=ZoneInfo("America/New_York"))
        week_ago = today - timedelta(days=7)
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={week_ago.strftime('%Y-%m-%d')}&to={today.strftime('%Y-%m-%d')}&token={FINNHUB_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return [], f"API error: {resp.status_code}"
        news_data = resp.json()
        return news_data[:5] if news_data else [], None
    except Exception as e:
        return [], str(e)

# ================== SIDEBAR INPUTS ==================
st.sidebar.header("⚙️ Trading Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="NVDA").upper()
investment_amount = st.sidebar.number_input("💵 Investment ($)", min_value=1.0, value=100.0, step=10.0)
stop_loss_pct = st.sidebar.number_input("🛡️ Stop Loss (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
slippage_pct = st.sidebar.slider("📉 Slippage (%)", 0.0, 10.0, 2.0, step=0.5)
calculate_button = st.sidebar.button("⚡ Analyze")

# ================== MAIN LOGIC ==================
if calculate_button and ticker:
    with st.spinner(f"Fetching data for {ticker}..."):
        price_data, price_err = fetch_finnhub_price(ticker)
        hist_data, hist_err = fetch_yfinance_15m(ticker)
        news_items, news_err = fetch_finnhub_news(ticker)

    if price_err:
        st.error(price_err)
        st.stop()
    if hist_err:
        st.error(hist_err)
        st.stop()

    current_price = price_data['c']
    ema_8 = hist_data['EMA_8'].iloc[-1]
    ema_20 = hist_data['EMA_20'].iloc[-1]
    rsi = hist_data['RSI'].iloc[-1]
    pivot, support, resistance = calculate_pivot_points(hist_data, current_price)
    probability, z_score, volatility = calculate_probability(hist_data, current_price, resistance)
    calc = calculate_investment_risk(current_price, resistance, investment_amount, stop_loss_pct, slippage_pct)
    verdict, verdict_type = get_verdict(ema_8, ema_20, rsi)

    # ================== DISPLAY ==================
    st.subheader("💰 Real-Time Price & Indicators")
    st.metric(f"{ticker} Price", f"${current_price:.2f}")
    st.metric("EMA 8", f"${ema_8:.2f}")
    st.metric("EMA 20", f"${ema_20:.2f}")
    st.metric("RSI (14)", f"{rsi:.2f}")
    st.markdown(f"Pivot: ${pivot:.2f} | Support: ${support:.2f} | Resistance: ${resistance:.2f}")
    
    st.subheader("🎯 Trading Verdict")
    if verdict_type == "success":
        st.success(f"# {verdict}")
    elif verdict_type == "error":
        st.error(f"# {verdict}")
    else:
        st.info(f"# {verdict}")

    st.subheader("🧮 Investment Breakdown")
    st.metric("Shares", f"{calc['shares']:.2f}")
    st.metric("Stop Loss Price", f"${calc['stop_loss_price']:.2f}")
    st.metric("Net Profit After Slippage", f"${calc['net_profit']:.2f}")

    st.subheader("📈 Charts")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, subplot_titles=(f"{ticker} Price & EMAs", "RSI"))
    fig.add_trace(go.Candlestick(x=hist_data.index, open=hist_data['Open'], high=hist_data['High'],
                                 low=hist_data['Low'], close=hist_data['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['EMA_8'], name="EMA 8", line=dict(color='gold')), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['EMA_20'], name="EMA 20", line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['RSI'], name="RSI", line=dict(color='cyan')), row=2, col=1)
    fig.update_layout(height=700, template='plotly_dark', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📰 Latest News")
    if news_err:
        st.warning(news_err)
    elif news_items:
        for n in news_items:
            st.markdown(f"**{n.get('headline', '')}**")
            st.markdown(n.get('summary', '')[:200]+"...")
            st.markdown(f"[Read More]({n.get('url','#')})")
            st.markdown("---")
    else:
        st.info("No recent news available.")
else:
    st.info("👈 Configure your settings and click Analyze")
