import streamlit as st
import yfinance as yf
import pandas as pd
import time

# Page configuration
st.set_page_config(page_title="Stock Signal Tracker", page_icon="📈", layout="wide")

st.title("📈 Advanced Stock Signal Tracker")
st.markdown("Complete trading analysis with price targets, volume, and master signals")

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_obv(data):
    """Calculate On-Balance Volume (OBV)"""
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return obv

def calculate_pivot_points(data):
    """Calculate Pivot Point, Support, and Resistance levels"""
    high = float(data['High'].iloc[-1])
    low = float(data['Low'].iloc[-1])
    close = float(data['Close'].iloc[-1])
    
    pivot = (high + low + close) / 3
    support = (2 * pivot) - high  # Buy Target
    resistance = (2 * pivot) - low  # Sell Target
    
    return pivot, support, resistance

@st.cache_data(ttl=300)  # Cache data for 5 minutes (300 seconds)
def fetch_stock_data(ticker):
    """
    Fetch stock data from Yahoo Finance with caching.
    Data is cached for 5 minutes to reduce API calls.
    """
    try:
        # Fetch 5 days of data with 15-minute interval
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d", interval="15m")
        
        if data.empty:
            return None, "No data found for this ticker"
        
        # Calculate EMA 8 and EMA 20
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        # Calculate RSI (14-period)
        data['RSI'] = calculate_rsi(data, period=14)
        
        # Calculate OBV
        data['OBV'] = calculate_obv(data)
        
        return data, None
        
    except Exception as e:
        error_msg = str(e)
        
        # Check for rate limit errors
        if "429" in error_msg or "rate limit" in error_msg.lower():
            return None, "⏳ Yahoo Finance rate limit reached. Please wait a few minutes and try again."
        elif "404" in error_msg or "not found" in error_msg.lower():
            return None, f"❌ Ticker '{ticker}' not found. Please check the symbol and try again."
        else:
            return None, f"❌ Error fetching data: {error_msg}"

# Sidebar with form to prevent auto-refresh on every keystroke
st.sidebar.header("Settings")

with st.sidebar.form("ticker_form"):
    st.markdown("### Enter Stock Ticker")
    ticker = st.text_input(
        "Stock Symbol",
        value="NVDA",
        help="Examples: AAPL, TSLA, MSFT, BTC-USD",
        placeholder="Enter ticker..."
    )
    
    submit_button = st.form_submit_button("🔍 Fetch Data")
    
    st.caption("💡 Data is cached for 5 minutes to reduce API calls")

# Only fetch data when form is submitted
if submit_button and ticker:
    with st.spinner(f"Fetching data for {ticker.upper()}..."):
        data, error = fetch_stock_data(ticker.upper())
    
    if error:
        st.error(error)
        if "rate limit" in error.lower():
            st.info("💡 **Tips to avoid rate limits:**\n"
                    "- Wait 2-5 minutes before trying again\n"
                    "- The app caches data for 5 minutes, so refreshing won't help\n"
                    "- Avoid submitting multiple tickers rapidly")
    elif data is not None:
        # Get the latest values and convert to float
        current_price = float(data['Close'].iloc[-1])
        ema_8_latest = float(data['EMA_8'].iloc[-1])
        ema_20_latest = float(data['EMA_20'].iloc[-1])
        rsi_latest = float(data['RSI'].iloc[-1])
        
        # Calculate Pivot Points
        pivot, support, resistance = calculate_pivot_points(data)
        
        # Calculate OBV trend (comparing last 20 periods)
        obv_recent = data['OBV'].iloc[-20:]
        obv_trend = obv_recent.iloc[-1] > obv_recent.iloc[0]
        volume_bullish = obv_trend
        
        # Determine signals
        ema_bullish = ema_8_latest > ema_20_latest
        ema_bearish = ema_8_latest < ema_20_latest
        
        # Master Signal Logic
        if ema_bullish and rsi_latest < 60 and volume_bullish:
            verdict = "🚀 STRONG BUY"
            verdict_color = "success"
        elif rsi_latest > 70:
            verdict = "💰 TAKE PROFIT / SELL"
            verdict_color = "warning"
        elif ema_bearish and rsi_latest > 40:
            verdict = "⚠️ WAIT / CAUTION"
            verdict_color = "error"
        else:
            verdict = "📊 NEUTRAL - Monitor"
            verdict_color = "info"
        
        # Display Master Verdict at the top
        st.markdown("### 🎯 Master Signal")
        if verdict_color == "success":
            st.success(f"**{verdict}**")
        elif verdict_color == "warning":
            st.warning(f"**{verdict}**")
        elif verdict_color == "error":
            st.error(f"**{verdict}**")
        else:
            st.info(f"**{verdict}**")
        
        st.markdown("---")
        
        # Top row: Current Price, Buy Target, Sell Target (side-by-side)
        st.subheader(f"💲 {ticker.upper()} Price & Targets")
        col_top1, col_top2, col_top3 = st.columns(3)
        
        with col_top1:
            st.metric(label="📍 Current Price", value=f"${current_price:.2f}")
        with col_top2:
            st.metric(label="🎯 Suggested Entry (Support)", value=f"${support:.2f}")
        with col_top3:
            st.metric(label="🚪 Suggested Exit (Resistance)", value=f"${resistance:.2f}")
        
        st.markdown("---")
        
        # Display indicators in columns
        st.subheader(f"📊 Technical Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="EMA 8 (Fast)", value=f"${ema_8_latest:.2f}")
        with col2:
            st.metric(label="EMA 20 (Slow)", value=f"${ema_20_latest:.2f}")
        with col3:
            st.metric(label="RSI (14)", value=f"{rsi_latest:.2f}")
        with col4:
            volume_status = "📈 Bullish" if volume_bullish else "📉 Bearish"
            st.metric(label="Volume (OBV)", value=volume_status)
        
        st.markdown("---")
        
        # Display individual signals
        st.subheader("🔍 Signal Breakdown")
        col_signal1, col_signal2, col_signal3 = st.columns(3)
        
        with col_signal1:
            if ema_8_latest > ema_20_latest:
                st.success("🟢 **EMA Signal**: Bullish\n\nEMA 8 > EMA 20")
            else:
                st.error("🔴 **EMA Signal**: Bearish\n\nEMA 8 < EMA 20")
        
        with col_signal2:
            if rsi_latest > 70:
                st.error("⚠️ **RSI**: Overbought\n\nRSI above 70")
            elif rsi_latest < 30:
                st.success("✅ **RSI**: Oversold\n\nRSI below 30")
            else:
                st.info(f"ℹ️ **RSI**: Neutral\n\nRSI at {rsi_latest:.2f}")
        
        with col_signal3:
            if volume_bullish:
                st.success("📈 **Volume**: Bullish\n\nOBV is rising")
            else:
                st.warning("📉 **Volume**: Bearish\n\nOBV is falling")
        
        st.markdown("---")
        
        # Prepare data for line charts
        st.subheader("📈 Price and EMA Trends")
        chart_data_price = pd.DataFrame({
            'Close': data['Close'],
            'EMA_8': data['EMA_8'],
            'EMA_20': data['EMA_20']
        })
        st.line_chart(chart_data_price)
        
        # RSI Chart
        st.subheader("📉 RSI Indicator")
        chart_data_rsi = pd.DataFrame({
            'RSI': data['RSI']
        })
        st.line_chart(chart_data_rsi)
        st.caption("RSI Reference: Above 70 = Overbought | Below 30 = Oversold")
        
        # OBV Chart
        st.subheader("📊 On-Balance Volume (OBV)")
        chart_data_obv = pd.DataFrame({
            'OBV': data['OBV']
        })
        st.line_chart(chart_data_obv)
        st.caption("Rising OBV indicates bullish volume, falling OBV indicates bearish volume")
        
        # Show cache status
        st.success(f"✅ Data loaded successfully and cached for 5 minutes")

elif not submit_button:
    # Initial state - show instructions
    st.info("👈 Enter a stock ticker in the sidebar and click 'Fetch Data' to begin")
    
    st.markdown("### 📋 How to Use:")
    st.markdown("""
    1. Enter a stock ticker symbol in the sidebar (e.g., AAPL, TSLA, NVDA)
    2. Click the **'Fetch Data'** button
    3. View the **Master Signal** for quick decision-making
    4. Check **Price Targets** for entry and exit points
    5. Review individual signals and charts for detailed analysis
    """)
    
    st.markdown("### 📊 Features:")
    st.markdown("""
    - **Master Signal**: Combined verdict from all indicators
    - **Price Targets**: Pivot-based support/resistance levels
    - **EMA Signals**: Fast (8) and Slow (20) moving averages
    - **RSI Indicator**: 14-period Relative Strength Index
    - **Volume Analysis**: On-Balance Volume (OBV) trend
    - **Smart Caching**: Reduces API calls and avoids rate limits
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**Trading Signals:**\n"
    "- **EMA 8 & 20**: Trend direction\n"
    "- **RSI 14**: Momentum strength\n"
    "- **OBV**: Volume confirmation\n"
    "- **Pivot Points**: Entry/exit targets\n\n"
    "**Rate Limit Protection:**\n"
    "- Data cached for 5 minutes\n"
    "- Manual fetch button\n"
    "- Smart error handling\n\n"
    "Data provided by Yahoo Finance"
)
