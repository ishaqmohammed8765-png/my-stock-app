import streamlit as st
import yfinance as yf
import pandas as pd
import time

# Page configuration
st.set_page_config(page_title="Stock Signal Tracker", page_icon="📈", layout="wide")

st.title("📈 Advanced Stock Signal Tracker")
st.markdown("Track stock signals with EMA and RSI indicators")

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
        
        # Display metrics in columns
        st.subheader(f"📊 {ticker.upper()} Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="Current Price", value=f"${current_price:.2f}")
        with col2:
            st.metric(label="EMA 8 (Fast)", value=f"${ema_8_latest:.2f}")
        with col3:
            st.metric(label="EMA 20 (Slow)", value=f"${ema_20_latest:.2f}")
        with col4:
            st.metric(label="RSI (14)", value=f"{rsi_latest:.2f}")
        
        st.markdown("---")
        
        # Display EMA signal
        st.subheader("🎯 Trading Signals")
        col_signal1, col_signal2 = st.columns(2)
        
        with col_signal1:
            if ema_8_latest > ema_20_latest:
                st.success("🟢 **BUY Signal**: EMA 8 > EMA 20")
            else:
                st.error("🔴 **WAIT Signal**: EMA 8 < EMA 20")
        
        # Display RSI signal
        with col_signal2:
            if rsi_latest > 70:
                st.error("⚠️ **Overbought**: RSI above 70")
            elif rsi_latest < 30:
                st.success("✅ **Oversold**: RSI below 30")
            else:
                st.info(f"ℹ️ **Neutral**: RSI at {rsi_latest:.2f}")
        
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
        
        # Add RSI reference lines info
        st.caption("RSI Reference: Above 70 = Overbought | Below 30 = Oversold")
        
        # Show cache status
        st.success(f"✅ Data loaded successfully and cached for 5 minutes")

elif not submit_button:
    # Initial state - show instructions
    st.info("👈 Enter a stock ticker in the sidebar and click 'Fetch Data' to begin")
    
    st.markdown("### 📋 How to Use:")
    st.markdown("""
    1. Enter a stock ticker symbol in the sidebar (e.g., AAPL, TSLA, NVDA)
    2. Click the **'Fetch Data'** button
    3. View the trading signals and charts
    4. Data is cached for 5 minutes to reduce API calls
    """)
    
    st.markdown("### 📊 Features:")
    st.markdown("""
    - **EMA Signals**: Fast (8) and Slow (20) moving averages
    - **RSI Indicator**: 14-period Relative Strength Index
    - **Real-time Analysis**: Overbought/Oversold detection
    - **Smart Caching**: Reduces API calls and avoids rate limits
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app tracks stock signals using:\n"
    "- **EMA 8 & 20**: Exponential Moving Averages\n"
    "- **RSI 14**: Relative Strength Index\n\n"
    "**Rate Limit Protection:**\n"
    "- Data cached for 5 minutes\n"
    "- Manual fetch button\n"
    "- Smart error handling\n\n"
    "Data provided by Yahoo Finance"
)
