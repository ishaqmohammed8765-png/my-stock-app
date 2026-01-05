import streamlit as st
import yfinance as yf
import pandas as pd

# Page configuration
st.set_page_config(page_title="Stock Signal Tracker", page_icon="📈", layout="wide")

st.title("📈 Advanced Stock Signal Tracker")
st.markdown("Track stock signals with EMA and RSI indicators")

# Sidebar for ticker input
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="NVDA", help="Examples: AAPL, TSLA, BTC-USD")

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if ticker:
    try:
        # Fetch 5 days of data with 15-minute interval
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d", interval="15m")
        
        if data.empty:
            st.error(f"❌ No data found for ticker: {ticker}")
        else:
            # Calculate EMA 8 and EMA 20
            data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
            data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
            
            # Calculate RSI (14-period)
            data['RSI'] = calculate_rsi(data, period=14)
            
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
            
    except Exception as e:
        st.error(f"❌ Error fetching data for {ticker}: {str(e)}")
        st.info("💡 Tip: Make sure you're using a valid ticker symbol (e.g., AAPL, TSLA, MSFT)")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app tracks stock signals using:\n"
    "- **EMA 8 & 20**: Exponential Moving Averages\n"
    "- **RSI 14**: Relative Strength Index\n\n"
    "Data provided by Yahoo Finance"
)
