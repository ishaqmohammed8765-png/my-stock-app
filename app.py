import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Daily Stock Tracker", layout="wide")

st.title("☀️ Daily Buy/Sell Signal App")
st.write("This version uses 5-day and 13-day moving averages for faster daily signals.")

# Input for the user
ticker = st.text_input("Enter Stock Ticker:", "NVDA").upper()

if ticker:
    # Fetch 1 month of daily data
    data = yf.download(ticker, period="1mo", interval="1d")
    
    if not data.empty:
        # 1. Calculate Fast Indicators for Daily Moves
        data['EMA5'] = data['Close'].ewm(span=5).mean()
        data['EMA13'] = data['Close'].ewm(span=13).mean()
        
        # 2. Calculate Daily RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Get latest values
        current_price = float(data['Close'].iloc[-1])
        last_ema5 = float(data['EMA5'].iloc[-1])
        last_ema13 = float(data['EMA13'].iloc[-1])
        last_rsi = float(data['RSI'].iloc[-1])
        prev_ema5 = float(data['EMA5'].iloc[-2])

        # Display Metrics in columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}")
        col2.metric("RSI (14d)", f"{last_rsi:.1f}")
        col3.metric("Trend", "UP" if last_ema5 > last_ema13 else "DOWN")

        # 3. Daily Logic Strategy
        st.divider()
        st.subheader("Action Recommendation")
        
        # Check for a "Crossover" (The moment the fast line crosses the slow line)
        is_crossover = last_ema5 > last_ema13 and prev_ema5 <= last_ema13

        if is_crossover:
            st.success("🔥 STRONG BUY: Price momentum just shifted UP today.")
        elif last_ema5 > last_ema13 and last_rsi < 70:
            st.info("✅ HOLD/BUY: Trend is positive for the day.")
        elif last_rsi > 75:
            st.warning("⚠️ CAUTION: Stock is overbought today. Might drop soon.")
        elif last_ema5 < last_ema13:
            st.error("🛑 SELL / WAIT: Daily trend is currently negative.")
        
        # 4. Charting
        st.subheader("Price Movement & Averages")
        st.line_chart(data[['Close', 'EMA5', 'EMA13']])
        
    else:
        st.error("Ticker not found. Try something like AAPL, BTC-USD, or TSLA.")

