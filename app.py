import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Daily Stock Signal", layout="wide")
st.title("🚀 Daily Buy/Sell Signals")

ticker = st.text_input("Enter Ticker", "NVDA").upper()

if ticker:
    try:
        # We fetch 5 days of data with 15-minute intervals for daily precision
        data = yf.download(ticker, period="5d", interval="15m")
        
        if not data.empty:
            # Short-term EMA for Day Trading (8 and 20 periods)
            data['EMA8'] = data['Close'].ewm(span=8).mean()
            data['EMA20'] = data['Close'].ewm(span=20).mean()
            
            last_price = data['Close'].iloc[-1]
            last_8 = data['EMA8'].iloc[-1]
            last_20 = data['EMA20'].iloc[-1]
            
            st.metric(f"Current {ticker} Price (15m)", f"${last_price:.2f}")
            
            # Daily Signal Logic
            if last_8 > last_21:
                st.success("🔥 DAILY BUY: Momentum is pushing UP today.")
            else:
                st.error("📉 DAILY WAIT: Momentum is currently DOWN.")
                
            # Charting the intraday movement
            st.line_chart(data[['Close', 'EMA8', 'EMA20']])
        else:
            st.warning("Could not find daily data. Try a major ticker like TSLA.")
    except Exception as e:
        st.error(f"Error: {e}")
