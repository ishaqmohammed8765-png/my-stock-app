import streamlit as st
import yfinance as yf
import pandas as pd

st.title("📈 Daily Stock Signal Tracker")

# Input for stock ticker
ticker = st.text_input("Enter Stock Ticker", value="NVDA")

if ticker:
    try:
        # Fetch 5 days of data with 15-minute interval
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d", interval="15m")
        
        if data.empty:
            st.error(f"No data found for ticker: {ticker}")
        else:
            # Calculate EMA 8 and EMA 20
            data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
            data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
            
            # Get the latest values and convert to float
            current_price = float(data['Close'].iloc[-1])
            ema_8_latest = float(data['EMA_8'].iloc[-1])
            ema_20_latest = float(data['EMA_20'].iloc[-1])
            
            # Display current price
            st.metric(label=f"{ticker} Current Price", value=f"${current_price:.2f}")
            
            # Display signal based on EMA comparison
            if ema_8_latest > ema_20_latest:
                st.success("🟢 BUY Signal: EMA 8 > EMA 20")
            else:
                st.error("🔴 WAIT Signal: EMA 8 < EMA 20")
            
            # Display EMA values
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="EMA 8 (Fast)", value=f"${ema_8_latest:.2f}")
            with col2:
                st.metric(label="EMA 20 (Slow)", value=f"${ema_20_latest:.2f}")
            
            # Prepare data for line chart
            chart_data = pd.DataFrame({
                'Close': data['Close'],
                'EMA_8': data['EMA_8'],
                'EMA_20': data['EMA_20']
            })
            
            # Display line chart
            st.subheader("Price and EMA Trends")
            st.line_chart(chart_data)
            
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
