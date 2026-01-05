import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============ PAGE CONFIG ============
st.set_page_config(page_title="Day Trading Dashboard", page_icon="📊", layout="wide")

# ============ TITLE - ALWAYS SHOWS ============
st.title("📊 Day Trading Dashboard")
st.markdown("---")

# ============ SIDEBAR SETTINGS ============
st.sidebar.header("⚙️ Trading Settings")
ticker = st.sidebar.text_input("Ticker", value="NVDA").upper()
investment_amount = st.sidebar.number_input("💵 Investment ($)", value=100.0, min_value=1.0, step=10.0)
stop_loss_pct = st.sidebar.number_input("🛡️ Stop Loss (%)", value=5.0, min_value=1.0, max_value=20.0, step=0.5)
slippage_pct = st.sidebar.slider("📉 Slippage (%)", 0.0, 10.0, 2.0, 0.5)
analyze = st.sidebar.button("⚡ Analyze", type="primary")

# ============ HELPER FUNCTIONS ============

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
if analyze:
    try:
        with st.spinner(f"🔄 Fetching data for {ticker}..."):
            # Fetch data
            ticker_obj = yf.Ticker(ticker)
            df_15m = ticker_obj.history(period="5d", interval="15m")
            
            if df_15m.empty:
                st.error(f"❌ No data found for {ticker}. Please check the ticker symbol.")
            else:
                # Calculate indicators
                df_15m['EMA_8'] = df_15m['Close'].ewm(span=8, adjust=False).mean()
                df_15m['EMA_20'] = df_15m['Close'].ewm(span=20, adjust=False).mean()
                df_15m['RSI'] = calculate_rsi(df_15m)
                df_15m['VWAP'] = calculate_vwap(df_15m)
                df_15m['MACD'], df_15m['MACD_SIGNAL'], df_15m['MACD_HIST'] = calculate_macd(df_15m)
                
                # Get current values
                current_price = df_15m['Close'].iloc[-1]
                current_volume = df_15m['Volume'].iloc[-1]
                ema_8 = df_15m['EMA_8'].iloc[-1]
                ema_20 = df_15m['EMA_20'].iloc[-1]
                rsi = df_15m['RSI'].iloc[-1]
                
                # Calculate pivot points
                high = df_15m['High'].iloc[-1]
                low = df_15m['Low'].iloc[-1]
                pivot = (high + low + current_price) / 3
                support = 2 * pivot - high
                resistance = 2 * pivot - low
                
                # Get trading verdict
                verdict, verdict_type = get_trading_verdict(ema_8, ema_20, rsi)
                
                # ============ DISPLAY RESULTS ============
                st.success(f"✅ Data loaded successfully for {ticker}")
                
                # Main metrics
                st.subheader(f"{ticker} Real-Time Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                prev_close = df_15m['Close'].iloc[-2] if len(df_15m) > 1 else current_price
                price_change = current_price - prev_close
                price_change_pct = (price_change / prev_close) * 100
                
                col1.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
                col2.metric("EMA 8", f"${ema_8:.2f}")
                col3.metric("EMA 20", f"${ema_20:.2f}")
                col4.metric("RSI", f"{rsi:.2f}")
                
                # Trading signal
                st.markdown("---")
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
                col8.metric("Volume", f"{current_volume:,.0f}")
                
                # Trade Calculator
                st.markdown("---")
                st.subheader("💰 Trade Calculator")
                col_calc1, col_calc2, col_calc3 = st.columns(3)
                
                shares = investment_amount / current_price
                stop_loss_price = current_price * (1 - stop_loss_pct / 100)
                max_loss = investment_amount * stop_loss_pct / 100
                
                col_calc1.metric("Shares to Buy", f"{shares:.2f}")
                col_calc2.metric("Stop Loss Price", f"${stop_loss_price:.2f}")
                col_calc3.metric("Max Loss", f"${max_loss:.2f}")
                
                # Charts
                st.markdown("---")
                st.subheader("📊 Price Chart")
                
                # Price chart with candlesticks
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_15m.index,
                    open=df_15m['Open'],
                    high=df_15m['High'],
                    low=df_15m['Low'],
                    close=df_15m['Close'],
                    name='Price'
                ))
                fig.add_trace(go.Scatter(x=df_15m.index, y=df_15m['EMA_8'], 
                                        name='EMA 8', line=dict(color='gold', width=2)))
                fig.add_trace(go.Scatter(x=df_15m.index, y=df_15m['EMA_20'], 
                                        name='EMA 20', line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=df_15m.index, y=df_15m['VWAP'], 
                                        name='VWAP', line=dict(color='green', dash='dot', width=2)))
                
                fig.add_hline(y=support, line_dash="dash", line_color="red", 
                             annotation_text="Support", annotation_position="right")
                fig.add_hline(y=resistance, line_dash="dash", line_color="green", 
                             annotation_text="Resistance", annotation_position="right")
                
                fig.update_layout(
                    title=f"{ticker} - 15 Minute Chart",
                    xaxis_title="Time",
                    yaxis_title="Price ($)",
                    height=500,
                    hovermode='x unified',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # RSI Chart
                st.subheader("📈 RSI Indicator")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df_15m.index, y=df_15m['RSI'], 
                                            name='RSI', line=dict(color='cyan', width=2), 
                                            fill='tozeroy'))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1)
                fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1)
                
                fig_rsi.update_layout(
                    title="RSI (14 period)",
                    xaxis_title="Time",
                    yaxis_title="RSI",
                    height=300,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # MACD Chart
                st.subheader("📉 MACD Indicator")
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df_15m.index, y=df_15m['MACD'], 
                                             name='MACD', line=dict(color='blue', width=2)))
                fig_macd.add_trace(go.Scatter(x=df_15m.index, y=df_15m['MACD_SIGNAL'], 
                                             name='Signal', line=dict(color='orange', width=2)))
                
                colors = ['green' if val >= 0 else 'red' for val in df_15m['MACD_HIST']]
                fig_macd.add_trace(go.Bar(x=df_15m.index, y=df_15m['MACD_HIST'], 
                                         name='Histogram', marker_color=colors))
                
                fig_macd.update_layout(
                    title="MACD (12, 26, 9)",
                    xaxis_title="Time",
                    yaxis_title="MACD",
                    height=300,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_macd, use_container_width=True)
                
                # Volume Chart
                st.subheader("📊 Volume Analysis")
                fig_vol = go.Figure()
                colors_vol = ['green' if df_15m['Close'].iloc[i] >= df_15m['Open'].iloc[i] 
                             else 'red' for i in range(len(df_15m))]
                fig_vol.add_trace(go.Bar(x=df_15m.index, y=df_15m['Volume'], 
                                        name='Volume', marker_color=colors_vol))
                
                fig_vol.update_layout(
                    title="Trading Volume",
                    xaxis_title="Time",
                    yaxis_title="Volume",
                    height=250,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_vol, use_container_width=True)
                
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("💡 Try a different ticker or check your internet connection.")

else:
    # Show welcome message when not analyzing
    st.info(f"👈 Click '⚡ Analyze' in the sidebar to analyze **{ticker}**")
    st.markdown("""
    ### Welcome to the Day Trading Dashboard! 🚀
    
    **Features:**
    - ✅ Real-time price data from Yahoo Finance (no API key needed!)
    - 📊 Technical indicators: EMA, RSI, MACD, VWAP
    - 🎯 Automated trading signals (BUY/SELL/WAIT)
    - 📈 Interactive candlestick charts
    - 💰 Trade calculator with stop loss
    - 📉 Volume analysis
    
    **How to use:**
    1. Enter a ticker symbol in the sidebar (e.g., NVDA, AAPL, TSLA)
    2. Adjust your investment amount and stop loss percentage
    3. Click the **Analyze** button
    4. View real-time metrics and trading signals
    
    **Popular Tickers to Try:**
    - **NVDA** - NVIDIA
    - **AAPL** - Apple
    - **TSLA** - Tesla
    - **MSFT** - Microsoft
    - **GOOGL** - Google
    - **AMZN** - Amazon
    - **SPY** - S&P 500 ETF
    
    ---
    
    ⚠️ **Disclaimer:** This dashboard is for educational purposes only. 
    Not financial advice. Always do your own research before trading.
    """)
