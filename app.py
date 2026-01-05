import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Day Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# ============================================================================
# ALWAYS SHOW HEADER
# ============================================================================
st.title("📊 Day Trading Dashboard")

# ============================================================================
# SESSION STATE
# ============================================================================
if 'mode' not in st.session_state:
    st.session_state.mode = 'Simple'

# ============================================================================
# FINANCIAL CALCULATIONS
# ============================================================================

def calculate_rsi_wilder(close_prices, period=14):
    """Calculate RSI using Wilder's smoothing"""
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(close_prices):
    """Calculate MACD"""
    ema_fast = close_prices.ewm(span=12, adjust=False).mean()
    ema_slow = close_prices.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def calculate_vwap(df):
    """Calculate VWAP"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

def calculate_indicators(df):
    """Calculate all indicators"""
    df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = calculate_rsi_wilder(df['Close'])
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    df['VWAP'] = calculate_vwap(df)
    return df

def generate_signal(ema_8, ema_20, rsi, macd, macd_signal, price, vwap):
    """Generate trading signal with reasoning"""
    score = 0
    reasons = []
    
    # Trend
    if ema_8 > ema_20:
        reasons.append("✅ **Uptrend**: Fast average ($%.2f) above slow average ($%.2f)" % (ema_8, ema_20))
        score += 1
    else:
        reasons.append("❌ **Downtrend**: Fast average ($%.2f) below slow average ($%.2f)" % (ema_8, ema_20))
        score -= 1
    
    # RSI
    if rsi < 30:
        reasons.append("✅ **Oversold**: RSI is %.0f (below 30) → Stock may be undervalued" % rsi)
        score += 2
    elif rsi < 45:
        reasons.append("✅ **Buy Zone**: RSI is %.0f (below 45)" % rsi)
        score += 1
    elif rsi > 70:
        reasons.append("❌ **Overbought**: RSI is %.0f (above 70) → Stock may be overvalued" % rsi)
        score -= 2
    elif rsi > 55:
        reasons.append("❌ **Sell Zone**: RSI is %.0f (above 55)" % rsi)
        score -= 1
    else:
        reasons.append("⚪ **Neutral**: RSI is %.0f (between 45-55)" % rsi)
    
    # MACD
    if macd > macd_signal:
        reasons.append("✅ **Momentum**: MACD above signal line (bullish)")
        score += 1
    else:
        reasons.append("❌ **Momentum**: MACD below signal line (bearish)")
        score -= 1
    
    # VWAP
    diff_pct = ((price - vwap) / vwap) * 100
    if price > vwap:
        reasons.append("✅ **Above Average**: Price is %.1f%% above VWAP ($%.2f)" % (diff_pct, vwap))
        score += 1
    else:
        reasons.append("❌ **Below Average**: Price is %.1f%% below VWAP ($%.2f)" % (diff_pct, vwap))
        score -= 1
    
    # Verdict
    if score >= 3:
        verdict = "STRONG BUY"
        verdict_type = "success"
    elif score >= 1:
        verdict = "BUY"
        verdict_type = "success"
    elif score <= -3:
        verdict = "STRONG SELL"
        verdict_type = "error"
    elif score <= -1:
        verdict = "SELL"
        verdict_type = "error"
    else:
        verdict = "WAIT"
        verdict_type = "info"
    
    return verdict, verdict_type, reasons, score

# ============================================================================
# UI COMPONENTS
# ============================================================================

def show_traffic_light(verdict, verdict_type):
    """Show traffic light signal"""
    if verdict_type == "success":
        st.markdown("""
        <div style='text-align: center; padding: 40px; background: #1a5f1a; border-radius: 15px;'>
            <h1 style='color: #4ade80; font-size: 80px; margin: 0;'>🟢</h1>
            <h1 style='color: white; margin: 10px 0;'>""" + verdict + """</h1>
        </div>
        """, unsafe_allow_html=True)
    elif verdict_type == "error":
        st.markdown("""
        <div style='text-align: center; padding: 40px; background: #5f1a1a; border-radius: 15px;'>
            <h1 style='color: #f87171; font-size: 80px; margin: 0;'>🔴</h1>
            <h1 style='color: white; margin: 10px 0;'>""" + verdict + """</h1>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; padding: 40px; background: #5f5f1a; border-radius: 15px;'>
            <h1 style='color: #fbbf24; font-size: 80px; margin: 0;'>🟡</h1>
            <h1 style='color: white; margin: 10px 0;'>""" + verdict + """</h1>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.header("⚙️ Settings")

# Mode selector
mode = st.sidebar.radio(
    "🎯 Mode",
    ["📱 Simple", "🔧 Advanced"],
    index=0 if st.session_state.mode == 'Simple' else 1
)
st.session_state.mode = 'Simple' if '📱' in mode else 'Advanced'
is_simple = st.session_state.mode == 'Simple'

st.sidebar.markdown("---")

# Inputs
ticker = st.sidebar.text_input("Stock Symbol", "NVDA").upper()
investment = st.sidebar.number_input("Investment ($)", 100.0, 100000.0, 1000.0, 100.0)
stop_loss_pct = st.sidebar.slider("Stop Loss %", 1.0, 10.0, 2.0, 0.5)

use_target = st.sidebar.checkbox("Set Target Price")
target_price = None
if use_target:
    target_price = st.sidebar.number_input("Target ($)", 0.01, 10000.0, 100.0, 0.01)

st.sidebar.markdown("---")
analyze = st.sidebar.button("🚀 ANALYZE", type="primary", use_container_width=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

if not analyze:
    # Welcome screen
    st.markdown("---")
    if is_simple:
        st.info("👈 Enter a stock symbol and click **ANALYZE** to get a BUY/SELL/WAIT signal")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✨ What you'll get:")
            st.markdown("- 🟢🔴🟡 Clear signal")
            st.markdown("- 📖 Plain explanation")
            st.markdown("- 💰 Trade setup")
        
        with col2:
            st.markdown("#### 💡 How it works:")
            st.markdown("- **Trend** (EMA)")
            st.markdown("- **RSI** (overbought/oversold)")
            st.markdown("- **Momentum** (MACD)")
            st.markdown("- **Average** (VWAP)")
    else:
        st.info("👈 Configure settings and click **ANALYZE**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📊 Features")
            st.markdown("- Technical indicators")
            st.markdown("- Interactive charts")
            st.markdown("- Trade calculator")
        
        with col2:
            st.markdown("### 📈 Indicators")
            st.markdown("- EMA (8, 20)")
            st.markdown("- RSI (Wilder's)")
            st.markdown("- MACD")
            st.markdown("- VWAP")
        
        with col3:
            st.markdown("### 💡 Tips")
            st.markdown("- Use stop losses")
            st.markdown("- Check multiple signals")
            st.markdown("- Risk 1-2% max")
    
    st.markdown("---")
    st.markdown("#### 🔥 Popular: SPY • QQQ • AAPL • TSLA • NVDA • MSFT • AMZN")
    st.warning("⚠️ Educational purposes only. Not financial advice.")

else:
    # Analysis
    try:
        with st.spinner(f"Analyzing {ticker}..."):
            # Fetch data
            df = yf.Ticker(ticker).history(period="5d", interval="15m")
            
            if df.empty:
                st.error(f"No data found for {ticker}")
                st.stop()
            
            # Calculate indicators
            df = calculate_indicators(df)
            
            # Get values
            current_price = df['Close'].iloc[-1]
            ema_8 = df['EMA_8'].iloc[-1]
            ema_20 = df['EMA_20'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_Signal'].iloc[-1]
            vwap = df['VWAP'].iloc[-1]
            
            # Generate signal
            verdict, verdict_type, reasons, score = generate_signal(
                ema_8, ema_20, rsi, macd, macd_signal, current_price, vwap
            )
            
            # Calculate trade info
            shares = investment / current_price
            stop_loss_price = current_price * (1 - stop_loss_pct / 100)
            max_loss = investment * (stop_loss_pct / 100)
            
            # Price change
            prev_close = df['Close'].iloc[-2]
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100
        
        # ========== SIMPLE MODE ==========
        if is_simple:
            st.markdown("---")
            
            # Price
            col1, col2 = st.columns([2, 1])
            with col1:
                st.metric(
                    f"{ticker} Price",
                    f"${current_price:.2f}",
                    f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                )
            with col2:
                st.caption(f"Score: {score}/6")
            
            # Signal
            show_traffic_light(verdict, verdict_type)
            
            # Reasoning
            st.markdown("---")
            st.subheader("📖 Why This Signal?")
            for reason in reasons:
                st.markdown(reason)
            
            # Trade setup
            st.markdown("---")
            st.subheader("💰 Your Trade Setup")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Buy", f"{shares:.2f} shares")
            with col2:
                st.metric("Stop Loss", f"${stop_loss_price:.2f}")
            with col3:
                st.metric("Max Loss", f"${max_loss:.2f}")
            
            # Risk/Reward
            if target_price:
                risk = abs(current_price - stop_loss_price)
                reward = abs(target_price - current_price)
                rr = reward / risk if risk > 0 else 0
                profit = (target_price - current_price) * shares
                
                if rr >= 2:
                    st.success(f"🎯 Risk/Reward: {rr:.2f}:1 → Potential: ${profit:.2f}")
                else:
                    st.warning(f"🎯 Risk/Reward: {rr:.2f}:1 → Potential: ${profit:.2f}")
            
            # Chart
            with st.expander("📈 View Chart", expanded=False):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Close'],
                    name='Price', line=dict(color='lightblue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['EMA_8'],
                    name='Fast EMA', line=dict(color='gold', width=1.5)
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['EMA_20'],
                    name='Slow EMA', line=dict(color='blue', width=1.5)
                ))
                fig.update_layout(height=400, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.info("💡 Always use stop losses • Only invest what you can afford to lose")
            
            if st.button("🔧 Switch to Advanced", use_container_width=True):
                st.session_state.mode = 'Advanced'
                st.rerun()
        
        # ========== ADVANCED MODE ==========
        else:
            st.markdown("---")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Price", f"${current_price:.2f}", f"{price_change_pct:+.2f}%")
            with col2:
                st.metric("RSI", f"{rsi:.2f}")
            with col3:
                st.metric("EMA 8", f"${ema_8:.2f}")
            with col4:
                st.metric("VWAP", f"${vwap:.2f}")
            
            # Signal
            st.markdown("---")
            if verdict_type == "success":
                st.success(f"## 🟢 {verdict}")
            elif verdict_type == "error":
                st.error(f"## 🔴 {verdict}")
            else:
                st.info(f"## 🟡 {verdict}")
            
            with st.expander("Signal Details", expanded=True):
                st.markdown(f"**Score: {score}/6**")
                for reason in reasons:
                    st.markdown(f"- {reason}")
            
            # Trade Calculator
            st.markdown("---")
            st.subheader("💰 Trade Calculator")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Shares", f"{shares:.2f}")
            with col2:
                st.metric("Stop Loss", f"${stop_loss_price:.2f}")
            with col3:
                st.metric("Max Loss", f"${max_loss:.2f}")
            
            if target_price:
                risk = abs(current_price - stop_loss_price)
                reward = abs(target_price - current_price)
                rr = reward / risk if risk > 0 else 0
                
                with col4:
                    st.metric("Risk/Reward", f"{rr:.2f}:1")
            
            # Charts
            st.markdown("---")
            st.subheader("📈 Technical Charts")
            
            # Price chart
            fig1 = go.Figure()
            fig1.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Price'
            ))
            fig1.add_trace(go.Scatter(
                x=df.index, y=df['EMA_8'],
                name='EMA 8', line=dict(color='gold', width=2)
            ))
            fig1.add_trace(go.Scatter(
                x=df.index, y=df['EMA_20'],
                name='EMA 20', line=dict(color='blue', width=2)
            ))
            fig1.add_trace(go.Scatter(
                x=df.index, y=df['VWAP'],
                name='VWAP', line=dict(color='purple', width=2, dash='dash')
            ))
            fig1.update_layout(height=500, hovermode='x unified')
            st.plotly_chart(fig1, use_container_width=True)
            
            # RSI
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df.index, y=df['RSI'],
                name='RSI', line=dict(color='cyan', width=2),
                fill='tozeroy'
            ))
            fig2.add_hline(y=70, line_dash="dash", line_color="red")
            fig2.add_hline(y=30, line_dash="dash", line_color="green")
            fig2.update_layout(height=300, title="RSI (14)", hovermode='x unified')
            st.plotly_chart(fig2, use_container_width=True)
            
            # MACD
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=df.index, y=df['MACD'],
                name='MACD', line=dict(color='blue', width=2)
            ))
            fig3.add_trace(go.Scatter(
                x=df.index, y=df['MACD_Signal'],
                name='Signal', line=dict(color='orange', width=2)
            ))
            fig3.update_layout(height=300, title="MACD", hovermode='x unified')
            st.plotly_chart(fig3, use_container_width=True)
            
            # Stats
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Indicator Values**")
                st.markdown(f"- EMA 8: ${ema_8:.2f}")
                st.markdown(f"- EMA 20: ${ema_20:.2f}")
                st.markdown(f"- RSI: {rsi:.2f}")
            with col2:
                st.markdown("**Levels**")
                st.markdown(f"- VWAP: ${vwap:.2f}")
                st.markdown(f"- MACD: {macd:.4f}")
                st.markdown(f"- Signal: {macd_signal:.4f}")
            
            st.markdown("---")
            if st.button("📱 Switch to Simple", use_container_width=True):
                st.session_state.mode = 'Simple'
                st.rerun()
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check the ticker symbol and try again")
