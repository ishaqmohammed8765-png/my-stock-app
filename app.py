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
# FINANCIAL CALCULATIONS WITH ERROR HANDLING
# ============================================================================

def safe_calculate_rsi_wilder(close_prices, period=14):
    """Calculate RSI using Wilder's smoothing with NaN handling"""
    try:
        if len(close_prices) < period + 1:
            return pd.Series([np.nan] * len(close_prices), index=close_prices.index)
        
        delta = close_prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        # Prevent division by zero
        avg_loss = avg_loss.replace(0, np.nan)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Default to neutral if calculation fails
    except Exception as e:
        return pd.Series([50] * len(close_prices), index=close_prices.index)

def safe_calculate_macd(close_prices):
    """Calculate MACD with error handling"""
    try:
        if len(close_prices) < 26:
            nan_series = pd.Series([np.nan] * len(close_prices), index=close_prices.index)
            return nan_series, nan_series
        
        ema_fast = close_prices.ewm(span=12, adjust=False).mean()
        ema_slow = close_prices.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        return macd_line.fillna(0), signal_line.fillna(0)
    except Exception as e:
        zero_series = pd.Series([0] * len(close_prices), index=close_prices.index)
        return zero_series, zero_series

def safe_calculate_vwap(df):
    """Calculate VWAP with error handling"""
    try:
        if 'High' not in df or 'Low' not in df or 'Close' not in df or 'Volume' not in df:
            return pd.Series([np.nan] * len(df), index=df.index)
        
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_volume = df['Volume'].cumsum()
        
        # Prevent division by zero
        cumulative_volume = cumulative_volume.replace(0, np.nan)
        vwap = (typical_price * df['Volume']).cumsum() / cumulative_volume
        
        # Fill NaN with current close price as fallback
        return vwap.fillna(df['Close'])
    except Exception as e:
        return df['Close'].copy()

def safe_calculate_indicators(df):
    """Calculate all indicators with comprehensive error handling"""
    try:
        # Validate minimum data
        if df is None or df.empty or len(df) < 2:
            return None
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                return None
        
        # Remove any rows with all NaN values
        df = df.dropna(how='all')
        
        # Fill NaN values in price columns with forward fill, then backward fill
        df[required_cols] = df[required_cols].fillna(method='ffill').fillna(method='bfill')
        
        # If still have NaN, fill with column mean or 0
        for col in required_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean() if df[col].mean() else 0)
        
        # Calculate EMAs
        df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean().fillna(df['Close'])
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean().fillna(df['Close'])
        
        # Calculate RSI
        df['RSI'] = safe_calculate_rsi_wilder(df['Close'])
        
        # Calculate MACD
        df['MACD'], df['MACD_Signal'] = safe_calculate_macd(df['Close'])
        
        # Calculate VWAP
        df['VWAP'] = safe_calculate_vwap(df)
        
        return df
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return None

def generate_signal(ema_8, ema_20, rsi, macd, macd_signal, price, vwap):
    """Generate trading signal with error handling"""
    try:
        # Handle NaN values
        if pd.isna(ema_8) or pd.isna(ema_20):
            ema_8 = price
            ema_20 = price
        if pd.isna(rsi):
            rsi = 50
        if pd.isna(macd) or pd.isna(macd_signal):
            macd = 0
            macd_signal = 0
        if pd.isna(vwap):
            vwap = price
        
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
        diff_pct = ((price - vwap) / vwap) * 100 if vwap != 0 else 0
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
    except Exception as e:
        return "WAIT", "info", ["⚠️ Unable to generate signal - insufficient data"], 0

# ============================================================================
# DATA FETCHING WITH ROBUST ERROR HANDLING
# ============================================================================

def fetch_stock_data(ticker, period="5d", interval="15m"):
    """Fetch stock data with comprehensive error handling"""
    try:
        # Attempt to fetch data
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        
        # Check if data is empty
        if df is None or df.empty:
            return None, "No data available. Ticker may be invalid or market is closed."
        
        # Check minimum data requirement
        if len(df) < 2:
            return None, f"Insufficient data: only {len(df)} data point(s) available. Need at least 2."
        
        # Validate data has required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, f"Missing required columns: {', '.join(missing_cols)}"
        
        # Check if all data is NaN
        if df[required_cols].isna().all().all():
            return None, "All data is missing or invalid."
        
        return df, None
        
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

# ============================================================================
# UI COMPONENTS
# ============================================================================

def show_traffic_light(verdict, verdict_type):
    """Show traffic light signal"""
    try:
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
    except Exception as e:
        st.warning(f"Signal: {verdict}")

def safe_create_price_chart(df, ticker):
    """Create price chart with error handling"""
    try:
        if df is None or df.empty or len(df) < 2:
            st.warning("Insufficient data for chart")
            return
        
        # Filter out NaN values
        df_clean = df.dropna(subset=['Close'])
        if df_clean.empty:
            st.warning("No valid price data for chart")
            return
        
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=df_clean.index,
            y=df_clean['Close'],
            name='Price',
            line=dict(color='lightblue', width=2)
        ))
        
        # Add EMAs if available
        if 'EMA_8' in df_clean.columns and not df_clean['EMA_8'].isna().all():
            fig.add_trace(go.Scatter(
                x=df_clean.index,
                y=df_clean['EMA_8'],
                name='Fast EMA (8)',
                line=dict(color='gold', width=1.5)
            ))
        
        if 'EMA_20' in df_clean.columns and not df_clean['EMA_20'].isna().all():
            fig.add_trace(go.Scatter(
                x=df_clean.index,
                y=df_clean['EMA_20'],
                name='Slow EMA (20)',
                line=dict(color='blue', width=1.5)
            ))
        
        fig.update_layout(
            height=400,
            title=f"{ticker} Price Chart",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Unable to display chart: {str(e)}")

def safe_create_candlestick_chart(df, ticker):
    """Create candlestick chart with error handling"""
    try:
        if df is None or df.empty or len(df) < 2:
            st.warning("Insufficient data for candlestick chart")
            return
        
        # Filter out rows with any NaN in OHLC
        df_clean = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        if df_clean.empty or len(df_clean) < 2:
            st.warning("No valid OHLC data for candlestick chart")
            return
        
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_clean.index,
            open=df_clean['Open'],
            high=df_clean['High'],
            low=df_clean['Low'],
            close=df_clean['Close'],
            name='Price'
        ))
        
        # Add EMAs if available
        if 'EMA_8' in df_clean.columns and not df_clean['EMA_8'].isna().all():
            fig.add_trace(go.Scatter(
                x=df_clean.index,
                y=df_clean['EMA_8'],
                name='EMA 8',
                line=dict(color='gold', width=2)
            ))
        
        if 'EMA_20' in df_clean.columns and not df_clean['EMA_20'].isna().all():
            fig.add_trace(go.Scatter(
                x=df_clean.index,
                y=df_clean['EMA_20'],
                name='EMA 20',
                line=dict(color='blue', width=2)
            ))
        
        if 'VWAP' in df_clean.columns and not df_clean['VWAP'].isna().all():
            fig.add_trace(go.Scatter(
                x=df_clean.index,
                y=df_clean['VWAP'],
                name='VWAP',
                line=dict(color='purple', width=2, dash='dash')
            ))
        
        fig.update_layout(
            height=500,
            title=f"{ticker} Price & Indicators",
            hovermode='x unified',
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Unable to display candlestick chart: {str(e)}")

def safe_create_rsi_chart(df):
    """Create RSI chart with error handling"""
    try:
        if df is None or df.empty or 'RSI' not in df.columns:
            return
        
        df_clean = df.dropna(subset=['RSI'])
        if df_clean.empty:
            return
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_clean.index,
            y=df_clean['RSI'],
            name='RSI',
            line=dict(color='cyan', width=2),
            fill='tozeroy'
        ))
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        fig.update_layout(
            height=300,
            title="RSI (14)",
            hovermode='x unified',
            yaxis_range=[0, 100]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Unable to display RSI chart: {str(e)}")

def safe_create_macd_chart(df):
    """Create MACD chart with error handling"""
    try:
        if df is None or df.empty or 'MACD' not in df.columns:
            return
        
        df_clean = df.dropna(subset=['MACD', 'MACD_Signal'])
        if df_clean.empty:
            return
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_clean.index,
            y=df_clean['MACD'],
            name='MACD',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_clean.index,
            y=df_clean['MACD_Signal'],
            name='Signal',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            height=300,
            title="MACD",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Unable to display MACD chart: {str(e)}")

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
    # Welcome screen - ALWAYS SHOWS
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
    # Analysis - WITH COMPREHENSIVE ERROR HANDLING
    with st.spinner(f"Analyzing {ticker}..."):
        # Fetch data
        df, error = fetch_stock_data(ticker, "5d", "15m")
        
        if error:
            st.error(f"❌ {error}")
            st.info("💡 Tips: Check ticker spelling • Try popular tickers like AAPL, TSLA, NVDA")
            st.stop()
        
        # Calculate indicators
        df = safe_calculate_indicators(df)
        
        if df is None:
            st.error("❌ Unable to calculate indicators - insufficient or invalid data")
            st.info("💡 Try a different ticker or check back when market is open")
            st.stop()
        
        # Extract values safely
        try:
            current_price = df['Close'].iloc[-1]
            if pd.isna(current_price):
                st.error("❌ Current price is not available")
                st.stop()
            
            ema_8 = df['EMA_8'].iloc[-1] if 'EMA_8' in df.columns else current_price
            ema_20 = df['EMA_20'].iloc[-1] if 'EMA_20' in df.columns else current_price
            rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
            macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
            macd_signal = df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns else 0
            vwap = df['VWAP'].iloc[-1] if 'VWAP' in df.columns else current_price
            
            # Handle NaN values
            if pd.isna(ema_8): ema_8 = current_price
            if pd.isna(ema_20): ema_20 = current_price
            if pd.isna(rsi): rsi = 50
            if pd.isna(macd): macd = 0
            if pd.isna(macd_signal): macd_signal = 0
            if pd.isna(vwap): vwap = current_price
            
        except Exception as e:
            st.error(f"❌ Error extracting values: {str(e)}")
            st.stop()
        
        # Generate signal
        verdict, verdict_type, reasons, score = generate_signal(
            ema_8, ema_20, rsi, macd, macd_signal, current_price, vwap
        )
        
        # Calculate trade info
        shares = investment / current_price if current_price > 0 else 0
        stop_loss_price = current_price * (1 - stop_loss_pct / 100)
        max_loss = investment * (stop_loss_pct / 100)
        
        # Price change
        try:
            prev_close = df['Close'].iloc[-2] if len(df) >= 2 else current_price
            if pd.isna(prev_close):
                prev_close = current_price
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close * 100) if prev_close != 0 else 0
        except:
            price_change = 0
            price_change_pct = 0
    
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
            st.caption(f"Signal Score: {score}/6")
        
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
            safe_create_price_chart(df, ticker)
        
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
        
        safe_create_candlestick_chart(df, ticker)
        safe_create_rsi_chart(df)
        safe_create_macd_chart(df)
        
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
