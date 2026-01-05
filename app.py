import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Day Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'mode' not in st.session_state:
    st.session_state.mode = 'Simple'

# ============================================================================
# CONSTANTS
# ============================================================================
NY_TZ = ZoneInfo("America/New_York")

# ============================================================================
# API KEY MANAGEMENT
# ============================================================================
def get_finnhub_api_key():
    """Get Finnhub API key from secrets or user input"""
    api_key = None
    if hasattr(st, 'secrets') and 'FINNHUB_API_KEY' in st.secrets:
        api_key = st.secrets['FINNHUB_API_KEY']
    return api_key

# ============================================================================
# FINANCIAL CALCULATIONS
# ============================================================================

def calculate_rsi_wilder(close_prices, period=14):
    """Calculate RSI using Wilder's smoothing method"""
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_obv_vectorized(df):
    """Vectorized On-Balance Volume calculation"""
    price_change = np.sign(df['Close'].diff())
    obv = (price_change * df['Volume']).fillna(0).cumsum()
    return obv

def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    """Calculate MACD, Signal line, and Histogram"""
    ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_vwap_daily(df):
    """Calculate VWAP that resets daily"""
    df = df.copy()
    
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)
    
    df['date'] = df.index.date
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tp_volume'] = df['typical_price'] * df['Volume']
    
    df['cumulative_tp_volume'] = df.groupby('date')['tp_volume'].cumsum()
    df['cumulative_volume'] = df.groupby('date')['Volume'].cumsum()
    
    vwap = df['cumulative_tp_volume'] / df['cumulative_volume']
    return vwap

def calculate_bollinger_bands(close_prices, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = close_prices.rolling(window=period).mean()
    std = close_prices.rolling(window=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_pivot_points(df):
    """Calculate pivot points"""
    high = df['High'].iloc[-1]
    low = df['Low'].iloc[-1]
    close = df['Close'].iloc[-1]
    
    pivot = (high + low + close) / 3
    support1 = (2 * pivot) - high
    resistance1 = (2 * pivot) - low
    support2 = pivot - (high - low)
    resistance2 = pivot + (high - low)
    
    return {
        'pivot': pivot,
        'r1': resistance1,
        'r2': resistance2,
        's1': support1,
        's2': support2
    }

# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_finnhub_quote(ticker, api_key):
    """Fetch real-time quote from Finnhub"""
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data or not data.get('c'):
            return None, "Invalid ticker or API error"
        
        return data, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_yfinance_data(ticker, period, interval):
    """Fetch historical data from Yahoo Finance"""
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        
        if df.empty:
            return None, f"No data available for {ticker}"
        
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        df.index = df.index.tz_convert(NY_TZ)
        
        return df, None
    except Exception as e:
        return None, str(e)

def get_current_price_fallback(ticker):
    """Get current price from Yahoo Finance as fallback"""
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="1d", interval="1m")
        if not data.empty:
            return data['Close'].iloc[-1], data['Volume'].iloc[-1]
        return None, None
    except:
        return None, None

# ============================================================================
# INDICATOR CALCULATION
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def calculate_basic_indicators(df):
    """Calculate essential indicators for simple mode"""
    df = df.copy()
    df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = calculate_rsi_wilder(df['Close'], period=14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['VWAP'] = calculate_vwap_daily(df)
    return df

@st.cache_data(ttl=300, show_spinner=False)
def calculate_all_indicators(df):
    """Calculate all indicators for advanced mode"""
    df = calculate_basic_indicators(df)
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['OBV'] = calculate_obv_vectorized(df)
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    return df

# ============================================================================
# TRADING LOGIC
# ============================================================================

def generate_trading_signal(ema_8, ema_20, rsi, macd, macd_signal, current_price, vwap):
    """Generate trading signal with reasoning"""
    signals = []
    score = 0
    reasoning = []
    
    # EMA Crossover
    if ema_8 > ema_20:
        signals.append("EMA: Bullish")
        reasoning.append(f"✅ **Uptrend**: Fast average (${ema_8:.2f}) is above slow average (${ema_20:.2f})")
        score += 1
    else:
        signals.append("EMA: Bearish")
        reasoning.append(f"❌ **Downtrend**: Fast average (${ema_8:.2f}) is below slow average (${ema_20:.2f})")
        score -= 1
    
    # RSI
    if rsi < 30:
        signals.append("RSI: Oversold (Strong Buy)")
        reasoning.append(f"✅ **Oversold**: RSI is {rsi:.0f} (below 30) → Stock may be undervalued")
        score += 2
    elif rsi < 45:
        signals.append("RSI: Buy Zone")
        reasoning.append(f"✅ **Buy Zone**: RSI is {rsi:.0f} (below 45) → Bullish momentum")
        score += 1
    elif rsi > 70:
        signals.append("RSI: Overbought (Strong Sell)")
        reasoning.append(f"❌ **Overbought**: RSI is {rsi:.0f} (above 70) → Stock may be overvalued")
        score -= 2
    elif rsi > 55:
        signals.append("RSI: Sell Zone")
        reasoning.append(f"❌ **Sell Zone**: RSI is {rsi:.0f} (above 55) → Bearish momentum")
        score -= 1
    else:
        signals.append("RSI: Neutral")
        reasoning.append(f"⚪ **Neutral**: RSI is {rsi:.0f} (between 45-55)")
    
    # MACD
    if macd > macd_signal:
        signals.append("MACD: Bullish")
        reasoning.append("✅ **Momentum**: MACD line is above signal line (bullish crossover)")
        score += 1
    else:
        signals.append("MACD: Bearish")
        reasoning.append("❌ **Momentum**: MACD line is below signal line (bearish crossover)")
        score -= 1
    
    # Price vs VWAP
    diff_pct = ((current_price - vwap) / vwap) * 100
    if current_price > vwap:
        signals.append("Price > VWAP (Bullish)")
        reasoning.append(f"✅ **Above Average**: Price is {diff_pct:+.1f}% above VWAP (${vwap:.2f})")
        score += 1
    else:
        signals.append("Price < VWAP (Bearish)")
        reasoning.append(f"❌ **Below Average**: Price is {diff_pct:.1f}% below VWAP (${vwap:.2f})")
        score -= 1
    
    # Final verdict
    if score >= 3:
        verdict = "STRONG BUY"
        verdict_type = "success"
        action = "Consider buying - strong bullish signals"
    elif score >= 1:
        verdict = "BUY"
        verdict_type = "success"
        action = "Consider buying - moderate bullish signals"
    elif score <= -3:
        verdict = "STRONG SELL"
        verdict_type = "error"
        action = "Consider selling - strong bearish signals"
    elif score <= -1:
        verdict = "SELL"
        verdict_type = "error"
        action = "Consider selling - moderate bearish signals"
    else:
        verdict = "WAIT"
        verdict_type = "info"
        action = "Wait for clearer signals - market is neutral"
    
    return verdict, verdict_type, signals, score, reasoning, action

# ============================================================================
# UI COMPONENTS - SIMPLE MODE
# ============================================================================

def show_traffic_light_signal(verdict, verdict_type):
    """Display traffic light style signal"""
    if verdict_type == "success":
        icon = "🟢"
        color = "#1a5f1a"
        text_color = "#4ade80"
    elif verdict_type == "error":
        icon = "🔴"
        color = "#5f1a1a"
        text_color = "#f87171"
    else:
        icon = "🟡"
        color = "#5f5f1a"
        text_color = "#fbbf24"
    
    st.markdown(f"""
    <div style='text-align: center; padding: 40px; background: {color}; border-radius: 15px; margin: 20px 0;'>
        <h1 style='color: {text_color}; font-size: 80px; margin: 0;'>{icon}</h1>
        <h1 style='color: white; margin: 10px 0;'>{verdict}</h1>
    </div>
    """, unsafe_allow_html=True)

def show_simple_mode(ticker, current_price, verdict, verdict_type, reasoning, action, 
                     trade_info, df, price_change, price_change_pct):
    """Simple beginner-friendly interface"""
    
    # Header
    st.title(f"📊 {ticker} Trading Signal")
    
    # Price with change
    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    with col2:
        current_time_ny = datetime.now(NY_TZ)
        st.caption(f"🕐 {current_time_ny.strftime('%I:%M %p ET')}")
    
    # Traffic light signal
    show_traffic_light_signal(verdict, verdict_type)
    
    # Action recommendation
    if verdict_type == "success":
        st.success(f"**💡 Recommendation:** {action}", icon="✅")
    elif verdict_type == "error":
        st.error(f"**💡 Recommendation:** {action}", icon="⚠️")
    else:
        st.info(f"**💡 Recommendation:** {action}", icon="⏸️")
    
    # Reasoning
    st.markdown("---")
    st.subheader("📖 Why This Signal?")
    st.markdown("*Here's what the indicators are telling us:*")
    
    for reason in reasoning:
        st.markdown(reason)
    
    # Trade Setup
    st.markdown("---")
    st.subheader("💰 Your Trade Setup")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("You Can Buy", f"{trade_info['shares']:.2f} shares")
        st.caption(f"With ${trade_info['investment']:.0f}")
    
    with col2:
        st.metric("Set Stop Loss", f"${trade_info['stop_loss']:.2f}")
        st.caption(f"{trade_info['stop_loss_pct']:.1f}% below entry")
    
    with col3:
        st.metric("Maximum Risk", f"${trade_info['max_loss']:.2f}", delta_color="inverse")
        st.caption("Never risk more!")
    
    # Risk/Reward if target set
    if trade_info.get('target_price'):
        st.markdown("---")
        rr_ratio = trade_info['risk_reward']
        potential_profit = trade_info['potential_profit']
        
        if rr_ratio >= 2:
            st.success(f"🎯 **Risk/Reward Ratio: {rr_ratio:.2f}:1** → Potential profit: ${potential_profit:.2f}")
            st.caption("✅ Good! Target is at least 2x your risk")
        else:
            st.warning(f"🎯 **Risk/Reward Ratio: {rr_ratio:.2f}:1** → Potential profit: ${potential_profit:.2f}")
            st.caption("⚠️ Consider a higher target (at least 2:1 ratio recommended)")
    
    # Educational tip
    st.markdown("---")
    st.info("💡 **Beginner Tips**: Always use stop losses • Only invest what you can afford to lose • Don't chase trades • Be patient")
    
    # Optional chart
    with st.expander("📈 View Price Chart (Optional)", expanded=False):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            name='Price', line=dict(color='lightblue', width=2),
            fill='tozeroy', fillcolor='rgba(173, 216, 230, 0.2)'
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_8'],
            name='Fast Average (EMA 8)', line=dict(color='gold', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_20'],
            name='Slow Average (EMA 20)', line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            height=400,
            title=f"{ticker} Price Trend",
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Switch to advanced
    st.markdown("---")
    if st.button("🔧 Switch to Advanced Mode", use_container_width=True):
        st.session_state.mode = 'Advanced'
        st.rerun()

# ============================================================================
# UI COMPONENTS - ADVANCED MODE
# ============================================================================

def create_comprehensive_chart(df, ticker, pivot_points):
    """Create multi-panel chart for advanced users"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(
            f'{ticker} - Price & Indicators',
            'RSI (14)',
            'MACD (12, 26, 9)',
            'Volume'
        )
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # EMAs
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_8'], name='EMA 8',
                  line=dict(color='gold', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20',
                  line=dict(color='blue', width=1.5)),
        row=1, col=1
    )
    
    # VWAP
    fig.add_trace(
        go.Scatter(x=df.index, y=df['VWAP'], name='VWAP',
                  line=dict(color='purple', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                  line=dict(color='gray', width=1, dash='dot'),
                  showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                  line=dict(color='gray', width=1, dash='dot'),
                  fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                  showlegend=False),
        row=1, col=1
    )
    
    # Pivot Points
    fig.add_hline(y=pivot_points['r2'], line_dash="dot", line_color="red",
                  annotation_text="R2", row=1, col=1, opacity=0.5)
    fig.add_hline(y=pivot_points['r1'], line_dash="dash", line_color="orange",
                  annotation_text="R1", row=1, col=1, opacity=0.7)
    fig.add_hline(y=pivot_points['pivot'], line_dash="solid", line_color="yellow",
                  annotation_text="Pivot", row=1, col=1, opacity=0.7)
    fig.add_hline(y=pivot_points['s1'], line_dash="dash", line_color="lightgreen",
                  annotation_text="S1", row=1, col=1, opacity=0.7)
    fig.add_hline(y=pivot_points['s2'], line_dash="dot", line_color="green",
                  annotation_text="S2", row=1, col=1, opacity=0.5)
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                  line=dict(color='cyan', width=2),
                  fill='tozeroy', fillcolor='rgba(0,255,255,0.1)'),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, row=2, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                  line=dict(color='blue', width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                  line=dict(color='orange', width=2)),
        row=3, col=1
    )
    colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
    fig.add_trace(
        go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram',
               marker_color=colors, showlegend=False),
        row=3, col=1
    )
    
    # Volume
    colors_vol = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i]
                  else 'red' for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume',
               marker_color=colors_vol, showlegend=False),
        row=4, col=1
    )
    
    # Layout
    fig.update_layout(
        height=1000,
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    fig.update_xaxes(title_text="Time (ET)", row=4, col=1)
    
    return fig

def show_advanced_mode(ticker, current_price, verdict, verdict_type, signals, score,
                       reasoning, pivot_points, latest, df, trade_info, price_change, price_change_pct):
    """Advanced interface with full charts and metrics"""
    
    st.title(f"📈 {ticker} - Advanced Analysis")
    
    current_time_ny = datetime.now(NY_TZ)
    st.caption(f"🕐 Market Time (ET): {current_time_ny.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        st.metric("RSI (14)", f"{latest['RSI']:.2f}")
    
    with col3:
        ema_diff = latest['EMA_8'] - latest['EMA_20']
        st.metric("EMA 8", f"${latest['EMA_8']:.2f}", f"{ema_diff:+.2f} vs EMA20")
    
    with col4:
        vwap_diff = ((current_price - latest['VWAP']) / latest['VWAP']) * 100
        st.metric("VWAP", f"${latest['VWAP']:.2f}", f"{vwap_diff:+.2f}%")
    
    with col5:
        st.metric("Volume", f"{trade_info['current_volume']:,.0f}")
    
    # Trading Signal
    st.markdown("---")
    st.subheader("🎯 Trading Signal")
    
    if verdict_type == "success":
        st.success(f"## 🟢 {verdict}")
    elif verdict_type == "error":
        st.error(f"## 🔴 {verdict}")
    else:
        st.info(f"## 🟡 {verdict}")
    
    # Signal details
    with st.expander("📋 Signal Breakdown", expanded=True):
        st.markdown(f"**Composite Score:** {score}/6")
        col_a, col_b = st.columns(2)
        with col_a:
            for signal in signals[:2]:
                st.markdown(f"- {signal}")
        with col_b:
            for signal in signals[2:]:
                st.markdown(f"- {signal}")
    
    # Pivot Points
    st.markdown("---")
    st.subheader("📍 Key Levels (Pivot Points)")
    
    pivot_cols = st.columns(5)
    with pivot_cols[0]:
        st.metric("R2", f"${pivot_points['r2']:.2f}")
    with pivot_cols[1]:
        st.metric("R1", f"${pivot_points['r1']:.2f}")
    with pivot_cols[2]:
        st.metric("Pivot", f"${pivot_points['pivot']:.2f}")
    with pivot_cols[3]:
        st.metric("S1", f"${pivot_points['s1']:.2f}")
    with pivot_cols[4]:
        st.metric("S2", f"${pivot_points['s2']:.2f}")
    
    # Trade Calculator
    st.markdown("---")
    st.subheader("💰 Trade Calculator")
    
    calc_cols = st.columns(4)
    
    with calc_cols[0]:
        st.metric("Shares", f"{trade_info['shares']:.2f}")
    with calc_cols[1]:
        st.metric("Stop Loss", f"${trade_info['stop_loss']:.2f}")
    with calc_cols[2]:
        st.metric("Max Loss", f"${trade_info['max_loss']:.2f}", delta_color="inverse")
    
    if trade_info.get('target_price'):
        with calc_cols[3]:
            rr_ratio = trade_info['risk_reward']
            potential_profit = trade_info['potential_profit']
            st.metric(
                "Risk/Reward",
                f"{rr_ratio:.2f}:1",
                f"+${potential_profit:.2f}" if potential_profit > 0 else f"${potential_profit:.2f}"
            )
        
        if rr_ratio < 2:
            st.warning("⚠️ Risk/Reward ratio below 2:1 - consider adjusting")
        else:
            st.success(f"✅ Good risk/reward ratio ({rr_ratio:.2f}:1)")
    
    # Charts
    st.markdown("---")
    st.subheader("📈 Technical Analysis Charts")
    
    fig = create_comprehensive_chart(df, ticker, pivot_points)
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional Stats
    st.markdown("---")
    st.subheader("📊 Additional Statistics")
    
    stats_col1, stats_col2 = st.columns(2)
    
    with stats_col1:
        st.markdown("**Indicator Values**")
        st.markdown(f"- **EMA 8:** ${latest['EMA_8']:.2f}")
        st.markdown(f"- **EMA 20:** ${latest['EMA_20']:.2f}")
        st.markdown(f"- **EMA 50:** ${latest['EMA_50']:.2f}")
        st.markdown(f"- **RSI:** {latest['RSI']:.2f}")
        st.markdown(f"- **MACD:** {latest['MACD']:.4f}")
        st.markdown(f"- **MACD Signal:** {latest['MACD_Signal']:.4f}")
        st.markdown(f"- **MACD Histogram:** {latest['MACD_Hist']:.4f}")
    
    with stats_col2:
        st.markdown("**Price Levels**")
        st.markdown(f"- **VWAP:** ${latest['VWAP']:.2f}")
        st.markdown(f"- **BB Upper:** ${latest['BB_Upper']:.2f}")
        st.markdown(f"- **BB Middle:** ${latest['BB_Middle']:.2f}")
        st.markdown(f"- **BB Lower:** ${latest['BB_Lower']:.2f}")
        st.markdown(f"- **OBV:** {latest['OBV']:,.0f}")
        
        dist_from_vwap = ((current_price - latest['VWAP']) / latest['VWAP']) * 100
        dist_from_bb_upper = ((current_price - latest['BB_Upper']) / latest['BB_Upper']) * 100
        
        st.markdown(f"- **Distance from VWAP:** {dist_from_vwap:+.2f}%")
        st.markdown(f"- **Distance from BB Upper:** {dist_from_bb_upper:+.2f}%")
    
    # Switch to simple
    st.markdown("---")
    if st.button("📱 Switch to Simple Mode", use_container_width=True):
        st.session_state.mode = 'Simple'
        st.rerun()

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.sidebar.header("⚙️ Settings")
    
    # Mode selector
    mode_options = ["📱 Simple (Beginner)", "🔧 Advanced (Expert)"]
    current_mode_display = "📱 Simple (Beginner)" if st.session_state.mode == 'Simple' else "🔧 Advanced (Expert)"
    
    selected_mode = st.sidebar.radio(
        "🎯 Display Mode",
        mode_options,
        index=mode_options.index(current_mode_display),
        help="Simple: Clean signal only | Advanced: Full charts & metrics"
    )
    
    st.session_state.mode = 'Simple' if 'Simple' in selected_mode else 'Advanced'
    is_simple_mode = st.session_state.mode == 'Simple'
    
    st.sidebar.markdown("---")
    
    # Trading inputs
    if is_simple_mode:
        st.sidebar.subheader("📊 Quick Setup")
        ticker = st.sidebar.text_input(
            "Stock Symbol",
            value="NVDA",
            help="e.g., AAPL, TSLA, SPY"
        ).upper()
        
        investment_amount = st.sidebar.number_input(
            "How much to invest?",
            min_value=100.0,
            max_value=100000.0,
            value=1000.0,
            step=100.0,
            help="Your total investment amount"
        )
        
        stop_loss_pct = st.sidebar.slider(
            "Stop loss %",
            1.0, 10.0, 2.0, 0.5,
            help="Maximum loss you'll accept"
        )
        
        interval = "15m"
        period = "5d"
        
    else:
        st.sidebar.subheader("📊 Configuration")
        ticker = st.sidebar.text_input(
            "Ticker Symbol",
            value="NVDA",
            help="Enter stock ticker"
        ).upper()
        
        interval_options = {
            "1 minute": ("1d", "1m"),
            "5 minutes": ("5d", "5m"),
            "15 minutes": ("5d", "15m")
        }
        interval_choice = st.sidebar.selectbox(
            "Chart Interval",
            options=list(interval_options.keys()),
            index=2
        )
        period, interval = interval_options[interval_choice]
        
        st.sidebar.markdown("### 💰 Position Sizing")
        
        investment_amount = st.sidebar.number_input(
            "Investment ($)",
            min_value=100.0,
            max_value=1000000.0,
            value=1000.0,
            step=100.0
        )
        
        stop_loss_pct = st.sidebar.number_input(
            "Stop Loss (%)",
            min_value=0.5,
            max_value=20.0,
            value=2.0,
            step=0.5
        )
    
    # Target price (both modes)
    use_target = st.sidebar.checkbox("Set Target Price", value=False)
    target_price = None
    if use_target:
        target_price = st.sidebar.number_input(
            "Target Price ($)",
            min_value=0.01,
            value=100.0,
            step=0.01
        )
    
    # API Key (advanced only)
    api_key = None
    if not is_simple_mode:
        api_key = get_finnhub_api_key()
        if not api_key:
            with st.sidebar.expander("🔑 Finnhub API (Optional)"):
                st.markdown("Get free key at [finnhub.io](https://finnhub.io)")
                api_key = st.text_input("API Key", type="password")
    
    st.sidebar.markdown("---")
    analyze = st.sidebar.button("🚀 ANALYZE", type="primary", use_container_width=True)
    
    # Main content
    if not analyze:
        # Welcome screen
        if is_simple_mode:
            st.title("📊 Simple Trading Dashboard")
            st.markdown("### Perfect for beginners!")
            
            st.info("👈 Enter a stock symbol and click **ANALYZE** to get a clear BUY, SELL, or WAIT signal")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✨ What you'll get:")
                st.markdown("""
                - 🟢🔴🟡 Clear traffic light signal
                - 📖 Plain English explanation
                - 💰 Simple trade setup
                - 📈 Optional price chart
                """)
            
            with col2:
                st.markdown("#### 💡 How it works:")
                st.markdown("""
                We analyze 4 key indicators:
                - **Trend** (price direction)
                - **RSI** (overbought/oversold)
                - **Momentum** (MACD)
                - **Average price** (VWAP)
                """)
            
            st.markdown("---")
            st.markdown("#### 🔥 Try these popular stocks:")
            ticker_cols = st.columns(7)
            for i, t in enumerate(["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]):
                with ticker_cols[i]:
                    st.code(t)
        
        else:
            st.title("📈 Professional Trading Dashboard")
            st.info("👈 Configure settings and click **ANALYZE**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📊 Features")
                st.markdown("""
                - Real-time quotes
                - Advanced indicators
                - Multi-timeframe analysis
                - Risk/reward calculator
                """)
            
            with col2:
                st.markdown("### 📈 Indicators")
                st.markdown("""
                - EMA (8, 20, 50)
                - RSI (Wilder's method)
                - MACD with histogram
                - VWAP (daily reset)
                - Bollinger Bands
                - OBV
                """)
            
            with col3:
                st.markdown("### 💡 Tips")
                st.markdown("""
                - Check multiple timeframes
                - Always use stop losses
                - Monitor volume
                - Wait for confirmation
                """)
        
        st.markdown("---")
        st.warning("⚠️ **Disclaimer**: Educational purposes only. Not financial advice.")
    
    else:
        # Analysis mode
        try:
            with st.spinner(f"🔄 Analyzing {ticker}..."):
                # Fetch data
                df, error = fetch_yfinance_data(ticker, period, interval)
                
                if error:
                    st.error(f"❌ {error}")
                    st.stop()
                
                # Get current price
                current_price = None
                current_volume = 0
                
                if api_key:
                    quote, error = fetch_finnhub_quote(ticker, api_key)
                    if quote and not error:
                        current_price = quote['c']
                        current_volume = quote.get('v', 0)
                
                if current_price is None:
                    current_price, current_volume = get_current_price_fallback(ticker)
                    if current_price is None:
                        current_price = df['Close'].iloc[-1]
                        current_volume = df['Volume'].iloc[-1]
                
                # Calculate indicators based on mode
                if is_simple_mode:
                    df = calculate_basic_indicators(df)
                else:
                    df = calculate_all_indicators(df)
                
                # Get latest values
                latest = df.iloc[-1]
                ema_8 = latest['EMA_8']
                ema_20 = latest['EMA_20']
                rsi = latest['RSI']
                macd = latest['MACD']
                macd_signal = latest['MACD_Signal']
                vwap = latest['VWAP']
                
                # Calculate pivot points
                pivot_points = calculate_pivot_points(df)
                
                # Generate signal
                verdict, verdict_type, signals, score, reasoning, action = generate_trading_signal(
                    ema_8, ema_20, rsi, macd, macd_signal, current_price, vwap
                )
                
                # Prepare trade info
                shares = investment_amount / current_price
                stop_loss_price = current_price * (1 - stop_loss_pct / 100)
                max_loss = investment_amount * (stop_loss_pct / 100)
                
                trade_info = {
                    'shares': shares,
                    'stop_loss': stop_loss_price,
                    'max_loss': max_loss,
                    'investment': investment_amount,
                    'stop_loss_pct': stop_loss_pct,
                    'current_volume': current_volume
                }
                
                if target_price:
                    risk = abs(current_price - stop_loss_price)
                    reward = abs(target_price - current_price)
                    rr_ratio = reward / risk if risk > 0 else 0
                    potential_profit = (target_price - current_price) * shares
                    
                    trade_info['target_price'] = target_price
                    trade_info['risk_reward'] = rr_ratio
                    trade_info['potential_profit'] = potential_profit
                
                # Price change
                prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
                price_change = current_price - prev_close
                price_change_pct = (price_change / prev_close) * 100
                
                # Display based on mode
                if is_simple_mode:
                    show_simple_mode(
                        ticker, current_price, verdict, verdict_type,
                        reasoning, action, trade_info, df,
                        price_change, price_change_pct
                    )
                else:
                    show_advanced_mode(
                        ticker, current_price, verdict, verdict_type,
                        signals, score, reasoning, pivot_points,
                        latest, df, trade_info,
                        price_change, price_change_pct
                    )
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("💡 Please check the ticker and try again")

if __name__ == "__main__":
    main()
