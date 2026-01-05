import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Professional Day Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================
NY_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

# ============================================================================
# API KEY MANAGEMENT
# ============================================================================
def get_finnhub_api_key():
    """Get Finnhub API key from secrets or user input"""
    api_key = None
    if hasattr(st, 'secrets') and 'FINNHUB_API_KEY' in st.secrets:
        api_key = st.secrets['FINNHUB_API_KEY']
    
    if not api_key:
        with st.sidebar.expander("🔑 Finnhub API Key (Optional)", expanded=False):
            st.markdown("Get a free API key at [finnhub.io](https://finnhub.io)")
            api_key = st.text_input(
                "Enter API Key",
                type="password",
                help="Optional: For enhanced real-time quotes"
            )
    
    return api_key if api_key else None

# ============================================================================
# FINANCIAL CALCULATIONS
# ============================================================================

def calculate_rsi_wilder(close_prices, period=14):
    """
    Calculate RSI using Wilder's smoothing method
    Uses exponential moving average with alpha=1/period
    """
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's smoothing: EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_obv_vectorized(df):
    """
    Vectorized On-Balance Volume calculation
    Much faster than iterative approach
    """
    price_change = np.sign(df['Close'].diff())
    obv = (price_change * df['Volume']).fillna(0).cumsum()
    return obv

def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD, Signal line, and Histogram
    """
    ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_vwap_daily(df):
    """
    Calculate VWAP that resets daily
    Groups by date for intraday data
    """
    df = df.copy()
    
    # Ensure timezone-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)
    
    # Add date column for grouping
    df['date'] = df.index.date
    
    # Calculate typical price
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tp_volume'] = df['typical_price'] * df['Volume']
    
    # Calculate cumulative sums within each day
    df['cumulative_tp_volume'] = df.groupby('date')['tp_volume'].cumsum()
    df['cumulative_volume'] = df.groupby('date')['Volume'].cumsum()
    
    # Calculate VWAP
    vwap = df['cumulative_tp_volume'] / df['cumulative_volume']
    
    return vwap

def calculate_bollinger_bands(close_prices, period=20, num_std=2):
    """
    Calculate Bollinger Bands
    """
    sma = close_prices.rolling(window=period).mean()
    std = close_prices.rolling(window=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_pivot_points(df):
    """
    Calculate pivot points (standard method)
    Uses previous day's high, low, close
    """
    # Get last complete bar
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

def calculate_risk_reward_ratio(entry_price, stop_loss, target_price):
    """
    Calculate risk/reward ratio
    """
    risk = abs(entry_price - stop_loss)
    reward = abs(target_price - entry_price)
    
    if risk == 0:
        return None
    
    return reward / risk

# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_finnhub_quote(ticker, api_key):
    """
    Fetch real-time quote from Finnhub
    Cached for 60 seconds
    """
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
    """
    Fetch historical data from Yahoo Finance
    Cached for 5 minutes
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        
        if df.empty:
            return None, f"No data available for {ticker}"
        
        # Ensure timezone awareness
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        df.index = df.index.tz_convert(NY_TZ)
        
        return df, None
    except Exception as e:
        return None, str(e)

def get_current_price_fallback(ticker):
    """
    Get current price from Yahoo Finance as fallback
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="1d", interval="1m")
        if not data.empty:
            return data['Close'].iloc[-1], data['Volume'].iloc[-1]
        return None, None
    except:
        return None, None

# ============================================================================
# INDICATOR CALCULATION WITH CACHING
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def calculate_all_indicators(df):
    """
    Calculate all technical indicators
    Cached to avoid recalculation
    """
    df = df.copy()
    
    # EMAs
    df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # RSI with Wilder's smoothing
    df['RSI'] = calculate_rsi_wilder(df['Close'], period=14)
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    
    # OBV (vectorized)
    df['OBV'] = calculate_obv_vectorized(df)
    
    # VWAP (daily reset)
    df['VWAP'] = calculate_vwap_daily(df)
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    
    return df

# ============================================================================
# TRADING LOGIC
# ============================================================================

def generate_trading_signal(ema_8, ema_20, rsi, macd, macd_signal, current_price, vwap):
    """
    Generate trading signal based on multiple indicators
    """
    signals = []
    score = 0
    
    # EMA Crossover
    if ema_8 > ema_20:
        signals.append("EMA: Bullish (8>20)")
        score += 1
    else:
        signals.append("EMA: Bearish (8<20)")
        score -= 1
    
    # RSI
    if rsi < 30:
        signals.append("RSI: Oversold (Strong Buy)")
        score += 2
    elif rsi < 45:
        signals.append("RSI: Buy Zone")
        score += 1
    elif rsi > 70:
        signals.append("RSI: Overbought (Strong Sell)")
        score -= 2
    elif rsi > 55:
        signals.append("RSI: Sell Zone")
        score -= 1
    else:
        signals.append("RSI: Neutral")
    
    # MACD
    if macd > macd_signal:
        signals.append("MACD: Bullish")
        score += 1
    else:
        signals.append("MACD: Bearish")
        score -= 1
    
    # Price vs VWAP
    if current_price > vwap:
        signals.append("Price > VWAP (Bullish)")
        score += 1
    else:
        signals.append("Price < VWAP (Bearish)")
        score -= 1
    
    # Final verdict
    if score >= 3:
        verdict = "🚀 STRONG BUY"
        verdict_type = "success"
    elif score >= 1:
        verdict = "✅ BUY"
        verdict_type = "success"
    elif score <= -3:
        verdict = "🔻 STRONG SELL"
        verdict_type = "error"
    elif score <= -1:
        verdict = "⚠️ SELL"
        verdict_type = "error"
    else:
        verdict = "⏸️ WAIT / NEUTRAL"
        verdict_type = "info"
    
    return verdict, verdict_type, signals, score

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comprehensive_chart(df, ticker, pivot_points):
    """
    Create multi-panel chart with price, indicators, and volume
    """
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
    
    # ========== ROW 1: CANDLESTICK + EMAs + VWAP + BOLLINGER BANDS ==========
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
    
    # ========== ROW 2: RSI ==========
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
    
    # ========== ROW 3: MACD ==========
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
    
    # ========== ROW 4: VOLUME ==========
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

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("📈 Professional Day Trading Dashboard")
    st.markdown("*Real-time market analysis with advanced technical indicators*")
    st.markdown("---")
    
    # ========== SIDEBAR ==========
    st.sidebar.header("⚙️ Trading Configuration")
    
    # Ticker input
    ticker = st.sidebar.text_input(
        "📊 Ticker Symbol",
        value="NVDA",
        help="Enter stock ticker (e.g., AAPL, TSLA, SPY)"
    ).upper()
    
    # Data interval
    interval_options = {
        "1 minute": ("1d", "1m"),
        "5 minutes": ("5d", "5m"),
        "15 minutes": ("5d", "15m")
    }
    interval_choice = st.sidebar.selectbox(
        "📅 Chart Interval",
        options=list(interval_options.keys()),
        index=2
    )
    period, interval = interval_options[interval_choice]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Position Sizing")
    
    # Investment parameters
    investment_amount = st.sidebar.number_input(
        "Investment Amount ($)",
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
        step=0.5,
        help="Maximum loss percentage you're willing to accept"
    )
    
    use_target = st.sidebar.checkbox("Set Target Price", value=False)
    target_price = None
    if use_target:
        target_price = st.sidebar.number_input(
            "Target Price ($)",
            min_value=0.01,
            value=100.0,
            step=0.01,
            help="Your profit target price"
        )
    
    # API Key
    api_key = get_finnhub_api_key()
    
    # Analyze button
    st.sidebar.markdown("---")
    analyze = st.sidebar.button("🚀 ANALYZE", type="primary", use_container_width=True)
    
    # ========== MAIN CONTENT ==========
    if not analyze:
        # Welcome screen
        st.info("👈 Configure your settings in the sidebar and click **ANALYZE** to begin")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Features")
            st.markdown("""
            - Real-time price quotes
            - Advanced technical indicators
            - Multi-timeframe analysis
            - Risk/reward calculator
            - Pivot points & support/resistance
            """)
        
        with col2:
            st.markdown("### 📈 Indicators")
            st.markdown("""
            - **EMA** (8, 20, 50)
            - **RSI** (Wilder's method)
            - **MACD** with histogram
            - **VWAP** (daily reset)
            - **Bollinger Bands**
            - **OBV** (vectorized)
            """)
        
        with col3:
            st.markdown("### 💡 Tips")
            st.markdown("""
            - Check multiple timeframes
            - Use stop losses always
            - Monitor volume patterns
            - Wait for confirmation signals
            - Risk max 1-2% per trade
            """)
        
        st.markdown("---")
        st.markdown("### 🔥 Popular Tickers")
        ticker_cols = st.columns(7)
        popular_tickers = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]
        for i, t in enumerate(popular_tickers):
            with ticker_cols[i]:
                st.code(t)
        
        st.markdown("---")
        st.warning("⚠️ **Disclaimer**: This is for educational purposes only. Not financial advice. Always do your own research.")
        
    else:
        # Analysis mode
        try:
            with st.spinner(f"🔄 Fetching market data for {ticker}..."):
                # Fetch historical data
                df, error = fetch_yfinance_data(ticker, period, interval)
                
                if error:
                    st.error(f"❌ {error}")
                    st.stop()
                
                # Get current price
                current_price = None
                current_volume = 0
                data_source = "Yahoo Finance (15min delayed)"
                
                # Try Finnhub if API key available
                if api_key:
                    quote, error = fetch_finnhub_quote(ticker, api_key)
                    if quote and not error:
                        current_price = quote['c']
                        current_volume = quote.get('v', 0)
                        data_source = "Finnhub (Real-time)"
                
                # Fallback to Yahoo Finance
                if current_price is None:
                    current_price, current_volume = get_current_price_fallback(ticker)
                    if current_price is None:
                        current_price = df['Close'].iloc[-1]
                        current_volume = df['Volume'].iloc[-1]
                
                # Calculate all indicators
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
                
                # Generate trading signal
                verdict, verdict_type, signals, score = generate_trading_signal(
                    ema_8, ema_20, rsi, macd, macd_signal, current_price, vwap
                )
            
            # ========== DISPLAY RESULTS ==========
            st.success(f"✅ Live data from {data_source}")
            
            # Current time in NY
            current_time_ny = datetime.now(NY_TZ)
            st.caption(f"🕐 Market Time (ET): {current_time_ny.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # Main metrics
            st.subheader(f"📊 {ticker} - Current Market Status")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                )
            
            with col2:
                st.metric("RSI (14)", f"{rsi:.2f}")
            
            with col3:
                ema_diff = ema_8 - ema_20
                st.metric("EMA 8", f"${ema_8:.2f}", f"{ema_diff:+.2f} vs EMA20")
            
            with col4:
                vwap_diff = ((current_price - vwap) / vwap) * 100
                st.metric("VWAP", f"${vwap:.2f}", f"{vwap_diff:+.2f}%")
            
            with col5:
                st.metric("Volume", f"{current_volume:,.0f}")
            
            # Trading Signal
            st.markdown("---")
            st.subheader("🎯 Trading Signal")
            
            if verdict_type == "success":
                st.success(f"## {verdict}")
            elif verdict_type == "error":
                st.error(f"## {verdict}")
            else:
                st.info(f"## {verdict}")
            
            # Signal details
            with st.expander("📋 Signal Breakdown", expanded=True):
                st.markdown(f"**Composite Score:** {score}/6")
                for signal in signals:
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
            
            shares = investment_amount / current_price
            stop_loss_price = current_price * (1 - stop_loss_pct / 100)
            max_loss = investment_amount * (stop_loss_pct / 100)
            
            calc_col1, calc_col2, calc_col3, calc_col4 = st.columns(4)
            
            with calc_col1:
                st.metric("Shares", f"{shares:.2f}")
            with calc_col2:
                st.metric("Stop Loss", f"${stop_loss_price:.2f}")
            with calc_col3:
                st.metric("Max Loss", f"${max_loss:.2f}", delta_color="inverse")
            
            if target_price:
                rr_ratio = calculate_risk_reward_ratio(current_price, stop_loss_price, target_price)
                potential_profit = (target_price - current_price) * shares
                
                with calc_col4:
                    if rr_ratio:
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
                st.markdown(f"- **EMA 8:** ${ema_8:.2f}")
                st.markdown(f"- **EMA 20:** ${ema_20:.2f}")
                st.markdown(f"- **EMA 50:** ${latest['EMA_50']:.2f}")
                st.markdown(f"- **RSI:** {rsi:.2f}")
                st.markdown(f"- **MACD:** {macd:.4f}")
                st.markdown(f"- **MACD Signal:** {macd_signal:.4f}")
                st.markdown(f"- **MACD Histogram:** {latest['MACD_Hist']:.4f}")
            
            with stats_col2:
                st.markdown("**Price Levels**")
                st.markdown(f"- **VWAP:** ${vwap:.2f}")
                st.markdown(f"- **BB Upper:** ${latest['BB_Upper']:.2f}")
                st.markdown(f"- **BB Middle:** ${latest['BB_Middle']:.2f}")
                st.markdown(f"- **BB Lower:** ${latest['BB_Lower']:.2f}")
                st.markdown(f"- **OBV:** {latest['OBV']:,.0f}")
                
                # Distance from levels
                dist_from_vwap = ((current_price - vwap) / vwap) * 100
                dist_from_bb_upper = ((current_price - latest['BB_Upper']) / latest['BB_Upper']) * 100
                
                st.markdown(f"- **Distance from VWAP:** {dist_from_vwap:+.2f}%")
                st.markdown(f"- **Distance from BB Upper:** {dist_from_bb_upper:+.2f}%")
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("💡 Please check the ticker symbol and try again")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
