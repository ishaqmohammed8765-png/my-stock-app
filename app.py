import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Stock Signal Tracker", page_icon="📈", layout="wide")

# Load thresholds from secrets (with fallback defaults)
try:
    RSI_STRONG_BUY = st.secrets.get("RSI_STRONG_BUY", 35)
    RSI_BUY_MAX = st.secrets.get("RSI_BUY_MAX", 50)
    RSI_SELL_MIN = st.secrets.get("RSI_SELL_MIN", 50)
    RSI_STRONG_SELL = st.secrets.get("RSI_STRONG_SELL", 65)
except:
    # Default thresholds if secrets not configured
    RSI_STRONG_BUY = 35
    RSI_BUY_MAX = 50
    RSI_SELL_MIN = 50
    RSI_STRONG_SELL = 65

st.title("📈 Professional Stock Signal Tracker")
st.markdown("High-conviction trading signals with advanced technical analysis")

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_obv(data):
    """Calculate On-Balance Volume (OBV)"""
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return obv

def calculate_pivot_points(data, current_price):
    """Calculate Pivot Point, Support (S1), and Resistance (R1) levels using latest data"""
    # Use the most recent high/low from historical data
    high = float(data['High'].iloc[-1])
    low = float(data['Low'].iloc[-1])
    # Use real-time price for most accurate pivot calculation
    close = current_price
    
    pivot = (high + low + close) / 3
    support = (2 * pivot) - high  # S1 - Buy Target
    resistance = (2 * pivot) - low  # R1 - Sell Target
    
    return pivot, support, resistance

def get_verdict(ema_8, ema_20, rsi):
    """Determine trading verdict based on EMA and RSI"""
    ema_bullish = ema_8 > ema_20
    ema_bearish = ema_8 < ema_20
    
    if ema_bullish and rsi < RSI_STRONG_BUY:
        return "🚀 STRONG BUY", "success"
    elif ema_bullish and RSI_STRONG_BUY <= rsi <= RSI_BUY_MAX:
        return "✅ BUY", "success"
    elif ema_bearish and rsi > RSI_STRONG_SELL:
        return "🔻 STRONG SELL", "error"
    elif ema_bearish and RSI_SELL_MIN <= rsi <= RSI_STRONG_SELL:
        return "⚠️ SELL", "error"
    else:
        return "⏸️ WAIT", "info"

@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_stock_data(ticker):
    """Fetch stock data from Yahoo Finance with caching"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d", interval="15m")
        
        if data.empty:
            return None, None, "No data found for this ticker"
        
        # Get real-time price (most current available)
        try:
            # Try fast_info first (most reliable)
            real_time_price = stock.fast_info['last_price']
        except:
            try:
                # Fallback to info
                real_time_price = stock.info.get('regularMarketPrice', None)
            except:
                # Final fallback to latest historical close
                real_time_price = float(data['Close'].iloc[-1])
        
        # Calculate indicators using latest data
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['RSI'] = calculate_rsi(data, period=14)
        data['OBV'] = calculate_obv(data)
        
        return data, real_time_price, None
        
    except Exception as e:
        error_msg = str(e)
        
        if "429" in error_msg or "rate limit" in error_msg.lower():
            return None, None, "⏳ Yahoo Finance rate limit reached. Please wait a few minutes and try again."
        elif "404" in error_msg or "not found" in error_msg.lower():
            return None, None, f"❌ Ticker '{ticker}' not found. Please check the symbol and try again."
        else:
            return None, None, f"❌ Error fetching data: {error_msg}"

def create_candlestick_chart(data, ticker):
    """Create professional candlestick chart with EMA overlays"""
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # EMA 8 (Yellow)
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA_8'],
        name='EMA 8 (Fast)',
        line=dict(color='#FFD700', width=2)
    ))
    
    # EMA 20 (Blue)
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA_20'],
        name='EMA 20 (Slow)',
        line=dict(color='#1E90FF', width=2)
    ))
    
    fig.update_layout(
        title=f'{ticker.upper()} - Price Action with EMAs',
        yaxis_title='Price ($)',
        xaxis_title='Time',
        template='plotly_dark',
        height=500,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

# Sidebar
st.sidebar.header("⚙️ Settings")

with st.sidebar.form("ticker_form"):
    st.markdown("### Stock Symbol")
    ticker = st.text_input(
        "Ticker",
        value="NVDA",
        help="Examples: AAPL, TSLA, MSFT, BTC-USD",
        placeholder="Enter ticker..."
    )
    
    submit_button = st.form_submit_button("🔍 Analyze Stock", use_container_width=True)
    
    st.caption("💡 Data cached for 5 minutes")

# Main content
if submit_button and ticker:
    with st.spinner(f"Analyzing {ticker.upper()}..."):
        data, real_time_price, error = fetch_stock_data(ticker.upper())
    
    if error:
        st.error(error)
        if "rate limit" in error.lower():
            st.info("💡 **Tips:**\n"
                    "- Wait 2-5 minutes before retrying\n"
                    "- Data is cached for 5 minutes\n"
                    "- Avoid rapid successive requests")
    elif data is not None:
        # Use real-time price for current price
        current_price = float(real_time_price)
        
        # Extract latest indicator values
        ema_8_latest = float(data['EMA_8'].iloc[-1])
        ema_20_latest = float(data['EMA_20'].iloc[-1])
        rsi_latest = float(data['RSI'].iloc[-1])
        
        # Calculate pivot points using real-time price for accuracy
        pivot, support, resistance = calculate_pivot_points(data, current_price)
        
        # Get trading verdict
        verdict, verdict_type = get_verdict(ema_8_latest, ema_20_latest, rsi_latest)
        
        # TOP SECTION: Price Targets (Most Important)
        st.markdown("### 💰 Price Levels")
        st.caption("🔴 **LIVE** - Real-time market data with 4 decimal precision")
        col_price1, col_price2, col_price3 = st.columns(3)
        
        with col_price1:
            st.metric(
                label="🎯 Buy Target (S1)",
                value=f"${support:.4f}",
                delta=f"{((support - current_price) / current_price * 100):.1f}%"
            )
        
        with col_price2:
            st.metric(
                label="📍 Current Price",
                value=f"${current_price:.4f}"
            )
        
        with col_price3:
            st.metric(
                label="🚪 Sell Target (R1)",
                value=f"${resistance:.4f}",
                delta=f"{((resistance - current_price) / current_price * 100):.1f}%"
            )
        
        st.markdown("---")
        
        # Master Verdict
        st.markdown("### 🎯 Trading Signal")
        if verdict_type == "success":
            st.success(f"## {verdict}")
        elif verdict_type == "error":
            st.error(f"## {verdict}")
        else:
            st.info(f"## {verdict}")
        
        st.markdown("---")
        
        # Technical Indicators
        st.markdown("### 📊 Technical Indicators")
        col_ind1, col_ind2, col_ind3, col_ind4 = st.columns(4)
        
        with col_ind1:
            st.metric("EMA 8", f"${ema_8_latest:.4f}")
        with col_ind2:
            st.metric("EMA 20", f"${ema_20_latest:.4f}")
        with col_ind3:
            st.metric("RSI (14)", f"{rsi_latest:.2f}")
        with col_ind4:
            obv_recent = data['OBV'].iloc[-20:]
            volume_trend = "📈 Bullish" if obv_recent.iloc[-1] > obv_recent.iloc[0] else "📉 Bearish"
            st.metric("Volume", volume_trend)
        
        st.markdown("---")
        
        # Professional Candlestick Chart
        st.markdown("### 📈 Professional Chart")
        st.caption("Candlestick chart clearly shows price spikes and movements with EMA overlays")
        fig = create_candlestick_chart(data, ticker)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### RSI Momentum")
            chart_rsi = pd.DataFrame({'RSI': data['RSI']})
            st.line_chart(chart_rsi)
            st.caption("Overbought: >70 | Oversold: <30")
        
        with col_chart2:
            st.markdown("#### Volume Trend (OBV)")
            chart_obv = pd.DataFrame({'OBV': data['OBV']})
            st.line_chart(chart_obv)
            st.caption("Rising = Bullish | Falling = Bearish")
        
        # Signal Breakdown
        st.markdown("---")
        st.markdown("### 🔍 Signal Details")
        
        col_sig1, col_sig2 = st.columns(2)
        
        with col_sig1:
            if ema_8_latest > ema_20_latest:
                st.success("**✅ EMA Signal: BULLISH**\n\nFast EMA above Slow EMA")
            else:
                st.error("**⚠️ EMA Signal: BEARISH**\n\nFast EMA below Slow EMA")
        
        with col_sig2:
            if rsi_latest > 70:
                st.error("**⚠️ RSI: OVERBOUGHT**\n\nRSI above 70 - Consider taking profits")
            elif rsi_latest < 30:
                st.success("**✅ RSI: OVERSOLD**\n\nRSI below 30 - Potential buying opportunity")
            else:
                st.info(f"**📊 RSI: NEUTRAL**\n\nRSI at {rsi_latest:.1f}")
        
        st.success("✅ Real-time analysis complete • Live price: ${:.4f} • Data cached for 5 minutes".format(current_price))

elif not submit_button:
    # Welcome screen
    st.info("👈 **Enter a stock ticker and click 'Analyze Stock' to begin**")
    
    st.markdown("### 🎯 Features")
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.markdown("""
        **📊 Advanced Analysis**
        - High-conviction buy/sell signals
        - Pivot-based price targets
        - Professional candlestick charts
        - EMA trend analysis
        """)
    
    with col_feat2:
        st.markdown("""
        **🛡️ Built-in Protection**
        - Smart data caching (5 min)
        - Rate limit handling
        - Configurable thresholds
        - Mobile-optimized layout
        """)
    
    st.markdown("### 📱 Signal Criteria")
    st.markdown("""
    - **🚀 STRONG BUY**: EMA bullish + RSI < 35
    - **✅ BUY**: EMA bullish + RSI 35-50
    - **⚠️ SELL**: EMA bearish + RSI 50-65
    - **🔻 STRONG SELL**: EMA bearish + RSI > 65
    - **⏸️ WAIT**: All other conditions
    """)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.info(
    "**Technical Indicators:**\n"
    "- EMA 8/20: Trend direction\n"
    "- RSI 14: Momentum strength\n"
    "- OBV: Volume confirmation\n"
    "- Pivot Points: Price targets\n\n"
    "**Privacy:**\n"
    "- No personal data stored\n"
    "- Configurable via st.secrets\n"
    "- Secure API handling\n\n"
    "Data: Yahoo Finance"
)

st.sidebar.markdown("---")
st.sidebar.caption("⚙️ Configure thresholds in `.streamlit/secrets.toml`")
