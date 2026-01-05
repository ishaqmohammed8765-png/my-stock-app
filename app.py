import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests

# Page configuration
st.set_page_config(page_title="Hybrid Trading Calculator", page_icon="⚡", layout="wide")

# Load thresholds
try:
    RSI_STRONG_BUY = st.secrets.get("RSI_STRONG_BUY", 35)
    RSI_BUY_MAX = st.secrets.get("RSI_BUY_MAX", 50)
    RSI_SELL_MIN = st.secrets.get("RSI_SELL_MIN", 50)
    RSI_STRONG_SELL = st.secrets.get("RSI_STRONG_SELL", 65)
except:
    RSI_STRONG_BUY = 35
    RSI_BUY_MAX = 50
    RSI_SELL_MIN = 50
    RSI_STRONG_SELL = 65

st.title("⚡ Hybrid Trading Calculator")
st.markdown("Real-time prices from Finnhub • Historical charts from Yahoo Finance")

# ============ HELPER FUNCTIONS ============
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
    """Calculate Pivot Point, Support, and Resistance levels"""
    high = float(data['High'].iloc[-1])
    low = float(data['Low'].iloc[-1])
    close = current_price
    
    pivot = (high + low + close) / 3
    support = (2 * pivot) - high
    resistance = (2 * pivot) - low
    
    return pivot, support, resistance

def calculate_probability(data, current_price, target_price):
    """Calculate probability using Z-score"""
    daily_data = data['Close'].resample('D').last().dropna()
    
    if len(daily_data) < 20:
        daily_returns = daily_data.pct_change().dropna()
    else:
        daily_returns = daily_data.tail(20).pct_change().dropna()
    
    volatility = daily_returns.std()
    expected_return = (target_price - current_price) / current_price
    
    if volatility > 0:
        z_score = expected_return / volatility
    else:
        z_score = 0
    
    from scipy import stats
    try:
        if z_score > 0:
            probability = stats.norm.cdf(abs(z_score)) * 100
        else:
            probability = (1 - stats.norm.cdf(abs(z_score))) * 100
    except:
        if abs(z_score) < 1:
            probability = 68.0
        elif abs(z_score) < 2:
            probability = 47.5
        else:
            probability = 30.0
    
    probability = min(probability, 95.0)
    return probability, z_score, volatility

def calculate_investment_risk(current_price, sell_target, investment_amount, stop_loss_pct, slippage_pct):
    """Calculate shares, risk, and profit including slippage"""
    shares = investment_amount / current_price if current_price > 0 else 0
    stop_loss_price = current_price * (1 - stop_loss_pct / 100)
    
    risk_per_share = current_price - stop_loss_price
    total_risk = risk_per_share * shares
    risk_percentage = (total_risk / investment_amount * 100) if investment_amount > 0 else 0
    
    # Calculate profit with slippage
    profit_per_share = sell_target - current_price
    gross_profit = profit_per_share * shares
    slippage_cost = gross_profit * (slippage_pct / 100)
    net_profit = gross_profit - slippage_cost
    
    return {
        'shares': shares,
        'stop_loss_price': stop_loss_price,
        'total_risk': total_risk,
        'risk_percentage': risk_percentage,
        'gross_profit': gross_profit,
        'slippage_cost': slippage_cost,
        'net_profit': net_profit
    }

def get_verdict(ema_8, ema_20, rsi):
    """Determine trading verdict"""
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

# ============ DATA FETCHING FUNCTIONS ============
@st.cache_data(ttl=5, show_spinner=False)
def fetch_finnhub_price(ticker, api_key):
    """Fetch real-time price from Finnhub (fast, 5-second cache)"""
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={api_key}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if data is valid
            if data.get('c', 0) == 0 and data.get('pc', 0) == 0:
                return None, None, None, f"Ticker '{ticker}' not found or invalid"
            
            current_price = data.get('c')  # Current price
            change = data.get('d')  # Change
            percent_change = data.get('dp')  # Percent change
            
            return current_price, change, percent_change, None
        elif response.status_code == 429:
            return None, None, None, "Finnhub rate limit reached. Please wait a moment."
        else:
            return None, None, None, f"Finnhub API error: {response.status_code}"
    except Exception as e:
        return None, None, None, f"Error fetching Finnhub data: {str(e)}"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_historical(ticker):
    """Fetch historical data from yfinance (1-hour cache)"""
    try:
        # Use standard yfinance without custom session
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d", interval="15m")
        
        if data.empty:
            return None, "No historical data available"
        
        # Calculate indicators
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['RSI'] = calculate_rsi(data, period=14)
        data['OBV'] = calculate_obv(data)
        
        return data, None
    except Exception as e:
        return None, f"Error fetching historical data: {str(e)}"

def create_candlestick_chart(data, ticker):
    """Create professional candlestick chart"""
    fig = go.Figure()
    
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
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA_8'],
        name='EMA 8 (Fast)',
        line=dict(color='#FFD700', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA_20'],
        name='EMA 20 (Slow)',
        line=dict(color='#1E90FF', width=3)
    ))
    
    fig.update_layout(
        title=f'{ticker.upper()} - Candlestick Chart with EMAs',
        yaxis_title='Price ($)',
        xaxis_title='Time',
        template='plotly_dark',
        height=500,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

# ============ SIDEBAR WITH FORM ============
st.sidebar.header("⚙️ Calculation Settings")

with st.sidebar.form("calculation_form", clear_on_submit=False):
    st.markdown("### API Configuration")
    
    # Try to load from secrets first, then allow manual input
    default_api_key = ""
    try:
        default_api_key = st.secrets.get("FINNHUB_API_KEY", "")
    except:
        pass
    
    finnhub_api_key = st.text_input(
        "Finnhub API Key",
        value=default_api_key,
        type="password",
        help="Get free key at finnhub.io or store in .streamlit/secrets.toml",
        key="api_key_input"
    )
    
    st.markdown("### Stock Symbol")
    ticker = st.text_input(
        "Ticker",
        value="NVDA",
        help="Examples: AAPL, TSLA, MSFT, MKDW",
        key="ticker_input"
    )
    
    st.markdown("### Investment Parameters")
    investment_amount = st.number_input(
        "💵 Total Investment ($)",
        min_value=1.0,
        value=100.0,
        step=10.0,
        key="investment_input"
    )
    
    stop_loss_pct = st.number_input(
        "🛡️ Stop Loss (%)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        key="stop_loss_input"
    )
    
    slippage_pct = st.slider(
        "📉 Slippage (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Expected price slippage for penny stocks",
        key="slippage_input"
    )
    
    calculate_button = st.form_submit_button(
        "⚡ Calculate", 
        use_container_width=True, 
        type="primary"
    )
    
    st.caption("Real-time: 5s cache | Historical: 1hr cache")

# ============ MAIN CONTENT ============
if calculate_button and ticker:
    if not finnhub_api_key:
        st.warning("⚠️ Please enter your Finnhub API Key in the sidebar")
    else:
        # Fetch real-time price from Finnhub
        with st.spinner(f"⚡ Fetching real-time data for {ticker.upper()}..."):
            current_price, change, percent_change, price_error = fetch_finnhub_price(ticker.upper(), finnhub_api_key)
        
        if price_error:
            st.warning(f"⚠️ {price_error}")
            st.info("💡 **Troubleshooting:**\n"
                   "- Verify your Finnhub API key is correct\n"
                   "- Check ticker symbol (use US market symbols)\n"
                   "- Try again in a moment if rate limited")
        elif current_price:
            # Fetch historical data from yfinance
            with st.spinner("📊 Loading historical data..."):
                hist_data, hist_error = fetch_yfinance_historical(ticker.upper())
            
            if hist_error:
                st.warning(f"⚠️ Historical data: {hist_error}")
                st.info("Charts will be unavailable, but calculations will proceed with live price")
            
            # ============ DISPLAY RESULTS ============
            
            # Current Price Display (Clean Metrics)
            st.markdown("### 💰 Real-Time Price")
            col_price1, col_price2, col_price3 = st.columns(3)
            
            with col_price1:
                st.metric(
                    label=f"{ticker.upper()} Current Price",
                    value=f"${current_price:.4f}"
                )
            
            with col_price2:
                st.metric(
                    label="Price Change",
                    value=f"${change:.4f}" if change else "N/A",
                    delta=f"{percent_change:.2f}%" if percent_change else None
                )
            
            with col_price3:
                st.metric(
                    label="Data Source",
                    value="Finnhub (Live)"
                )
            
            st.markdown("---")
            
            # Calculate metrics if we have historical data
            if hist_data is not None:
                ema_8_latest = float(hist_data['EMA_8'].iloc[-1])
                ema_20_latest = float(hist_data['EMA_20'].iloc[-1])
                rsi_latest = float(hist_data['RSI'].iloc[-1])
                
                pivot, support, resistance = calculate_pivot_points(hist_data, current_price)
                probability, z_score, volatility = calculate_probability(hist_data, current_price, resistance)
                
                # Calculate investment with slippage
                calc = calculate_investment_risk(
                    current_price, resistance, investment_amount, stop_loss_pct, slippage_pct
                )
                
                verdict, verdict_type = get_verdict(ema_8_latest, ema_20_latest, rsi_latest)
                
                # VERDICT BOX
                st.markdown("## 🎯 TRADING SIGNAL")
                if verdict_type == "success":
                    st.success(f"# {verdict}")
                elif verdict_type == "error":
                    st.error(f"# {verdict}")
                else:
                    st.info(f"# {verdict}")
                
                st.markdown("---")
                
                # PRICE TARGETS
                st.markdown("### 🎯 Price Targets")
                col_target1, col_target2, col_target3 = st.columns(3)
                
                with col_target1:
                    st.metric(
                        "🎯 Buy Target (S1)",
                        f"${support:.4f}",
                        delta=f"{((support - current_price) / current_price * 100):.2f}%"
                    )
                
                with col_target2:
                    st.metric(
                        "📍 Current Price",
                        f"${current_price:.4f}"
                    )
                
                with col_target3:
                    st.metric(
                        "🚪 Sell Target (R1)",
                        f"${resistance:.4f}",
                        delta=f"{((resistance - current_price) / current_price * 100):.2f}%"
                    )
                
                st.markdown("---")
                
                # CONVICTION SCORE
                st.markdown("### 🎲 Conviction Score")
                col_conv1, col_conv2, col_conv3 = st.columns(3)
                
                with col_conv1:
                    if probability >= 70:
                        st.success(f"# {probability:.1f}%")
                        st.markdown("### 🟢 High Conviction")
                    elif probability >= 50:
                        st.warning(f"# {probability:.1f}%")
                        st.markdown("### 🟡 Medium Conviction")
                    else:
                        st.error(f"# {probability:.1f}%")
                        st.markdown("### 🔴 Low Conviction")
                
                with col_conv2:
                    st.metric("Z-Score", f"{z_score:.2f}σ")
                    st.caption("Standard deviations to target")
                
                with col_conv3:
                    st.metric("Daily Volatility", f"{volatility*100:.2f}%")
                    st.caption("20-day standard deviation")
                
                st.markdown("---")
                
                # INVESTMENT BREAKDOWN
                st.markdown("### 🧮 Investment Breakdown")
                st.caption(f"Based on Finnhub live price: ${current_price:.4f} | Stop Loss: {stop_loss_pct}% | Slippage: {slippage_pct}%")
                
                col_calc1, col_calc2, col_calc3, col_calc4 = st.columns(4)
                
                with col_calc1:
                    st.metric(
                        "📦 Shares",
                        f"{calc['shares']:.2f}"
                    )
                    st.caption(f"@ ${current_price:.4f}/share")
                
                with col_calc2:
                    if calc['risk_percentage'] <= 5:
                        st.success(f"## {calc['risk_percentage']:.1f}%")
                    elif calc['risk_percentage'] <= 10:
                        st.warning(f"## {calc['risk_percentage']:.1f}%")
                    else:
                        st.error(f"## {calc['risk_percentage']:.1f}%")
                    st.markdown("**Risk %**")
                    st.caption(f"${calc['total_risk']:.2f} at risk")
                
                with col_calc3:
                    st.metric(
                        "💰 Net Profit",
                        f"${calc['net_profit']:.2f}"
                    )
                    st.caption(f"Gross: ${calc['gross_profit']:.2f}")
                    st.caption(f"Slippage: -${calc['slippage_cost']:.2f}")
                
                with col_calc4:
                    rr_ratio = calc['net_profit'] / calc['total_risk'] if calc['total_risk'] > 0 else 0
                    if rr_ratio >= 2:
                        st.success(f"**{rr_ratio:.2f}:1**")
                        st.caption("✅ Good R/R")
                    elif rr_ratio >= 1:
                        st.warning(f"**{rr_ratio:.2f}:1**")
                        st.caption("⚠️ Fair R/R")
                    else:
                        st.error(f"**{rr_ratio:.2f}:1**")
                        st.caption("❌ Poor R/R")
                
                st.info(f"💡 Net profit includes {slippage_pct}% slippage cost. Adjust slider for penny stock volatility.")
                
                st.markdown("---")
                
                # TECHNICAL INDICATORS
                st.markdown("### 📊 Technical Indicators")
                col_ind1, col_ind2, col_ind3, col_ind4 = st.columns(4)
                
                with col_ind1:
                    st.metric("EMA 8", f"${ema_8_latest:.4f}")
                with col_ind2:
                    st.metric("EMA 20", f"${ema_20_latest:.4f}")
                with col_ind3:
                    st.metric("RSI (14)", f"{rsi_latest:.2f}")
                with col_ind4:
                    obv_recent = hist_data['OBV'].iloc[-20:]
                    volume_trend = "📈 Bullish" if obv_recent.iloc[-1] > obv_recent.iloc[0] else "📉 Bearish"
                    st.metric("Volume", volume_trend)
                
                st.markdown("---")
                
                # CANDLESTICK CHART
                st.markdown("### 📈 Historical Chart (Yahoo Finance)")
                fig = create_candlestick_chart(hist_data, ticker)
                st.plotly_chart(fig, use_container_width=True)
                
                # ADDITIONAL CHARTS
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown("#### RSI Momentum")
                    chart_rsi = pd.DataFrame({'RSI': hist_data['RSI']})
                    st.line_chart(chart_rsi)
                
                with col_chart2:
                    st.markdown("#### Volume (OBV)")
                    chart_obv = pd.DataFrame({'OBV': hist_data['OBV']})
                    st.line_chart(chart_obv)
                
                st.success(f"✅ Live price: ${current_price:.4f} (Finnhub) | Charts: Yahoo Finance (1hr cache)")
            else:
                # No historical data, but we have live price
                st.markdown("### 🧮 Basic Calculation (No Historical Data)")
                
                # Simple calculation without targets
                shares = investment_amount / current_price if current_price > 0 else 0
                stop_loss_price = current_price * (1 - stop_loss_pct / 100)
                total_risk = (current_price - stop_loss_price) * shares
                risk_pct = (total_risk / investment_amount * 100) if investment_amount > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📦 Shares", f"{shares:.2f}")
                with col2:
                    st.metric("🛡️ Risk", f"${total_risk:.2f}")
                    st.caption(f"{risk_pct:.1f}%")
                with col3:
                    st.metric("Stop Loss", f"${stop_loss_price:.4f}")

elif not calculate_button:
    st.info("👈 **Complete the form in the sidebar and click Calculate**")
    
    st.markdown("### ⚡ Hybrid Data System")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔴 Real-Time (Finnhub)**
        - Current price (5s cache)
        - Price change & %
        - Perfect for penny stocks
        - Fast & accurate
        """)
    
    with col2:
        st.markdown("""
        **📊 Historical (Yahoo Finance)**
        - 5-day charts (1hr cache)
        - EMA 8 & 20 indicators
        - RSI & OBV analysis
        - No rate limit issues
        """)
    
    st.markdown("### 🧮 Features")
    st.markdown("""
    - **Live Pricing**: Finnhub API for accurate penny stock prices
    - **Slippage Calculator**: Adjust for realistic penny stock spreads
    - **Smart Caching**: 5s for prices, 1hr for charts
    - **Clean Error Handling**: No blue code errors, only warnings
    - **4 Decimal Precision**: Perfect for stocks like MKDW
    """)
    
    st.markdown("### 🔑 Setup")
    st.markdown("""
    1. Get free API key at [finnhub.io](https://finnhub.io)
    2. Enter API key in sidebar
    3. Enter ticker and investment amount
    4. Adjust stop loss and slippage
    5. Click Calculate!
    """)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(
    "**Hybrid System:**\n"
    "🔴 Finnhub: Real-time (5s)\n"
    "📊 Yahoo: Historical (1hr)\n\n"
    "**Why Hybrid?**\n"
    "- Accurate penny stock prices\n"
    "- No Yahoo rate limits\n"
    "- Fast performance\n"
    "- Professional data\n\n"
    "Get Finnhub key: finnhub.io"
)
