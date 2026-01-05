import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(page_title="Professional Stock Dashboard", page_icon="📊", layout="wide")

# ============ SECURITY: API KEY VALIDATION ============
try:
    FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]
except Exception:
    st.error("🔑 **API Key not found.** Please add `FINNHUB_API_KEY` to your Streamlit Secrets.")
    st.info("""
    **How to add secrets:**
    1. Create `.streamlit/secrets.toml` in your project
    2. Add: `FINNHUB_API_KEY = "your_key_here"`
    3. For deployed apps: Add in Streamlit Cloud settings
    """)
    st.stop()

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

st.title("📊 Professional Stock Dashboard")
st.markdown("Real-time data • Technical analysis • Market news")

# ============ HELPER FUNCTIONS ============
def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_vwap(data):
    """Calculate Volume Weighted Average Price"""
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

def calculate_rvol(current_volume, historical_data):
    """Calculate Relative Volume (RVOL)"""
    if historical_data is None or historical_data.empty:
        return None
    
    # Calculate average volume
    avg_volume = historical_data['Volume'].mean()
    
    if avg_volume > 0 and current_volume:
        rvol = current_volume / avg_volume
        return rvol
    return None

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=period).mean()

def calculate_obv(data):
    """Calculate On-Balance Volume"""
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
    """Calculate Pivot Points"""
    high = float(data['High'].iloc[-1])
    low = float(data['Low'].iloc[-1])
    close = current_price
    
    pivot = (high + low + close) / 3
    support = (2 * pivot) - high
    resistance = (2 * pivot) - low
    
    return pivot, support, resistance

def calculate_probability(data, current_price, target_price):
    """Calculate Z-score probability"""
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
    """Calculate investment metrics with slippage"""
    shares = investment_amount / current_price if current_price > 0 else 0
    stop_loss_price = current_price * (1 - stop_loss_pct / 100)
    
    risk_per_share = current_price - stop_loss_price
    total_risk = risk_per_share * shares
    risk_percentage = (total_risk / investment_amount * 100) if investment_amount > 0 else 0
    
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

# ============ DATA FETCHING ============
@st.cache_data(ttl=5, show_spinner=False)
def fetch_finnhub_price(ticker):
    """Fetch real-time price and volume from Finnhub"""
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('c', 0) == 0 and data.get('pc', 0) == 0:
                return None, None, None, None, f"Ticker '{ticker}' not found"
            
            current_price = data.get('c')  # Current price
            change = data.get('d')  # Change
            percent_change = data.get('dp')  # Percent change
            volume = data.get('v')  # Volume
            
            return current_price, change, percent_change, volume, None
        elif response.status_code == 429:
            return None, None, None, None, "Rate limit reached. Please wait."
        else:
            return None, None, None, None, f"API error: {response.status_code}"
    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_historical(ticker):
    """Fetch historical data from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get 1 year of daily data for SMAs
        data_1y = stock.history(period="1y", interval="1d")
        
        # Get 5 days of 15m data for short-term analysis
        data_5d = stock.history(period="5d", interval="15m")
        
        if data_5d.empty:
            return None, None, "No historical data available"
        
        # Calculate indicators on 15m data
        data_5d['EMA_8'] = data_5d['Close'].ewm(span=8, adjust=False).mean()
        data_5d['EMA_20'] = data_5d['Close'].ewm(span=20, adjust=False).mean()
        data_5d['RSI'] = calculate_rsi(data_5d, period=14)
        data_5d['OBV'] = calculate_obv(data_5d)
        data_5d['VWAP'] = calculate_vwap(data_5d)
        
        # Calculate SMAs on daily data
        if not data_1y.empty:
            data_1y['SMA_50'] = calculate_sma(data_1y, 50)
            data_1y['SMA_200'] = calculate_sma(data_1y, 200)
            data_1y['RSI_14'] = calculate_rsi(data_1y, 14)
        
        return data_5d, data_1y, None
    except Exception as e:
        return None, None, f"Error: {str(e)}"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_finnhub_news(ticker):
    """Fetch latest news from Finnhub"""
    try:
        # Get news from last 7 days
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        
        from_date = week_ago.strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')
        
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            news_data = response.json()
            # Return top 5 news items
            return news_data[:5] if news_data else [], None
        else:
            return [], f"Error fetching news: {response.status_code}"
    except Exception as e:
        return [], f"Error: {str(e)}"

def create_price_chart_with_sma(data_5d, data_1y, ticker):
    """Create candlestick chart with EMAs and SMAs"""
    fig = go.Figure()
    
    # Candlestick on 15m data
    fig.add_trace(go.Candlestick(
        x=data_5d.index,
        open=data_5d['Open'],
        high=data_5d['High'],
        low=data_5d['Low'],
        close=data_5d['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # EMA 8 and 20 on 15m data
    fig.add_trace(go.Scatter(
        x=data_5d.index,
        y=data_5d['EMA_8'],
        name='EMA 8',
        line=dict(color='#FFD700', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=data_5d.index,
        y=data_5d['EMA_20'],
        name='EMA 20',
        line=dict(color='#1E90FF', width=2)
    ))
    
    # Add VWAP for day trading
    fig.add_trace(go.Scatter(
        x=data_5d.index,
        y=data_5d['VWAP'],
        name='VWAP',
        line=dict(color='#00FF00', width=2.5, dash='dot')
    ))
    
    # Add SMAs if daily data available
    if data_1y is not None and not data_1y.empty:
        # Only show last 5 days of SMA data to match the chart timeframe
        recent_1y = data_1y.tail(5)
        
        fig.add_trace(go.Scatter(
            x=recent_1y.index,
            y=recent_1y['SMA_50'],
            name='SMA 50',
            line=dict(color='#FF6B6B', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=recent_1y.index,
            y=recent_1y['SMA_200'],
            name='SMA 200',
            line=dict(color='#9B59B6', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f'{ticker.upper()} - Price with EMAs, VWAP & SMAs',
        yaxis_title='Price ($)',
        xaxis_title='Time',
        template='plotly_dark',
        height=500,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_rsi_chart(data, ticker):
    """Create RSI momentum chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI'],
        name='RSI (14)',
        line=dict(color='#00D9FF', width=2)
    ))
    
    # Add overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")
    
    fig.update_layout(
        title=f'{ticker.upper()} - RSI Momentum (14-day)',
        yaxis_title='RSI',
        xaxis_title='Time',
        template='plotly_dark',
        height=300,
        hovermode='x unified',
        yaxis_range=[0, 100]
    )
    
    return fig

# ============ SIDEBAR ============
st.sidebar.header("⚙️ Trading Settings")

with st.sidebar.form("trading_form", clear_on_submit=False):
    st.markdown("### Stock Symbol")
    ticker = st.text_input(
        "Ticker",
        value="NVDA",
        help="US stock symbols (e.g., AAPL, TSLA, MSFT)",
        key="ticker_input"
    )
    
    st.markdown("### Investment Parameters")
    investment_amount = st.number_input(
        "💵 Investment ($)",
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
        key="slippage_input"
    )
    
    calculate_button = st.form_submit_button(
        "⚡ Analyze", 
        use_container_width=True, 
        type="primary"
    )

st.sidebar.markdown("---")
st.sidebar.success("🔐 Secure API Connection")
st.sidebar.info(
    "**Day Trading Features:**\n"
    "✅ VWAP (Volume Weighted Avg)\n"
    "✅ RVOL (Relative Volume)\n"
    "✅ Real-time volume data\n"
    "✅ Persistent calculations\n\n"
    "Results stay visible when\n"
    "switching between tabs!"
)

# ============ MAIN CONTENT ============
if calculate_button and ticker:
    with st.spinner(f"⚡ Fetching data for {ticker.upper()}..."):
        # Fetch real-time price with volume
        current_price, change, percent_change, current_volume, price_error = fetch_finnhub_price(ticker.upper())
        
        # Fetch historical data
        data_5d, data_1y, hist_error = fetch_yfinance_historical(ticker.upper())
        
        # Fetch news
        news_items, news_error = fetch_finnhub_news(ticker.upper())
    
    if price_error:
        st.warning(f"⚠️ {price_error}")
    elif current_price:
        # Calculate RVOL if we have volume data
        rvol = calculate_rvol(current_volume, data_5d) if current_volume and data_5d is not None else None
        
        # Calculate all metrics and store in session_state
        if data_5d is not None:
            ema_8 = float(data_5d['EMA_8'].iloc[-1])
            ema_20 = float(data_5d['EMA_20'].iloc[-1])
            rsi = float(data_5d['RSI'].iloc[-1])
            
            pivot, support, resistance = calculate_pivot_points(data_5d, current_price)
            probability, z_score, volatility = calculate_probability(data_5d, current_price, resistance)
            calc = calculate_investment_risk(current_price, resistance, investment_amount, stop_loss_pct, slippage_pct)
            verdict, verdict_type = get_verdict(ema_8, ema_20, rsi)
            
            # Store in session_state for persistence across tabs
            st.session_state['ticker'] = ticker.upper()
            st.session_state['current_price'] = current_price
            st.session_state['change'] = change
            st.session_state['percent_change'] = percent_change
            st.session_state['current_volume'] = current_volume
            st.session_state['rvol'] = rvol
            st.session_state['ema_8'] = ema_8
            st.session_state['ema_20'] = ema_20
            st.session_state['rsi'] = rsi
            st.session_state['support'] = support
            st.session_state['resistance'] = resistance
            st.session_state['pivot'] = pivot
            st.session_state['probability'] = probability
            st.session_state['z_score'] = z_score
            st.session_state['volatility'] = volatility
            st.session_state['calc'] = calc
            st.session_state['verdict'] = verdict
            st.session_state['verdict_type'] = verdict_type
            st.session_state['data_5d'] = data_5d
            st.session_state['data_1y'] = data_1y
            st.session_state['news_items'] = news_items
            st.session_state['news_error'] = news_error
            st.session_state['stop_loss_pct'] = stop_loss_pct
            st.session_state['slippage_pct'] = slippage_pct
            st.session_state['investment_amount'] = investment_amount

# Display results from session_state
if 'ticker' in st.session_state:
    # Retrieve from session_state
    ticker = st.session_state['ticker']
    current_price = st.session_state['current_price']
    change = st.session_state.get('change')
    percent_change = st.session_state.get('percent_change')
    current_volume = st.session_state.get('current_volume')
    rvol = st.session_state.get('rvol')
    stop_loss_pct = st.session_state.get('stop_loss_pct', 5.0)
    
    # Display current price at top
    st.markdown("### 💰 Real-Time Price")
    col_price1, col_price2, col_price3, col_price4 = st.columns(4)
    
    with col_price1:
        st.metric(
            f"{ticker} Price",
            f"${current_price:.4f}"
        )
    
    with col_price2:
        st.metric(
            "Change",
            f"${change:.4f}" if change else "N/A",
            delta=f"{percent_change:.2f}%" if percent_change else None
        )
    
    with col_price3:
        suggested_stop = current_price * (1 - stop_loss_pct / 100)
        st.metric(
            "Suggested Stop Loss",
            f"${suggested_stop:.4f}",
            delta=f"-{stop_loss_pct}%"
        )
    
    with col_price4:
        # Display volume with RVOL
        if current_volume:
            vol_display = f"{current_volume:,.0f}"
            if rvol:
                st.metric(
                    "Volume (RVOL)",
                    vol_display,
                    delta=f"{rvol:.2f}x"
                )
            else:
                st.metric("Volume", vol_display)
        else:
            # Fallback to historical average
            data_5d = st.session_state.get('data_5d')
            if data_5d is not None:
                avg_volume = data_5d['Volume'].tail(10).mean()
                st.metric("Avg Volume (10d)", f"{avg_volume:,.0f}")
            else:
                st.metric("Volume", "N/A")
    
    st.markdown("---")
    
    # ============ TABS FOR ORGANIZATION ============
    tab1, tab2, tab3 = st.tabs(["📊 Calculator", "📈 Analysis", "📰 News"])
    
    # TAB 1: CALCULATOR (Always visible with session_state)
    with tab1:
        verdict = st.session_state.get('verdict')
        verdict_type = st.session_state.get('verdict_type')
        support = st.session_state.get('support')
        resistance = st.session_state.get('resistance')
        calc = st.session_state.get('calc')
        probability = st.session_state.get('probability')
        z_score = st.session_state.get('z_score')
        volatility = st.session_state.get('volatility')
        slippage_pct = st.session_state.get('slippage_pct', 2.0)
        
        if verdict and calc:
            # VERDICT
            st.markdown("## 🎯 Trading Signal")
            if verdict_type == "success":
                st.success(f"# {verdict}")
            elif verdict_type == "error":
                st.error(f"# {verdict}")
            else:
                st.info(f"# {verdict}")
            
            st.markdown("---")
            
            # PRICE TARGETS
            st.markdown("### 🎯 Price Targets")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Buy Target (S1)", f"${support:.4f}",
                         delta=f"{((support - current_price) / current_price * 100):.2f}%")
            with col2:
                st.metric("Current", f"${current_price:.4f}")
            with col3:
                st.metric("Sell Target (R1)", f"${resistance:.4f}",
                         delta=f"{((resistance - current_price) / current_price * 100):.2f}%")
            
            st.markdown("---")
            
            # INVESTMENT BREAKDOWN
            st.markdown("### 🧮 Investment Breakdown")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📦 Shares", f"{calc['shares']:.2f}")
            
            with col2:
                if calc['risk_percentage'] <= 5:
                    st.success(f"## {calc['risk_percentage']:.1f}%")
                elif calc['risk_percentage'] <= 10:
                    st.warning(f"## {calc['risk_percentage']:.1f}%")
                else:
                    st.error(f"## {calc['risk_percentage']:.1f}%")
                st.markdown("**Risk %**")
                st.caption(f"${calc['total_risk']:.2f} at risk")
            
            with col3:
                st.metric("💰 Net Profit", f"${calc['net_profit']:.2f}")
                st.caption(f"After {slippage_pct}% slippage")
            
            with col4:
                rr = calc['net_profit'] / calc['total_risk'] if calc['total_risk'] > 0 else 0
                if rr >= 2:
                    st.success(f"**{rr:.2f}:1**")
                elif rr >= 1:
                    st.warning(f"**{rr:.2f}:1**")
                else:
                    st.error(f"**{rr:.2f}:1**")
                st.caption("R/R Ratio")
            
            st.markdown("---")
            
            # CONVICTION SCORE
            st.markdown("### 🎲 Conviction Score")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if probability >= 70:
                    st.success(f"# {probability:.1f}%")
                    st.markdown("### 🟢 High")
                elif probability >= 50:
                    st.warning(f"# {probability:.1f}%")
                    st.markdown("### 🟡 Medium")
                else:
                    st.error(f"# {probability:.1f}%")
                    st.markdown("### 🔴 Low")
            
            with col2:
                st.metric("Z-Score", f"{z_score:.2f}σ")
            
            with col3:
                st.metric("Volatility", f"{volatility*100:.2f}%")
        else:
            st.warning("⚠️ Calculator data unavailable. Historical data may be missing.")
    
    # TAB 2: ANALYSIS
    with tab2:
        data_5d = st.session_state.get('data_5d')
        data_1y = st.session_state.get('data_1y')
        
        if data_5d is not None:
            st.markdown("### 📈 Price Chart with VWAP & Technical Indicators")
            st.caption("Green dotted line = VWAP (Volume Weighted Average Price)")
            fig_price = create_price_chart_with_sma(data_5d, data_1y, ticker)
            st.plotly_chart(fig_price, use_container_width=True)
            
            st.markdown("### 📊 RSI Momentum Indicator")
            fig_rsi = create_rsi_chart(data_5d, ticker)
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Technical Summary
            st.markdown("### 📋 Technical Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            ema_8 = st.session_state.get('ema_8')
            ema_20 = st.session_state.get('ema_20')
            rsi = st.session_state.get('rsi')
            
            with col1:
                st.metric("EMA 8", f"${ema_8:.4f}")
            with col2:
                st.metric("EMA 20", f"${ema_20:.4f}")
            with col3:
                st.metric("RSI (14)", f"{rsi:.2f}")
            with col4:
                obv = data_5d['OBV'].iloc[-20:]
                trend = "📈 Bullish" if obv.iloc[-1] > obv.iloc[0] else "📉 Bearish"
                st.metric("Volume", trend)
            
            # VWAP Analysis
            st.markdown("### 💹 VWAP Analysis (Day Trading)")
            vwap_latest = data_5d['VWAP'].iloc[-1]
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("VWAP", f"${vwap_latest:.4f}")
            with col2:
                if current_price > vwap_latest:
                    st.success("✅ Price above VWAP (Bullish)")
                else:
                    st.error("⚠️ Price below VWAP (Bearish)")
            
            # SMA Analysis
            if data_1y is not None and not data_1y.empty:
                st.markdown("### 📊 Long-Term Moving Averages")
                col1, col2 = st.columns(2)
                
                with col1:
                    sma_50 = data_1y['SMA_50'].iloc[-1]
                    if not pd.isna(sma_50):
                        st.metric("SMA 50", f"${sma_50:.4f}")
                        if current_price > sma_50:
                            st.success("✅ Price above SMA 50 (Bullish)")
                        else:
                            st.error("⚠️ Price below SMA 50 (Bearish)")
                
                with col2:
                    sma_200 = data_1y['SMA_200'].iloc[-1]
                    if not pd.isna(sma_200):
                        st.metric("SMA 200", f"${sma_200:.4f}")
                        if current_price > sma_200:
                            st.success("✅ Price above SMA 200 (Bullish)")
                        else:
                            st.error("⚠️ Price below SMA 200 (Bearish)")
        else:
            st.warning("⚠️ Historical data unavailable for analysis")
    
    # TAB 3: NEWS
    with tab3:
        st.markdown("### 📰 Latest Market News")
        
        news_items = st.session_state.get('news_items', [])
        news_error = st.session_state.get('news_error')
        
        if news_error:
            st.warning(f"⚠️ {news_error}")
        elif news_items:
            for i, news in enumerate(news_items, 1):
                with st.container():
                    st.markdown(f"#### {i}. {news.get('headline', 'No headline')}")
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        summary = news.get('summary', 'No summary available')
                        if len(summary) > 200:
                            summary = summary[:200] + "..."
                        st.markdown(summary)
                    
                    with col2:
                        news_url = news.get('url', '#')
                        news_date = datetime.fromtimestamp(news.get('datetime', 0)).strftime('%Y-%m-%d %H:%M')
                        st.caption(f"📅 {news_date}")
                        st.link_button("Read More", news_url, use_container_width=True)
                    
                    st.markdown("---")
        else:
            st.info("📭 No recent news available for this ticker")

elif 'ticker' not in st.session_state:
    st.info("👈 **Configure your settings and click Analyze**")
    
    st.markdown("### 🚀 Day Trading Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Calculator**
        - Real-time pricing
        - Investment breakdown
        - Risk analysis
        - Conviction scoring
        - Persistent results
        """)
    
    with col2:
        st.markdown("""
        **📈 Analysis**
        - VWAP indicator
        - RVOL (Relative Volume)
        - EMA 8 & 20
        - SMA 50 & 200
        - RSI momentum
        """)
    
    with col3:
        st.markdown("""
        **📰 News**
        - Latest headlines
        - Article summaries
        - Direct links
        - Real-time updates
        """)
