import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Page configuration
st.set_page_config(page_title="Trading Calculator", page_icon="⚡", layout="wide")

# Load thresholds from secrets (with fallback defaults)
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

st.title("⚡ High-Speed Trading Calculator")
st.markdown("Instant analysis with automated position sizing and profit calculations")

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
    """Calculate Pivot Point, Support (S1), and Resistance (R1) levels"""
    high = float(data['High'].iloc[-1])
    low = float(data['Low'].iloc[-1])
    close = current_price
    
    pivot = (high + low + close) / 3
    support = (2 * pivot) - high
    resistance = (2 * pivot) - low
    
    return pivot, support, resistance

def calculate_probability(data, current_price, target_price):
    """
    Calculate probability of hitting target using Z-score
    Returns: probability percentage (0-100%), z-score, volatility
    """
    # Get daily closes for volatility (last 20 days)
    daily_data = data['Close'].resample('D').last().dropna()
    
    if len(daily_data) < 20:
        daily_returns = daily_data.pct_change().dropna()
    else:
        daily_returns = daily_data.tail(20).pct_change().dropna()
    
    # Calculate daily volatility (standard deviation)
    volatility = daily_returns.std()
    
    # Calculate expected return to target
    expected_return = (target_price - current_price) / current_price
    
    # Calculate Z-score
    if volatility > 0:
        z_score = expected_return / volatility
    else:
        z_score = 0
    
    # Convert Z-score to probability
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
    
    # Cap probability at 95% for display purposes
    probability = min(probability, 95.0)
    
    return probability, z_score, volatility

def calculate_investment_risk(current_price, sell_target, investment_amount, stop_loss_pct=0.05):
    """
    Calculate shares, risk, and profit based on total investment
    """
    # Calculate shares you can buy with investment
    shares = investment_amount / current_price if current_price > 0 else 0
    
    # Calculate stop loss price (5% below entry)
    stop_loss_price = current_price * (1 - stop_loss_pct)
    
    # Calculate total risk in dollars
    risk_per_share = current_price - stop_loss_price
    total_risk = risk_per_share * shares
    
    # Calculate risk as percentage of investment
    risk_percentage = (total_risk / investment_amount * 100) if investment_amount > 0 else 0
    
    # Expected profit if target hits
    profit_per_share = sell_target - current_price
    expected_profit = profit_per_share * shares
    
    return {
        'shares': shares,
        'stop_loss_price': stop_loss_price,
        'total_risk': total_risk,
        'risk_percentage': risk_percentage,
        'expected_profit': expected_profit,
        'risk_per_share': risk_per_share
    }

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

@st.cache_data(ttl=300)
def fetch_stock_data(ticker):
    """Fetch stock data from Yahoo Finance with caching"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d", interval="15m")
        
        if data.empty:
            return None, None, "No data found for this ticker"
        
        # Get real-time price
        try:
            real_time_price = stock.fast_info['last_price']
        except:
            try:
                real_time_price = stock.info.get('regularMarketPrice', None)
            except:
                real_time_price = float(data['Close'].iloc[-1])
        
        # Calculate indicators
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
    
    # EMA 8 (Yellow) - Fast
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA_8'],
        name='EMA 8 (Fast)',
        line=dict(color='#FFD700', width=3)
    ))
    
    # EMA 20 (Blue) - Slow
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

# Sidebar
st.sidebar.header("⚙️ Quick Settings")

with st.sidebar.form("trading_form"):
    st.markdown("### Stock Symbol")
    ticker = st.text_input(
        "Ticker",
        value="NVDA",
        help="Examples: AAPL, TSLA, MSFT, MKDW, BTC-USD",
        placeholder="Enter ticker..."
    )
    
    st.markdown("### Investment Amount")
    investment_amount = st.number_input(
        "💵 Total Investment ($)",
        min_value=1.0,
        value=100.0,
        step=10.0,
        help="Total dollars you want to invest in this position"
    )
    
    submit_button = st.form_submit_button("⚡ Calculate", use_container_width=True, type="primary")
    
    st.caption("💡 Data cached for 5 minutes")

# Main content
if submit_button and ticker:
    with st.spinner(f"Calculating {ticker.upper()}..."):
        data, real_time_price, error = fetch_stock_data(ticker.upper())
    
    if error:
        st.error(error)
        if "rate limit" in error.lower():
            st.info("💡 **Tips:**\n"
                    "- Wait 2-5 minutes before retrying\n"
                    "- Data is cached for 5 minutes")
    elif data is not None:
        # Use real-time price with 4 decimal precision
        current_price = float(real_time_price)
        
        # Extract latest indicator values
        ema_8_latest = float(data['EMA_8'].iloc[-1])
        ema_20_latest = float(data['EMA_20'].iloc[-1])
        rsi_latest = float(data['RSI'].iloc[-1])
        
        # Calculate pivot points
        pivot, support, resistance = calculate_pivot_points(data, current_price)
        
        # Calculate probability for sell target
        probability, z_score, volatility = calculate_probability(data, current_price, resistance)
        
        # Calculate investment risk and profit
        calc = calculate_investment_risk(current_price, resistance, investment_amount)
        
        # Get trading verdict
        verdict, verdict_type = get_verdict(ema_8_latest, ema_20_latest, rsi_latest)
        
        # ============ BIG VERDICT BOX AT TOP ============
        st.markdown("## 🎯 TRADING SIGNAL")
        if verdict_type == "success":
            st.success(f"# {verdict}")
        elif verdict_type == "error":
            st.error(f"# {verdict}")
        else:
            st.info(f"# {verdict}")
        
        st.markdown("---")
        
        # ============ PRICE LEVELS (4 DECIMAL PRECISION) ============
        st.markdown("### 💰 Price Levels")
        st.caption("🔴 LIVE - Real-time with 4 decimal precision (perfect for MKDW)")
        
        col_price1, col_price2, col_price3 = st.columns(3)
        
        with col_price1:
            st.metric(
                label="🎯 Buy Target (S1)",
                value=f"${support:.4f}",
                delta=f"{((support - current_price) / current_price * 100):.2f}%"
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
                delta=f"{((resistance - current_price) / current_price * 100):.2f}%"
            )
        
        st.markdown("---")
        
        # ============ CONVICTION METER ============
        st.markdown("### 🎲 Conviction Score (Probability Gauge)")
        st.caption("Z-Score probability of hitting the Sell Target")
        
        col_conv1, col_conv2, col_conv3 = st.columns(3)
        
        with col_conv1:
            # Visual gauge with larger display
            if probability >= 70:
                st.success(f"# {probability:.1f}%")
                st.markdown("### 🟢 High Conviction")
            elif probability >= 50:
                st.warning(f"# {probability:.1f}%")
                st.markdown("### 🟡 Medium Conviction")
            else:
                st.error(f"# {probability:.1f}%")
                st.markdown("### 🔴 Low Conviction")
            st.caption("Probability of hitting target")
        
        with col_conv2:
            st.metric("Z-Score", f"{z_score:.2f}σ")
            st.caption("Standard deviations to target")
            if abs(z_score) < 1:
                st.info("Within 1σ - High probability")
            elif abs(z_score) < 2:
                st.info("Within 2σ - Medium probability")
            else:
                st.info("Beyond 2σ - Low probability")
        
        with col_conv3:
            st.metric("Daily Volatility", f"{volatility*100:.2f}%")
            st.caption("20-day standard deviation")
        
        st.caption("💡 High Conviction = Target within 1σ (68% probability) | Medium = within 2σ (47%) | Low = beyond 2σ")
        
        st.markdown("---")
        
        # ============ AUTO-CALCULATED POSITION ============
        st.markdown("### 🧮 Investment Breakdown")
        st.caption(f"Total Investment: ${investment_amount:.2f} | Stop Loss: 5% below entry (${calc['stop_loss_price']:.4f})")
        
        col_calc1, col_calc2, col_calc3, col_calc4 = st.columns(4)
        
        with col_calc1:
            st.metric(
                "📦 Shares to Buy",
                f"{calc['shares']:.2f}",
                help="Shares you can buy with your investment"
            )
            st.caption(f"@ ${current_price:.4f}/share")
        
        with col_calc2:
            # THE RISK THING - Show risk clearly
            if calc['risk_percentage'] <= 5:
                st.success(f"## {calc['risk_percentage']:.1f}%")
            elif calc['risk_percentage'] <= 10:
                st.warning(f"## {calc['risk_percentage']:.1f}%")
            else:
                st.error(f"## {calc['risk_percentage']:.1f}%")
            st.markdown("**Risk Percentage**")
            st.caption(f"${calc['total_risk']:.2f} at risk")
        
        with col_calc3:
            st.metric(
                "💰 Expected Profit",
                f"${calc['expected_profit']:.2f}",
                help="Profit if sell target hits"
            )
            gain_pct = (calc['expected_profit'] / investment_amount * 100) if investment_amount > 0 else 0
            st.caption(f"+{gain_pct:.1f}% gain")
        
        with col_calc4:
            rr_ratio = calc['expected_profit'] / calc['total_risk'] if calc['total_risk'] > 0 else 0
            if rr_ratio >= 2:
                st.success(f"**{rr_ratio:.2f}:1**")
                st.caption("✅ Good R/R")
            elif rr_ratio >= 1:
                st.warning(f"**{rr_ratio:.2f}:1**")
                st.caption("⚠️ Fair R/R")
            else:
                st.error(f"**{rr_ratio:.2f}:1**")
                st.caption("❌ Poor R/R")
        
        # Risk explanation box
        st.info(f"💡 **What this means:** If you invest ${investment_amount:.2f} and the price drops 5% to your stop loss (${calc['stop_loss_price']:.4f}), you'll lose ${calc['total_risk']:.2f} which is {calc['risk_percentage']:.1f}% of your investment.")
        
        st.markdown("---")
        
        # ============ TECHNICAL INDICATORS ============
        st.markdown("### 📊 Technical Indicators")
        
        col_ind1, col_ind2, col_ind3, col_ind4 = st.columns(4)
        
        with col_ind1:
            st.metric("EMA 8 (Fast)", f"${ema_8_latest:.4f}")
        with col_ind2:
            st.metric("EMA 20 (Slow)", f"${ema_20_latest:.4f}")
        with col_ind3:
            st.metric("RSI (14)", f"{rsi_latest:.2f}")
        with col_ind4:
            obv_recent = data['OBV'].iloc[-20:]
            volume_trend = "📈 Bullish" if obv_recent.iloc[-1] > obv_recent.iloc[0] else "📉 Bearish"
            st.metric("Volume (OBV)", volume_trend)
        
        st.markdown("---")
        
        # ============ CANDLESTICK CHART WITH EMA OVERLAYS ============
        st.markdown("### 📈 Professional Chart")
        st.caption("Candlestick with EMA 8 (Yellow) and EMA 20 (Blue)")
        
        fig = create_candlestick_chart(data, ticker)
        st.plotly_chart(fig, use_container_width=True)
        
        # ============ ADDITIONAL CHARTS ============
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### 📉 RSI Momentum")
            chart_rsi = pd.DataFrame({'RSI': data['RSI']})
            st.line_chart(chart_rsi)
            st.caption("Overbought >70 | Oversold <30")
        
        with col_chart2:
            st.markdown("#### 📊 Volume (OBV)")
            chart_obv = pd.DataFrame({'OBV': data['OBV']})
            st.line_chart(chart_obv)
            st.caption("Rising = Bullish | Falling = Bearish")
        
        st.success(f"⚡ Analysis complete • Live: ${current_price:.4f} • Cached for 5 min")

elif not submit_button:
    # Welcome screen
    st.info("👈 **Enter ticker and investment amount, then click Calculate**")
    
    st.markdown("### ⚡ High-Speed Features")
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.markdown("""
        **📊 Instant Analysis**
        - Real-time prices (4 decimals)
        - EMA 8 & 20 trend signals
        - RSI momentum indicator
        - OBV volume confirmation
        - Pivot point targets
        """)
    
    with col_feat2:
        st.markdown("""
        **🧮 Investment Calculator**
        - Enter total investment amount
        - Get shares to buy
        - See your total risk ($)
        - Risk as % of investment
        - Expected profit calculation
        """)
    
    st.markdown("### 🎲 Conviction Score")
    st.markdown("""
    Z-Score probability of hitting Sell Target:
    - **🟢 High**: ≥70% (target within 1 standard deviation)
    - **🟡 Medium**: 50-69% (target within 2 standard deviations)
    - **🔴 Low**: <50% (target beyond 2 standard deviations)
    
    The Conviction Score shows how likely the stock is to reach your sell target based on historical volatility.
    """)
    
    st.markdown("### 💰 How Investment Risk Works")
    st.markdown("""
    **Example:** You invest $100 in MKDW at $0.2680
    
    1. **Shares**: $100 / $0.2680 = 373 shares
    2. **Stop Loss**: 5% below entry = $0.2546
    3. **Risk per share**: $0.2680 - $0.2546 = $0.0134
    4. **Total Risk**: $0.0134 × 373 = $5.00
    5. **Risk %**: $5.00 / $100 = **5.0%** of your investment
    
    This means if the price drops 5% and hits your stop loss, you lose $5 (5% of your $100 investment).
    """)
    
    st.markdown("### 📱 Trading Signals")
    st.markdown("""
    - **🚀 STRONG BUY**: EMA bullish + RSI < 35
    - **✅ BUY**: EMA bullish + RSI 35-50
    - **⚠️ SELL**: EMA bearish + RSI 50-65
    - **🔻 STRONG SELL**: EMA bearish + RSI > 65
    - **⏸️ WAIT**: All other conditions
    """)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Quick Guide")
st.sidebar.info(
    "**How It Works:**\n"
    "1. Enter ticker symbol\n"
    "2. Set total investment\n"
    "3. Click Calculate\n"
    "4. See your risk & profit!\n\n"
    "**Indicators:**\n"
    "- EMA 8/20: Trend\n"
    "- RSI: Momentum\n"
    "- Z-Score: Probability\n"
    "- OBV: Volume\n\n"
    "**Investment Logic:**\n"
    "- Shares = Investment / Price\n"
    "- Risk = (Price - Stop) × Shares\n"
    "- Risk % = Risk / Investment\n"
    "- Stop Loss: 5% below entry\n\n"
    "Data: Yahoo Finance"
)
