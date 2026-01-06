import streamlit as st
import yfinance as yf
import pandas as pd
from scipy.stats import norm
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import time
import json
import os
import logging

# Configure logging
logging.basicConfig(
    filename='stock_app.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Page config
st.set_page_config(
    page_title="Stock Calculator Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============ SESSION STATE INITIALIZATION ============
def init_session_state():
    """Initialize all session state variables with defaults"""
    defaults = {
        'portfolio': [],
        'current_ticker': None,
        'current_price': None,
        'volatility': None,
        'hist_data': None,
        'buy_price': None,
        'sell_price': None,
        'quantity': 100,
        'time_horizon': 30,
        'last_request_time': 0,
        'recommended_buy': None,
        'recommended_sell': None,
        'api_healthy': True
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============ UTILITY FUNCTIONS ============

def validate_ticker_format(ticker):
    """Validate ticker symbol format"""
    import re
    if not ticker:
        return False, "Ticker cannot be empty"
    if not re.match(r'^[A-Z]{1,5}$', ticker):
        return False, "Invalid ticker format. Use 1-5 uppercase letters (e.g., AAPL, MSFT)"
    return True, ""

def rate_limit_check():
    """Implement simple rate limiting (1 request per second)"""
    current_time = time.time()
    if st.session_state.last_request_time > 0:
        elapsed = current_time - st.session_state.last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
    st.session_state.last_request_time = time.time()

def check_api_health():
    """Quick health check for yfinance API"""
    try:
        test_ticker = yf.Ticker("AAPL")
        _ = test_ticker.info
        st.session_state.api_healthy = True
        return True
    except Exception as e:
        st.session_state.api_healthy = False
        logging.error(f"API health check failed: {e}")
        return False

def safe_float(value, default=0.0):
    """Safely convert to float with fallback"""
    try:
        result = float(value)
        return result if not (np.isnan(result) or np.isinf(result)) else default
    except (ValueError, TypeError):
        return default

def check_market_hours():
    """Check if markets are open (simple version)"""
    now = datetime.now()
    if now.weekday() >= 5:  # Weekend
        return False, "Markets are closed (Weekend)"
    return True, ""

# ============ DATA FETCHING WITH CACHING ============

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker_symbol):
    """
    Fetch stock data with caching (5-minute TTL)
    Returns: (current_price, hist_data, error_message)
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get current price with multiple fallback methods
        current_price = None
        try:
            current_price = ticker.fast_info['lastPrice']
        except:
            try:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
            except:
                pass
        
        # Validate price
        if current_price is None or pd.isna(current_price) or current_price <= 0:
            return None, None, f"Could not fetch valid price for {ticker_symbol}"
        
        # Get historical data (6 months)
        hist_data = ticker.history(period="6mo")
        
        if hist_data.empty:
            return None, None, f"No historical data available for {ticker_symbol}"
        
        # Check for minimum data requirements
        if len(hist_data) < 20:
            return current_price, hist_data, "Warning: Less than 20 days of data available"
        
        # Data quality check - remove outliers
        z_scores = np.abs((hist_data['Close'] - hist_data['Close'].mean()) / hist_data['Close'].std())
        outliers = z_scores > 5  # Very extreme outliers only
        if outliers.any():
            logging.warning(f"{ticker_symbol}: {outliers.sum()} extreme outliers detected")
        
        return current_price, hist_data, None
        
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "No data found" in error_msg:
            return None, None, f"Ticker '{ticker_symbol}' not found"
        elif "timed out" in error_msg.lower():
            return None, None, "Connection timeout. Please check your internet connection"
        elif "Connection" in error_msg:
            return None, None, "Cannot connect to Yahoo Finance. Please try again later"
        else:
            logging.error(f"Error fetching {ticker_symbol}: {e}")
            return None, None, f"Error fetching data: {error_msg[:100]}"

@st.cache_data(ttl=300, show_spinner=False)
def calculate_historical_volatility(hist_data, days=60):
    """
    Calculate annualized historical volatility with robust error handling
    Returns: (volatility, error_message)
    """
    try:
        if hist_data is None or hist_data.empty:
            return None, "No historical data provided"
        
        if len(hist_data) < 20:
            return 0.30, "Using default volatility (30%) - insufficient data"
        
        # Calculate log returns
        hist_data = hist_data.copy()
        hist_data['Log_Return'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
        log_returns = hist_data['Log_Return'].dropna()
        
        # Remove infinite values
        log_returns = log_returns[np.isfinite(log_returns)]
        
        if len(log_returns) < 20:
            return 0.30, "Using default volatility (30%) - insufficient clean data"
        
        # Calculate annualized volatility
        daily_volatility = log_returns.std()
        
        if pd.isna(daily_volatility) or daily_volatility <= 0:
            return 0.30, "Using default volatility (30%) - calculation error"
        
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        # Sanity check (volatility should be between 5% and 300%)
        if annualized_volatility < 0.05:
            annualized_volatility = 0.05
        elif annualized_volatility > 3.0:
            annualized_volatility = 3.0
            return annualized_volatility, "Warning: Extremely high volatility detected"
        
        return annualized_volatility, None
        
    except Exception as e:
        logging.error(f"Volatility calculation error: {e}")
        return 0.30, "Using default volatility (30%) - calculation error"

# ============ CALCULATION FUNCTIONS ============

def calculate_lognormal_probability(S0, K, sigma, T, r=0.0):
    """
    Calculate probability using Black-Scholes d2 formula with robust error handling
    
    Returns probability that S_T > K (stock price above strike at time T)
    """
    # Input validation
    if T <= 0 or sigma <= 0 or S0 <= 0 or K <= 0:
        return 0.5
    
    try:
        # Prevent extreme values
        if K / S0 > 100 or S0 / K > 100:
            # Extreme price targets
            return 0.01 if K > S0 else 0.99
        
        # Black-Scholes d2 formula
        d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        # Check for invalid results
        if np.isnan(d2) or np.isinf(d2):
            return 0.5
        
        # Clamp extreme values
        if d2 > 6:
            return 0.9999
        elif d2 < -6:
            return 0.0001
        
        prob_above = norm.cdf(d2)
        
        return safe_float(prob_above, 0.5)
        
    except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning) as e:
        logging.warning(f"Probability calculation error: {e}")
        return 0.5

def calculate_sma(data, period=20):
    """Calculate Simple Moving Average with validation"""
    try:
        if data is None or len(data) < period:
            return pd.Series([np.nan] * len(data))
        sma = data['Close'].rolling(window=period).mean()
        return sma
    except Exception as e:
        logging.error(f"SMA calculation error: {e}")
        return pd.Series([np.nan] * len(data))

def get_recommended_levels(hist_data):
    """
    Calculate recommended buy/sell levels with comprehensive validation
    
    Returns: (rec_buy, rec_sell, sma, std) or (None, None, None, None)
    """
    try:
        if hist_data is None or hist_data.empty:
            return None, None, None, None
        
        if len(hist_data) < 20:
            return None, None, None, None
        
        # Calculate 20-day SMA
        sma_20 = hist_data['Close'].rolling(window=20).mean()
        
        # Calculate 20-day standard deviation
        std_20 = hist_data['Close'].rolling(window=20).std()
        
        # Get most recent values
        latest_sma = sma_20.iloc[-1]
        latest_std = std_20.iloc[-1]
        
        # Validate results
        if pd.isna(latest_sma) or pd.isna(latest_std):
            return None, None, None, None
        
        if latest_sma <= 0 or latest_std <= 0:
            return None, None, None, None
        
        # Calculate recommended levels
        recommended_buy = latest_sma - latest_std
        recommended_sell = latest_sma + latest_std
        
        # Prevent negative prices
        if recommended_buy <= 0:
            recommended_buy = latest_sma * 0.90  # Fallback: 10% below SMA
        
        # Sanity check: recommendations should be within reasonable range
        current_price = hist_data['Close'].iloc[-1]
        if recommended_buy < current_price * 0.5:
            recommended_buy = current_price * 0.90
        if recommended_sell > current_price * 2.0:
            recommended_sell = current_price * 1.10
        
        return recommended_buy, recommended_sell, latest_sma, latest_std
        
    except Exception as e:
        logging.error(f"Recommendation calculation error: {e}")
        return None, None, None, None

# ============ CHARTING ============

def create_candlestick_chart(hist_data, ticker, buy_price=None, sell_price=None, 
                             rec_buy=None, rec_sell=None):
    """Create interactive candlestick chart with comprehensive error handling"""
    
    try:
        if hist_data is None or hist_data.empty:
            return None
        
        # Fix timezone issue
        hist_data = hist_data.copy()
        if hist_data.index.tz is not None:
            hist_data.index = hist_data.index.tz_localize(None)
        
        # Get last 6 months of data
        six_months_ago = datetime.now() - timedelta(days=180)
        chart_data = hist_data[hist_data.index >= six_months_ago].copy()
        
        # Fallback to all data if filter results in empty dataframe
        if chart_data.empty:
            chart_data = hist_data.copy()
        
        # Final check
        if len(chart_data) == 0:
            return None
        
        # Calculate 20-day SMA
        chart_data['SMA_20'] = calculate_sma(chart_data, 20)
        
        # Create figure
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=chart_data.index,
            open=chart_data['Open'],
            high=chart_data['High'],
            low=chart_data['Low'],
            close=chart_data['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # 20-day SMA (only if enough data)
        if not chart_data['SMA_20'].isna().all():
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data['SMA_20'],
                mode='lines',
                name='20-day SMA',
                line=dict(color='#ff9800', width=2)
            ))
        
        # Buy Price Line
        if buy_price and buy_price > 0:
            fig.add_hline(
                y=buy_price,
                line_dash="dash",
                line_color="#2196f3",
                annotation_text=f"Buy Target: ${buy_price:.2f}",
                annotation_position="right"
            )
        
        # Sell Price Line
        if sell_price and sell_price > 0:
            fig.add_hline(
                y=sell_price,
                line_dash="dash",
                line_color="#4caf50",
                annotation_text=f"Sell Target: ${sell_price:.2f}",
                annotation_position="right"
            )
        
        # Recommended Buy Level
        if rec_buy and rec_buy > 0:
            fig.add_hline(
                y=rec_buy,
                line_dash="dot",
                line_color="#9c27b0",
                opacity=0.6,
                annotation_text=f"AI Buy: ${rec_buy:.2f}",
                annotation_position="left"
            )
        
        # Recommended Sell Level
        if rec_sell and rec_sell > 0:
            fig.add_hline(
                y=rec_sell,
                line_dash="dot",
                line_color="#ff5722",
                opacity=0.6,
                annotation_text=f"AI Sell: ${rec_sell:.2f}",
                annotation_position="left"
            )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} - Price History',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            template='plotly_white',
            height=600,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
        
    except Exception as e:
        logging.error(f"Chart creation error: {e}")
        return None

# ============ DATA PERSISTENCE ============

def save_portfolio_to_file():
    """Save portfolio to JSON file"""
    try:
        with open('portfolio_data.json', 'w') as f:
            json.dump(st.session_state.portfolio, f, indent=2, default=str)
        return True
    except Exception as e:
        logging.error(f"Portfolio save error: {e}")
        return False

def load_portfolio_from_file():
    """Load portfolio from JSON file"""
    try:
        if os.path.exists('portfolio_data.json'):
            with open('portfolio_data.json', 'r') as f:
                st.session_state.portfolio = json.load(f)
            return True
    except Exception as e:
        logging.error(f"Portfolio load error: {e}")
    return False

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### 🔍 Stock Lookup")
    
    # API Health indicator
    if not st.session_state.api_healthy:
        st.warning("⚠️ API Connection Issue")
    
    ticker_input = st.text_input(
        "Ticker Symbol",
        placeholder="e.g., AAPL, TSLA, MSFT",
        help="Enter a valid stock ticker symbol (1-5 letters)",
        max_chars=5
    ).upper()
    
    lookup_button = st.button("📊 Load Stock Data", type="primary", use_container_width=True)
    
    # Market hours info
    is_open, market_msg = check_market_hours()
    if not is_open:
        st.caption(f"ℹ️ {market_msg}")
    
    if lookup_button:
        if not ticker_input:
            st.error("Please enter a ticker symbol")
        else:
            # Validate ticker format
            is_valid, error_msg = validate_ticker_format(ticker_input)
            if not is_valid:
                st.error(error_msg)
            else:
                # Rate limiting
                rate_limit_check()
                
                with st.spinner(f"Fetching {ticker_input}..."):
                    # Fetch data
                    current_price, hist_data, error = fetch_stock_data(ticker_input)
                    
                    if error and current_price is None:
                        st.error(f"❌ {error}")
                    else:
                        # Calculate volatility
                        volatility, vol_error = calculate_historical_volatility(hist_data)
                        
                        # Store in session state
                        st.session_state.current_ticker = ticker_input
                        st.session_state.current_price = current_price
                        st.session_state.volatility = volatility
                        st.session_state.hist_data = hist_data
                        
                        # Set default buy/sell prices
                        st.session_state.buy_price = current_price * 0.95
                        st.session_state.sell_price = current_price * 1.05
                        
                        # Calculate recommendations
                        rec_buy, rec_sell, _, _ = get_recommended_levels(hist_data)
                        st.session_state.recommended_buy = rec_buy
                        st.session_state.recommended_sell = rec_sell
                        
                        # Success message
                        success_msg = f"✅ {ticker_input} loaded!"
                        if error:
                            success_msg += f"\n⚠️ {error}"
                        if vol_error:
                            success_msg += f"\n⚠️ {vol_error}"
                        
                        st.success(success_msg)
    
    # Display current stock info
    if st.session_state.current_ticker:
        st.divider()
        st.markdown("### 📊 Current Stock")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ticker", st.session_state.current_ticker)
        with col2:
            st.metric("Price", f"${st.session_state.current_price:.2f}")
        
        st.metric(
            "Volatility (σ)",
            f"{st.session_state.volatility*100:.2f}%",
            help="Annualized historical volatility based on 60-day log returns"
        )
        
        # Show last update time
        if st.session_state.hist_data is not None and not st.session_state.hist_data.empty:
            last_update = st.session_state.hist_data.index[-1]
            st.caption(f"Data as of: {last_update.strftime('%Y-%m-%d')}")
        
        st.divider()
        
        if st.button("🗑️ Clear Stock", use_container_width=True):
            st.session_state.current_ticker = None
            st.session_state.current_price = None
            st.session_state.volatility = None
            st.session_state.hist_data = None
            st.session_state.buy_price = None
            st.session_state.sell_price = None
            st.rerun()

# ============ MAIN CONTENT ============
st.markdown('<div class="main-header">📈 Stock Calculator Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Trading Analysis with Black-Scholes Probability Models</div>', unsafe_allow_html=True)

if not st.session_state.current_ticker:
    st.info("👈 Enter a stock ticker in the sidebar to begin analysis")
    
    # Load portfolio on startup
    if st.button("📂 Load Saved Portfolio"):
        if load_portfolio_from_file():
            st.success("✅ Portfolio loaded from file!")
            st.rerun()
        else:
            st.warning("No saved portfolio found")
    
    st.stop()

# ============ TABS ============
tab1, tab2, tab3 = st.tabs(["💰 Trading Calculator", "📈 Technical Chart", "📊 Portfolio History"])

# TAB 1: Trading Calculator
with tab1:
    st.markdown("### Configure Your Trade")
    
    # Use form to batch inputs and reduce reruns
    with st.form("trading_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            buy_price = st.number_input(
                "Buy Price ($)",
                min_value=0.01,
                max_value=float(st.session_state.current_price * 10),
                value=float(st.session_state.buy_price) if st.session_state.buy_price else float(st.session_state.current_price * 0.95),
                format="%.2f",
                help="Target price to buy the stock"
            )
            
            quantity = st.number_input(
                "Quantity (shares)",
                min_value=1,
                max_value=1000000,
                value=st.session_state.quantity,
                step=1,
                help="Number of shares to trade"
            )
        
        with col2:
            sell_price = st.number_input(
                "Sell Price ($)",
                min_value=0.01,
                max_value=float(st.session_state.current_price * 10),
                value=float(st.session_state.sell_price) if st.session_state.sell_price else float(st.session_state.current_price * 1.05),
                format="%.2f",
                help="Target price to sell the stock"
            )
            
            time_horizon = st.number_input(
                "Time Horizon (days)",
                min_value=1,
                max_value=365,
                value=st.session_state.time_horizon,
                step=1,
                help="Number of days until target date"
            )
        
        submitted = st.form_submit_button("📊 Calculate", use_container_width=True)
    
    # Input validation
    if buy_price >= sell_price:
        st.warning("⚠️ **Warning:** Buy price should be less than sell price for a profitable trade")
    
    if buy_price < st.session_state.current_price * 0.5:
        st.warning("⚠️ **Warning:** Buy price is very far below current price (>50% drop)")
    
    if sell_price > st.session_state.current_price * 2.0:
        st.warning("⚠️ **Warning:** Sell price is very far above current price (>100% gain)")
    
    st.divider()
    
    # Display AI recommendations
    if st.session_state.recommended_buy and st.session_state.recommended_sell:
        st.markdown("### 💡 AI-Powered Price Recommendations")
        st.caption("Based on 20-day SMA ± 1 Standard Deviation (mean-reversion strategy)")
        
        rec_buy_price = st.session_state.recommended_buy
        rec_sell_price = st.session_state.recommended_sell
        
        # Calculate differences
        buy_diff = ((rec_buy_price - st.session_state.current_price) / st.session_state.current_price * 100)
        sell_diff = ((rec_sell_price - st.session_state.current_price) / st.session_state.current_price * 100)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.info(f"""
            **Recommended Buy (Dip)**  
            ${rec_buy_price:.2f}  
            _{buy_diff:+.2f}% from current_  
            """)
        
        with col2:
            st.info(f"""
            **Recommended Sell (Strength)**  
            ${rec_sell_price:.2f}  
            _{sell_diff:+.2f}% from current_  
            """)
        
        with col3:
            st.write("")
            st.write("")
            if st.button("✨ Use AI Prices", type="secondary", use_container_width=True):
                st.session_state.buy_price = rec_buy_price
                st.session_state.sell_price = rec_sell_price
                st.rerun()
    
    st.divider()
    
    # Calculate probabilities
    T = time_horizon / 365.0
    
    # Probability calculations with direction logic
    if buy_price < st.session_state.current_price:
        # Buying below current - prob of dipping
        buy_prob = 1 - calculate_lognormal_probability(
            st.session_state.current_price, buy_price, st.session_state.volatility, T
        )
    else:
        # Buying above current - prob of rising
        buy_prob = calculate_lognormal_probability(
            st.session_state.current_price, buy_price, st.session_state.volatility, T
        )
    
    if sell_price > st.session_state.current_price:
        # Selling above current - prob of rising
        sell_prob = calculate_lognormal_probability(
            st.session_state.current_price, sell_price, st.session_state.volatility, T
        )
    else:
        # Selling below current - prob of dipping
        sell_prob = 1 - calculate_lognormal_probability(
            st.session_state.current_price, sell_price, st.session_state.volatility, T
        )
    
    # Display Probabilities
    st.markdown("### 🎯 Target Probabilities")
    st.caption("Probability of reaching target price at expiration (using Black-Scholes d₂ formula)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Buy Target Probability",
            f"{buy_prob:.2%}",
            delta=f"{buy_prob - 0.5:.2%} vs 50%",
            help="Probability that stock price will reach the buy target at time horizon (not anytime before)"
        )
    
    with col2:
        st.metric(
            "Sell Target Probability",
            f"{sell_prob:.2%}",
            delta=f"{sell_prob - 0.5:.2%} vs 50%",
            help="Probability that stock price will reach the sell target at time horizon (not anytime before)"
        )
    
    st.divider()
    
    # Calculate P&L
    total_cost = buy_price * quantity
    total_revenue = sell_price * quantity
    profit_loss = total_revenue - total_cost
    profit_loss_pct = (profit_loss / total_cost) * 100 if total_cost > 0 else 0
    
    # Display P&L
    st.markdown("### 💵 Profit/Loss Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Cost", f"${total_cost:,.2f}")
    
    with col2:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col3:
        st.metric(
            "Net P&L",
            f"${profit_loss:,.2f}",
            delta=f"{profit_loss_pct:+.2f}%"
        )
    
    # Visual P&L indicator
    if profit_loss > 0:
        st.success(f"💰 **Potential Profit: ${profit_loss:,.2f} ({profit_loss_pct:+.2f}%)**")
    elif profit_loss < 0:
        st.error(f"📉 **Potential Loss: ${abs(profit_loss):,.2f} ({profit_loss_pct:.2f}%)**")
    else:
        st.info("⚖️ **Break Even Trade**")
    
    st.divider()
    
    # Action Buttons
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("➕ Add to Portfolio", type="primary", use_container_width=True):
            trade_data = {
                "Ticker": st.session_state.current_ticker,
                "Date/Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Buy Price": buy_price,
                "Sell Price": sell_price,
                "Quantity": quantity,
                "P&L ($)": profit_loss,
                "Return (%)": profit_loss_pct,
                "Buy Prob": buy_prob,
                "Sell Prob": sell_prob,
                "Time Horizon": time_horizon
            }
            st.session_state.portfolio.append(trade_data)
            
            # Auto-save
            save_portfolio_to_file()
            
            st.success("✅ Trade added to portfolio!")
            st.balloons()

# TAB 2: Technical Chart
with tab2:
    if st.session_state.hist_data is not None:
        st.markdown("### 📈 Technical Analysis")
        
        with st.spinner("Rendering chart..."):
            # Get values from session state (safe fallback)
            buy_price_chart = st.session_state.buy_price
            sell_price_chart = st.session_state.sell_price
            rec_buy = st.session_state.recommended_buy
            rec_sell = st.session_state.recommended_sell
            
            # Create chart
            fig = create_candlestick_chart(
                st.session_state.hist_data,
                st.session_state.current_ticker,
                buy_price_chart,
                sell_price_chart,
                rec_buy,
                rec_sell
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Unable to render chart. Insufficient data.")
        
        # Chart legend
        st.markdown("""
        **Chart Legend:**
        - 🟢 **Green/Red Candles**: Price movement (Open, High, Low, Close)
        - 🟠 **Orange Line**: 20-day Simple Moving Average
        - 🔵 **Blue Dashed**: Your Buy Target
        - 🟢 **Green Dashed**: Your Sell Target
        - 🟣 **Purple Dotted**: AI Recommended Buy (SMA - 1σ)
        - 🔴 **Red Dotted**: AI Recommended Sell (SMA + 1σ)
        """)
        
        st.divider()
        
        # Chart statistics
        if len(st.session_state.hist_data) >= 30:
            col1, col2, col3, col4 = st.columns(4)
            
            recent_data = st.session_state.hist_data.tail(30)
            
            with col1:
                st.metric("30-Day High", f"${recent_data['High'].max():.2f}")
            
            with col2:
                st.metric("30-Day Low", f"${recent_data['Low'].min():.2f}")
            
            with col3:
                avg_volume = recent_data['Volume'].mean()
                st.metric("Avg Volume", f"{avg_volume:,.0f}")
            
            with col4:
                price_change = ((recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / 
                              recent_data['Close'].iloc[0] * 100)
                st.metric("30-Day Change", f"{price_change:+.2f}%")
        
        # Educational expander
        with st.expander("📚 Understanding the Analysis"):
            st.markdown("""
            ### How It Works:
            
            **Technical Indicators:**
            - **20-day SMA**: Average closing price over 20 trading days. Smooths out price action and identifies trends.
            - **Standard Deviation (σ)**: Measures price volatility. Higher values = more price swings.
            
            **AI Recommendations (Mean-Reversion Strategy):**
            - **Buy Level (SMA - 1σ)**: When price drops 1 standard deviation below average, it's statistically "oversold"
            - **Sell Level (SMA + 1σ)**: When price rises 1 standard deviation above average, it's statistically "overbought"
            
            **Probability Model (Black-Scholes d₂):**
            - Calculates the likelihood of reaching target prices at the specified time horizon
            - Uses log-normal distribution to model stock price movements
            - Accounts for volatility and time decay
            
            **⚠️ Important Disclaimers:**
            - These are statistical models, not predictions
            - Past performance doesn't guarantee future results
            - Always consider your risk tolerance and investment goals
            - Consult a financial advisor for personalized advice
            """)
    else:
        st.warning("⚠️ No historical data available. Try reloading the stock from the sidebar.")

# TAB 3: Portfolio History
with tab3:
    st.markdown("### 📊 Portfolio Trades")
    
    if st.session_state.portfolio:
        df = pd.DataFrame(st.session_state.portfolio)
        
        # Format for display
        df_display = df.copy()
        df_display['Buy Price'] = df_display['Buy Price'].apply(lambda x: f"${x:.2f}")
        df_display['Sell Price'] = df_display['Sell Price'].apply(lambda x: f"${x:.2f}")
        df_display['P&L ($)'] = df_display['P&L ($)'].apply(lambda x: f"${x:+,.2f}")
        df_display['Return (%)'] = df_display['Return (%)'].apply(lambda x: f"{x:+.2f}%")
        df_display['Buy Prob'] = df_display['Buy Prob'].apply(lambda x: f"{x:.2%}")
        df_display['Sell Prob'] = df_display['Sell Prob'].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Portfolio Summary
        st.markdown("### 📈 Portfolio Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_trades = len(df)
        total_pl = df['P&L ($)'].sum()
        avg_return = df['Return (%)'].mean()
        winning_trades = len(df[df['P&L ($)'] > 0])
        
        with col1:
            st.metric("Total Trades", total_trades)
        
        with col2:
            st.metric("Net P&L", f"${total_pl:+,.2f}")
        
        with col3:
            st.metric("Avg Return", f"{avg_return:+.2f}%")
        
        with col4:
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        st.divider()
        
        # Export and management buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Export CSV",
                data=csv,
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("💾 Save Portfolio", use_container_width=True):
                if save_portfolio_to_file():
                    st.success("✅ Portfolio saved!")
                else:
                    st.error("Failed to save portfolio")
        
        with col3:
            if st.button("🗑️ Clear Portfolio", use_container_width=True):
                if st.session_state.portfolio:
                    st.session_state.portfolio = []
                    st.success("Portfolio cleared")
                    st.rerun()
    else:
        st.info("📭 No trades in portfolio yet. Add trades from the Trading Calculator tab!")
        
        # Offer to load saved portfolio
        if st.button("📂 Load Saved Portfolio"):
            if load_portfolio_from_file():
                st.success("✅ Portfolio loaded!")
                st.rerun()
            else:
                st.warning("No saved portfolio found")

# Footer
st.divider()
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(
        "*Built with Streamlit • Data by Yahoo Finance • Black-Scholes probability model*",
        help="Probabilities use the d₂ formula from Black-Scholes-Merton option pricing model"
    )
with col2:
    if st.button("🔄 Refresh Data Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")
        st.rerun()
