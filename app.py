"""
Stock Calculator Pro - Professional Trading Dashboard
A Streamlit application for statistical trade analysis using Black-Scholes probability models
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import plotly.graph_objects as go
import json
import os
from typing import Tuple, Optional

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Stock Calculator Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables with safe defaults"""
    
    default_state = {
        # Stock Data
        'current_ticker': None,
        'current_price': None,
        'hist_data': None,
        'volatility': None,
        
        # User Inputs
        'buy_price': None,
        'sell_price': None,
        'quantity': 100,
        'time_horizon': 30,
        
        # AI Recommendations
        'rec_buy': None,
        'rec_sell': None,
        'rec_sma': None,
        'rec_std': None,
        
        # Portfolio
        'portfolio': [],
        
        # Flags
        'data_loaded': False,
        'use_ai_prices': False
    }
    
    for key, default_value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# ============================================================================
# DATA FETCHING & CACHING
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str) -> Tuple[Optional[float], Optional[pd.DataFrame], Optional[str]]:
    """
    Fetch stock data with comprehensive error handling
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Tuple of (current_price, historical_data, error_message)
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Fetch current price
        try:
            current_price = stock.fast_info['lastPrice']
        except:
            hist_1d = stock.history(period="1d")
            if hist_1d.empty:
                return None, None, f"No data found for ticker '{ticker}'"
            current_price = hist_1d['Close'].iloc[-1]
        
        # Validate price
        if pd.isna(current_price) or current_price <= 0:
            return None, None, "Invalid price data received"
        
        # Fetch 6 months of historical data
        hist_data = stock.history(period="6mo")
        
        if hist_data.empty:
            return None, None, "No historical data available"
        
        # CRITICAL: Fix timezone issues before any operations
        hist_data.index = hist_data.index.tz_localize(None)
        
        # Data quality check
        if len(hist_data) < 20:
            return current_price, hist_data, "Warning: Less than 20 days of data available"
        
        return current_price, hist_data, None
        
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "No data found" in error_str:
            return None, None, f"Ticker '{ticker}' not found"
        elif "timed out" in error_str.lower():
            return None, None, "Connection timeout - check internet"
        else:
            return None, None, f"Error: {error_str[:100]}"

# ============================================================================
# FINANCIAL CALCULATIONS
# ============================================================================

def calculate_annualized_volatility(hist_data: pd.DataFrame, days: int = 60) -> Tuple[float, Optional[str]]:
    """
    Calculate annualized historical volatility using log returns
    
    Args:
        hist_data: DataFrame with historical price data
        days: Number of days to use for calculation
        
    Returns:
        Tuple of (annualized_volatility, error_message)
    """
    try:
        if hist_data is None or len(hist_data) < 20:
            return 0.30, "Using default 30% volatility (insufficient data)"
        
        # Take last 'days' of data
        recent_data = hist_data.tail(min(days, len(hist_data)))
        
        # Calculate log returns
        log_returns = np.log(recent_data['Close'] / recent_data['Close'].shift(1))
        log_returns = log_returns.dropna()
        
        # Remove infinite values
        log_returns = log_returns[np.isfinite(log_returns)]
        
        if len(log_returns) < 20:
            return 0.30, "Using default 30% volatility (insufficient clean data)"
        
        # Calculate daily volatility
        daily_vol = log_returns.std()
        
        if pd.isna(daily_vol) or daily_vol <= 0:
            return 0.30, "Using default 30% volatility (calculation error)"
        
        # Annualize (252 trading days per year)
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sanity check: volatility should be between 5% and 300%
        annual_vol = np.clip(annual_vol, 0.05, 3.0)
        
        return annual_vol, None
        
    except Exception as e:
        return 0.30, f"Using default 30% volatility (error: {str(e)[:50]})"


def calculate_black_scholes_probability(S0: float, K: float, sigma: float, T: float, r: float = 0.0) -> float:
    """
    Calculate probability using Black-Scholes d2 formula
    
    Formula: d2 = [ln(S0/K) + (r - 0.5*σ²)T] / (σ√T)
    P(S_T > K) = N(d2)
    
    Args:
        S0: Current stock price
        K: Target price (strike)
        sigma: Annualized volatility
        T: Time to expiration (in years)
        r: Risk-free rate (default 0%)
        
    Returns:
        Probability that stock price will be above K at time T
    """
    # Input validation
    if any(x <= 0 for x in [S0, K, sigma, T]):
        return 0.5
    
    try:
        # Handle extreme scenarios
        if K / S0 > 100 or S0 / K > 100:
            return 0.01 if K > S0 else 0.99
        
        # Black-Scholes d2 calculation
        d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        # Validate result
        if not np.isfinite(d2):
            return 0.5
        
        # Calculate probability
        probability = norm.cdf(d2)
        
        return float(np.clip(probability, 0.0001, 0.9999))
        
    except Exception:
        return 0.5


def calculate_mean_reversion_levels(hist_data: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Calculate AI-recommended buy/sell levels using mean-reversion strategy
    
    Strategy:
    - Buy recommendation: 20-day SMA - 1 Standard Deviation (buy the dip)
    - Sell recommendation: 20-day SMA + 1 Standard Deviation (sell the rally)
    
    Args:
        hist_data: DataFrame with historical price data
        
    Returns:
        Tuple of (rec_buy, rec_sell, sma_20, std_20)
    """
    try:
        if hist_data is None or len(hist_data) < 20:
            return None, None, None, None
        
        # Calculate 20-day SMA
        sma_20 = hist_data['Close'].rolling(window=20).mean().iloc[-1]
        
        # Calculate 20-day rolling standard deviation
        std_20 = hist_data['Close'].rolling(window=20).std().iloc[-1]
        
        # Validate calculations
        if pd.isna(sma_20) or pd.isna(std_20) or sma_20 <= 0 or std_20 <= 0:
            return None, None, None, None
        
        # Calculate recommendations
        rec_buy = sma_20 - std_20
        rec_sell = sma_20 + std_20
        
        # Prevent negative buy recommendations
        if rec_buy <= 0:
            rec_buy = sma_20 * 0.90  # Fallback: 10% below SMA
        
        # Get current price for sanity checks
        current_price = hist_data['Close'].iloc[-1]
        
        # Ensure recommendations are within reasonable bounds (50% to 200% of current)
        rec_buy = max(rec_buy, current_price * 0.50)
        rec_sell = min(rec_sell, current_price * 2.00)
        
        return rec_buy, rec_sell, sma_20, std_20
        
    except Exception:
        return None, None, None, None


def calculate_sma(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average"""
    if data is None or len(data) < period:
        return pd.Series([np.nan] * len(data), index=data.index)
    return data['Close'].rolling(window=period).mean()

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_professional_chart(
    hist_data: pd.DataFrame,
    ticker: str,
    buy_target: Optional[float] = None,
    sell_target: Optional[float] = None,
    rec_buy: Optional[float] = None,
    rec_sell: Optional[float] = None
) -> Optional[go.Figure]:
    """
    Create professional candlestick chart with all overlays
    
    Args:
        hist_data: Historical price data
        ticker: Stock ticker symbol
        buy_target: User's buy target price
        sell_target: User's sell target price
        rec_buy: AI recommended buy price
        rec_sell: AI recommended sell price
        
    Returns:
        Plotly figure object or None if error
    """
    try:
        if hist_data is None or hist_data.empty:
            return None
        
        # Get 6-month window
        six_months_ago = datetime.now() - timedelta(days=180)
        chart_data = hist_data[hist_data.index >= six_months_ago].copy()
        
        if chart_data.empty:
            chart_data = hist_data.copy()
        
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
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350'
        ))
        
        # 20-day SMA
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['SMA_20'],
            mode='lines',
            name='20-day SMA',
            line=dict(color='#ff9800', width=2.5),
            hovertemplate='SMA: $%{y:.2f}<extra></extra>'
        ))
        
        # User's Buy Target (Blue Dashed)
        if buy_target and buy_target > 0:
            fig.add_hline(
                y=buy_target,
                line_dash="dash",
                line_color="#2196f3",
                line_width=2,
                annotation_text=f"Your Buy: ${buy_target:.2f}",
                annotation_position="right",
                annotation_font_size=11,
                annotation_font_color="#2196f3"
            )
        
        # User's Sell Target (Green Dashed)
        if sell_target and sell_target > 0:
            fig.add_hline(
                y=sell_target,
                line_dash="dash",
                line_color="#4caf50",
                line_width=2,
                annotation_text=f"Your Sell: ${sell_target:.2f}",
                annotation_position="right",
                annotation_font_size=11,
                annotation_font_color="#4caf50"
            )
        
        # AI Recommended Buy (Purple Dotted)
        if rec_buy and rec_buy > 0:
            fig.add_hline(
                y=rec_buy,
                line_dash="dot",
                line_color="#9c27b0",
                line_width=2,
                opacity=0.7,
                annotation_text=f"AI Buy: ${rec_buy:.2f}",
                annotation_position="left",
                annotation_font_size=11,
                annotation_font_color="#9c27b0"
            )
        
        # AI Recommended Sell (Red Dotted)
        if rec_sell and rec_sell > 0:
            fig.add_hline(
                y=rec_sell,
                line_dash="dot",
                line_color="#ff5722",
                line_width=2,
                opacity=0.7,
                annotation_text=f"AI Sell: ${rec_sell:.2f}",
                annotation_position="left",
                annotation_font_size=11,
                annotation_font_color="#ff5722"
            )
        
        # Layout
        fig.update_layout(
            title={
                'text': f'{ticker} - Technical Analysis (6-Month View)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            yaxis_title='Price (USD)',
            xaxis_title='Date',
            template='plotly_white',
            height=650,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(size=12)
        )
        
        # Grid styling
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
        
        return fig
        
    except Exception:
        return None

# ============================================================================
# PORTFOLIO PERSISTENCE
# ============================================================================

PORTFOLIO_FILE = "portfolio_data.json"

def save_portfolio():
    """Save portfolio to JSON file"""
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(st.session_state.portfolio, f, indent=2, default=str)
        return True
    except Exception:
        return False

def load_portfolio():
    """Load portfolio from JSON file"""
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, 'r') as f:
                st.session_state.portfolio = json.load(f)
            return True
    except Exception:
        pass
    return False

# ============================================================================
# SIDEBAR - STOCK LOOKUP
# ============================================================================

with st.sidebar:
    st.markdown("## 🔍 Stock Lookup")
    
    # Ticker input
    ticker_input = st.text_input(
        "Enter Ticker Symbol",
        placeholder="e.g., AAPL, MSFT, TSLA",
        max_chars=5,
        help="Enter a valid stock ticker (1-5 uppercase letters)"
    ).strip().upper()
    
    # Load button
    if st.button("📊 Load Stock Data", type="primary", use_container_width=True):
        if not ticker_input:
            st.error("Please enter a ticker symbol")
        elif not ticker_input.isalpha() or len(ticker_input) > 5:
            st.error("Invalid ticker format")
        else:
            with st.spinner(f"Loading {ticker_input}..."):
                # Fetch data
                price, hist_data, error = fetch_stock_data(ticker_input)
                
                if error and price is None:
                    st.error(f"❌ {error}")
                else:
                    # Calculate volatility
                    volatility, vol_warning = calculate_annualized_volatility(hist_data)
                    
                    # Calculate recommendations
                    rec_buy, rec_sell, rec_sma, rec_std = calculate_mean_reversion_levels(hist_data)
                    
                    # Update session state
                    st.session_state.current_ticker = ticker_input
                    st.session_state.current_price = price
                    st.session_state.hist_data = hist_data
                    st.session_state.volatility = volatility
                    st.session_state.rec_buy = rec_buy
                    st.session_state.rec_sell = rec_sell
                    st.session_state.rec_sma = rec_sma
                    st.session_state.rec_std = rec_std
                    st.session_state.data_loaded = True
                    
                    # Set default buy/sell prices
                    st.session_state.buy_price = price * 0.95
                    st.session_state.sell_price = price * 1.05
                    
                    # Success message
                    msg = f"✅ {ticker_input} loaded successfully!"
                    if error:
                        msg += f"\n⚠️ {error}"
                    if vol_warning:
                        msg += f"\n⚠️ {vol_warning}"
                    st.success(msg)
    
    # Display current stock info
    if st.session_state.data_loaded and st.session_state.current_ticker:
        st.divider()
        st.markdown("### 📊 Current Stock")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ticker", st.session_state.current_ticker)
        with col2:
            st.metric("Price", f"${st.session_state.current_price:.2f}")
        
        st.metric(
            "Volatility (σ)",
            f"{st.session_state.volatility*100:.1f}%",
            help="Annualized historical volatility (60-day)"
        )
        
        # Data timestamp
        if st.session_state.hist_data is not None:
            last_date = st.session_state.hist_data.index[-1]
            st.caption(f"📅 Data as of: {last_date.strftime('%Y-%m-%d')}")
        
        st.divider()
        
        # Clear button
        if st.button("🗑️ Clear Stock", use_container_width=True):
            st.session_state.data_loaded = False
            st.session_state.current_ticker = None
            st.session_state.current_price = None
            st.session_state.hist_data = None
            st.rerun()
    
    # Portfolio quick actions
    st.divider()
    st.markdown("### 💼 Portfolio")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save", use_container_width=True):
            if save_portfolio():
                st.success("Saved!")
            else:
                st.error("Save failed")
    
    with col2:
        if st.button("📂 Load", use_container_width=True):
            if load_portfolio():
                st.success("Loaded!")
                st.rerun()
            else:
                st.warning("No file found")

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<div class="main-header">📈 Stock Calculator Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Professional Trading Dashboard with Black-Scholes Probability Models</div>', unsafe_allow_html=True)

# Check if data is loaded
if not st.session_state.data_loaded:
    st.info("👈 **Get Started:** Enter a stock ticker in the sidebar to begin your analysis")
    st.stop()

# ============================================================================
# TABS: Trading Calculator | Technical Chart | Portfolio History
# ============================================================================

tab1, tab2, tab3 = st.tabs(["💰 Trading Calculator", "📈 Technical Chart", "📊 Portfolio History"])

# ============================================================================
# TAB 1: TRADING CALCULATOR
# ============================================================================

with tab1:
    st.markdown("### 🎯 Configure Your Trade")
    
    # Trading inputs in a form
    with st.form("trading_inputs"):
        col1, col2 = st.columns(2)
        
        with col1:
            buy_price = st.number_input(
                "Buy Price ($)",
                min_value=0.01,
                value=float(st.session_state.buy_price),
                format="%.2f",
                help="Your target entry price"
            )
            
            quantity = st.number_input(
                "Quantity (shares)",
                min_value=1,
                value=st.session_state.quantity,
                step=1,
                help="Number of shares to trade"
            )
        
        with col2:
            sell_price = st.number_input(
                "Sell Price ($)",
                min_value=0.01,
                value=float(st.session_state.sell_price),
                format="%.2f",
                help="Your target exit price"
            )
            
            time_horizon = st.number_input(
                "Time Horizon (days)",
                min_value=1,
                max_value=365,
                value=st.session_state.time_horizon,
                step=1,
                help="Days until target date"
            )
        
        calculate_btn = st.form_submit_button("🔢 Calculate Probabilities", use_container_width=True)
    
    # Validation warnings
    if buy_price >= sell_price:
        st.warning("⚠️ Buy price should be less than sell price for profit")
    
    st.divider()
    
    # AI Recommendations Section
    if st.session_state.rec_buy and st.session_state.rec_sell:
        st.markdown("### 🤖 AI-Powered Recommendations")
        st.caption("Mean-reversion strategy: 20-day SMA ± 1 Standard Deviation")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        rec_buy = st.session_state.rec_buy
        rec_sell = st.session_state.rec_sell
        current_price = st.session_state.current_price
        
        buy_diff = ((rec_buy - current_price) / current_price) * 100
        sell_diff = ((rec_sell - current_price) / current_price) * 100
        
        with col1:
            st.info(f"""
            **🎯 Recommended Buy**  
            **${rec_buy:.2f}**  
            _{buy_diff:+.1f}% from current_
            """)
        
        with col2:
            st.info(f"""
            **🎯 Recommended Sell**  
            **${rec_sell:.2f}**  
            _{sell_diff:+.1f}% from current_
            """)
        
        with col3:
            st.write("")
            st.write("")
            if st.button("✨ Use AI Prices", type="secondary", use_container_width=True):
                st.session_state.buy_price = rec_buy
                st.session_state.sell_price = rec_sell
                st.rerun()
    
    st.divider()
    
    # Probability Calculations
    st.markdown("### 📊 Probability Analysis")
    st.caption("Using Black-Scholes d₂ formula to calculate target probabilities")
    
    T = time_horizon / 365.0  # Convert to years
    
    # Buy probability
    if buy_price < st.session_state.current_price:
        buy_prob = 1 - calculate_black_scholes_probability(
            st.session_state.current_price, buy_price, st.session_state.volatility, T
        )
    else:
        buy_prob = calculate_black_scholes_probability(
            st.session_state.current_price, buy_price, st.session_state.volatility, T
        )
    
    # Sell probability
    if sell_price > st.session_state.current_price:
        sell_prob = calculate_black_scholes_probability(
            st.session_state.current_price, sell_price, st.session_state.volatility, T
        )
    else:
        sell_prob = 1 - calculate_black_scholes_probability(
            st.session_state.current_price, sell_price, st.session_state.volatility, T
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Buy Target Probability",
            f"{buy_prob:.1%}",
            delta=f"{buy_prob - 0.5:+.1%} vs 50%",
            help="Probability of reaching buy price at expiration"
        )
    
    with col2:
        st.metric(
            "Sell Target Probability",
            f"{sell_prob:.1%}",
            delta=f"{sell_prob - 0.5:+.1%} vs 50%",
            help="Probability of reaching sell price at expiration"
        )
    
    st.divider()
    
    # Profit/Loss Analysis
    st.markdown("### 💵 Profit/Loss Analysis")
    
    total_cost = buy_price * quantity
    total_revenue = sell_price * quantity
    profit_loss = total_revenue - total_cost
    profit_loss_pct = (profit_loss / total_cost) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Investment", f"${total_cost:,.2f}")
    
    with col2:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col3:
        st.metric(
            "Net P&L",
            f"${profit_loss:,.2f}",
            delta=f"{profit_loss_pct:+.1f}%"
        )
    
    # Visual P&L indicator
    if profit_loss > 0:
        st.success(f"💰 **Expected Profit: ${profit_loss:,.2f} ({profit_loss_pct:+.2f}%)**")
    elif profit_loss < 0:
        st.error(f"📉 **Expected Loss: ${abs(profit_loss):,.2f} ({profit_loss_pct:.2f}%)**")
    else:
        st.info("⚖️ **Break Even Trade**")
    
    st.divider()
    
    # Add to Portfolio Button
    if st.button("➕ Add to Portfolio", type="primary", use_container_width=True):
        trade = {
            "Ticker": st.session_state.current_ticker,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Buy Price": buy_price,
            "Sell Price": sell_price,
            "Quantity": quantity,
            "P&L": profit_loss,
            "Return": profit_loss_pct,
            "Buy Prob": buy_prob,
            "Sell Prob": sell_prob,
            "Time Horizon": time_horizon
        }
        
        st.session_state.portfolio.append(trade)
        save_portfolio()  # Auto-save
        
        st.success("✅ Trade added to portfolio!")
        st.balloons()

# ============================================================================
# TAB 2: TECHNICAL CHART
# ============================================================================

with tab2:
    st.markdown("### 📈 Technical Chart Analysis")
    
    with st.spinner("Rendering chart..."):
        fig = create_professional_chart(
            st.session_state.hist_data,
            st.session_state.current_ticker,
            st.session_state.buy_price,
            st.session_state.sell_price,
            st.session_state.rec_buy,
            st.session_state.rec_sell
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Unable to render chart")
    
    # Chart Legend
    st.markdown("""
    **📌 Chart Legend:**
    - 🟢🔴 **Candlesticks**: Daily OHLC (Open, High, Low, Close) prices
    - 🟠 **Orange Line**: 20-day Simple Moving Average (trend indicator)
    - 🔵 **Blue Dashed**: Your manual buy target
    - 🟢 **Green Dashed**: Your manual sell target
    - 🟣 **Purple Dotted**: AI recommended buy level (SMA - 1σ)
    - 🔴 **Red Dotted**: AI recommended sell level (SMA + 1σ)
    """)
    
    st.divider()
    
    # Statistical Summary
    if len(st.session_state.hist_data) >= 30:
        st.markdown("### 📊 30-Day Statistics")
        
        recent_data = st.session_state.hist_data.tail(30)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("30-Day High", f"${recent_data['High'].max():.2f}")
        
        with col2:
            st.metric("30-Day Low", f"${recent_data['Low'].min():.2f}")
        
        with col3:
            avg_volume = recent_data['Volume'].mean()
            st.metric("Avg Daily Volume", f"{avg_volume:,.0f}")
        
        with col4:
            price_change = ((recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / 
                          recent_data['Close'].iloc[0] * 100)
            st.metric("30-Day Return", f"{price_change:+.2f}%")
    
    # Educational Expander
    with st.expander("📚 Understanding the Analysis"):
        st.markdown("""
        ### Technical Analysis Explained
        
        **Moving Averages:**
        - **20-day SMA** smooths out price action and identifies the prevailing trend
        - Price above SMA = bullish signal | Price below SMA = bearish signal
        
        **Mean-Reversion Strategy:**
        - **AI Buy Level (SMA - 1σ)**: When price drops 1 standard deviation below average, 
          it's statistically "oversold" and may bounce back
        - **AI Sell Level (SMA + 1σ)**: When price rises 1 standard deviation above average, 
          it's statistically "overbought" and may pull back
        
        **Black-Scholes Probability Model:**
        - Uses log-normal distribution to model future stock prices
        - Calculates the probability of reaching target prices at expiration
        - Accounts for volatility (σ) and time decay
        
        **⚠️ Important Disclaimers:**
        - These are statistical models based on historical data
        - Past performance does not guarantee future results
        - Always consider your personal risk tolerance
        - Consult a licensed financial advisor for investment decisions
        """)

# ============================================================================
# TAB 3: PORTFOLIO HISTORY
# ============================================================================

with tab3:
    st.markdown("### 📊 Portfolio Trades")
    
    if st.session_state.portfolio:
        df = pd.DataFrame(st.session_state.portfolio)
        
        # Format display
        df_display = df.copy()
        df_display['Buy Price'] = df_display['Buy Price'].apply(lambda x: f"${x:.2f}")
        df_display['Sell Price'] = df_display['Sell Price'].apply(lambda x: f"${x:.2f}")
        df_display['P&L'] = df_display['P&L'].apply(lambda x: f"${x:+,.2f}")
        df_display['Return'] = df_display['Return'].apply(lambda x: f"{x:+.2f}%")
        df_display['Buy Prob'] = df_display['Buy Prob'].apply(lambda x: f"{x:.1%}")
        df_display['Sell Prob'] = df_display['Sell Prob'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Portfolio Analytics
        st.markdown("### 📈 Portfolio Summary")
        
        total_trades = len(df)
        total_pl = df['P&L'].sum()
        avg_return = df['Return'].mean()
        winning_trades = len(df[df['P&L'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", total_trades)
        
        with col2:
            st.metric("Net P&L", f"${total_pl:+,.2f}")
        
        with col3:
            st.metric("Avg Return", f"{avg_return:+.2f}%")
        
        with col4:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        st.divider()
        
        # Export and Management
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Export CSV",
                data=csv,
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("💾 Save to File", use_container_width=True):
                if save_portfolio():
                    st.success("Portfolio saved!")
                else:
                    st.error("Save failed")
        
        with col3:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.portfolio = []
                save_portfolio()
                st.success("Portfolio cleared")
                st.rerun()
    
    else:
        st.info("📭 **No trades yet.** Add trades from the Trading Calculator tab!")
        
        if st.button("📂 Load Saved Portfolio"):
            if load_portfolio():
                st.success("Portfolio loaded!")
                st.rerun()
            else:
                st.warning("No saved portfolio found")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

col1, col2 = st.columns([4, 1])

with col1:
    st.caption("""
    **Stock Calculator Pro** | Data by Yahoo Finance | Probabilities calculated using Black-Scholes d₂ formula  
    ⚠️ For educational purposes only. Not financial advice.
    """)

with col2:
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")
        st.rerun()
