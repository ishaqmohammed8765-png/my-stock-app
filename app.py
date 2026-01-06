import streamlit as st
import yfinance as yf
import pandas as pd
from scipy.stats import norm
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Stock Calculator Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'current_price' not in st.session_state:
    st.session_state.current_price = None
if 'volatility' not in st.session_state:
    st.session_state.volatility = None
if 'hist_data' not in st.session_state:
    st.session_state.hist_data = None

def calculate_historical_volatility(ticker_symbol, days=60):
    """Calculate annualized historical volatility using log returns"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)
        
        ticker = yf.Ticker(ticker_symbol)
        hist_data = ticker.history(start=start_date, end=end_date)
        
        if hist_data.empty or len(hist_data) < 20:
            return None, None, "Insufficient historical data"
        
        # Calculate log returns
        hist_data['Log_Return'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
        log_returns = hist_data['Log_Return'].dropna()
        
        if len(log_returns) < 20:
            return None, None, "Insufficient data for volatility calculation"
        
        # Annualized volatility (252 trading days)
        daily_volatility = log_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return annualized_volatility, hist_data, None
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def calculate_lognormal_probability(S0, K, sigma, T, r=0.0):
    """
    Calculate probability using Black-Scholes d2 formula
    
    P(S_T > K) where S_T is the stock price at time T
    
    Using the risk-neutral framework:
    d2 = [ln(S0/K) + (r - 0.5*σ²)T] / (σ√T)
    
    P(S_T > K) = N(d2)
    
    Parameters:
    - S0: Current stock price
    - K: Target price (strike)
    - sigma: Annualized volatility
    - T: Time to expiration (in years)
    - r: Risk-free rate (default 0 for simplicity)
    """
    if T <= 0 or sigma <= 0 or S0 <= 0 or K <= 0:
        return 0.5
    
    try:
        # Black-Scholes d2 formula
        d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        # Probability that stock price > K at time T
        prob_above = norm.cdf(d2)
        
        return prob_above
    except:
        return 0.5

def calculate_sma(data, period=20):
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=period).mean()

def create_candlestick_chart(hist_data, ticker, buy_price=None, sell_price=None):
    """Create interactive candlestick chart with overlays"""
    
    # Get last 6 months of data
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
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # 20-day SMA
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data['SMA_20'],
        mode='lines',
        name='20-day SMA',
        line=dict(color='#ff9800', width=2)
    ))
    
    # Buy Price Line
    if buy_price:
        fig.add_hline(
            y=buy_price,
            line_dash="dash",
            line_color="#2196f3",
            annotation_text=f"Buy Target: ${buy_price:.2f}",
            annotation_position="right"
        )
    
    # Sell Price Line
    if sell_price:
        fig.add_hline(
            y=sell_price,
            line_dash="dash",
            line_color="#4caf50",
            annotation_text=f"Sell Target: ${sell_price:.2f}",
            annotation_position="right"
        )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} - 6 Month Price History',
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

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### 🔍 Stock Lookup")
    
    ticker_input = st.text_input(
        "Ticker Symbol",
        placeholder="e.g., AAPL, TSLA, MSFT",
        help="Enter a valid stock ticker symbol"
    ).upper()
    
    lookup_button = st.button("📊 Load Stock Data", type="primary", use_container_width=True)
    
    if lookup_button and ticker_input:
        with st.spinner(f"Fetching {ticker_input}..."):
            try:
                ticker = yf.Ticker(ticker_input)
                
                # Get current price
                try:
                    current_price = ticker.fast_info['lastPrice']
                except:
                    hist = ticker.history(period="1d")
                    if hist.empty:
                        st.error("❌ No price data available")
                        st.stop()
                    current_price = hist['Close'].iloc[-1]
                
                if pd.isna(current_price) or current_price <= 0:
                    st.error("❌ Invalid price data")
                    st.stop()
                
                # Calculate volatility
                volatility, hist_data, error = calculate_historical_volatility(ticker_input)
                
                if error:
                    st.warning(error)
                    volatility = 0.3  # Default 30%
                    # Still try to get historical data
                    hist_data = ticker.history(period="6mo")
                
                # Store in session state
                st.session_state.current_ticker = ticker_input
                st.session_state.current_price = current_price
                st.session_state.volatility = volatility
                st.session_state.hist_data = hist_data
                
                st.success(f"✅ {ticker_input} loaded!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
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
            help="Annualized historical volatility"
        )
        
        st.divider()
        
        if st.button("🗑️ Clear Stock", use_container_width=True):
            st.session_state.current_ticker = None
            st.session_state.current_price = None
            st.session_state.volatility = None
            st.session_state.hist_data = None
            st.rerun()

# ============ MAIN CONTENT ============
st.markdown('<div class="main-header">📈 Stock Calculator Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Trading Analysis with Log-Normal Probability Models</div>', unsafe_allow_html=True)

if not st.session_state.current_ticker:
    st.info("👈 Enter a stock ticker in the sidebar to begin analysis")
    st.stop()

# ============ TABS ============
tab1, tab2, tab3 = st.tabs(["💰 Trading Calculator", "📈 Technical Chart", "📊 Portfolio History"])

# TAB 1: Trading Calculator
with tab1:
    st.markdown("### Configure Your Trade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        buy_price = st.number_input(
            "Buy Price ($)",
            min_value=0.01,
            value=float(st.session_state.current_price * 0.95),
            format="%.2f",
            help="Target price to buy the stock"
        )
        
        quantity = st.number_input(
            "Quantity (shares)",
            min_value=1,
            value=100,
            step=1,
            help="Number of shares to trade"
        )
    
    with col2:
        sell_price = st.number_input(
            "Sell Price ($)",
            min_value=0.01,
            value=float(st.session_state.current_price * 1.05),
            format="%.2f",
            help="Target price to sell the stock"
        )
        
        time_horizon = st.number_input(
            "Time Horizon (days)",
            min_value=1,
            value=30,
            step=1,
            help="Number of days until target date"
        )
    
    st.divider()
    
    # Calculate probabilities using Black-Scholes d2
    T = time_horizon / 365.0
    
    # Probability of reaching buy price
    if buy_price < st.session_state.current_price:
        # Prob of going DOWN to buy price
        buy_prob = 1 - calculate_lognormal_probability(
            st.session_state.current_price, buy_price, st.session_state.volatility, T
        )
    else:
        # Prob of going UP to buy price
        buy_prob = calculate_lognormal_probability(
            st.session_state.current_price, buy_price, st.session_state.volatility, T
        )
    
    # Probability of reaching sell price
    if sell_price > st.session_state.current_price:
        # Prob of going UP to sell price
        sell_prob = calculate_lognormal_probability(
            st.session_state.current_price, sell_price, st.session_state.volatility, T
        )
    else:
        # Prob of going DOWN to sell price
        sell_prob = 1 - calculate_lognormal_probability(
            st.session_state.current_price, sell_price, st.session_state.volatility, T
        )
    
    # Display Probabilities
    st.markdown("### 🎯 Target Probabilities")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Buy Target Probability",
            f"{buy_prob:.2%}",
            delta=f"{buy_prob - 0.5:.2%} vs 50%",
            help="Probability of reaching buy price within time horizon"
        )
    
    with col2:
        st.metric(
            "Sell Target Probability",
            f"{sell_prob:.2%}",
            delta=f"{sell_prob - 0.5:.2%} vs 50%",
            help="Probability of reaching sell price within time horizon"
        )
    
    st.divider()
    
    # Calculate P&L
    total_cost = buy_price * quantity
    total_revenue = sell_price * quantity
    profit_loss = total_revenue - total_cost
    profit_loss_pct = (profit_loss / total_cost) * 100
    
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
    col1, col2, col3 = st.columns([1, 1, 2])
    
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
                "Sell Prob": sell_prob
            }
            st.session_state.portfolio.append(trade_data)
            st.success("✅ Trade added!")
            st.balloons()

# TAB 2: Technical Chart
with tab2:
    if st.session_state.hist_data is not None:
        st.markdown("### 📈 Technical Analysis")
        
        # Get current values from calculator (if available)
        buy_price_chart = buy_price if 'buy_price' in locals() else None
        sell_price_chart = sell_price if 'sell_price' in locals() else None
        
        # Create and display chart
        fig = create_candlestick_chart(
            st.session_state.hist_data,
            st.session_state.current_ticker,
            buy_price_chart,
            sell_price_chart
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Chart statistics
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
    else:
        st.warning("⚠️ No historical data available. Try reloading the stock.")

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
        
        # Export buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
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
            if st.button("🗑️ Clear Portfolio", use_container_width=True):
                if st.session_state.portfolio:
                    st.session_state.portfolio = []
                    st.rerun()
    else:
        st.info("📭 No trades in portfolio yet. Add trades from the Trading Calculator tab!")

# Footer
st.divider()
st.markdown(
    "*Built with Streamlit • Market data by Yahoo Finance • Using Black-Scholes probability model*",
    help="Probabilities calculated using the d2 formula from Black-Scholes model"
)
