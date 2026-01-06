import streamlit as st
import yfinance as yf
import pandas as pd
from scipy.stats import norm
import numpy as np
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Stock Calculator Pro", page_icon="📈", layout="wide")

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'current_price' not in st.session_state:
    st.session_state.current_price = None
if 'volatility' not in st.session_state:
    st.session_state.volatility = None

def calculate_historical_volatility(ticker_symbol, days=60):
    """Calculate annualized historical volatility"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)
        
        ticker = yf.Ticker(ticker_symbol)
        hist_data = ticker.history(start=start_date, end=end_date)
        
        if hist_data.empty or len(hist_data) < 20:
            return None, "Insufficient historical data"
        
        hist_data['Log_Return'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
        log_returns = hist_data['Log_Return'].dropna()
        
        if len(log_returns) < 20:
            return None, "Insufficient data for volatility calculation"
        
        daily_volatility = log_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return annualized_volatility, None
    except Exception as e:
        return None, f"Error: {str(e)}"

def calculate_lognormal_probability(S0, ST, sigma, T, mu=0.0):
    """Calculate probability using Log-Normal distribution"""
    if T <= 0 or sigma <= 0 or S0 <= 0 or ST <= 0:
        return 0.5
    
    try:
        d = (np.log(ST / S0) - (mu - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        prob_above = 1 - norm.cdf(d)
        return prob_above
    except:
        return 0.5

# Title
st.markdown("# 📈 Stock Calculator Pro")
st.markdown("### Advanced Trading Analysis with Log-Normal Probability Models")
st.divider()

# Stock Lookup Section
st.markdown("## 🔍 Stock Lookup")
col1, col2 = st.columns([3, 1])

with col1:
    ticker_input = st.text_input("Stock Ticker Symbol", placeholder="e.g., AAPL, TSLA, MSFT").upper()

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    lookup_button = st.button("🔍 Lookup Stock", type="primary")

if lookup_button and ticker_input:
    with st.spinner(f"Fetching data for {ticker_input}..."):
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
            volatility, error = calculate_historical_volatility(ticker_input)
            
            if error:
                volatility = 0.3  # Default 30%
                vol_note = " (default)"
            else:
                vol_note = ""
            
            # Store in session state
            st.session_state.current_ticker = ticker_input
            st.session_state.current_price = current_price
            st.session_state.volatility = volatility
            
            st.success(f"✅ Successfully loaded {ticker_input}!")
            
        except Exception as e:
            st.error(f"❌ Error fetching stock data: {str(e)}")
            st.stop()

# Stock Information Display
if st.session_state.current_ticker:
    st.divider()
    st.markdown("## 📊 Stock Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ticker", st.session_state.current_ticker)
    
    with col2:
        st.metric("Current Price", f"${st.session_state.current_price:.2f}")
    
    with col3:
        vol_display = f"{st.session_state.volatility*100:.2f}%"
        st.metric("Ann. Volatility (σ)", vol_display)
    
    # Trading Calculator
    st.divider()
    st.markdown("## 💰 Trading Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        buy_price = st.number_input(
            "Buy Price ($)", 
            min_value=0.01, 
            value=float(st.session_state.current_price * 0.95),
            format="%.2f"
        )
        quantity = st.number_input("Quantity", min_value=1, value=100, step=1)
    
    with col2:
        sell_price = st.number_input(
            "Sell Price ($)", 
            min_value=0.01, 
            value=float(st.session_state.current_price * 1.05),
            format="%.2f"
        )
        time_horizon = st.number_input("Time Horizon (days)", min_value=1, value=30, step=1)
    
    # Calculate probabilities
    T = time_horizon / 365.0
    
    if buy_price < st.session_state.current_price:
        buy_prob = 1 - calculate_lognormal_probability(
            st.session_state.current_price, buy_price, st.session_state.volatility, T
        )
    else:
        buy_prob = calculate_lognormal_probability(
            st.session_state.current_price, buy_price, st.session_state.volatility, T
        )
    
    if sell_price > st.session_state.current_price:
        sell_prob = calculate_lognormal_probability(
            st.session_state.current_price, sell_price, st.session_state.volatility, T
        )
    else:
        sell_prob = 1 - calculate_lognormal_probability(
            st.session_state.current_price, sell_price, st.session_state.volatility, T
        )
    
    with col4:
        st.metric("Buy Target Probability", f"{buy_prob:.2%}")
        st.metric("Sell Target Probability", f"{sell_prob:.2%}")
    
    # Calculate P&L
    total_cost = buy_price * quantity
    total_revenue = sell_price * quantity
    profit_loss = total_revenue - total_cost
    profit_loss_pct = (profit_loss / total_cost) * 100
    
    # Display P&L
    st.markdown("### Profit/Loss Calculation")
    if profit_loss > 0:
        st.success(f"💰 **Profit: ${profit_loss:,.2f} ({profit_loss_pct:+.2f}%)**")
    elif profit_loss < 0:
        st.error(f"📉 **Loss: ${abs(profit_loss):,.2f} ({profit_loss_pct:.2f}%)**")
    else:
        st.info("⚖️ **Break Even**")
    
    # Buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("➕ Add to Portfolio", type="primary"):
            trade_data = {
                "Ticker": st.session_state.current_ticker,
                "Date/Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Buy Price": buy_price,
                "Sell Price": sell_price,
                "Quantity": quantity,
                "P&L ($)": profit_loss,
                "Return (%)": profit_loss_pct
            }
            st.session_state.portfolio.append(trade_data)
            st.success("✅ Trade added to portfolio!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Inputs"):
            st.session_state.current_ticker = None
            st.session_state.current_price = None
            st.session_state.volatility = None
            st.rerun()

# Portfolio Display
if st.session_state.portfolio:
    st.divider()
    st.markdown("## 📈 Portfolio History")
    
    df = pd.DataFrame(st.session_state.portfolio)
    
    # Format for display
    df_display = df.copy()
    df_display['Buy Price'] = df_display['Buy Price'].apply(lambda x: f"${x:.2f}")
    df_display['Sell Price'] = df_display['Sell Price'].apply(lambda x: f"${x:.2f}")
    df_display['P&L ($)'] = df_display['P&L ($)'].apply(lambda x: f"${x:+,.2f}")
    df_display['Return (%)'] = df_display['Return (%)'].apply(lambda x: f"{x:+.2f}%")
    
    st.dataframe(df_display, use_container_width=True)
    
    # Summary
    total_trades = len(df)
    total_pl = df['P&L ($)'].sum()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.metric("Total Trades", total_trades)
        st.metric("Net P&L", f"${total_pl:+,.2f}", delta=f"{(total_pl/abs(total_pl) if total_pl != 0 else 0):.0%}")
    
    with col2:
        if st.button("📥 Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🗑️ Clear Portfolio"):
            st.session_state.portfolio = []
            st.rerun()

# Footer
st.divider()
st.markdown("*Built with Streamlit • Data provided by Yahoo Finance*")
