import streamlit as st
import pandas as pd

# 1. IMPORT YOUR ENGINES
from utils.config import CFG, validate_keys
from utils.data_loader import load_historical, sanity_check_bars
from utils.indicators import add_indicators_inplace
from utils.backtester import backtest_strategy
from utils.live_stream import RealtimeStream

# 2. PAGE SETUP
st.set_page_config(page_title="Pro Algo Trader", layout="wide")
st.title("📈 Modular Algorithmic Dashboard")

# 3. SIDEBAR: CREDENTIALS & SETTINGS
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Alpaca API Key", type="default")
    sec_key = st.text_input("Alpaca Secret Key", type="password")
    symbol = st.text_input("Stock Symbol", value="AAPL").upper()
    
    st.divider()
    run_backtest = st.button("🚀 Run Backtest", use_container_width=True)

# 4. MAIN LOGIC
if validate_keys(api_key, sec_key):
    # Load Data using your utility
    df, debug_info = load_historical(symbol, api_key, sec_key)
    
    if df is not None and not df.empty:
        # Add Math using your utility
        add_indicators_inplace(df)
        
        # Display results in Tabs
        tab1, tab2 = st.tabs(["📊 Historical Analysis", "📡 Live Stream"])
        
        with tab1:
            st.subheader(f"Analysis for {symbol}")
            st.line_chart(df[['close', 'ma50', 'ma200']])
            
            if run_backtest:
                # Call your backtest engine
                results, trades = backtest_strategy(df, None, horizon=20, mode="limit", 
                                                   atr_entry=1.0, atr_stop=2.0, atr_target=3.0,
                                                   require_risk_on=True, rsi_min=30, rsi_max=70,
                                                   rvol_min=1.2, vol_max=1.0, cooldown_bars=5,
                                                   include_spread_penalty=True, assumed_spread_bps=5.0,
                                                   start_equity=100000)
                st.write("Backtest Complete!", results.tail())
        
        with tab2:
            st.info("Live data will appear here once the stream is started.")
            # Your live_stream.py logic would be initialized here
    else:
        st.error(f"Could not find data for {symbol}. Check your API keys or ticker.")
else:
    st.warning("Please enter valid Alpaca API keys in the sidebar to begin.")
