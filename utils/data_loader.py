import pandas as pd
import numpy as np
import time
import streamlit as st
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from utils.config import CFG, parse_ts, validate_keys # Pulling from your first file

# Try to handle different Alpaca versions for DataFeed
try:
    from alpaca.data.enums import DataFeed
    HAS_DATAFEED = True
except ImportError:
    HAS_DATAFEED = False

@st.cache_data(ttl=21600, show_spinner="Fetching market data...")
def load_historical(ticker: str, api_key: str, secret_key: str, days_back: int = 1200) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Fetches historical daily bars from Alpaca."""
    dbg = {"ticker": ticker, "steps": []}
    if not validate_keys(api_key, secret_key) or not ticker:
        return None, {"error": "Invalid keys or ticker"}

    client = StockHistoricalDataClient(api_key, secret_key)
    request_params = {
        "symbol_or_symbols": [ticker.upper()],
        "timeframe": TimeFrame.Day,
        "start": datetime.utcnow() - timedelta(days=days_back),
        "end": datetime.utcnow(),
    }
    
    # Apply IEX feed for free-tier users if possible
    if HAS_DATAFEED:
        request_params["feed"] = DataFeed.IEX

    try:
        bars = client.get_stock_bars(StockBarsRequest(**request_params))
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        return df, {"status": "success"}
    except Exception as e:
        return None, {"error": str(e)}

def sanity_check_bars(df: pd.DataFrame) -> list[str]:
    """Checks for data gaps or price errors."""
    warns = []
    if df is None or df.empty:
        return ["No data found."]
    if (df['close'] <= 0).any():
        warns.append("Found zero or negative prices.")
    return warns
