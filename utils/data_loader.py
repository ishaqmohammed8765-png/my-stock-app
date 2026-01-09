import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Try to handle different Alpaca versions for DataFeed
try:
    from alpaca.data.enums import DataFeed  # type: ignore
    HAS_DATAFEED = True
except Exception:
    HAS_DATAFEED = False


def validate_keys(api_key: str, secret_key: str) -> bool:
    """Simple local validation to avoid circular imports."""
    return bool(api_key and secret_key and str(api_key).strip() and str(secret_key).strip())


def _normalize_bars_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Alpaca bars often come back as a MultiIndex df with (symbol, timestamp) index.
    This normalizes into a flat df with columns:
      timestamp, open, high, low, close, volume, trade_count, vwap (if present)
    """
    if df is None or df.empty:
        return df

    # If MultiIndex -> reset to columns
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Common Alpaca column names after reset_index:
    # symbol, timestamp, open, high, low, close, volume, trade_count, vwap
    # Some versions: "time" instead of "timestamp"
    if "time" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"time": "timestamp"})

    # If still missing timestamp, try to find any datetime-ish column
    if "timestamp" not in df.columns:
        for c in df.columns:
            if "time" in c.lower() or "date" in c.lower():
                df = df.rename(columns={c: "timestamp"})
                break

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
    else:
        # If we truly can't find a timestamp column, just return as-is
        return df

    # Ensure standard OHLCV columns exist if present under weird casing
    rename_map = {}
    for col in ["open", "high", "low", "close", "volume", "trade_count", "vwap", "symbol"]:
        if col not in df.columns:
            # try case-insensitive match
            matches = [c for c in df.columns if c.lower() == col]
            if matches:
                rename_map[matches[0]] = col
    if rename_map:
        df = df.rename(columns=rename_map)

    # Filter to ticker (sometimes multiple symbols can sneak in)
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper() == ticker.upper()]

    # Sort, drop duplicates
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    # Keep numeric columns numeric
    for c in ["open", "high", "low", "close", "volume", "trade_count", "vwap"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing core OHLC
    core = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if core:
        df = df.dropna(subset=core)

    df = df.reset_index(drop=True)
    return df


@st.cache_data(ttl=21600, show_spinner="Fetching market data...")
def load_historical(
    ticker: str,
    api_key: str,
    secret_key: str,
    days_back: int = 1200,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Fetch historical daily bars from Alpaca."""
    ticker = (ticker or "").upper().strip()
    dbg: Dict[str, Any] = {"ticker": ticker, "status": "init"}

    if not ticker:
        dbg["status"] = "error"
        dbg["error"] = "Empty ticker"
        return None, dbg

    if not validate_keys(api_key, secret_key):
        dbg["status"] = "error"
        dbg["error"] = "Invalid/missing keys"
        return None, dbg

    start = datetime.utcnow() - timedelta(days=int(days_back))
    end = datetime.utcnow()

    request_params: Dict[str, Any] = {
        "symbol_or_symbols": [ticker],
        "timeframe": TimeFrame.Day,
        "start": start,
        "end": end,
    }

    # Use IEX feed for free-tier users if supported by installed SDK
    if HAS_DATAFEED:
        try:
            request_params["feed"] = DataFeed.IEX
            dbg["feed"] = "IEX"
        except Exception:
            dbg["feed"] = "default"
    else:
        dbg["feed"] = "default"

    t0 = time.time()
    try:
        client = StockHistoricalDataClient(api_key, secret_key)
        bars = client.get_stock_bars(StockBarsRequest(**request_params))
        raw_df = bars.df
        df = _normalize_bars_df(raw_df, ticker)

        dbg["status"] = "success"
        dbg["rows"] = int(len(df)) if df is not None else 0
        if df is not None and not df.empty and "timestamp" in df.columns:
            dbg["from"] = str(df["timestamp"].min())
            dbg["to"] = str(df["timestamp"].max())

        dbg["elapsed_sec"] = round(time.time() - t0, 3)
        return df, dbg

    except Exception as e:
        dbg["status"] = "error"
        dbg["error"] = str(e)
        dbg["elapsed_sec"] = round(time.time() - t0, 3)
        return None, dbg


def sanity_check_bars(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns a dict that app.py can display:
      {"ok": bool, "warnings": [...], "stats": {...}}
    """
    out: Dict[str, Any] = {"ok": True, "warnings": [], "stats": {}}

    if df is None or df.empty:
        out["ok"] = False
        out["warnings"].append("No data returned.")
        return out

    # Required columns check
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        out["ok"] = False
        out["warnings"].append(f"Missing required columns: {missing}")
        return out

    # Basic validity
    if (df["close"] <= 0).any():
        out["ok"] = False
        out["warnings"].append("Found zero or negative close prices.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        out["warnings"].append("Found non-positive OHLC values (may indicate bad data).")

    # Timestamp monotonicity / duplicates
    if not df["timestamp"].is_monotonic_increasing:
        out["warnings"].append("Timestamps not strictly increasing (sorting applied or source issue).")
    dupes = int(df["timestamp"].duplicated().sum())
    if dupes > 0:
        out["warnings"].append(f"Duplicate timestamps found: {dupes}")

    # Simple gap check (daily bars): look for > 7 day gaps (to avoid flagging weekends/holidays too aggressively)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if len(ts) >= 2:
        gaps = ts.diff().dt.days.fillna(0)
        big_gaps = int((gaps > 7).sum())
        if big_gaps > 0:
            out["warnings"].append(f"Detected {big_gaps} large time gaps (>7 days).")

    # Stats
    out["stats"] = {
        "rows": int(len(df)),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
    }

    if out["warnings"]:
        # ok can still be True if warnings are mild, but here keep ok True unless we explicitly set False above
        pass

    return out
