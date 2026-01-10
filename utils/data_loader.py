from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
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


# ---------------------------
# Key validation (local, no circular imports)
# ---------------------------
def validate_keys(api_key: str, secret_key: str) -> bool:
    """
    Best-effort local validation.
    This does NOT prove entitlements; only an API call can.
    """
    if not api_key or not secret_key:
        return False
    k = str(api_key).strip()
    s = str(secret_key).strip()
    # Alpaca key id is often ~20 chars, secret is much longer, but keep soft
    if len(k) < 12 or len(s) < 20:
        return False
    return True


# ---------------------------
# Dataframe normalization
# ---------------------------
def _normalize_bars_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Alpaca bars often come back as a MultiIndex df with (symbol, timestamp) index.
    This normalizes into a flat df with columns:
      timestamp, open, high, low, close, volume, trade_count, vwap (if present), symbol (if present)
    """
    if df is None or df.empty:
        return df

    # If MultiIndex -> reset to columns
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Standardize column names to lowercase strings for easier matching
    df = df.rename(columns={c: str(c).lower() for c in df.columns})

    # Common variants
    if "time" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"date": "timestamp"})

    # If still missing timestamp, try to find any datetime-ish column
    if "timestamp" not in df.columns:
        for c in list(df.columns):
            if "time" in c.lower() or "date" in c.lower():
                df = df.rename(columns={c: "timestamp"})
                break

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
    else:
        # Can't normalize without a timestamp; return as-is (caller will handle)
        return df

    # Ensure symbol column exists if provided by Alpaca
    if "symbol" not in df.columns:
        # some versions may use 's'
        if "s" in df.columns:
            df["symbol"] = df["s"]

    # Filter to ticker (sometimes multiple symbols can sneak in)
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper() == ticker.upper()]

    # Coerce numeric columns
    for c in ["open", "high", "low", "close", "volume", "trade_count", "vwap"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing core OHLC
    core = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if core:
        df = df.dropna(subset=core)

    # Sort + dedupe timestamps
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    return df


def _fetch_bars_df(
    *,
    ticker: str,
    api_key: str,
    secret_key: str,
    request_params: Dict[str, Any],
) -> pd.DataFrame:
    client = StockHistoricalDataClient(api_key, secret_key)
    bars = client.get_stock_bars(StockBarsRequest(**request_params))
    return bars.df


# ---------------------------
# Cached loader
# ---------------------------
@st.cache_data(ttl=21600, show_spinner="Fetching market data...")
def load_historical(
    ticker: str,
    api_key: str,
    secret_key: str,
    days_back: int = 900,
    *,
    prefer_iex: bool = True,
    force_refresh: int = 0,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Fetch historical daily bars from Alpaca.

    Parameters
    ----------
    ticker : str
        Stock ticker (e.g., AAPL)
    api_key/secret_key : str
        Alpaca credentials
    days_back : int
        How many calendar days back to request (daily bars). 900 ~= ~3.5 years.
    prefer_iex : bool
        If True and DataFeed is supported by installed alpaca-py, try IEX first.
    force_refresh : int
        Cache-buster knob. Pass a changing int (e.g., int(time.time())) to bypass cache on demand.

    Returns
    -------
    (df, dbg)
        df: normalized dataframe or None
        dbg: debug metadata for UI display
    """
    _ = force_refresh  # used only for cache keying

    ticker = (ticker or "").upper().strip()
    dbg: Dict[str, Any] = {"ticker": ticker, "status": "init", "feed": "default"}

    if not ticker:
        dbg["status"] = "error"
        dbg["error"] = "Empty ticker"
        return None, dbg

    if not validate_keys(api_key, secret_key):
        dbg["status"] = "error"
        dbg["error"] = "Invalid/missing keys (format check)."
        return None, dbg

    # Use timezone-aware UTC datetimes
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=int(days_back))
    end = now

    base_params: Dict[str, Any] = {
        "symbol_or_symbols": [ticker],
        "timeframe": TimeFrame.Day,
        "start": start,
        "end": end,
    }

    # Decide whether to try IEX first
    try_iex = bool(prefer_iex and HAS_DATAFEED)

    t0 = time.time()

    try:
        # Attempt #1: IEX feed (if available)
        if try_iex:
            req1 = dict(base_params)
            try:
                req1["feed"] = DataFeed.IEX  # type: ignore[name-defined]
                dbg["feed"] = "IEX"
            except Exception:
                dbg["feed"] = "default"
                req1 = base_params

            raw_df = _fetch_bars_df(ticker=ticker, api_key=api_key, secret_key=secret_key, request_params=req1)
            df = _normalize_bars_df(raw_df, ticker)

            # Retry if empty (common entitlement/version edge cases)
            if df is None or df.empty:
                dbg["retry"] = "no_feed"
                dbg["feed"] = "default"
                raw_df = _fetch_bars_df(ticker=ticker, api_key=api_key, secret_key=secret_key, request_params=base_params)
                df = _normalize_bars_df(raw_df, ticker)
        else:
            # Attempt #1: default feed
            raw_df = _fetch_bars_df(ticker=ticker, api_key=api_key, secret_key=secret_key, request_params=base_params)
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
        dbg["error"] = f"{type(e).__name__}: {e}"
        dbg["elapsed_sec"] = round(time.time() - t0, 3)
        return None, dbg


# ---------------------------
# Sanity checks
# ---------------------------
def sanity_check_bars(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns a dict that app.py can display:
      {"ok": bool, "warnings": [...], "stats": {...}}

    Notes for beginners:
    - Warnings do not always mean "bad data" — holidays, IPOs, halts can create gaps.
    - This is a lightweight safety net, not a perfect validator.
    """
    out: Dict[str, Any] = {"ok": True, "warnings": [], "stats": {}}

    if df is None or df.empty:
        out["ok"] = False
        out["warnings"].append("No data returned.")
        return out

    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        out["ok"] = False
        out["warnings"].append(f"Missing required columns: {missing}")
        return out

    # Numeric checks
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if (df["close"] <= 0).any():
        out["ok"] = False
        out["warnings"].append("Found zero or negative close prices.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        out["warnings"].append("Found non-positive OHLC values (may indicate bad data).")

    # Timestamp monotonicity / duplicates
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if len(ts) < 2:
        out["warnings"].append("Very few rows; indicators/backtest may not work yet.")
    else:
        if not ts.is_monotonic_increasing:
            out["warnings"].append("Timestamps not increasing (source issue; app will sort).")
        dupes = int(ts.duplicated().sum())
        if dupes > 0:
            out["warnings"].append(f"Duplicate timestamps found: {dupes}")

        # Gap check: > 7 day gaps (informational)
        gaps = ts.diff().dt.days.fillna(0)
        big_gaps = int((gaps > 7).sum())
        if big_gaps > 0:
            out["warnings"].append(f"Detected {big_gaps} large time gaps (>7 days). (Often normal: IPO/halts/holidays)")

    out["stats"] = {
        "rows": int(len(df)),
        "start": str(ts.min()) if len(ts) else None,
        "end": str(ts.max()) if len(ts) else None,
    }

    return out
