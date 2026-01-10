from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

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
    if len(k) < 8 or len(s) < 12:
        return False
    if any(ch.isspace() for ch in k) or any(ch.isspace() for ch in s):
        return False
    return True


# ---------------------------
# Dataframe normalization
# ---------------------------
def _normalize_bars_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize Alpaca bars into a flat df with columns:
      timestamp, open, high, low, close, volume, trade_count, vwap (if present), symbol (if present)

    Handles:
    - MultiIndex (symbol, timestamp)
    - DatetimeIndex
    - various timestamp column names
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # If MultiIndex -> reset to columns
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()

    # If DatetimeIndex and no timestamp column, promote index
    if isinstance(out.index, pd.DatetimeIndex) and "timestamp" not in [str(c).lower() for c in out.columns]:
        out = out.reset_index().rename(columns={"index": "timestamp"})

    # Standardize column names to lowercase strings for easier matching
    out = out.rename(columns={c: str(c).lower() for c in out.columns})

    # Common variants -> timestamp
    for alt in ("time", "date", "datetime"):
        if alt in out.columns and "timestamp" not in out.columns:
            out = out.rename(columns={alt: "timestamp"})

    # If still missing timestamp, try to find any datetime-ish column
    if "timestamp" not in out.columns:
        for c in list(out.columns):
            if "time" in c or "date" in c:
                out = out.rename(columns={c: "timestamp"})
                break

    if "timestamp" not in out.columns:
        return out

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])

    # Ensure symbol exists if available
    if "symbol" not in out.columns and "s" in out.columns:
        out["symbol"] = out["s"]

    # Filter to ticker if symbol is present
    tkr = ticker.upper().strip()
    if "symbol" in out.columns:
        out = out[out["symbol"].astype(str).str.upper() == tkr]

    # Coerce numeric columns
    for c in ["open", "high", "low", "close", "volume", "trade_count", "vwap"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop rows missing core OHLC
    core = [c for c in ["open", "high", "low", "close"] if c in out.columns]
    if core:
        out = out.dropna(subset=core)

    # Sort + dedupe timestamps
    out = (
        out.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    return out


def _fetch_bars_df(
    *,
    ticker: str,
    api_key: str,
    secret_key: str,
    request_params: dict[str, Any],
    retries: int = 2,
    backoff_sec: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch bars with light retry for transient issues (e.g., 429 rate limit).
    """
    client = StockHistoricalDataClient(api_key, secret_key)

    last_err: Exception | None = None
    for i in range(max(1, int(retries) + 1)):
        try:
            bars = client.get_stock_bars(StockBarsRequest(**request_params))
            return bars.df
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            is_rate = ("429" in msg) or ("rate" in msg and "limit" in msg)
            if i < retries and is_rate:
                time.sleep(backoff_sec * (i + 1))
                continue
            raise

    # Should never reach here
    raise RuntimeError(f"Failed to fetch bars: {last_err}")


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
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """
    Fetch historical daily bars from Alpaca.
    Returns (df, dbg).
    """
    _ = force_refresh  # used only for cache keying

    ticker = (ticker or "").upper().strip()
    dbg: dict[str, Any] = {"ticker": ticker, "status": "init", "feed": "default"}

    if not ticker:
        dbg["status"] = "error"
        dbg["error"] = "Empty ticker"
        return None, dbg

    if not validate_keys(api_key, secret_key):
        dbg["status"] = "error"
        dbg["error"] = "Invalid/missing keys (format check)."
        return None, dbg

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=int(days_back))
    end = now

    base_params: dict[str, Any] = {
        "symbol_or_symbols": [ticker],
        "timeframe": TimeFrame.Day,
        "start": start,
        "end": end,
    }

    try_iex = bool(prefer_iex and HAS_DATAFEED)
    t0 = time.time()

    try:
        if try_iex:
            req1 = dict(base_params)
            try:
                req1["feed"] = DataFeed.IEX  # type: ignore[name-defined]
                dbg["feed"] = "IEX"
            except Exception:
                req1 = base_params
                dbg["feed"] = "default"

            raw_df = _fetch_bars_df(
                ticker=ticker,
                api_key=api_key,
                secret_key=secret_key,
                request_params=req1,
            )
            df = _normalize_bars_df(raw_df, ticker)

            if df is None or df.empty:
                dbg["retry"] = "fallback_default_feed"
                dbg["feed"] = "default"
                raw_df = _fetch_bars_df(
                    ticker=ticker,
                    api_key=api_key,
                    secret_key=secret_key,
                    request_params=base_params,
                )
                df = _normalize_bars_df(raw_df, ticker)
        else:
            raw_df = _fetch_bars_df(
                ticker=ticker,
                api_key=api_key,
                secret_key=secret_key,
                request_params=base_params,
            )
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
def sanity_check_bars(df: pd.DataFrame) -> dict[str, Any]:
    """
    Lightweight checks. Does NOT mutate the input df.
    Returns: {"ok": bool, "warnings": [...], "stats": {...}}
    """
    out: dict[str, Any] = {"ok": True, "warnings": [], "stats": {}}

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

    # Work on local series (don’t mutate original df)
    close = pd.to_numeric(df["close"], errors="coerce")
    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")

    if (close <= 0).any():
        out["ok"] = False
        out["warnings"].append("Found zero or negative close prices.")
    if ((pd.concat([o, h, l, close], axis=1) <= 0).any().any()):
        out["warnings"].append("Found non-positive OHLC values (may indicate bad data).")

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if len(ts) < 2:
        out["warnings"].append("Very few rows; indicators/backtest may not work yet.")
    else:
        if not ts.is_monotonic_increasing:
            out["warnings"].append("Timestamps not increasing (source issue; caller should sort).")
        dupes = int(ts.duplicated().sum())
        if dupes > 0:
            out["warnings"].append(f"Duplicate timestamps found: {dupes}")

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
