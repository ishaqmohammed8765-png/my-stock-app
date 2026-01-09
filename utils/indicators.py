# utils/indicators.py
from __future__ import annotations

import numpy as np
import pandas as pd


# ---- Defaults (kept here to avoid config import / circular deps) ----
TRADING_DAYS_DEFAULT = 252
VOL_FLOOR_DEFAULT = 0.01
VOL_CAP_DEFAULT = 5.0
VOL_DEFAULT_DEFAULT = 1.0


def _fs(s: pd.Series) -> pd.Series:
    """Convert a series to numeric float safely."""
    return pd.to_numeric(s, errors="coerce").astype(float)


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing."""
    c = _fs(close)
    d = c.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_dn = dn.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / (roll_dn + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.clip(0.0, 100.0)


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR using Wilder's smoothing."""
    h = _fs(high)
    l = _fs(low)
    c = _fs(close)
    prev = c.shift(1)

    tr = pd.concat(
        [(h - l).abs(), (h - prev).abs(), (l - prev).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rvol_ratio(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Relative Volume: today's vol divided by prior rolling avg vol."""
    v = _fs(volume).fillna(0.0)
    avg = v.shift(1).rolling(lookback, min_periods=lookback).mean()
    out = (v / avg.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return out.fillna(1.0).clip(lower=0.0)


def add_indicators_inplace(
    df: pd.DataFrame,
    *,
    trading_days: int = TRADING_DAYS_DEFAULT,
    vol_floor: float = VOL_FLOOR_DEFAULT,
    vol_cap: float = VOL_CAP_DEFAULT,
    vol_default: float = VOL_DEFAULT_DEFAULT,
) -> None:
    """
    Adds indicator columns in-place.
    Expected base columns: high, low, close, volume
    Adds:
      - ma50, ma200
      - rsi14
      - rvol
      - vol_ann (annualized vol proxy)
      - atr14
    """
    required = {"high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"add_indicators_inplace missing required columns: {sorted(missing)}")

    c = _fs(df["close"])

    df["ma50"] = c.rolling(50, min_periods=50).mean()
    df["ma200"] = c.rolling(200, min_periods=200).mean()

    df["rsi14"] = rsi_wilder(c, 14)
    df["rvol"] = rvol_ratio(df["volume"], 20)

    # Annualized vol proxy from log returns
    r = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan)
    v = r.ewm(span=20, adjust=False, min_periods=20).std() * np.sqrt(float(trading_days))
    df["vol_ann"] = v.clip(lower=float(vol_floor), upper=float(vol_cap)).fillna(float(vol_default))

    df["atr14"] = atr_wilder(df["high"], df["low"], c, 14)


def market_regime_at(
    market_df: pd.DataFrame,
    idx: int,
    ma_len: int = 200,
    price_col: str = "close",
) -> bool:
    """
    Market regime filter at bar index `idx`.
    Returns True if risk-on (market close > MA(ma_len)), else False.

    Notes:
    - Expects market_df sorted oldest->newest, same cadence as your trading df.
    - If not enough data, defaults to True (don't block early history).
    """
    if market_df is None or market_df.empty:
        return True

    if price_col not in market_df.columns:
        raise KeyError(f"market_regime_at expected market_df['{price_col}']")

    if idx < 0 or idx >= len(market_df):
        return True

    close = _fs(market_df[price_col])
    ma = close.rolling(ma_len, min_periods=ma_len).mean()

    m = ma.iloc[idx]
    c = close.iloc[idx]

    if np.isnan(m) or np.isnan(c):
        return True

    return bool(c > m)
