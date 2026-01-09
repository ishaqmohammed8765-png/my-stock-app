from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---- Defaults (kept here to avoid config import / circular deps) ----
TRADING_DAYS_DEFAULT = 252

# These defaults are “neutral but not crazy”
VOL_FLOOR_DEFAULT = 0.05   # 5% annualized
VOL_CAP_DEFAULT = 2.00     # 200% annualized
VOL_DEFAULT_DEFAULT = 0.30 # 30% annualized


@dataclass(frozen=True)
class IndicatorParams:
    ma_fast: int = 50
    ma_slow: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    rvol_lookback: int = 20
    vol_ewm_span: int = 20


def _fs(s: pd.Series, *, name: str = "series", strict: bool = False, max_nan_frac: float = 0.25) -> pd.Series:
    """
    Convert a series to numeric float safely.

    strict:
      - if True, raises if NaN fraction after coercion exceeds max_nan_frac.
    """
    out = pd.to_numeric(s, errors="coerce").astype(float)
    if strict:
        n = len(out)
        if n > 0:
            nan_frac = float(out.isna().mean())
            if nan_frac > max_nan_frac:
                raise ValueError(f"{name}: too many NaNs after numeric coercion ({nan_frac:.0%} > {max_nan_frac:.0%})")
    return out


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing."""
    c = _fs(close, name="close")
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
    h = _fs(high, name="high")
    l = _fs(low, name="low")
    c = _fs(close, name="close")
    prev = c.shift(1)

    tr = pd.concat(
        [(h - l).abs(), (h - prev).abs(), (l - prev).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rvol_ratio(volume: pd.Series, lookback: int = 20, *, neutral_fill: Optional[float] = None) -> pd.Series:
    """
    Relative Volume: today's vol divided by prior rolling avg vol.

    neutral_fill:
      - None -> keep NaN until enough history
      - number -> fill NaN with that neutral value (e.g. 1.0)
    """
    v = _fs(volume, name="volume").fillna(0.0)
    avg = v.shift(1).rolling(lookback, min_periods=lookback).mean()

    out = v / avg.replace(0.0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan).clip(lower=0.0)

    if neutral_fill is not None:
        out = out.fillna(float(neutral_fill))

    return out


def add_indicators_inplace(
    df: pd.DataFrame,
    *,
    params: IndicatorParams = IndicatorParams(),
    trading_days: int = TRADING_DAYS_DEFAULT,
    vol_floor: float = VOL_FLOOR_DEFAULT,
    vol_cap: float = VOL_CAP_DEFAULT,
    vol_default: float = VOL_DEFAULT_DEFAULT,
    strict: bool = False,
) -> None:
    """
    Adds indicator columns in-place.

    Expected base columns: high, low, close, volume
    Adds:
      - ma{fast}, ma{slow}
      - rsi{period}
      - rvol
      - vol_ann (annualized vol proxy)
      - atr{period}
    """
    required = {"high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"add_indicators_inplace missing required columns: {sorted(missing)}")

    c = _fs(df["close"], name="close", strict=strict)

    fast, slow = params.ma_fast, params.ma_slow
    df[f"ma{fast}"] = c.rolling(fast, min_periods=fast).mean()
    df[f"ma{slow}"] = c.rolling(slow, min_periods=slow).mean()

    rp = params.rsi_period
    df[f"rsi{rp}"] = rsi_wilder(c, rp)

    df["rvol"] = rvol_ratio(df["volume"], params.rvol_lookback, neutral_fill=None)

    # Annualized vol proxy from log returns (EWM std)
    r = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan)
    v = r.ewm(span=params.vol_ewm_span, adjust=False, min_periods=params.vol_ewm_span).std()
    v = v * np.sqrt(float(trading_days))
    df["vol_ann"] = v.clip(lower=float(vol_floor), upper=float(vol_cap)).fillna(float(vol_default))

    ap = params.atr_period
    df[f"atr{ap}"] = atr_wilder(df["high"], df["low"], c, ap)


def market_regime_series(
    market_df: pd.DataFrame,
    *,
    ma_len: int = 200,
    price_col: str = "close",
) -> pd.Series:
    """
    Returns a boolean Series indexed like market_df: True if close > MA(ma_len), else False.
    """
    if market_df is None or market_df.empty:
        return pd.Series(dtype=bool)

    if price_col not in market_df.columns:
        raise KeyError(f"market_regime_series expected market_df['{price_col}']")

    close = _fs(market_df[price_col], name=f"market_{price_col}")
    ma = close.rolling(ma_len, min_periods=ma_len).mean()
    regime = close > ma
    # If MA/close missing early, treat as risk-on (do not block early history)
    return regime.fillna(True)


def market_regime_at(
    market_df: pd.DataFrame,
    idx: int,
    ma_len: int = 200,
    price_col: str = "close",
) -> bool:
    """
    Backwards-compatible wrapper: uses integer idx into market_df.
    Prefer aligning by date and using market_regime_series().
    """
    if market_df is None or market_df.empty:
        return True
    if idx < 0 or idx >= len(market_df):
        return True

    regime = market_regime_series(market_df, ma_len=ma_len, price_col=price_col)
    if regime.empty:
        return True
    return bool(regime.iloc[idx])
