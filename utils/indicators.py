from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS_DEFAULT = 252

VOL_FLOOR_DEFAULT = 0.05
VOL_CAP_DEFAULT = 2.00
VOL_DEFAULT_DEFAULT = 0.30


@dataclass(frozen=True)
class IndicatorParams:
    ma_fast: int = 50
    ma_slow: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    rvol_lookback: int = 20
    vol_ewm_span: int = 20

    # beginner-friendly behavior
    rvol_neutral_fill: Optional[float] = 1.0   # set None for raw NaN until ready


def _fs(s: pd.Series, *, name: str = "series", strict: bool = False, max_nan_frac: float = 0.25) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").astype(float)
    if strict:
        n = len(out)
        if n > 0:
            nan_frac = float(out.isna().mean())
            if nan_frac > max_nan_frac:
                raise ValueError(f"{name}: too many NaNs after coercion ({nan_frac:.0%} > {max_nan_frac:.0%})")
    return out


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
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
    h = _fs(high, name="high")
    l = _fs(low, name="low")
    c = _fs(close, name="close")
    prev = c.shift(1)

    tr = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rvol_ratio(volume: pd.Series, lookback: int = 20, *, neutral_fill: Optional[float] = 1.0) -> pd.Series:
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
    Expected base columns: high, low, close, volume
    Adds canonical columns used by your app/backtester:
      - ma50, ma200
      - rsi14
      - atr14
      - rvol
      - vol_ann
    Also adds beginner-friendly extras:
      - atr_pct, range_pct
      - trend_state (Up / Down / Mixed)
    """
    required = {"high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"add_indicators_inplace missing required columns: {sorted(missing)}")

    # Coerce core series
    c = _fs(df["close"], name="close", strict=strict)
    h = _fs(df["high"], name="high", strict=strict)
    l = _fs(df["low"], name="low", strict=strict)

    # Defensive OHLC cleanup
    # If high < low due to bad data, swap (keeps the bar usable)
    bad_hl = (h < l) & np.isfinite(h) & np.isfinite(l)
    if bad_hl.any():
        hh = h.copy()
        h[bad_hl] = l[bad_hl]
        l[bad_hl] = hh[bad_hl]

    # Moving averages (param + canonical)
    fast, slow = int(params.ma_fast), int(params.ma_slow)
    df[f"ma{fast}"] = c.rolling(fast, min_periods=fast).mean()
    df[f"ma{slow}"] = c.rolling(slow, min_periods=slow).mean()

    # Canonical names expected elsewhere
    if fast != 50 and "ma50" not in df.columns:
        df["ma50"] = c.rolling(50, min_periods=50).mean()
    if slow != 200 and "ma200" not in df.columns:
        df["ma200"] = c.rolling(200, min_periods=200).mean()
    if fast == 50:
        df["ma50"] = df[f"ma{fast}"]
    if slow == 200:
        df["ma200"] = df[f"ma{slow}"]

    # RSI (param + canonical)
    rp = int(params.rsi_period)
    df[f"rsi{rp}"] = rsi_wilder(c, rp)
    df["rsi14"] = df[f"rsi{rp}"] if rp == 14 else rsi_wilder(c, 14)

    # RVOL (canonical)
    df["rvol"] = rvol_ratio(df["volume"], int(params.rvol_lookback), neutral_fill=params.rvol_neutral_fill)

    # Annualized vol proxy (canonical)
    r = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan)
    v = r.ewm(span=int(params.vol_ewm_span), adjust=False, min_periods=int(params.vol_ewm_span)).std()
    v = v * np.sqrt(float(trading_days))
    df["vol_ann"] = v.clip(lower=float(vol_floor), upper=float(vol_cap)).fillna(float(vol_default))

    # ATR (param + canonical)
    ap = int(params.atr_period)
    df[f"atr{ap}"] = atr_wilder(h, l, c, ap)
    df["atr14"] = df[f"atr{ap}"] if ap == 14 else atr_wilder(h, l, c, 14)

    # Beginner-friendly extras
    df["atr_pct"] = (df["atr14"] / c.replace(0.0, np.nan)).clip(lower=0.0)
    df["range_pct"] = ((h - l).abs() / c.replace(0.0, np.nan)).clip(lower=0.0)

    # Simple trend label (helps explain why BUY/HOLD/SELL)
    ma50 = pd.to_numeric(df["ma50"], errors="coerce")
    ma200 = pd.to_numeric(df["ma200"], errors="coerce")
    trend = np.where((c > ma50) & (ma50 > ma200), "Up",
             np.where((c < ma50) & (ma50 < ma200), "Down", "Mixed"))
    df["trend_state"] = pd.Series(trend, index=df.index, dtype="object")


def market_regime_series(
    market_df: pd.DataFrame,
    *,
    ma_len: int = 200,
    price_col: str = "close",
) -> pd.Series:
    if market_df is None or market_df.empty:
        return pd.Series(dtype=bool)

    m = market_df.copy()
    if price_col not in m.columns:
        raise KeyError(f"market_regime_series expected market_df['{price_col}']")

    # Sort to avoid MA on unsorted frames
    if isinstance(m.index, pd.DatetimeIndex):
        m = m.sort_index()

    close = _fs(m[price_col], name=f"market_{price_col}")
    ma = close.rolling(int(ma_len), min_periods=int(ma_len)).mean()
    regime = close > ma
    return regime.fillna(True)


def market_regime_at(
    market_df: pd.DataFrame,
    idx: int,
    ma_len: int = 200,
    price_col: str = "close",
) -> bool:
    if market_df is None or market_df.empty:
        return True
    if idx < 0 or idx >= len(market_df):
        return True

    regime = market_regime_series(market_df, ma_len=ma_len, price_col=price_col)
    if regime.empty:
        return True
    return bool(regime.iloc[idx])
