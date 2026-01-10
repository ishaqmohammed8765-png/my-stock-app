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

    # ADX gate (automatic, used by backtester if present)
    adx_period: int = 14

    # beginner-friendly behavior
    rvol_neutral_fill: Optional[float] = 1.0  # set None for raw NaN until ready


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


def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    ADX (Wilder): trend strength indicator.
    Output is 0..100-ish. Higher = stronger trend.
    """
    h = _fs(high, name="high")
    l = _fs(low, name="low")
    c = _fs(close, name="close")

    up_move = h.diff()
    down_move = (-l.diff())

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=c.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / (atr + 1e-12)
    minus_di = 100.0 * pd.Series(minus_dm, index=c.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / (atr + 1e-12)

    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx.clip(lower=0.0, upper=100.0)


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
    Adds canonical columns:
      - ma50, ma200
      - rsi14
      - atr14
      - rvol
      - vol_ann
      - adx14   (NEW)
    Extras:
      - atr_pct, range_pct
      - trend_state (Up / Down / Mixed)
      - ind_ready (NEW: if enough history for core indicators)
    """
    required = {"high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"add_indicators_inplace missing required columns: {sorted(missing)}")

    c = _fs(df["close"], name="close", strict=strict)
    h = _fs(df["high"], name="high", strict=strict)
    l = _fs(df["low"], name="low", strict=strict)

    # Defensive HL cleanup (keep series consistent)
    bad_hl = (h < l) & np.isfinite(h) & np.isfinite(l)
    if bad_hl.any():
        hh = h.copy()
        h[bad_hl] = l[bad_hl]
        l[bad_hl] = hh[bad_hl]
        # optionally sync back to df so plots match indicators
        df.loc[bad_hl, "high"] = h[bad_hl]
        df.loc[bad_hl, "low"] = l[bad_hl]

    fast, slow = int(params.ma_fast), int(params.ma_slow)
    df[f"ma{fast}"] = c.rolling(fast, min_periods=fast).mean()
    df[f"ma{slow}"] = c.rolling(slow, min_periods=slow).mean()

    df["ma50"] = df[f"ma{fast}"] if fast == 50 else c.rolling(50, min_periods=50).mean()
    df["ma200"] = df[f"ma{slow}"] if slow == 200 else c.rolling(200, min_periods=200).mean()

    rp = int(params.rsi_period)
    df[f"rsi{rp}"] = rsi_wilder(c, rp)
    df["rsi14"] = df[f"rsi{rp}"] if rp == 14 else rsi_wilder(c, 14)

    df["rvol"] = rvol_ratio(df["volume"], int(params.rvol_lookback), neutral_fill=params.rvol_neutral_fill)

    r = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan)
    span = int(params.vol_ewm_span)
    v = r.ewm(span=span, adjust=False, min_periods=span).std()
    v = v * np.sqrt(float(trading_days))
    df["vol_ann"] = v.clip(lower=float(vol_floor), upper=float(vol_cap)).fillna(float(vol_default))

    ap = int(params.atr_period)
    df[f"atr{ap}"] = atr_wilder(h, l, c, ap)
    df["atr14"] = df[f"atr{ap}"] if ap == 14 else atr_wilder(h, l, c, 14)

    # NEW: ADX
    adxp = int(params.adx_period)
    df[f"adx{adxp}"] = adx_wilder(h, l, c, adxp)
    df["adx14"] = df[f"adx{adxp}"] if adxp == 14 else adx_wilder(h, l, c, 14)

    # Extras
    df["atr_pct"] = (df["atr14"] / c.replace(0.0, np.nan)).clip(lower=0.0)
    df["range_pct"] = ((h - l).abs() / c.replace(0.0, np.nan)).clip(lower=0.0)

    ma50 = pd.to_numeric(df["ma50"], errors="coerce")
    ma200 = pd.to_numeric(df["ma200"], errors="coerce")
    trend = np.where((c > ma50) & (ma50 > ma200), "Up",
             np.where((c < ma50) & (ma50 < ma200), "Down", "Mixed"))
    df["trend_state"] = pd.Series(trend, index=df.index, dtype="object")

    # NEW: readiness (avoids trading on default-filled early indicators)
    ready_cols = [
        pd.to_numeric(df["ma200"], errors="coerce").notna(),
        pd.to_numeric(df["rsi14"], errors="coerce").notna(),
        pd.to_numeric(df["atr14"], errors="coerce").notna(),
        pd.to_numeric(df["vol_ann"], errors="coerce").notna(),
        pd.to_numeric(df["adx14"], errors="coerce").notna(),
    ]
    df["ind_ready"] = pd.concat(ready_cols, axis=1).all(axis=1)


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
