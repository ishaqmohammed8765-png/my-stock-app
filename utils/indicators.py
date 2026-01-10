from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

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
    rvol_neutral_fill: Optional[float] = 1.0  # set None for raw NaN until ready


def _fs(
    s: pd.Series,
    *,
    name: str = "series",
    strict: bool = False,
    max_nan_frac: float = 0.25,
) -> pd.Series:
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
    Adds canonical columns:
      - ma50, ma200
      - rsi14
      - atr14
      - rvol
      - vol_ann
    Extras:
      - atr_pct, range_pct
      - trend_state (Up / Down / Mixed)
      - readiness flags: ma_ready, rsi_ready, atr_ready, rvol_ready, vol_ready, ind_ready
      - vol_ann_raw (unclamped)
      - hl_fixed (bool flag if high/low were swapped)
    """
    required = {"high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"add_indicators_inplace missing required columns: {sorted(missing)}")

    # Coerce core series (local)
    c = _fs(df["close"], name="close", strict=strict)
    h = _fs(df["high"], name="high", strict=strict)
    l = _fs(df["low"], name="low", strict=strict)

    # If high < low, swap AND write back to df so plots and indicators agree
    bad_hl = (h < l) & np.isfinite(h) & np.isfinite(l)
    df["hl_fixed"] = bool(bad_hl.any())
    if bad_hl.any():
        h2 = h.copy()
        h2[bad_hl] = l[bad_hl]
        l2 = l.copy()
        l2[bad_hl] = h[bad_hl]
        h, l = h2, l2
        df.loc[bad_hl, "high"] = h[bad_hl]
        df.loc[bad_hl, "low"] = l[bad_hl]

    # Moving averages
    fast, slow = int(params.ma_fast), int(params.ma_slow)
    df[f"ma{fast}"] = c.rolling(fast, min_periods=fast).mean()
    df[f"ma{slow}"] = c.rolling(slow, min_periods=slow).mean()

    # Canonical names
    df["ma50"] = df[f"ma{fast}"] if fast == 50 else c.rolling(50, min_periods=50).mean()
    df["ma200"] = df[f"ma{slow}"] if slow == 200 else c.rolling(200, min_periods=200).mean()

    # RSI
    rp = int(params.rsi_period)
    df[f"rsi{rp}"] = rsi_wilder(c, rp)
    df["rsi14"] = df[f"rsi{rp}"] if rp == 14 else rsi_wilder(c, 14)

    # RVOL
    df["rvol"] = rvol_ratio(df["volume"], int(params.rvol_lookback), neutral_fill=params.rvol_neutral_fill)

    # Volatility (raw + clamped/fill)
    r = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan)
    span = int(params.vol_ewm_span)
    v_raw = r.ewm(span=span, adjust=False, min_periods=span).std() * np.sqrt(float(trading_days))
    df["vol_ann_raw"] = v_raw
    df["vol_ann"] = v_raw.clip(lower=float(vol_floor), upper=float(vol_cap)).fillna(float(vol_default))

    # ATR
    ap = int(params.atr_period)
    df[f"atr{ap}"] = atr_wilder(h, l, c, ap)
    df["atr14"] = df[f"atr{ap}"] if ap == 14 else atr_wilder(h, l, c, 14)

    # Extras
    df["atr_pct"] = (df["atr14"] / c.replace(0.0, np.nan)).clip(lower=0.0)
    df["range_pct"] = ((h - l).abs() / c.replace(0.0, np.nan)).clip(lower=0.0)

    ma50 = pd.to_numeric(df["ma50"], errors="coerce")
    ma200 = pd.to_numeric(df["ma200"], errors="coerce")
    trend = np.where((c > ma50) & (ma50 > ma200), "Up",
             np.where((c < ma50) & (ma50 < ma200), "Down", "Mixed"))
    df["trend_state"] = pd.Series(trend, index=df.index, dtype="object")

    # Readiness flags (so your app/backtest can avoid default-filled early periods)
    df["ma_ready"] = ma200.notna()
    df["rsi_ready"] = pd.to_numeric(df["rsi14"], errors="coerce").notna()
    df["atr_ready"] = pd.to_numeric(df["atr14"], errors="coerce").notna()
    # rvol may be filled with neutral_fill, so "ready" means avg exists
    avg_vol = _fs(df["volume"], name="volume").shift(1).rolling(int(params.rvol_lookback), min_periods=int(params.rvol_lookback)).mean()
    df["rvol_ready"] = avg_vol.notna()
    df["vol_ready"] = df["vol_ann_raw"].notna()

    df["ind_ready"] = df[["ma_ready", "rsi_ready", "atr_ready", "rvol_ready", "vol_ready"]].all(axis=1)


def market_regime_series(
    market_df: pd.DataFrame,
    *,
    ma_len: int = 200,
    price_col: str = "close",
    unknown: Literal["risk_on", "risk_off"] = "risk_off",
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

    fill_val = True if unknown == "risk_on" else False
    return regime.fillna(fill_val)


def market_regime_at(
    market_df: pd.DataFrame,
    idx: int,
    ma_len: int = 200,
    price_col: str = "close",
    unknown: Literal["risk_on", "risk_off"] = "risk_off",
) -> bool:
    if market_df is None or market_df.empty:
        return True
    if idx < 0 or idx >= len(market_df):
        return True

    regime = market_regime_series(market_df, ma_len=ma_len, price_col=price_col, unknown=unknown)
    if regime.empty:
        return True
    return bool(regime.iloc[idx])
