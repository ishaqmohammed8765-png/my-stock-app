"""Causal daily indicators; no future rows or default-filled readiness."""
import numpy as np
import pandas as pd


def wilder(series: pd.Series, period: int = 14) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    previous = np.nan
    for i, value in enumerate(values):
        if not np.isfinite(value):
            previous = np.nan
        elif np.isfinite(previous):
            previous = (previous * (period - 1) + value) / period
        elif i >= period - 1 and np.isfinite(values[i-period+1:i+1]).all():
            previous = float(values[i-period+1:i+1].mean())
        out[i] = previous
    return pd.Series(out, index=series.index)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c, h, low, v = (out[k] for k in ("close", "high", "low", "volume"))
    out["ma50"], out["ma200"] = c.rolling(50).mean(), c.rolling(200).mean()
    delta = c.diff()
    up, down = wilder(delta.clip(lower=0)), wilder(-delta.clip(upper=0))
    out["rsi14"] = 100 * up / (up+down).replace(0, np.nan)
    out.loc[(up == 0) & (down == 0), "rsi14"] = 50.
    tr = pd.concat([h-low, (h-c.shift()).abs(), (low-c.shift()).abs()], axis=1).max(axis=1)
    out["atr14"] = wilder(tr)
    out["rvol"] = v / v.shift().rolling(20).mean().replace(0, np.nan)
    out["vol_ann"] = np.log(c/c.shift()).ewm(span=20, min_periods=20, adjust=False).std() * np.sqrt(252)
    up_move, down_move = h.diff(), -low.diff()
    plus = wilder(up_move.where((up_move > down_move) & (up_move > 0), 0.))
    minus = wilder(down_move.where((down_move > up_move) & (down_move > 0), 0.))
    dx = 100 * (plus-minus).abs() / (plus+minus).replace(0, np.nan)
    out["adx14"] = wilder(dx.mask((plus == 0) & (minus == 0), 0.))
    out["macd"] = c.ewm(span=12, adjust=False, min_periods=12).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    out["macd_signal"] = out.macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out.macd-out.macd_signal
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_upper"], out["bb_lower"] = mid+2*sd, mid-2*sd
    out["ind_ready"] = np.isfinite(out[["ma200", "ma50", "rsi14", "atr14", "rvol", "vol_ann", "adx14"]]).all(axis=1) & (out.atr14 > 0)
    return out
