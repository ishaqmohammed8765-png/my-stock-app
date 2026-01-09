import pandas as pd
import numpy as np
from utils.config import CFG  # This pulls in the settings you just saved

def _fs(s: pd.Series) -> pd.Series:
    """Converts a series to numeric/float safely."""
    return pd.to_numeric(s, errors="coerce").astype(float)

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index using Wilder's EMA."""
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
    """Calculates Average True Range."""
    h = _fs(high); l = _fs(low); c = _fs(close)
    prev = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

def rvol_ratio(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Calculates Relative Volume compared to its average."""
    v = _fs(volume).fillna(0.0)
    avg = v.shift(1).rolling(lookback, min_periods=lookback).mean()
    out = (v / avg.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return out.fillna(1.0).clip(lower=0.0)

def add_indicators_inplace(df: pd.DataFrame) -> None:
    """Adds all technical indicators to the dataframe at once."""
    c = _fs(df["close"])
    df["ma50"] = c.rolling(50, min_periods=50).mean()
    df["ma200"] = c.rolling(200, min_periods=200).mean()
    df["rsi14"] = rsi_wilder(c, 14)
    df["rvol"] = rvol_ratio(df["volume"], 20)
    # This uses the VOL_FLOOR and VOL_CAP from your config file!
    r = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan)
    v = r.ewm(span=20, adjust=False, min_periods=20).std() * np.sqrt(CFG.TRADING_DAYS)
    df["vol_ann"] = v.clip(lower=CFG.VOL_FLOOR, upper=CFG.VOL_CAP).fillna(CFG.VOL_DEFAULT)
    df["atr14"] = atr_wilder(df["high"], df["low"], c, 14)
