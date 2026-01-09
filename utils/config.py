# utils/backtester.py
from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from .config import CFG
from .indicators import add_indicators_inplace, market_regime_at


def fill_limit_buy(open_px: float, low_px: float, limit_px: float) -> Optional[float]:
    return float(limit_px) if low_px <= limit_px else None


def fill_stop_buy(open_px: float, high_px: float, stop_px: float) -> Optional[float]:
    if high_px >= stop_px:
        return float(open_px) if open_px > stop_px else float(stop_px)
    return None


def fill_limit_sell(open_px: float, high_px: float, limit_px: float) -> Optional[float]:
    return float(limit_px) if high_px >= limit_px else None


def fill_stop_sell(open_px: float, low_px: float, stop_px: float) -> Optional[float]:
    if low_px <= stop_px:
        return float(open_px) if open_px < stop_px else float(stop_px)
    return None


def backtest_strategy(
    df: pd.DataFrame,
    market_df: Optional[pd.DataFrame],
    horizon: int,
    mode: str,
    atr_entry: float,
    atr_stop: float,
    atr_target: float,
    require_risk_on: bool,
    rsi_min: float,
    rsi_max: float,
    rvol_min: float,
    vol_max: float,
    cooldown_bars: int,
    include_spread_penalty: bool,
    assumed_spread_bps: float,
    start_equity: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if df is None or df.empty or len(df) < CFG.MIN_HIST_DAYS:
        return pd.DataFrame(), pd.DataFrame()

    df_bt = df.copy()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df_bt.columns)
    if missing:
        raise KeyError(f"backtest_strategy missing required columns in df: {sorted(missing)}")

    if isinstance(df_bt.index, pd.DatetimeIndex):
        df_bt = df_bt.sort_index()

    add_indicators_inplace(df_bt)

    horizon = int(max(1, horizon))
    cooldown_bars = int(max(0, cooldown_bars))

    trades: List[Dict[str, Any]] = []

    in_pos = False
    entry_px = np.nan
    entry_i = -1
    stop_px = np.nan
    target_px = np.nan
    cooldown = 0

    for i in range(len(df_bt) - 1):
        row = df_bt.iloc[i]

        if cooldown > 0:
            cooldown -= 1

        if in_pos:
            nxt = df_bt.iloc[i + 1]
            o, h, l = float(nxt["open"]), float(nxt["high"]), float(nxt["low"])

            stop_fill = fill_stop_sell(o, l, float(stop_px))
            if stop_fill is not None:
                exit_px = stop_fill
                exit_reason = "stop"
            else:
                tgt_fill = fill_limit_sell(o, h, float(target_px))
                if tgt_fill is not None:
                    exit_px = tgt_fill
                    exit_reason = "target"
                else:
                    if (i - entry_i) >= horizon:
                        exit_px = float(nxt["open"])
                        exit_reason = "time"
                    else:
                        continue

            if include_spread_penalty and assumed_spread_bps > 0:
                exit_px *= (1.0 - assumed_spread_bps / 10000.0)

            pnl = (exit_px - entry_px)
            trades.append(
                {
                    "entry_i": entry_i,
                    "exit_i": i + 1,
                    "entry_px": float(entry_px),
                    "exit_px": float(exit_px),
                    "reason": exit_reason,
                    "pnl_per_share": float(pnl),
                }
            )

            in_pos = False
            cooldown = cooldown_bars
            entry_px = np.nan
            entry_i = -1
            stop_px = np.nan
            target_px = np.nan
            continue

        if cooldown > 0:
            continue

        if require_risk_on and market_df is not None:
            if not market_regime_at(market_df, i, ma_len=200, price_col="close"):
                continue

        rsi = float(row.get("rsi14", np.nan))
        rvol = float(row.get("rvol", np.nan))
        vol_ann = float(row.get("vol_ann", np.nan))
        atr = float(row.get("atr14", np.nan))
        close = float(row["close"])

        if not np.isfinite([rsi, rvol, vol_ann, atr, close]).all():
            continue

        if rsi < rsi_min or rsi > rsi_max:
            continue
        if rvol < rvol_min:
            continue
        if vol_ann > vol_max:
            continue

        nxt = df_bt.iloc[i + 1]
        o, h, l = float(nxt["open"]), float(nxt["high"]), float(nxt["low"])

        mode_l = str(mode).lower()
        if mode_l == "pullback":
            limit_px = close - atr_entry * atr
            fill = fill_limit_buy(o, l, limit_px)
            if fill is None:
                continue
            entry = fill
        elif mode_l == "breakout":
            stop_buy = close + atr_entry * atr
            fill = fill_stop_buy(o, h, stop_buy)
            if fill is None:
                continue
            entry = fill
        else:
            raise ValueError("mode must be 'pullback' or 'breakout'")

        if include_spread_penalty and assumed_spread_bps > 0:
            entry *= (1.0 + assumed_spread_bps / 10000.0)

        stop_px = float(entry - atr_stop * atr)
        target_px = float(entry + atr_target * atr)

        in_pos = True
        entry_px = float(entry)
        entry_i = i + 1

    return df_bt, pd.DataFrame(trades)
