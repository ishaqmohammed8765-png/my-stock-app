# utils/backtester.py
from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from .indicators import add_indicators_inplace, market_regime_at


# ---------- Defaults (avoid config import / circular deps) ----------
MIN_HIST_DAYS_DEFAULT = 50


# ---------- Fill helpers (bar-based approximations) ----------

def fill_limit_buy(open_px: float, low_px: float, limit_px: float) -> Optional[float]:
    """Limit buy fills if bar low trades through limit."""
    return float(limit_px) if low_px <= limit_px else None


def fill_stop_buy(open_px: float, high_px: float, stop_px: float) -> Optional[float]:
    """
    Stop buy (breakout) fills if bar high crosses stop.
    If gap up above stop, assume fill at open; else at stop.
    """
    if high_px >= stop_px:
        return float(open_px) if open_px > stop_px else float(stop_px)
    return None


def fill_limit_sell(open_px: float, high_px: float, limit_px: float) -> Optional[float]:
    """Limit sell fills if bar high trades through limit."""
    return float(limit_px) if high_px >= limit_px else None


def fill_stop_sell(open_px: float, low_px: float, stop_px: float) -> Optional[float]:
    """
    Stop sell (stop-loss) fills if bar low crosses stop.
    If gap down below stop, assume fill at open; else at stop.
    """
    if low_px <= stop_px:
        return float(open_px) if open_px < stop_px else float(stop_px)
    return None


# ---------- Core backtest ----------

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
    start_equity: float,  # kept for API compatibility (not used in per-share pnl version)
    *,
    min_hist_days: int = MIN_HIST_DAYS_DEFAULT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates trading logic over historical data with bar-based fills.

    Expects df columns: open, high, low, close, volume
    Optional: timestamp column for nicer reporting.

    Adds indicators (inplace on copy): ma50, ma200, rsi14, rvol, vol_ann, atr14

    Returns:
      - df_bt (price+indicator frame, plus optional debug columns)
      - trades (one row per trade)
    """
    if df is None or df.empty or len(df) < int(min_hist_days):
        return pd.DataFrame(), pd.DataFrame()

    df_bt = df.copy()

    # Ensure required columns exist early (clear error > silent failures)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df_bt.columns)
    if missing:
        raise KeyError(f"backtest_strategy missing required columns in df: {sorted(missing)}")

    # Sort by timestamp if present; else by datetime index if present
    if "timestamp" in df_bt.columns:
        df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"], utc=True, errors="coerce")
        df_bt = df_bt.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    elif isinstance(df_bt.index, pd.DatetimeIndex):
        df_bt = df_bt.sort_index().reset_index(drop=False)

    # Add indicators
    add_indicators_inplace(df_bt)

    # Basic safety
    horizon = int(max(1, horizon))
    cooldown_bars = int(max(0, cooldown_bars))

    # Trade log
    trades: List[Dict[str, Any]] = []

    in_pos = False
    entry_px = np.nan
    entry_i = -1
    stop_px = np.nan
    target_px = np.nan
    cooldown = 0

    # Helpers for pretty timestamps
    def ts_at(i: int) -> Optional[str]:
        if "timestamp" in df_bt.columns and 0 <= i < len(df_bt):
            t = df_bt.loc[i, "timestamp"]
            try:
                return str(pd.to_datetime(t, utc=True))
            except Exception:
                return str(t)
        return None

    # Iterate bars; we reference i+1 for execution, so stop at len-2
    for i in range(len(df_bt) - 1):
        row = df_bt.iloc[i]

        if cooldown > 0:
            cooldown -= 1

        # ---------------------------
        # Exit logic (uses NEXT bar)
        # ---------------------------
        if in_pos:
            nxt = df_bt.iloc[i + 1]
            o, h, l = float(nxt["open"]), float(nxt["high"]), float(nxt["low"])

            # Conservative: stop priority
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
                    # Time-based exit
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
                    "entry_ts": ts_at(entry_i),
                    "exit_ts": ts_at(i + 1),
                    "entry_px": float(entry_px),
                    "exit_px": float(exit_px),
                    "stop_px": float(stop_px),
                    "target_px": float(target_px),
                    "reason": exit_reason,
                    "bars_held": int((i + 1) - entry_i),
                    "pnl_per_share": float(pnl),
                    "r_multiple": float(pnl / max(1e-12, (entry_px - stop_px))),
                }
            )

            in_pos = False
            cooldown = cooldown_bars
            entry_px = np.nan
            entry_i = -1
            stop_px = np.nan
            target_px = np.nan
            continue

        # ---------------------------
        # Entry logic (flat)
        # ---------------------------
        if cooldown > 0:
            continue

        # Optional regime filter
        if require_risk_on:
            if market_df is None or market_df.empty:
                # If user enabled it but no market_df provided, don't silently block everything.
                # Just treat as pass.
                pass
            else:
                if not market_regime_at(market_df, i, ma_len=200, price_col="close"):
                    continue

        # Indicator filters
        rsi = float(row.get("rsi14", np.nan))
        rvol = float(row.get("rvol", np.nan))
        vol_ann = float(row.get("vol_ann", np.nan))
        atr = float(row.get("atr14", np.nan))
        close = float(row["close"])

        if not np.isfinite([rsi, rvol, vol_ann, atr, close]).all():
            continue

        if rsi < float(rsi_min) or rsi > float(rsi_max):
            continue
        if rvol < float(rvol_min):
            continue
        if vol_ann > float(vol_max):
            continue

        # Next bar execution values
        nxt = df_bt.iloc[i + 1]
        o, h, l = float(nxt["open"]), float(nxt["high"]), float(nxt["low"])

        mode_l = str(mode).lower().strip()

        if mode_l == "pullback":
            limit_px = close - float(atr_entry) * atr
            fill = fill_limit_buy(o, l, limit_px)
            if fill is None:
                continue
            entry = fill
            entry_type = "limit"
        elif mode_l == "breakout":
            stop_buy = close + float(atr_entry) * atr
            fill = fill_stop_buy(o, h, stop_buy)
            if fill is None:
                continue
            entry = fill
            entry_type = "stop"
        else:
            raise ValueError("mode must be 'pullback' or 'breakout'")

        if include_spread_penalty and assumed_spread_bps > 0:
            entry *= (1.0 + assumed_spread_bps / 10000.0)

        stop = entry - float(atr_stop) * atr
        target = entry + float(atr_target) * atr

        # Basic guard: avoid nonsensical stop/target
        if not (np.isfinite(entry) and np.isfinite(stop) and np.isfinite(target)):
            continue
        if stop >= entry:
            continue
        if target <= entry:
            continue

        in_pos = True
        entry_px = float(entry)
        entry_i = i + 1
        stop_px = float(stop)
        target_px = float(target)

        # Optional: annotate df_bt with signals (nice for later plotting/debug)
        df_bt.loc[entry_i, "signal"] = 1.0
        df_bt.loc[entry_i, "entry_type"] = entry_type
        df_bt.loc[entry_i, "bt_stop"] = stop_px
        df_bt.loc[entry_i, "bt_target"] = target_px

    trades_df = pd.DataFrame(trades)

    # Make trades table nice even if empty
    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=[
                "entry_i",
                "exit_i",
                "entry_ts",
                "exit_ts",
                "entry_px",
                "exit_px",
                "stop_px",
                "target_px",
                "reason",
                "bars_held",
                "pnl_per_share",
                "r_multiple",
            ]
        )

    return df_bt, trades_df
