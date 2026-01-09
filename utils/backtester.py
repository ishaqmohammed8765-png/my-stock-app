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
    # open_px currently unused but kept for interface consistency / future enhancements
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
    # open_px unused; kept for symmetry / future enhancements
    return float(limit_px) if high_px >= limit_px else None


def fill_stop_sell(open_px: float, low_px: float, stop_px: float) -> Optional[float]:
    """
    Stop sell (stop-loss) fills if bar low crosses stop.
    If gap down below stop, assume fill at open; else at stop.
    """
    if low_px <= stop_px:
        return float(open_px) if open_px < stop_px else float(stop_px)
    return None


# ---------- Internal helpers ----------

def _ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a normalized UTC `timestamp` column, sorted ascending.
    Works if df has:
      - a 'timestamp' column
      - a DatetimeIndex
      - neither (then timestamp will be NaT)
    """
    out = df.copy()

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.copy()
        out["timestamp"] = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out.reset_index(drop=True)
    else:
        out["timestamp"] = pd.NaT

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


def _apply_costs_px(
    px: float,
    side: str,
    *,
    slippage_bps: float,
    spread_bps: float,
    spread_mode: str,
    fill_type: str,
    reason: str,
) -> float:
    """
    Apply slippage + spread penalties to a fill price.

    side: "buy" or "sell"
    spread_mode:
      - "taker_only": spread applies only to taker-like fills (stops, time exits)
      - "always": spread applies to all fills
      - "never": spread applies never
    fill_type: "limit" or "stop" (entry), for exit we pass something like "stop/target/time"
    reason: exit reason ("stop", "target", "time") or "entry"
    """
    px = float(px)
    side_l = side.lower().strip()
    spread_mode = str(spread_mode).lower().strip()
    fill_type = str(fill_type).lower().strip()
    reason = str(reason).lower().strip()

    # Slippage: always adverse
    # buy -> higher, sell -> lower
    if slippage_bps and slippage_bps > 0:
        if side_l == "buy":
            px *= (1.0 + slippage_bps / 10000.0)
        else:
            px *= (1.0 - slippage_bps / 10000.0)

    # Spread:
    # - For "taker_only": apply to stop entries and time/stop exits; targets may be maker-ish.
    apply_spread = False
    if spread_bps and spread_bps > 0:
        if spread_mode == "always":
            apply_spread = True
        elif spread_mode == "never":
            apply_spread = False
        else:  # taker_only (default)
            if side_l == "buy":
                apply_spread = (fill_type == "stop")
            else:
                # stop/time exits are usually taker-ish; target could be limit/maker-ish
                apply_spread = (reason in {"stop", "time"})

    if apply_spread:
        if side_l == "buy":
            px *= (1.0 + spread_bps / 10000.0)
        else:
            px *= (1.0 - spread_bps / 10000.0)

    return float(px)


def _market_regime_on_timestamp(
    market_df: pd.DataFrame,
    ts: pd.Timestamp,
    *,
    ma_len: int = 200,
    price_col: str = "close",
) -> bool:
    """
    Align market regime by timestamp rather than integer index.
    Uses the last market bar with timestamp <= ts.
    """
    if market_df is None or market_df.empty or ts is pd.NaT:
        return True

    m = market_df.copy()
    if "timestamp" in m.columns:
        m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True, errors="coerce")
        m = m.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    elif isinstance(m.index, pd.DatetimeIndex):
        m = m.copy()
        m["timestamp"] = pd.to_datetime(m.index, utc=True, errors="coerce")
        m = m.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    else:
        # Can't align properly; fail open rather than block all trades
        return True

    # Find last index where market timestamp <= ts
    idx = int(m["timestamp"].searchsorted(ts, side="right") - 1)
    if idx < 0:
        return True  # no prior market data; fail open

    # Use existing helper on the correctly aligned bar index
    try:
        return bool(market_regime_at(m, idx, ma_len=ma_len, price_col=price_col))
    except Exception:
        return True


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
    start_equity: float,  # now used if sizing enabled
    *,
    min_hist_days: int = MIN_HIST_DAYS_DEFAULT,

    # --- Upgrades (keyword-only, backward compatible) ---
    slippage_bps: float = 0.0,
    spread_mode: str = "taker_only",  # "taker_only" | "always" | "never"
    enable_position_sizing: bool = False,
    risk_pct: float = 0.02,           # risk per trade as % of equity (if sizing enabled)
    max_alloc_pct: float = 0.10,      # max position value as % of equity
    commission_per_trade: float = 0.0,
    min_risk_per_share: float = 1e-6, # skip trades with tiny risk
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates trading logic over historical data with bar-based fills.

    Expects df columns: open, high, low, close, volume
    Optional: timestamp column OR a DatetimeIndex.

    Adds indicators (inplace on copy): rsi14, rvol, vol_ann, atr14 (plus any others your indicators add)

    Returns:
      - df_bt (price+indicator frame, plus optional signal/targets, and optional equity curve)
      - trades (one row per trade)
    """
    if df is None or df.empty or len(df) < int(min_hist_days):
        return pd.DataFrame(), pd.DataFrame()

    # Ensure required columns exist early (clear error > silent failures)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"backtest_strategy missing required columns in df: {sorted(missing)}")

    # Normalize timestamp + sort
    df_bt = _ensure_timestamp_column(df)

    if df_bt.empty or len(df_bt) < int(min_hist_days):
        return pd.DataFrame(), pd.DataFrame()

    # Add indicators
    add_indicators_inplace(df_bt)

    # Validate required indicators exist; if not, fail loud (better than skipping everything)
    needed_ind_cols = {"rsi14", "rvol", "vol_ann", "atr14"}
    missing_ind = needed_ind_cols - set(df_bt.columns)
    if missing_ind:
        raise KeyError(
            f"Indicators missing expected columns: {sorted(missing_ind)}. "
            f"Check add_indicators_inplace()."
        )

    # Basic safety
    horizon = int(max(1, horizon))
    cooldown_bars = int(max(0, cooldown_bars))

    # Cost controls
    spread_bps = float(assumed_spread_bps) if include_spread_penalty else 0.0
    slip_bps = float(slippage_bps)

    # Trade log
    trades: List[Dict[str, Any]] = []

    # Position/equity
    equity = float(start_equity) if (enable_position_sizing and np.isfinite(start_equity) and start_equity > 0) else float(start_equity or 0.0)

    in_pos = False
    entry_px = np.nan
    entry_i = -1
    stop_px = np.nan
    target_px = np.nan
    cooldown = 0
    qty = 0  # shares if sizing enabled

    def ts_at(i: int) -> Optional[str]:
        if 0 <= i < len(df_bt):
            t = df_bt.loc[i, "timestamp"]
            try:
                return str(pd.to_datetime(t, utc=True))
            except Exception:
                return str(t)
        return None

    # Optional equity curve columns
    if enable_position_sizing:
        df_bt["equity"] = np.nan
        df_bt.loc[0, "equity"] = equity

    mode_l = str(mode).lower().strip()

    # Iterate bars; we reference i+1 for execution, so stop at len-2
    for i in range(len(df_bt) - 1):
        row = df_bt.iloc[i]

        if enable_position_sizing and np.isfinite(df_bt.loc[i, "equity"]) is False:
            # forward-fill equity line
            df_bt.loc[i, "equity"] = equity

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
                raw_exit_px = stop_fill
                exit_reason = "stop"
                exit_fill_type = "stop"
            else:
                tgt_fill = fill_limit_sell(o, h, float(target_px))
                if tgt_fill is not None:
                    raw_exit_px = tgt_fill
                    exit_reason = "target"
                    exit_fill_type = "limit"
                else:
                    # Time-based exit
                    if (i + 1 - entry_i) >= horizon:
                        raw_exit_px = float(nxt["open"])
                        exit_reason = "time"
                        exit_fill_type = "time"
                    else:
                        continue

            exit_px = _apply_costs_px(
                raw_exit_px,
                "sell",
                slippage_bps=slip_bps,
                spread_bps=spread_bps,
                spread_mode=spread_mode,
                fill_type=exit_fill_type,
                reason=exit_reason,
            )

            pnl_per_share = float(exit_px - entry_px)
            risk_per_share = float(entry_px - stop_px)
            r_mult = float(pnl_per_share / risk_per_share) if risk_per_share > min_risk_per_share else np.nan

            pnl = float(pnl_per_share * qty) if enable_position_sizing else np.nan
            equity_before = float(equity)

            if enable_position_sizing:
                # Commission assumed per round trip exit (can adjust if you prefer entry+exit)
                equity = float(equity + pnl - float(commission_per_trade))
                df_bt.loc[i + 1, "equity"] = equity

            trades.append(
                {
                    "entry_i": int(entry_i),
                    "exit_i": int(i + 1),
                    "entry_ts": ts_at(entry_i),
                    "exit_ts": ts_at(i + 1),
                    "entry_px": float(entry_px),
                    "exit_px": float(exit_px),
                    "stop_px": float(stop_px),
                    "target_px": float(target_px),
                    "reason": exit_reason,
                    "bars_held": int((i + 1) - entry_i),
                    "pnl_per_share": float(pnl_per_share),
                    "r_multiple": float(r_mult) if np.isfinite(r_mult) else np.nan,
                    "qty": int(qty) if enable_position_sizing else np.nan,
                    "pnl": float(pnl) if enable_position_sizing else np.nan,
                    "equity_before": float(equity_before) if enable_position_sizing else np.nan,
                    "equity_after": float(equity) if enable_position_sizing else np.nan,
                }
            )

            in_pos = False
            cooldown = cooldown_bars
            entry_px = np.nan
            entry_i = -1
            stop_px = np.nan
            target_px = np.nan
            qty = 0
            continue

        # ---------------------------
        # Entry logic (flat)
        # ---------------------------
        if cooldown > 0:
            continue

        # Optional regime filter (aligned by timestamp now)
        if require_risk_on and market_df is not None and not market_df.empty:
            ts = df_bt.loc[i, "timestamp"]
            if not _market_regime_on_timestamp(market_df, ts, ma_len=200, price_col="close"):
                continue

        # Indicator filters
        rsi = float(row["rsi14"])
        rvol = float(row["rvol"])
        vol_ann = float(row["vol_ann"])
        atr = float(row["atr14"])
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

        if mode_l == "pullback":
            limit_px = close - float(atr_entry) * atr
            fill = fill_limit_buy(o, l, limit_px)
            if fill is None:
                continue
            raw_entry_px = float(fill)
            entry_type = "limit"
        elif mode_l == "breakout":
            stop_buy = close + float(atr_entry) * atr
            fill = fill_stop_buy(o, h, stop_buy)
            if fill is None:
                continue
            raw_entry_px = float(fill)
            entry_type = "stop"
        else:
            raise ValueError("mode must be 'pullback' or 'breakout'")

        entry = _apply_costs_px(
            raw_entry_px,
            "buy",
            slippage_bps=slip_bps,
            spread_bps=spread_bps,
            spread_mode=spread_mode,
            fill_type=entry_type,
            reason="entry",
        )

        stop = entry - float(atr_stop) * atr
        target = entry + float(atr_target) * atr

        # Basic guard: avoid nonsensical stop/target
        if not (np.isfinite(entry) and np.isfinite(stop) and np.isfinite(target)):
            continue
        if stop >= entry:
            continue
        if target <= entry:
            continue

        risk_per_share = float(entry - stop)
        if risk_per_share <= float(min_risk_per_share):
            continue

        # Sizing (optional)
        new_qty = 1
        if enable_position_sizing:
            # Risk-based sizing + allocation cap
            eq = float(equity)
            rpct = float(risk_pct)
            apct = float(max_alloc_pct)

            if eq <= 0 or rpct <= 0 or apct <= 0:
                # fail-safe: if inputs nonsensical, just do 1 share
                new_qty = 1
            else:
                risk_budget = eq * rpct
                qty_risk = int(np.floor(risk_budget / risk_per_share))

                alloc_budget = eq * apct
                qty_alloc = int(np.floor(alloc_budget / max(1e-12, entry)))

                new_qty = int(max(0, min(qty_risk, qty_alloc)))

            if new_qty <= 0:
                continue

            # Commission assumed at entry too? (keep simple: charge on exit only by default)
            # If you want commission on entry+exit, subtract half here and half on exit.
            # equity -= commission_per_trade * 0.5

        in_pos = True
        entry_px = float(entry)
        entry_i = i + 1
        stop_px = float(stop)
        target_px = float(target)
        qty = int(new_qty)

        # Annotate df_bt with signals (nice for plotting/debug)
        df_bt.loc[entry_i, "signal"] = 1.0
        df_bt.loc[entry_i, "entry_type"] = entry_type
        df_bt.loc[entry_i, "bt_stop"] = stop_px
        df_bt.loc[entry_i, "bt_target"] = target_px
        if enable_position_sizing:
            df_bt.loc[entry_i, "qty"] = qty

    trades_df = pd.DataFrame(trades)

    # Make trades table nice even if empty
    if trades_df.empty:
        base_cols = [
            "entry_i", "exit_i", "entry_ts", "exit_ts",
            "entry_px", "exit_px", "stop_px", "target_px",
            "reason", "bars_held", "pnl_per_share", "r_multiple",
        ]
        if enable_position_sizing:
            base_cols += ["qty", "pnl", "equity_before", "equity_after"]
        trades_df = pd.DataFrame(columns=base_cols)

    return df_bt, trades_df
