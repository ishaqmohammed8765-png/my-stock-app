from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Literal

import numpy as np
import pandas as pd

from .indicators import add_indicators_inplace, market_regime_at

MIN_HIST_DAYS_DEFAULT = 50

ExitPriority = Literal["stop_first", "target_first", "worst_case"]


# ---------- Fill helpers (bar-based approximations) ----------

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


# ---------- Internal helpers ----------

def _ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    elif isinstance(out.index, pd.DatetimeIndex):
        out["timestamp"] = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out.reset_index(drop=True)
    else:
        out["timestamp"] = pd.NaT
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _to_numeric_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) < 2:
        return float("nan")
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 2:
        return float("nan")
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def _sharpe_from_returns(ret: pd.Series, periods_per_year: int = 252) -> float:
    r = pd.to_numeric(ret, errors="coerce").dropna()
    if len(r) < 2:
        return float("nan")
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return float((mu / sd) * np.sqrt(periods_per_year))


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
    px = float(px)
    side_l = str(side).lower().strip()
    spread_mode = str(spread_mode).lower().strip()
    fill_type = str(fill_type).lower().strip()
    reason = str(reason).lower().strip()

    if slippage_bps and slippage_bps > 0:
        if side_l == "buy":
            px *= (1.0 + slippage_bps / 10000.0)
        else:
            px *= (1.0 - slippage_bps / 10000.0)

    apply_spread = False
    if spread_bps and spread_bps > 0:
        if spread_mode == "always":
            apply_spread = True
        elif spread_mode == "never":
            apply_spread = False
        else:
            # taker_only
            if side_l == "buy":
                apply_spread = (fill_type == "stop") or (fill_type == "time")
            else:
                apply_spread = (reason in {"stop", "time"})

    if apply_spread:
        if side_l == "buy":
            px *= (1.0 + spread_bps / 10000.0)
        else:
            px *= (1.0 - spread_bps / 10000.0)

    return float(px)


@dataclass
class _MarketRegimeIndex:
    df: pd.DataFrame

    @classmethod
    def from_market_df(cls, market_df: pd.DataFrame) -> "_MarketRegimeIndex":
        m = _ensure_timestamp_column(market_df)
        m = _to_numeric_ohlcv(m)
        return cls(df=m)

    def risk_on_at(self, ts: pd.Timestamp, *, ma_len: int = 200, price_col: str = "close") -> bool:
        if self.df is None or self.df.empty or ts is pd.NaT:
            return True
        idx = int(self.df["timestamp"].searchsorted(ts, side="right") - 1)
        if idx < 0:
            return True
        try:
            return bool(market_regime_at(self.df, idx, ma_len=ma_len, price_col=price_col))
        except Exception:
            return True


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
    *,
    min_hist_days: int = MIN_HIST_DAYS_DEFAULT,

    # realism / execution
    slippage_bps: float = 0.0,
    spread_mode: str = "taker_only",
    commission_per_order: float = 0.0,
    charge_commission_on: str = "both",
    exit_priority: ExitPriority = "stop_first",

    # sizing
    enable_position_sizing: bool = False,
    risk_pct: float = 0.02,
    max_alloc_pct: float = 0.10,
    min_risk_per_share: float = 1e-6,

    # equity display
    mark_to_market: bool = False,

    # --- AUTOMATIC EDGE BOOSTERS (no UI needed) ---
    auto_adx_gate: bool = True,
    adx_min: float = 20.0,
    auto_trail_after_1r: bool = True,
    trail_atr_mult: float = 2.0,  # stop = max(stop, close - trail_atr_mult * atr)
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if df is None or df.empty or len(df) < int(min_hist_days):
        return {"error": "Not enough history", "df_bt": pd.DataFrame()}, pd.DataFrame()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"backtest_strategy missing required columns: {sorted(missing)}")

    horizon = int(max(1, horizon))
    cooldown_bars = int(max(0, cooldown_bars))

    mode_l = str(mode).lower().strip()
    if mode_l not in {"pullback", "breakout"}:
        raise ValueError("mode must be 'pullback' or 'breakout'")

    charge_commission_on = str(charge_commission_on).lower().strip()
    if charge_commission_on not in {"entry", "exit", "both"}:
        charge_commission_on = "both"

    exit_priority = str(exit_priority).lower().strip()
    if exit_priority not in {"stop_first", "target_first", "worst_case"}:
        exit_priority = "stop_first"

    df_bt = _ensure_timestamp_column(df)
    df_bt = _to_numeric_ohlcv(df_bt)
    if df_bt.empty or len(df_bt) < int(min_hist_days):
        return {"error": "Not enough usable bars after normalization", "df_bt": pd.DataFrame()}, pd.DataFrame()

    add_indicators_inplace(df_bt)

    needed_ind_cols = {"rsi14", "rvol", "vol_ann", "atr14"}
    missing_ind = needed_ind_cols - set(df_bt.columns)
    if missing_ind:
        raise KeyError(f"Indicators missing expected columns: {sorted(missing_ind)}")

    # Create columns used by UI/plots
    df_bt["signal"] = 0.0
    df_bt["entry_type"] = ""
    df_bt["bt_stop"] = np.nan
    df_bt["bt_target"] = np.nan
    df_bt["qty"] = np.nan
    df_bt["equity"] = np.nan

    regime_index: Optional[_MarketRegimeIndex] = None
    if require_risk_on and market_df is not None and not market_df.empty:
        regime_index = _MarketRegimeIndex.from_market_df(market_df)

    spread_bps = float(assumed_spread_bps) if include_spread_penalty else 0.0
    slip_bps = float(slippage_bps)

    equity = float(start_equity) if (np.isfinite(start_equity) and start_equity > 0) else 0.0
    df_bt.loc[0, "equity"] = equity

    in_pos = False
    entry_px = np.nan
    entry_i = -1
    stop_px = np.nan
    target_px = np.nan
    qty = 0
    cooldown = 0
    initial_risk_per_share = np.nan  # store 1R at entry

    def ts_at(i: int) -> str:
        if 0 <= i < len(df_bt):
            return str(pd.to_datetime(df_bt.loc[i, "timestamp"], utc=True, errors="coerce"))
        return ""

    def commission_due(which: str) -> float:
        if commission_per_order <= 0:
            return 0.0
        if charge_commission_on == "both":
            return float(commission_per_order)
        return float(commission_per_order) if charge_commission_on == which else 0.0

    trades: List[Dict[str, Any]] = []

    for i in range(len(df_bt) - 1):
        # Equity series display
        if in_pos and mark_to_market:
            cur_close = float(df_bt.loc[i, "close"])
            eff_qty = int(qty) if enable_position_sizing else 1
            if np.isfinite(cur_close) and np.isfinite(entry_px) and eff_qty > 0:
                df_bt.loc[i, "equity"] = float(equity + (cur_close - entry_px) * eff_qty)
            else:
                df_bt.loc[i, "equity"] = equity
        else:
            if not np.isfinite(df_bt.loc[i, "equity"]):
                df_bt.loc[i, "equity"] = equity

        if cooldown > 0:
            cooldown -= 1

        # ---------------- EXIT ----------------
        if in_pos:
            # AUTOMATIC TRAILING STOP UPDATE (based on current bar i close, affects next bar exit)
            if auto_trail_after_1r and np.isfinite(initial_risk_per_share) and initial_risk_per_share > 0:
                cur_close = float(df_bt.loc[i, "close"])
                cur_atr = float(df_bt.loc[i, "atr14"]) if "atr14" in df_bt.columns else np.nan

                # breakeven after +1R
                if np.isfinite(cur_close) and cur_close >= (entry_px + initial_risk_per_share):
                    stop_px = float(max(stop_px, entry_px))

                # ATR trail (optional)
                if np.isfinite(cur_close) and np.isfinite(cur_atr) and cur_atr > 0 and trail_atr_mult > 0:
                    trail_stop = float(cur_close - float(trail_atr_mult) * cur_atr)
                    stop_px = float(max(stop_px, trail_stop))

            nxt = df_bt.iloc[i + 1]
            o, h, l = float(nxt["open"]), float(nxt["high"]), float(nxt["low"])

            stop_hit = (l <= float(stop_px))
            target_hit = (h >= float(target_px))

            raw_exit_px: Optional[float] = None
            exit_reason = ""
            exit_fill_type = "time"

            if stop_hit and target_hit:
                if exit_priority in {"stop_first", "worst_case"}:
                    raw_exit_px = fill_stop_sell(o, l, float(stop_px))
                    exit_reason = "stop"
                    exit_fill_type = "stop"
                else:
                    raw_exit_px = fill_limit_sell(o, h, float(target_px))
                    exit_reason = "target"
                    exit_fill_type = "limit"
            elif stop_hit:
                raw_exit_px = fill_stop_sell(o, l, float(stop_px))
                exit_reason = "stop"
                exit_fill_type = "stop"
            elif target_hit:
                raw_exit_px = fill_limit_sell(o, h, float(target_px))
                exit_reason = "target"
                exit_fill_type = "limit"
            else:
                if (i + 1 - entry_i) >= horizon:
                    raw_exit_px = float(nxt["open"])
                    exit_reason = "time"
                    exit_fill_type = "time"

            if raw_exit_px is None:
                continue

            exit_px = _apply_costs_px(
                float(raw_exit_px),
                "sell",
                slippage_bps=slip_bps,
                spread_bps=spread_bps,
                spread_mode=spread_mode,
                fill_type=exit_fill_type,
                reason=exit_reason,
            )

            equity_before = float(equity)
            equity -= commission_due("exit")

            pnl_per_share = float(exit_px - entry_px)
            risk_per_share = float(entry_px - stop_px)
            r_mult = float(pnl_per_share / initial_risk_per_share) if (np.isfinite(initial_risk_per_share) and initial_risk_per_share > min_risk_per_share) else np.nan

            eff_qty = int(qty) if enable_position_sizing else 1
            pnl = float(pnl_per_share * eff_qty)
            equity = float(equity + pnl)

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
                    "qty": int(eff_qty),
                    "pnl": float(pnl),
                    "equity_before": float(equity_before),
                    "equity_after": float(equity),
                }
            )

            in_pos = False
            cooldown = cooldown_bars
            entry_px = np.nan
            entry_i = -1
            stop_px = np.nan
            target_px = np.nan
            qty = 0
            initial_risk_per_share = np.nan
            continue

        # ---------------- ENTRY ----------------
        if cooldown > 0:
            continue

        if require_risk_on and regime_index is not None:
            ts = df_bt.loc[i, "timestamp"]
            if not regime_index.risk_on_at(ts, ma_len=200, price_col="close"):
                continue

        # If ind_ready exists, use it automatically
        if "ind_ready" in df_bt.columns:
            if not bool(df_bt.loc[i, "ind_ready"]):
                continue

        row = df_bt.iloc[i]
        rsi = float(row["rsi14"])
        rvol = float(row["rvol"])
        vol_ann = float(row["vol_ann"])
        atr = float(row["atr14"])
        close = float(row["close"])

        if not np.isfinite([rsi, rvol, vol_ann, atr, close]).all():
            continue

        # Automatic ADX chop filter if available
        if auto_adx_gate and "adx14" in df_bt.columns:
            adx = float(row["adx14"])
            if np.isfinite(adx) and adx < float(adx_min):
                continue

        if rsi < float(rsi_min) or rsi > float(rsi_max):
            continue
        if rvol < float(rvol_min):
            continue
        if vol_ann > float(vol_max):
            continue

        nxt = df_bt.iloc[i + 1]
        o, h, l = float(nxt["open"]), float(nxt["high"]), float(nxt["low"])

        if mode_l == "pullback":
            limit_px = close - float(atr_entry) * atr
            fill = fill_limit_buy(o, l, float(limit_px))
            if fill is None:
                continue
            raw_entry_px = float(fill)
            entry_type = "limit"
        else:
            stop_buy = close + float(atr_entry) * atr
            fill = fill_stop_buy(o, h, float(stop_buy))
            if fill is None:
                continue
            raw_entry_px = float(fill)
            entry_type = "stop"

        entry = _apply_costs_px(
            raw_entry_px,
            "buy",
            slippage_bps=slip_bps,
            spread_bps=spread_bps,
            spread_mode=spread_mode,
            fill_type=entry_type,
            reason="entry",
        )

        stop = float(entry - float(atr_stop) * atr)
        target = float(entry + float(atr_target) * atr)

        if not np.isfinite([entry, stop, target]).all():
            continue
        if stop >= entry or target <= entry:
            continue

        risk_per_share = float(entry - stop)
        if risk_per_share <= float(min_risk_per_share):
            continue

        # Size qty before charging commission
        new_qty = 1
        if enable_position_sizing:
            eq = float(equity)
            rpct = float(risk_pct)
            apct = float(max_alloc_pct)
            if eq > 0 and rpct > 0 and apct > 0:
                risk_budget = eq * rpct
                qty_risk = int(np.floor(risk_budget / risk_per_share))

                alloc_budget = eq * apct
                qty_alloc = int(np.floor(alloc_budget / max(1e-12, entry)))

                new_qty = int(max(0, min(qty_risk, qty_alloc)))
            else:
                new_qty = 1

            if new_qty <= 0:
                continue

        equity_before = float(equity)
        equity -= commission_due("entry")

        in_pos = True
        entry_px = float(entry)
        entry_i = i + 1
        stop_px = float(stop)
        target_px = float(target)
        qty = int(new_qty)
        initial_risk_per_share = float(risk_per_share)

        df_bt.loc[entry_i, "signal"] = 1.0
        df_bt.loc[entry_i, "entry_type"] = entry_type
        df_bt.loc[entry_i, "bt_stop"] = stop_px
        df_bt.loc[entry_i, "bt_target"] = target_px
        df_bt.loc[entry_i, "qty"] = float(qty)
        df_bt.loc[entry_i, "equity"] = equity

    df_bt["equity"] = df_bt["equity"].ffill()

    trades_df = pd.DataFrame(trades)
    results: Dict[str, Any] = {"df_bt": df_bt}

    if trades_df.empty:
        results.update(
            {
                "trades": 0,
                "win_rate": float("nan"),
                "total_return": float("nan"),
                "max_drawdown": float("nan"),
                "sharpe": float("nan"),
                "avg_r_multiple": float("nan"),
                "assumptions": {
                    "mode": mode_l,
                    "horizon_bars": int(horizon),
                    "slippage_bps": float(slip_bps),
                    "spread_bps": float(spread_bps),
                    "spread_mode": str(spread_mode),
                    "commission_per_order": float(commission_per_order),
                    "commission_charged_on": str(charge_commission_on),
                    "exit_priority": str(exit_priority),
                    "position_sizing": bool(enable_position_sizing),
                    "risk_pct": float(risk_pct),
                    "max_alloc_pct": float(max_alloc_pct),
                    "mark_to_market": bool(mark_to_market),
                    "auto_adx_gate": bool(auto_adx_gate),
                    "adx_min": float(adx_min),
                    "auto_trail_after_1r": bool(auto_trail_after_1r),
                    "trail_atr_mult": float(trail_atr_mult),
                },
                "notes_for_beginners": [
                    "No trades were taken. With ADX gate on, it may be filtering out choppy markets.",
                    "Try widening RSI range or lowering RVOL min if trade count is too low.",
                ],
            }
        )
        return results, pd.DataFrame(columns=[
            "entry_i","exit_i","entry_ts","exit_ts","entry_px","exit_px","stop_px","target_px",
            "reason","bars_held","pnl_per_share","r_multiple","qty","pnl","equity_before","equity_after"
        ])

    pnlps = pd.to_numeric(trades_df["pnl_per_share"], errors="coerce")
    win_rate = float((pnlps > 0).sum()) / max(1, len(pnlps))

    eq0 = float(start_equity) if (np.isfinite(start_equity) and start_equity > 0) else float("nan")
    eqN = float(df_bt["equity"].iloc[-1]) if len(df_bt) else float("nan")
    total_return = (eqN / eq0 - 1.0) if np.isfinite(eq0) and eq0 > 0 and np.isfinite(eqN) else float("nan")

    max_dd = _max_drawdown(df_bt["equity"])
    eq = pd.to_numeric(df_bt["equity"], errors="coerce")
    eq_ret = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = _sharpe_from_returns(eq_ret, periods_per_year=252)
    avg_r = float(pd.to_numeric(trades_df["r_multiple"], errors="coerce").dropna().mean())

    results.update(
        {
            "trades": int(len(trades_df)),
            "win_rate": float(win_rate),
            "total_return": float(total_return) if np.isfinite(total_return) else float("nan"),
            "max_drawdown": float(max_dd) if np.isfinite(max_dd) else float("nan"),
            "sharpe": float(sharpe) if np.isfinite(sharpe) else float("nan"),
            "avg_r_multiple": float(avg_r) if np.isfinite(avg_r) else float("nan"),
            "assumptions": {
                "mode": mode_l,
                "horizon_bars": int(horizon),
                "slippage_bps": float(slip_bps),
                "spread_bps": float(spread_bps),
                "spread_mode": str(spread_mode),
                "commission_per_order": float(commission_per_order),
                "commission_charged_on": str(charge_commission_on),
                "exit_priority": str(exit_priority),
                "position_sizing": bool(enable_position_sizing),
                "risk_pct": float(risk_pct),
                "max_alloc_pct": float(max_alloc_pct),
                "mark_to_market": bool(mark_to_market),
                "auto_adx_gate": bool(auto_adx_gate),
                "adx_min": float(adx_min),
                "auto_trail_after_1r": bool(auto_trail_after_1r),
                "trail_atr_mult": float(trail_atr_mult),
            },
        }
    )

    col_order = [
        "entry_i","exit_i","entry_ts","exit_ts",
        "entry_px","exit_px","stop_px","target_px",
        "reason","bars_held","pnl_per_share","r_multiple",
        "qty","pnl","equity_before","equity_after"
    ]
    trades_df = trades_df[[c for c in col_order if c in trades_df.columns]]
    return results, trades_df
