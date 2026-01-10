# utils/backtester.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .indicators import add_indicators_inplace, market_regime_at

MIN_HIST_DAYS_DEFAULT = 50

ExitPriority = Literal["stop_first", "target_first", "worst_case"]
SpreadMode = Literal["taker_only", "always", "never"]
CommissionChargeOn = Literal["entry", "exit", "both"]
TimeExitPrice = Literal["open", "close"]
GateMode = Literal["hard", "soft"]  # hard = skip trades, soft = size-down / score-down


# =============================================================================
# Fill helpers (bar-based approximations)
# =============================================================================

def fill_limit_buy(open_px: float, low_px: float, limit_px: float) -> Optional[float]:
    """Limit buy fills if bar low trades through limit. Fill assumed at limit price."""
    return float(limit_px) if low_px <= limit_px else None


def fill_stop_buy(open_px: float, high_px: float, stop_px: float) -> Optional[float]:
    """
    Stop buy fills if bar high crosses stop.
    If gap up above stop, assume fill at open; else at stop.
    """
    if high_px >= stop_px:
        return float(open_px) if open_px > stop_px else float(stop_px)
    return None


def fill_limit_sell(open_px: float, high_px: float, limit_px: float) -> Optional[float]:
    """Limit sell fills if bar high trades through limit. Fill assumed at limit price."""
    return float(limit_px) if high_px >= limit_px else None


def fill_stop_sell(open_px: float, low_px: float, stop_px: float) -> Optional[float]:
    """
    Stop sell fills if bar low crosses stop.
    If gap down below stop, assume fill at open; else at stop.
    """
    if low_px <= stop_px:
        return float(open_px) if open_px < stop_px else float(stop_px)
    return None


# =============================================================================
# Data prep helpers
# =============================================================================

def _ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a normalized UTC `timestamp` column, sorted ascending."""
    out = df.copy()

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    elif isinstance(out.index, pd.DatetimeIndex):
        out["timestamp"] = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out.reset_index(drop=True)
    else:
        out["timestamp"] = pd.NaT

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


def _to_numeric_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce OHLCV to numeric floats; invalid values become NaN."""
    out = df.copy()
    for c in ("open", "high", "low", "close", "volume"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# =============================================================================
# Costs / execution helpers
# =============================================================================

def _apply_costs_px(
    px: float,
    side: Literal["buy", "sell"],
    *,
    slippage_bps: float,
    spread_bps: float,
    spread_mode: SpreadMode,
    fill_type: Literal["limit", "stop", "time"],
) -> float:
    """
    Apply slippage + spread penalties to a fill price (always adverse).

    Beginner note:
      - Slippage: pay a bit worse than the raw fill price.
      - Spread: taker (market/stop/time) pays spread, maker (limit) often doesn't.
    """
    px = float(px)
    side_l = str(side).lower().strip()
    fill_type_l = str(fill_type).lower().strip()
    spread_mode_l = str(spread_mode).lower().strip()

    # 1) Slippage: always adverse
    if slippage_bps and slippage_bps > 0:
        if side_l == "buy":
            px *= (1.0 + slippage_bps / 10000.0)
        else:
            px *= (1.0 - slippage_bps / 10000.0)

    # 2) Spread: optionally applied
    apply_spread = False
    if spread_bps and spread_bps > 0:
        if spread_mode_l == "always":
            apply_spread = True
        elif spread_mode_l == "never":
            apply_spread = False
        else:
            # taker_only (default): apply spread only to taker-like fills
            apply_spread = (fill_type_l in {"stop", "time"})

    if apply_spread:
        if side_l == "buy":
            px *= (1.0 + spread_bps / 10000.0)
        else:
            px *= (1.0 - spread_bps / 10000.0)

    return float(px)


def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) < 2:
        return float("nan")
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _sharpe_from_returns(ret: pd.Series, periods_per_year: int = 252) -> float:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if len(ret) < 2:
        return float("nan")
    mu = float(ret.mean())
    sd = float(ret.std(ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return float((mu / sd) * np.sqrt(periods_per_year))


# =============================================================================
# Market regime helper (optional filter)
# =============================================================================

@dataclass
class _MarketRegimeIndex:
    df: pd.DataFrame

    @classmethod
    def from_market_df(cls, market_df: pd.DataFrame) -> "_MarketRegimeIndex":
        m = _ensure_timestamp_column(market_df)
        m = _to_numeric_ohlcv(m)
        return cls(df=m)

    def risk_on_at(self, ts: pd.Timestamp, *, ma_len: int = 200, price_col: str = "close") -> bool:
        # If anything is missing, default to "risk-on" so we don't accidentally block everything.
        if self.df is None or self.df.empty or ts is pd.NaT:
            return True

        idx = int(self.df["timestamp"].searchsorted(ts, side="right") - 1)
        if idx < 0:
            return True

        try:
            return bool(market_regime_at(self.df, idx, ma_len=ma_len, price_col=price_col))
        except Exception:
            return True


# =============================================================================
# Probability gating buckets (simple and beginner-friendly)
# =============================================================================

def _trend_bucket(row: pd.Series) -> str:
    c = float(row["close"])
    m50 = float(row["ma50"])
    m200 = float(row["ma200"])
    if np.isfinite(c) and np.isfinite(m50) and np.isfinite(m200):
        if (c > m50) and (m50 > m200):
            return "Up"
        if (c < m50) and (m50 < m200):
            return "Down"
    return "Mixed"


def _rsi_bucket(rsi: float) -> str:
    if not np.isfinite(rsi):
        return "NA"
    if rsi < 35:
        return "Low"
    if rsi > 65:
        return "High"
    return "Mid"


def _bucket_key(row: pd.Series) -> str:
    # 9 buckets total: Trend (Up/Down/Mixed) x RSI (Low/Mid/High)
    return "|".join([_trend_bucket(row), _rsi_bucket(float(row["rsi14"]))])


def _choose_entry_fill(
    *,
    mode_l: str,
    close: float,
    atr: float,
    atr_entry: float,
    next_open: float,
    next_high: float,
    next_low: float,
) -> Tuple[Optional[float], str]:
    """
    Decide entry fill price on the next bar.
    Returns (raw_entry_px, entry_fill_type) where fill_type ∈ {"limit","stop"}.
    """
    if mode_l == "pullback":
        limit_px = float(close - float(atr_entry) * float(atr))
        fill = fill_limit_buy(next_open, next_low, limit_px)
        return (float(fill) if fill is not None else None), "limit"

    # breakout
    stop_px = float(close + float(atr_entry) * float(atr))
    fill = fill_stop_buy(next_open, next_high, stop_px)
    return (float(fill) if fill is not None else None), "stop"


def _choose_exit_fill(
    *,
    next_open: float,
    next_high: float,
    next_low: float,
    stop_px: float,
    target_px: float,
    bars_held: int,
    horizon: int,
    exit_priority: ExitPriority,
    time_exit_price: TimeExitPrice,
    next_close: float,
) -> Tuple[Optional[float], str]:
    """
    Decide exit fill price on next bar.
    Returns (raw_exit_px, exit_fill_type) where fill_type ∈ {"limit","stop","time"}.
    """
    stop_hit = (next_low <= float(stop_px))
    target_hit = (next_high >= float(target_px))

    if stop_hit and target_hit:
        if exit_priority in {"stop_first", "worst_case"}:
            raw = fill_stop_sell(next_open, next_low, float(stop_px))
            return (float(raw) if raw is not None else float(stop_px)), "stop"
        raw = fill_limit_sell(next_open, next_high, float(target_px))
        return (float(raw) if raw is not None else float(target_px)), "limit"

    if stop_hit:
        raw = fill_stop_sell(next_open, next_low, float(stop_px))
        return (float(raw) if raw is not None else float(stop_px)), "stop"

    if target_hit:
        raw = fill_limit_sell(next_open, next_high, float(target_px))
        return (float(raw) if raw is not None else float(target_px)), "limit"

    # Time exit
    if bars_held >= int(horizon):
        if time_exit_price == "close":
            return float(next_close), "time"
        return float(next_open), "time"

    return None, "time"


def _simulate_outcome_r_multiple(
    df: pd.DataFrame,
    *,
    entry_i: int,
    entry_px: float,
    stop_px: float,
    target_px: float,
    horizon: int,
    exit_priority: ExitPriority,
    time_exit_price: TimeExitPrice,
) -> Tuple[str, float]:
    """
    Used for bucket stats (in-sample learning).
    Returns (reason, r_multiple) where reason ∈ {"target","stop","time","none"}.
    Uses RAW prices (no costs) because it’s just estimating setup quality.
    """
    n = len(df)
    if entry_i < 0 or entry_i >= n:
        return "none", float("nan")

    risk = float(entry_px - stop_px)
    if not np.isfinite(risk) or risk <= 1e-12:
        return "none", float("nan")

    last_i = min(n - 1, entry_i + int(horizon))
    for j in range(entry_i, last_i + 1):
        bar = df.iloc[j]
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])

        raw_exit, fill_type = _choose_exit_fill(
            next_open=o,
            next_high=h,
            next_low=l,
            stop_px=stop_px,
            target_px=target_px,
            bars_held=(j - entry_i),
            horizon=horizon,
            exit_priority=exit_priority,
            time_exit_price=time_exit_price,
            next_close=c,
        )
        if raw_exit is None:
            continue

        if fill_type == "stop":
            reason = "stop"
        elif fill_type == "limit":
            reason = "target"
        else:
            reason = "time"

        r = (float(raw_exit) - entry_px) / risk
        return reason, float(r)

    return "none", float("nan")


def _build_bucket_stats(
    df: pd.DataFrame,
    *,
    mode_l: str,
    atr_entry: float,
    atr_stop: float,
    atr_target: float,
    rsi_min: float,
    rsi_max: float,
    rvol_min: float,
    vol_max: float,
    horizon: int,
    exit_priority: ExitPriority,
    time_exit_price: TimeExitPrice,
    is_end_i: int,
) -> Dict[str, Dict[str, float]]:
    """
    Build in-sample bucket stats from rows [0 .. is_end_i).
    Each stat contains: n, p_win, avg_r
    """
    rs_by_key: Dict[str, List[float]] = {}
    wins: Dict[str, int] = {}
    counts: Dict[str, int] = {}

    end = int(max(2, min(is_end_i, len(df) - 1)))

    for i in range(end - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]

        rsi = float(row["rsi14"])
        rvol = float(row["rvol"])
        vol_ann = float(row["vol_ann"])
        atr = float(row["atr14"])
        close = float(row["close"])

        if not (np.isfinite(rsi) and np.isfinite(rvol) and np.isfinite(vol_ann) and np.isfinite(atr) and np.isfinite(close)):
            continue

        # same base filters as strategy
        if rsi < float(rsi_min) or rsi > float(rsi_max):
            continue
        if rvol < float(rvol_min):
            continue
        if vol_ann > float(vol_max):
            continue

        o = float(nxt["open"])
        h = float(nxt["high"])
        l = float(nxt["low"])

        raw_entry, _ = _choose_entry_fill(
            mode_l=mode_l,
            close=close,
            atr=atr,
            atr_entry=float(atr_entry),
            next_open=o,
            next_high=h,
            next_low=l,
        )
        if raw_entry is None:
            continue

        entry_px = float(raw_entry)
        stop_px = float(entry_px - float(atr_stop) * atr)
        target_px = float(entry_px + float(atr_target) * atr)

        if not (np.isfinite(stop_px) and np.isfinite(target_px)):
            continue
        if stop_px >= entry_px or target_px <= entry_px:
            continue

        reason, r_mult = _simulate_outcome_r_multiple(
            df,
            entry_i=i + 1,
            entry_px=entry_px,
            stop_px=stop_px,
            target_px=target_px,
            horizon=int(horizon),
            exit_priority=exit_priority,
            time_exit_price=time_exit_price,
        )
        if reason == "none" or not np.isfinite(r_mult):
            continue

        key = _bucket_key(row)
        counts[key] = counts.get(key, 0) + 1
        rs_by_key.setdefault(key, []).append(float(r_mult))
        if reason == "target":
            wins[key] = wins.get(key, 0) + 1

    out: Dict[str, Dict[str, float]] = {}
    for key, rs in rs_by_key.items():
        n = int(counts.get(key, 0))
        if n <= 0:
            continue
        p_win = float(wins.get(key, 0)) / float(n)
        avg_r = float(np.mean(rs)) if rs else float("nan")
        out[key] = {"n": float(n), "p_win": float(p_win), "avg_r": float(avg_r)}
    return out


def _gate_multiplier(
    row: pd.Series,
    bucket_stats: Dict[str, Dict[str, float]],
    *,
    prob_min: float,
    min_bucket_trades: int,
    min_avg_r: float,
) -> Tuple[float, str]:
    """
    Returns (multiplier, reason).

    Multiplier meaning (for soft gating):
      - 0.0 means "skip completely"
      - 0.25 means "take but 25% size"
      - 1.0 means "no penalty"

    For hard gating, caller can treat multiplier == 0.0 as skip.
    """
    key = _bucket_key(row)
    stt = bucket_stats.get(key)
    if stt is None:
        return 0.0, "unknown_bucket"

    n = float(stt.get("n", 0.0))
    p = float(stt.get("p_win", np.nan))
    ar = float(stt.get("avg_r", np.nan))

    if n < float(min_bucket_trades):
        return 0.0, "low_n"
    if (not np.isfinite(p)) or (p < float(prob_min)):
        # soft penalty for weak win rate
        return 0.25, "low_p"
    if (not np.isfinite(ar)) or (ar < float(min_avg_r)):
        # soft penalty for weak average outcome
        return 0.25, "low_avg_r"

    return 1.0, "ok"


# =============================================================================
# Backtest
# =============================================================================

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

    # --- realism / execution ---
    slippage_bps: float = 0.0,
    spread_mode: SpreadMode = "taker_only",
    commission_per_order: float = 0.0,
    charge_commission_on: CommissionChargeOn = "both",
    exit_priority: ExitPriority = "stop_first",
    time_exit_price: TimeExitPrice = "open",

    # --- sizing ---
    enable_position_sizing: bool = False,
    risk_pct: float = 0.02,
    max_alloc_pct: float = 0.10,
    min_risk_per_share: float = 1e-6,

    # --- accounting model ---
    use_cash_ledger: bool = False,   # NEW: if True, buys consume cash; sells restore cash
    allow_margin: bool = False,      # if cash ledger on: allow cash to go negative or not

    # --- equity display only ---
    mark_to_market: bool = False,

    # --- probability gating ---
    prob_gating: bool = True,
    prob_is_frac: float = 0.85,
    prob_min: float = 0.50,
    min_bucket_trades: int = 8,
    min_avg_r: float = -0.05,
    gate_mode: GateMode = "hard",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Beginner-friendly next-bar execution backtest (pullback or breakout).

    Key features:
      - Next-bar fills with limit/stop approximations
      - Optional slippage/spread costs
      - Optional commissions
      - Optional position sizing (risk cap + allocation cap)
      - Optional cash ledger (more realistic for small accounts)
      - Optional market regime filter (risk-on only)
      - Optional probability gating (hard or soft)

    IMPORTANT:
      - This is educational and simplified. Daily bars cannot capture intraday path perfectly.
    """
    if df is None or df.empty or len(df) < int(min_hist_days):
        return {"error": "Not enough history", "df_bt": pd.DataFrame()}, pd.DataFrame()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"backtest_strategy missing required columns: {sorted(missing)}")

    horizon = int(max(1, horizon))
    cooldown_bars = int(max(0, cooldown_bars))
    slip_bps = float(slippage_bps)
    spread_bps = float(assumed_spread_bps) if include_spread_penalty else 0.0

    mode_l = str(mode).lower().strip()
    if mode_l not in {"pullback", "breakout"}:
        raise ValueError("mode must be 'pullback' or 'breakout'")

    spread_mode = str(spread_mode).lower().strip()  # type: ignore[assignment]
    if spread_mode not in {"taker_only", "always", "never"}:
        spread_mode = "taker_only"  # type: ignore[assignment]

    charge_commission_on = str(charge_commission_on).lower().strip()  # type: ignore[assignment]
    if charge_commission_on not in {"entry", "exit", "both"}:
        charge_commission_on = "both"  # type: ignore[assignment]

    exit_priority = str(exit_priority).lower().strip()  # type: ignore[assignment]
    if exit_priority not in {"stop_first", "target_first", "worst_case"}:
        exit_priority = "stop_first"  # type: ignore[assignment]

    time_exit_price = str(time_exit_price).lower().strip()  # type: ignore[assignment]
    if time_exit_price not in {"open", "close"}:
        time_exit_price = "open"  # type: ignore[assignment]

    gate_mode = str(gate_mode).lower().strip()  # type: ignore[assignment]
    if gate_mode not in {"hard", "soft"}:
        gate_mode = "hard"  # type: ignore[assignment]

    df_bt = _ensure_timestamp_column(df)
    df_bt = _to_numeric_ohlcv(df_bt)

    if df_bt.empty or len(df_bt) < int(min_hist_days):
        return {"error": "Not enough usable bars after timestamp normalization", "df_bt": pd.DataFrame()}, pd.DataFrame()

    # Add indicators in-place
    add_indicators_inplace(df_bt)
    needed_ind_cols = {"rsi14", "rvol", "vol_ann", "atr14", "ma50", "ma200"}
    missing_ind = needed_ind_cols - set(df_bt.columns)
    if missing_ind:
        raise KeyError(
            f"Indicators missing expected columns: {sorted(missing_ind)}. "
            f"Check add_indicators_inplace()."
        )

    # Market regime index (optional)
    regime_index: Optional[_MarketRegimeIndex] = None
    if require_risk_on and market_df is not None and not market_df.empty:
        regime_index = _MarketRegimeIndex.from_market_df(market_df)

    # Probability gating stats (optional)
    bucket_stats: Dict[str, Dict[str, float]] = {}
    if prob_gating:
        is_end_i = int(max(10, min(len(df_bt) - 2, int(len(df_bt) * float(prob_is_frac)))))
        bucket_stats = _build_bucket_stats(
            df_bt,
            mode_l=mode_l,
            atr_entry=float(atr_entry),
            atr_stop=float(atr_stop),
            atr_target=float(atr_target),
            rsi_min=float(rsi_min),
            rsi_max=float(rsi_max),
            rvol_min=float(rvol_min),
            vol_max=float(vol_max),
            horizon=int(horizon),
            exit_priority=exit_priority,  # type: ignore[arg-type]
            time_exit_price=time_exit_price,  # type: ignore[arg-type]
            is_end_i=is_end_i,
        )

    # Output columns
    df_bt["signal"] = 0.0
    df_bt["entry_type"] = ""
    df_bt["bt_stop"] = np.nan
    df_bt["bt_target"] = np.nan
    df_bt["qty"] = np.nan
    df_bt["equity"] = np.nan
    df_bt["cash"] = np.nan  # only meaningful when use_cash_ledger True

    trades: List[Dict[str, Any]] = []

    # Accounting state
    equity = float(start_equity) if (np.isfinite(start_equity) and start_equity > 0) else 0.0
    cash = float(equity)  # cash ledger starts as full equity

    df_bt.loc[0, "equity"] = equity
    df_bt.loc[0, "cash"] = cash

    # Position state
    in_pos = False
    entry_px = np.nan
    entry_i = -1
    stop_px = np.nan
    target_px = np.nan
    qty = 0
    cooldown = 0

    # Gating counters (useful to show in UI)
    gated_total = 0
    gated_unknown = 0
    gated_low_n = 0
    gated_low_p = 0
    gated_low_r = 0

    # Helper: timestamp string
    def ts_at(i: int) -> str:
        if 0 <= i < len(df_bt):
            return str(pd.to_datetime(df_bt.loc[i, "timestamp"], utc=True, errors="coerce"))
        return ""

    def can_pay_commission(which: CommissionChargeOn) -> bool:
        if commission_per_order <= 0:
            return True
        if not use_cash_ledger:
            # In "equity-only" mode, we still prevent going negative for beginner sanity
            return (equity - float(commission_per_order)) >= 0.0
        # In cash-ledger mode, commission is paid from cash
        if allow_margin:
            return True
        return (cash - float(commission_per_order)) >= 0.0

    def charge_commission(which: CommissionChargeOn) -> bool:
        """
        Returns True if commission charged, False if not possible (insufficient funds).
        """
        nonlocal equity, cash

        if commission_per_order <= 0:
            return True

        do_charge = (charge_commission_on == "both") or (charge_commission_on == which)
        if not do_charge:
            return True

        if not can_pay_commission(which):
            return False

        if use_cash_ledger:
            cash -= float(commission_per_order)
            # equity will be updated later (equity = cash + position_value)
        else:
            equity -= float(commission_per_order)

        return True

    def update_equity_mark_to_market(i: int) -> None:
        """
        Update df_bt equity/cash display at row i.
        In equity-only mode: equity just carries forward (or marks MTM if enabled).
        In cash-ledger mode: equity = cash + position_value (if mark_to_market), else cash + realized only.
        """
        nonlocal equity, cash

        if use_cash_ledger:
            df_bt.loc[i, "cash"] = float(cash)

            if in_pos and mark_to_market:
                cur_close = float(df_bt.loc[i, "close"])
                pos_val = float(cur_close * int(qty)) if (np.isfinite(cur_close) and qty > 0) else 0.0
                df_bt.loc[i, "equity"] = float(cash + pos_val)
            else:
                # If not MTM, equity shown as cash + (realized PnL already embedded in cash)
                df_bt.loc[i, "equity"] = float(cash)
        else:
            # equity-only mode
            if in_pos and mark_to_market:
                cur_close = float(df_bt.loc[i, "close"])
                eff_qty = int(qty) if enable_position_sizing else 1
                if np.isfinite(cur_close) and np.isfinite(entry_px) and eff_qty > 0:
                    df_bt.loc[i, "equity"] = float(equity + (cur_close - entry_px) * eff_qty)
                else:
                    df_bt.loc[i, "equity"] = float(equity)
            else:
                if not np.isfinite(df_bt.loc[i, "equity"]):
                    df_bt.loc[i, "equity"] = float(equity)

    # Main loop: we trade using next bar fills, so iterate to len-2 at most for entries
    for i in range(len(df_bt) - 1):
        update_equity_mark_to_market(i)

        if cooldown > 0:
            cooldown -= 1

        # ---------------------------------------------------------
        # EXIT (we check next bar OHLC and decide if stop/target/time)
        # ---------------------------------------------------------
        if in_pos:
            nxt = df_bt.iloc[i + 1]
            o = float(nxt["open"])
            h = float(nxt["high"])
            l = float(nxt["low"])
            c = float(nxt["close"])

            bars_held = int((i + 1) - entry_i)
            raw_exit, exit_fill_type = _choose_exit_fill(
                next_open=o,
                next_high=h,
                next_low=l,
                stop_px=float(stop_px),
                target_px=float(target_px),
                bars_held=bars_held,
                horizon=int(horizon),
                exit_priority=exit_priority,  # type: ignore[arg-type]
                time_exit_price=time_exit_price,  # type: ignore[arg-type]
                next_close=c,
            )
            if raw_exit is None:
                continue

            exit_px = _apply_costs_px(
                float(raw_exit),
                "sell",
                slippage_bps=slip_bps,
                spread_bps=spread_bps,
                spread_mode=spread_mode,  # type: ignore[arg-type]
                fill_type=exit_fill_type,  # type: ignore[arg-type]
            )

            # Commission (exit)
            if not charge_commission("exit"):
                # If you can't afford commission, we still exit but clamp for beginner sanity
                # (Real brokers would reject the order or liquidate; we keep it simple.)
                pass

            equity_before = float(df_bt.loc[i, "equity"]) if np.isfinite(df_bt.loc[i, "equity"]) else float(equity)
            eff_qty = int(qty) if enable_position_sizing else 1

            pnl_per_share = float(exit_px - float(entry_px))
            pnl = float(pnl_per_share * eff_qty)

            risk_per_share = float(float(entry_px) - float(stop_px))
            r_mult = float(pnl_per_share / risk_per_share) if risk_per_share > float(min_risk_per_share) else np.nan

            # Update accounting
            if use_cash_ledger:
                # Sell returns cash
                cash += float(exit_px) * eff_qty
                # PnL is implicitly realized in cash (because we paid entry cost earlier)
                # Equity will be updated by update_equity_mark_to_market
            else:
                equity = float(equity + pnl)

            # Write equity on exit bar
            if use_cash_ledger:
                df_bt.loc[i + 1, "cash"] = float(cash)
                df_bt.loc[i + 1, "equity"] = float(cash)  # if MTM, will update next iteration
            else:
                df_bt.loc[i + 1, "equity"] = float(equity)

            # Determine reason label
            if exit_fill_type == "stop":
                reason = "stop"
            elif exit_fill_type == "limit":
                reason = "target"
            else:
                reason = "time"

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
                    "reason": str(reason),
                    "bars_held": int(bars_held),
                    "pnl_per_share": float(pnl_per_share),
                    "r_multiple": float(r_mult) if np.isfinite(r_mult) else np.nan,
                    "qty": int(eff_qty),
                    "pnl": float(pnl),
                    "equity_before": float(equity_before),
                    "equity_after": float(df_bt.loc[i + 1, "equity"]),
                }
            )

            # Reset position state
            in_pos = False
            cooldown = int(cooldown_bars)
            entry_px = np.nan
            entry_i = -1
            stop_px = np.nan
            target_px = np.nan
            qty = 0
            continue

        # ---------------------------------------------------------
        # ENTRY
        # ---------------------------------------------------------
        if cooldown > 0:
            continue

        # Market regime filter (optional)
        if require_risk_on and regime_index is not None:
            ts = df_bt.loc[i, "timestamp"]
            if not regime_index.risk_on_at(ts, ma_len=200, price_col="close"):
                continue

        row = df_bt.iloc[i]
        rsi = float(row["rsi14"])
        rvol = float(row["rvol"])
        vol_ann = float(row["vol_ann"])
        atr = float(row["atr14"])
        close = float(row["close"])

        if not (np.isfinite(rsi) and np.isfinite(rvol) and np.isfinite(vol_ann) and np.isfinite(atr) and np.isfinite(close)):
            continue

        if rsi < float(rsi_min) or rsi > float(rsi_max):
            continue
        if rvol < float(rvol_min):
            continue
        if vol_ann > float(vol_max):
            continue

        # Probability gating (optional)
        gate_mult = 1.0
        gate_reason = "ok"
        if prob_gating:
            gm, gr = _gate_multiplier(
                row,
                bucket_stats,
                prob_min=float(prob_min),
                min_bucket_trades=int(min_bucket_trades),
                min_avg_r=float(min_avg_r),
            )
            gate_mult, gate_reason = float(gm), str(gr)

            # Count gate outcomes (for reporting)
            if gate_mult <= 0.0:
                gated_total += 1
                if gate_reason == "unknown_bucket":
                    gated_unknown += 1
                elif gate_reason == "low_n":
                    gated_low_n += 1
                elif gate_reason == "low_p":
                    gated_low_p += 1
                elif gate_reason == "low_avg_r":
                    gated_low_r += 1

            if gate_mode == "hard" and gate_mult <= 0.0:
                continue

        # Look at next bar for entry fill
        nxt = df_bt.iloc[i + 1]
        o = float(nxt["open"])
        h = float(nxt["high"])
        l = float(nxt["low"])

        raw_entry, entry_fill_type = _choose_entry_fill(
            mode_l=mode_l,
            close=close,
            atr=atr,
            atr_entry=float(atr_entry),
            next_open=o,
            next_high=h,
            next_low=l,
        )
        if raw_entry is None:
            continue

        entry_px_eff = _apply_costs_px(
            float(raw_entry),
            "buy",
            slippage_bps=slip_bps,
            spread_bps=spread_bps,
            spread_mode=spread_mode,  # type: ignore[arg-type]
            fill_type=entry_fill_type,  # type: ignore[arg-type]
        )

        stop = float(entry_px_eff - float(atr_stop) * atr)
        target = float(entry_px_eff + float(atr_target) * atr)

        if not (np.isfinite(entry_px_eff) and np.isfinite(stop) and np.isfinite(target)):
            continue
        if stop >= entry_px_eff or target <= entry_px_eff:
            continue

        risk_per_share = float(entry_px_eff - stop)
        if risk_per_share <= float(min_risk_per_share):
            continue

        # Choose quantity (default 1 share), then apply gating as size-down if soft mode
        new_qty = 1
        if enable_position_sizing:
            # Base sizing by risk + allocation
            account_base = float(cash) if use_cash_ledger else float(equity)
            if account_base <= 0:
                continue

            rpct = float(risk_pct)
            apct = float(max_alloc_pct)

            risk_budget = account_base * max(0.0, rpct)
            qty_risk = int(np.floor(risk_budget / risk_per_share))

            alloc_budget = account_base * max(0.0, apct)
            qty_alloc = int(np.floor(alloc_budget / max(1e-12, entry_px_eff)))

            new_qty = int(max(0, min(qty_risk, qty_alloc)))
        else:
            new_qty = 1

        # Apply soft gating size-down (only if sizing is enabled; otherwise you still get 1 share)
        if prob_gating and gate_mode == "soft":
            if enable_position_sizing:
                new_qty = int(np.floor(new_qty * max(0.0, min(1.0, gate_mult))))
            else:
                # If not sizing, you can still choose to skip unknown buckets in soft mode
                if gate_mult <= 0.0:
                    continue

        if new_qty <= 0:
            continue

        # Commission (entry) - must be affordable
        if not charge_commission("entry"):
            # Can't pay commission => skip trade (beginner-friendly)
            continue

        # Cash ledger: buy consumes cash
        if use_cash_ledger:
            cost = float(entry_px_eff) * int(new_qty)
            if (not allow_margin) and (cash - cost) < 0.0:
                # Can't afford the shares
                continue
            cash -= cost

        # Enter position
        in_pos = True
        entry_px = float(entry_px_eff)
        entry_i = int(i + 1)
        stop_px = float(stop)
        target_px = float(target)
        qty = int(new_qty)

        df_bt.loc[entry_i, "signal"] = 1.0
        df_bt.loc[entry_i, "entry_type"] = str(entry_fill_type)
        df_bt.loc[entry_i, "bt_stop"] = float(stop_px)
        df_bt.loc[entry_i, "bt_target"] = float(target_px)
        df_bt.loc[entry_i, "qty"] = float(qty)

        # Update equity/cash display at entry bar
        if use_cash_ledger:
            df_bt.loc[entry_i, "cash"] = float(cash)
            df_bt.loc[entry_i, "equity"] = float(cash)  # MTM updates on next iterations
        else:
            df_bt.loc[entry_i, "equity"] = float(equity)

    # Final fill-forward
    df_bt["equity"] = df_bt["equity"].ffill()
    df_bt["cash"] = df_bt["cash"].ffill()

    trades_df = pd.DataFrame(trades)
    results: Dict[str, Any] = {"df_bt": df_bt}

    gating_info = {
        "enabled": bool(prob_gating),
        "mode": str(gate_mode),
        "is_frac": float(prob_is_frac),
        "prob_min": float(prob_min),
        "min_bucket_trades": int(min_bucket_trades),
        "min_avg_r": float(min_avg_r),
        "gated_total": int(gated_total),
        "gated_unknown_bucket": int(gated_unknown),
        "gated_low_n": int(gated_low_n),
        "gated_low_p": int(gated_low_p),
        "gated_low_avg_r": int(gated_low_r),
        "buckets_learned": int(len(bucket_stats)),
    }

    # Compute summary stats
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
                    "time_exit_price": str(time_exit_price),
                    "position_sizing": bool(enable_position_sizing),
                    "risk_pct": float(risk_pct),
                    "max_alloc_pct": float(max_alloc_pct),
                    "use_cash_ledger": bool(use_cash_ledger),
                    "allow_margin": bool(allow_margin),
                    "mark_to_market": bool(mark_to_market),
                    "probability_gating": gating_info,
                },
                "notes_for_beginners": [
                    "No trades were taken. This can happen if filters or gating are strict for this ticker/time period.",
                    "Try: widen RSI filter, lower rvol_min, raise vol_max, reduce min_bucket_trades, or switch gate_mode to 'soft'.",
                ],
            }
        )
        return results, pd.DataFrame(
            columns=[
                "entry_i", "exit_i", "entry_ts", "exit_ts", "entry_px", "exit_px", "stop_px", "target_px",
                "reason", "bars_held", "pnl_per_share", "r_multiple", "qty", "pnl", "equity_before", "equity_after",
            ]
        )

    pnlps = pd.to_numeric(trades_df["pnl_per_share"], errors="coerce")
    win_rate = float((pnlps > 0).sum()) / max(1, len(pnlps))

    eq0 = float(start_equity) if (np.isfinite(start_equity) and start_equity > 0) else float("nan")
    eqN = float(df_bt["equity"].iloc[-1]) if len(df_bt) else float("nan")
    total_return = (eqN / eq0 - 1.0) if np.isfinite(eq0) and eq0 > 0 and np.isfinite(eqN) else float("nan")

    max_dd = _max_drawdown(pd.to_numeric(df_bt["equity"], errors="coerce"))
    eq_series = pd.to_numeric(df_bt["equity"], errors="coerce")
    eq_ret = eq_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
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
                "time_exit_price": str(time_exit_price),
                "position_sizing": bool(enable_position_sizing),
                "risk_pct": float(risk_pct),
                "max_alloc_pct": float(max_alloc_pct),
                "use_cash_ledger": bool(use_cash_ledger),
                "allow_margin": bool(allow_margin),
                "mark_to_market": bool(mark_to_market),
                "probability_gating": gating_info,
            },
        }
    )

    col_order = [
        "entry_i", "exit_i", "entry_ts", "exit_ts",
        "entry_px", "exit_px", "stop_px", "target_px",
        "reason", "bars_held", "pnl_per_share", "r_multiple",
        "qty", "pnl", "equity_before", "equity_after",
    ]
    trades_df = trades_df[[c for c in col_order if c in trades_df.columns]]
    return results, trades_df
