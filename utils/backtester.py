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
GateMode = Literal["hard", "soft"]  # hard = skip trades, soft = size-down weak setups
SizingMode = Literal["percent", "fixed_amount"]


# =============================================================================
# Fill helpers (bar-based approximations)
# =============================================================================

def fill_limit_buy(open_px: float, low_px: float, limit_px: float) -> Optional[float]:
    """Limit buy fills if bar low trades through limit. Fill assumed at limit price."""
    return float(limit_px) if float(low_px) <= float(limit_px) else None


def fill_stop_buy(open_px: float, high_px: float, stop_px: float) -> Optional[float]:
    """
    Stop buy fills if bar high crosses stop.
    If gap up above stop, assume fill at open; else at stop.
    """
    if float(high_px) >= float(stop_px):
        return float(open_px) if float(open_px) > float(stop_px) else float(stop_px)
    return None


def fill_limit_sell(open_px: float, high_px: float, limit_px: float) -> Optional[float]:
    """Limit sell fills if bar high trades through limit. Fill assumed at limit price."""
    return float(limit_px) if float(high_px) >= float(limit_px) else None


def fill_stop_sell(open_px: float, low_px: float, stop_px: float) -> Optional[float]:
    """
    Stop sell fills if bar low crosses stop.
    If gap down below stop, assume fill at open; else at stop.
    """
    if float(low_px) <= float(stop_px):
        return float(open_px) if float(open_px) < float(stop_px) else float(stop_px)
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
    """Coerce OHLCV to numeric; invalid values become NaN."""
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

    - Slippage: always adverse (buys higher, sells lower).
    - Spread: applied depending on mode. Default "taker_only" applies spread to stop/time fills.
    """
    p = float(px)
    if not np.isfinite(p) or p <= 0:
        return float("nan")

    side_l = str(side).lower().strip()
    fill_type_l = str(fill_type).lower().strip()
    spread_mode_l = str(spread_mode).lower().strip()

    # 1) Slippage (always adverse)
    sb = float(slippage_bps) if np.isfinite(slippage_bps) else 0.0
    if sb > 0:
        if side_l == "buy":
            p *= (1.0 + sb / 10000.0)
        else:
            p *= (1.0 - sb / 10000.0)

    # 2) Spread
    sp = float(spread_bps) if np.isfinite(spread_bps) else 0.0
    apply_spread = False
    if sp > 0:
        if spread_mode_l == "always":
            apply_spread = True
        elif spread_mode_l == "never":
            apply_spread = False
        else:
            # taker_only
            apply_spread = (fill_type_l in {"stop", "time"})

    if apply_spread:
        if side_l == "buy":
            p *= (1.0 + sp / 10000.0)
        else:
            p *= (1.0 - sp / 10000.0)

    return float(p)


def _max_drawdown(equity: pd.Series) -> float:
    s = pd.to_numeric(equity, errors="coerce").dropna()
    if len(s) < 2:
        return float("nan")
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return float(dd.min())


def _sharpe_from_returns(ret: pd.Series, periods_per_year: int = 252) -> float:
    r = pd.to_numeric(ret, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return float("nan")
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        return float("nan")
    return float((mu / sd) * np.sqrt(int(periods_per_year)))


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
        # If anything is missing, default to risk-on so we don't accidentally block everything.
        if self.df is None or self.df.empty or ts is pd.NaT:
            return True

        try:
            idx = int(self.df["timestamp"].searchsorted(ts, side="right") - 1)
        except Exception:
            return True

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
    try:
        c = float(row["close"])
        m50 = float(row["ma50"])
        m200 = float(row["ma200"])
    except Exception:
        return "Mixed"

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
    try:
        rsi = float(row["rsi14"])
    except Exception:
        rsi = float("nan")
    return "|".join([_trend_bucket(row), _rsi_bucket(rsi)])


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
    """Return (raw_entry_px, entry_fill_type) where fill_type ∈ {"limit","stop"}."""
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
    """Return (raw_exit_px, exit_fill_type) where fill_type ∈ {"limit","stop","time"}."""
    stop_hit = (float(next_low) <= float(stop_px))
    target_hit = (float(next_high) >= float(target_px))

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

    if int(bars_held) >= int(horizon):
        if time_exit_price == "close":
            return float(next_close), "time"
        return float(next_open), "time"

    return None, "time"


def _simulate_outcome_r_multiple_capped(
    df: pd.DataFrame,
    *,
    entry_i: int,
    entry_px: float,
    stop_px: float,
    target_px: float,
    horizon: int,
    exit_priority: ExitPriority,
    time_exit_price: TimeExitPrice,
    max_i_inclusive: int,
) -> Tuple[str, float]:
    """
    Used for in-sample bucket stats.

    Returns (reason, r_multiple) where reason ∈ {"target","stop","time","none"}.
    Uses RAW prices (no costs): estimates setup quality, not execution quality.

    IMPORTANT: capped so it never reads past max_i_inclusive (prevents leakage).
    """
    n = len(df)
    if n <= 0:
        return "none", float("nan")

    if entry_i < 0 or entry_i >= n:
        return "none", float("nan")

    risk = float(entry_px - stop_px)
    if not np.isfinite(risk) or risk <= 1e-12:
        return "none", float("nan")

    last_i = min(int(max_i_inclusive), n - 1, int(entry_i + horizon))
    if last_i < entry_i:
        return "none", float("nan")

    for j in range(int(entry_i), int(last_i) + 1):
        bar = df.iloc[j]
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])

        raw_exit, fill_type = _choose_exit_fill(
            next_open=o,
            next_high=h,
            next_low=l,
            stop_px=float(stop_px),
            target_px=float(target_px),
            bars_held=int(j - entry_i),
            horizon=int(horizon),
            exit_priority=exit_priority,
            time_exit_price=time_exit_price,
            next_close=c,
        )
        if raw_exit is None:
            continue

        reason = "stop" if fill_type == "stop" else ("target" if fill_type == "limit" else "time")
        r = (float(raw_exit) - float(entry_px)) / risk
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

    Each stat: n, p_win, avg_r.

    IMPORTANT: outcome simulation is capped so it cannot read beyond is_end_i-1
    (prevents information leakage).
    """
    rs_by_key: Dict[str, List[float]] = {}
    wins: Dict[str, int] = {}
    counts: Dict[str, int] = {}

    n = len(df)
    # In-sample window ends at is_end_i-1 (inclusive)
    end = int(max(2, min(is_end_i, n - 1)))
    max_i_inclusive = int(end - 1)

    # We use row i for signal, next bar (i+1) for entry attempt
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

        reason, r_mult = _simulate_outcome_r_multiple_capped(
            df,
            entry_i=int(i + 1),
            entry_px=float(entry_px),
            stop_px=float(stop_px),
            target_px=float(target_px),
            horizon=int(horizon),
            exit_priority=exit_priority,
            time_exit_price=time_exit_price,
            max_i_inclusive=int(max_i_inclusive),
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
        nn = int(counts.get(key, 0))
        if nn <= 0:
            continue
        p_win = float(wins.get(key, 0)) / float(nn)
        avg_r = float(np.mean(rs)) if rs else float("nan")
        out[key] = {"n": float(nn), "p_win": float(p_win), "avg_r": float(avg_r)}
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

    multiplier (soft gating):
      - 0.0 => skip completely
      - 0.25 => take but 25% size
      - 1.0 => no penalty
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
        return 0.25, "low_p"
    if (not np.isfinite(ar)) or (ar < float(min_avg_r)):
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
    sizing_mode: SizingMode = "percent",
    invest_amount: float = 25.0,     # per trade when sizing_mode="fixed_amount"
    risk_pct: float = 0.02,          # risk cap (shares) when sizing enabled
    max_alloc_pct: float = 0.10,     # allocation cap when sizing_mode="percent"
    min_risk_per_share: float = 1e-6,

    # --- accounting model ---
    use_cash_ledger: bool = False,   # if True, buys consume cash; sells restore cash
    allow_margin: bool = False,      # if cash ledger on: allow cash to go negative

    # --- equity series behavior ---
    mark_to_market: bool = False,    # if True, equity includes unrealized PnL (or position value with cash ledger)

    # --- probability gating ---
    prob_gating: bool = True,
    prob_is_frac: float = 0.85,
    prob_min: float = 0.50,
    min_bucket_trades: int = 8,
    min_avg_r: float = -0.05,
    gate_mode: GateMode = "hard",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Beginner-friendly next-bar execution backtest (pullback or breakout). Long-only, single position.

    Key honesty notes:
      - Daily OHLC bars cannot tell intraday path; stop vs target order is ambiguous.
      - If mark_to_market is False, Sharpe/DD are less meaningful (equity updates mainly at exits).
      - Probability gating is a historical bucket filter (not a true probability model).
    """
    warnings: List[str] = []

    if df is None or df.empty or len(df) < int(min_hist_days):
        return {"error": "Not enough history", "df_bt": pd.DataFrame()}, pd.DataFrame()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"backtest_strategy missing required columns: {sorted(missing)}")

    horizon = int(max(1, horizon))
    cooldown_bars = int(max(0, cooldown_bars))

    slip_bps = float(slippage_bps) if np.isfinite(slippage_bps) else 0.0
    spread_bps = float(assumed_spread_bps) if include_spread_penalty else 0.0

    mode_l = str(mode).lower().strip()
    if mode_l not in {"pullback", "breakout"}:
        raise ValueError("mode must be 'pullback' or 'breakout'")

    spread_mode_l = str(spread_mode).lower().strip()
    if spread_mode_l not in {"taker_only", "always", "never"}:
        spread_mode_l = "taker_only"

    charge_commission_on_l = str(charge_commission_on).lower().strip()
    if charge_commission_on_l not in {"entry", "exit", "both"}:
        charge_commission_on_l = "both"

    exit_priority_l = str(exit_priority).lower().strip()
    if exit_priority_l not in {"stop_first", "target_first", "worst_case"}:
        exit_priority_l = "stop_first"

    time_exit_price_l = str(time_exit_price).lower().strip()
    if time_exit_price_l not in {"open", "close"}:
        time_exit_price_l = "open"

    gate_mode_l = str(gate_mode).lower().strip()
    if gate_mode_l not in {"hard", "soft"}:
        gate_mode_l = "hard"

    sizing_mode_l = str(sizing_mode).lower().strip()
    if sizing_mode_l not in {"percent", "fixed_amount"}:
        sizing_mode_l = "percent"

    invest_amount_f = float(invest_amount)
    if not np.isfinite(invest_amount_f) or invest_amount_f <= 0:
        invest_amount_f = 25.0

    start_eq = float(start_equity) if (np.isfinite(start_equity) and start_equity > 0) else 0.0

    # Prepare data
    df_bt = _ensure_timestamp_column(df)
    df_bt = _to_numeric_ohlcv(df_bt)
    df_bt = df_bt.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    if df_bt.empty or len(df_bt) < int(min_hist_days):
        return {"error": "Not enough usable bars after cleaning", "df_bt": pd.DataFrame()}, pd.DataFrame()

    add_indicators_inplace(df_bt)
    needed_ind_cols = {"rsi14", "rvol", "vol_ann", "atr14", "ma50", "ma200"}
    missing_ind = needed_ind_cols - set(df_bt.columns)
    if missing_ind:
        raise KeyError(f"Indicators missing expected columns: {sorted(missing_ind)}. Check add_indicators_inplace().")

    # Market regime index (optional)
    regime_index: Optional[_MarketRegimeIndex] = None
    if require_risk_on and market_df is not None and not market_df.empty:
        regime_index = _MarketRegimeIndex.from_market_df(market_df)

    # Probability gating stats (optional) — leakage-protected
    bucket_stats: Dict[str, Dict[str, float]] = {}
    is_end_i = 0
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
            exit_priority=exit_priority_l,        # type: ignore[arg-type]
            time_exit_price=time_exit_price_l,    # type: ignore[arg-type]
            is_end_i=int(is_end_i),
        )

    # Output columns
    df_bt["signal"] = 0.0
    df_bt["entry_type"] = ""
    df_bt["bt_stop"] = np.nan
    df_bt["bt_target"] = np.nan
    df_bt["qty"] = np.nan
    df_bt["equity"] = np.nan
    df_bt["cash"] = np.nan  # meaningful when use_cash_ledger True (still populated for convenience)

    trades: List[Dict[str, Any]] = []

    # -----------------------
    # Accounting state
    # -----------------------
    # Two modes:
    # - use_cash_ledger=False: "equity_base" is realized equity; we can optionally add unrealized PnL if mark_to_market.
    # - use_cash_ledger=True : cash changes on buys/sells; equity is cash + position_value (always coherent).
    equity_base = float(start_eq)
    cash = float(start_eq)  # only "real" if use_cash_ledger=True

    # Position state
    in_pos = False
    entry_px = float("nan")
    entry_i = -1
    stop_px = float("nan")
    target_px = float("nan")
    qty = 0
    cooldown = 0

    # Gating counters
    gated_total = 0
    gated_unknown = 0
    gated_low_n = 0
    gated_low_p = 0
    gated_low_r = 0

    # Other skip counters (useful for honest diagnostics)
    skipped_qty_amt_zero = 0
    forced_negative_cash_on_exit = 0

    def ts_at(i: int) -> str:
        if 0 <= i < len(df_bt):
            return str(pd.to_datetime(df_bt.loc[i, "timestamp"], utc=True, errors="coerce"))
        return ""

    def _do_charge(which: CommissionChargeOn) -> bool:
        nonlocal equity_base, cash, forced_negative_cash_on_exit

        c = float(commission_per_order) if np.isfinite(commission_per_order) else 0.0
        if c <= 0:
            return True

        do_charge = (charge_commission_on_l == "both") or (charge_commission_on_l == which)
        if not do_charge:
            return True

        if use_cash_ledger:
            # Entry: must be payable unless allow_margin
            if (which == "entry") and (not allow_margin) and (cash - c) < 0.0:
                return False

            # Exit: we always charge (broker will debit), even if it makes cash negative.
            if (which == "exit") and (not allow_margin) and (cash - c) < 0.0:
                forced_negative_cash_on_exit += 1
            cash -= c
            return True

        # No cash ledger: commission reduces realized equity_base.
        if (which == "entry") and (equity_base - c) < 0.0:
            return False
        equity_base -= c
        return True

    def _equity_at_bar(i: int) -> Tuple[float, float]:
        """
        Returns (equity, cash_display) for bar i.

        - cash_display is:
          - real cash when use_cash_ledger=True
          - equity_base when use_cash_ledger=False (for a simple "account value" display)
        """
        if not in_pos:
            if use_cash_ledger:
                return float(cash), float(cash)
            return float(equity_base), float(equity_base)

        # In position:
        cur_close = float(df_bt.loc[i, "close"])
        if use_cash_ledger:
            # Equity must always include holdings to be coherent.
            if mark_to_market and np.isfinite(cur_close) and qty > 0:
                pos_val = float(cur_close) * int(qty)
            else:
                # If not MTM, value holdings at cost basis (keeps equity stable during hold).
                pos_val = float(entry_px) * int(qty) if (np.isfinite(entry_px) and qty > 0) else 0.0
            eq = float(cash + pos_val)
            return eq, float(cash)

        # No cash ledger:
        # equity_base is realized; optionally add unrealized PnL.
        eq = float(equity_base)
        if mark_to_market and np.isfinite(cur_close) and np.isfinite(entry_px) and qty > 0:
            eq += (float(cur_close) - float(entry_px)) * int(qty)
        return float(eq), float(equity_base)

    # Init first row
    eq0, cash0 = _equity_at_bar(0)
    df_bt.loc[0, "equity"] = float(eq0)
    df_bt.loc[0, "cash"] = float(cash0)

    # -----------------------
    # Main loop
    # -----------------------
    for i in range(len(df_bt) - 1):
        # Update equity/cash display at bar i
        eq_i, cash_i = _equity_at_bar(i)
        df_bt.loc[i, "equity"] = float(eq_i)
        df_bt.loc[i, "cash"] = float(cash_i)

        if cooldown > 0:
            cooldown -= 1

        # -------------------- EXIT --------------------
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
                exit_priority=exit_priority_l,      # type: ignore[arg-type]
                time_exit_price=time_exit_price_l,  # type: ignore[arg-type]
                next_close=c,
            )
            if raw_exit is None:
                continue

            exit_px_eff = _apply_costs_px(
                float(raw_exit),
                "sell",
                slippage_bps=float(slip_bps),
                spread_bps=float(spread_bps),
                spread_mode=spread_mode_l,          # type: ignore[arg-type]
                fill_type=exit_fill_type,           # type: ignore[arg-type]
            )
            if not np.isfinite(exit_px_eff):
                continue

            # Charge commission on exit (never silently ignored)
            _do_charge("exit")

            eff_qty = int(qty) if enable_position_sizing else 1

            pnl_per_share = float(exit_px_eff - float(entry_px))
            pnl = float(pnl_per_share * eff_qty)

            risk_per_share = float(float(entry_px) - float(stop_px))
            r_mult = float(pnl_per_share / risk_per_share) if risk_per_share > float(min_risk_per_share) else np.nan

            equity_before = float(eq_i)

            # Accounting update
            if use_cash_ledger:
                cash += float(exit_px_eff) * eff_qty
            else:
                equity_base = float(equity_base + pnl)

            # Record equity at i+1 after exit
            # (Note: we haven't updated in_pos yet, so compute carefully.)
            # We'll temporarily flip out of position for correct display.
            reason = "stop" if exit_fill_type == "stop" else ("target" if exit_fill_type == "limit" else "time")

            # Close position
            in_pos = False
            exit_i = int(i + 1)

            eq_after, cash_after = _equity_at_bar(exit_i)
            df_bt.loc[exit_i, "equity"] = float(eq_after)
            df_bt.loc[exit_i, "cash"] = float(cash_after)

            trades.append(
                {
                    "entry_i": int(entry_i),
                    "exit_i": int(exit_i),
                    "entry_ts": ts_at(entry_i),
                    "exit_ts": ts_at(exit_i),
                    "entry_px": float(entry_px),
                    "exit_px": float(exit_px_eff),
                    "stop_px": float(stop_px),
                    "target_px": float(target_px),
                    "reason": str(reason),
                    "bars_held": int(bars_held),
                    "pnl_per_share": float(pnl_per_share),
                    "r_multiple": float(r_mult) if np.isfinite(r_mult) else np.nan,
                    "qty": int(eff_qty),
                    "pnl": float(pnl),
                    "equity_before": float(equity_before),
                    "equity_after": float(eq_after),
                }
            )

            # Reset / cooldown
            cooldown = int(cooldown_bars)
            entry_px = float("nan")
            entry_i = -1
            stop_px = float("nan")
            target_px = float("nan")
            qty = 0
            continue

        # -------------------- ENTRY --------------------
        if cooldown > 0:
            continue

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

            if gate_mode_l == "hard" and gate_mult <= 0.0:
                continue

        nxt = df_bt.iloc[i + 1]
        o = float(nxt["open"])
        h = float(nxt["high"])
        l = float(nxt["low"])

        raw_entry, entry_fill_type = _choose_entry_fill(
            mode_l=mode_l,
            close=float(close),
            atr=float(atr),
            atr_entry=float(atr_entry),
            next_open=float(o),
            next_high=float(h),
            next_low=float(l),
        )
        if raw_entry is None:
            continue

        entry_px_eff = _apply_costs_px(
            float(raw_entry),
            "buy",
            slippage_bps=float(slip_bps),
            spread_bps=float(spread_bps),
            spread_mode=spread_mode_l,       # type: ignore[arg-type]
            fill_type=entry_fill_type,       # type: ignore[arg-type]
        )
        if not np.isfinite(entry_px_eff) or entry_px_eff <= 0:
            continue

        stop = float(entry_px_eff - float(atr_stop) * float(atr))
        target = float(entry_px_eff + float(atr_target) * float(atr))

        if not (np.isfinite(stop) and np.isfinite(target)):
            continue
        if stop >= entry_px_eff or target <= entry_px_eff:
            continue

        risk_per_share = float(entry_px_eff - stop)
        if not np.isfinite(risk_per_share) or risk_per_share <= float(min_risk_per_share):
            continue

        # -------- Choose quantity --------
        new_qty = 1

        if enable_position_sizing:
            # Base for sizing
            if use_cash_ledger:
                account_base = float(cash)
            else:
                account_base = float(equity_base)

            if not np.isfinite(account_base) or account_base <= 0:
                continue

            rpct = max(0.0, float(risk_pct))
            risk_budget = account_base * rpct
            qty_risk = int(np.floor(risk_budget / risk_per_share)) if rpct > 0 else 10**9

            if sizing_mode_l == "fixed_amount":
                qty_amt = int(np.floor(float(invest_amount_f) / max(1e-12, float(entry_px_eff))))
                if qty_amt <= 0:
                    skipped_qty_amt_zero += 1
                    continue
                new_qty = int(max(0, min(qty_amt, qty_risk)))
            else:
                apct = max(0.0, float(max_alloc_pct))
                alloc_budget = account_base * apct
                qty_alloc = int(np.floor(alloc_budget / max(1e-12, float(entry_px_eff))))
                new_qty = int(max(0, min(qty_risk, qty_alloc)))
        else:
            new_qty = 1

        # Apply soft gating size-down
        if prob_gating and gate_mode_l == "soft":
            if enable_position_sizing:
                new_qty = int(np.floor(new_qty * max(0.0, min(1.0, float(gate_mult)))))
            else:
                # If not sizing, "soft" gating can't scale; skip if it says 0.
                if float(gate_mult) <= 0.0:
                    continue

        if new_qty <= 0:
            continue

        # Charge commission on entry; if cannot pay, skip trade
        if not _do_charge("entry"):
            continue

        # Cash ledger: buy consumes cash
        if use_cash_ledger:
            cost = float(entry_px_eff) * int(new_qty)
            if (not allow_margin) and (cash - cost) < 0.0:
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

        eq_e, cash_e = _equity_at_bar(entry_i)
        df_bt.loc[entry_i, "equity"] = float(eq_e)
        df_bt.loc[entry_i, "cash"] = float(cash_e)

    # Final fill-forward (equity/cash)
    df_bt["equity"] = df_bt["equity"].ffill()
    df_bt["cash"] = df_bt["cash"].ffill()

    trades_df = pd.DataFrame(trades)
    results: Dict[str, Any] = {"df_bt": df_bt}

    gating_info = {
        "enabled": bool(prob_gating),
        "mode": str(gate_mode_l),
        "is_frac": float(prob_is_frac),
        "is_end_i": int(is_end_i) if prob_gating else 0,
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

    # Honesty / diagnostics warnings
    if not mark_to_market:
        warnings.append("mark_to_market=False: equity updates mainly on exits; Sharpe/MaxDD are less meaningful.")
    if exit_priority_l != "worst_case":
        warnings.append("Daily OHLC cannot determine stop vs target order within a bar; results depend on exit_priority.")
    if prob_gating:
        warnings.append("probability_gating uses in-sample bucket performance (heuristic), not a true probability model.")
    if forced_negative_cash_on_exit > 0 and use_cash_ledger and not allow_margin:
        warnings.append("Some exits forced cash negative to pay commissions (real brokers would debit your account).")
    if sizing_mode_l == "fixed_amount" and skipped_qty_amt_zero > 0:
        warnings.append("Some signals were skipped because invest_amount was too small to buy 1 share at entry price.")

    # Summary stats
    assumptions = {
        "mode": mode_l,
        "horizon_bars": int(horizon),
        "slippage_bps": float(slip_bps),
        "spread_bps": float(spread_bps),
        "spread_mode": str(spread_mode_l),
        "commission_per_order": float(commission_per_order),
        "commission_charged_on": str(charge_commission_on_l),
        "exit_priority": str(exit_priority_l),
        "time_exit_price": str(time_exit_price_l),
        "position_sizing": bool(enable_position_sizing),
        "sizing_mode": str(sizing_mode_l),
        "invest_amount": float(invest_amount_f),
        "risk_pct": float(risk_pct),
        "max_alloc_pct": float(max_alloc_pct),
        "use_cash_ledger": bool(use_cash_ledger),
        "allow_margin": bool(allow_margin),
        "mark_to_market": bool(mark_to_market),
        "probability_gating": gating_info,
        "single_position_long_only": True,
    }

    if trades_df.empty:
        results.update(
            {
                "trades": 0,
                "win_rate": float("nan"),
                "total_return": float("nan"),
                "max_drawdown": float("nan"),
                "sharpe": float("nan"),
                "avg_r_multiple": float("nan"),
                "assumptions": assumptions,
                "warnings": warnings,
                "diagnostics": {
                    "skipped_qty_amt_zero": int(skipped_qty_amt_zero),
                    "forced_negative_cash_on_exit": int(forced_negative_cash_on_exit),
                },
                "notes_for_beginners": [
                    "No trades were taken. This can happen if filters or gating are strict for this ticker/time period.",
                    "Try: widen RSI range, lower rvol_min, raise vol_max, reduce min_bucket_trades, or switch gate_mode to 'soft' / turn gating off.",
                    "If using fixed amount sizing: invest_amount must be large enough to buy at least 1 share.",
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

    eq_series = pd.to_numeric(df_bt["equity"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    eq0 = float(start_eq) if start_eq > 0 else float("nan")
    eqN = float(eq_series.iloc[-1]) if len(eq_series) else float("nan")
    total_return = (eqN / eq0 - 1.0) if np.isfinite(eq0) and eq0 > 0 and np.isfinite(eqN) else float("nan")

    max_dd = _max_drawdown(eq_series) if len(eq_series) else float("nan")
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
            "assumptions": assumptions,
            "warnings": warnings,
            "diagnostics": {
                "skipped_qty_amt_zero": int(skipped_qty_amt_zero),
                "forced_negative_cash_on_exit": int(forced_negative_cash_on_exit),
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
