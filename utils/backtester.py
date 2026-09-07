"""Daily long-only execution with a single cash ledger and conservative fills."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Execution, Strategy
from .strategy import assess


@dataclass
class Backtest:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict
    diagnostics: dict


def market_fill(price: float, side: str, cfg: Execution) -> float:
    cost = (cfg.slippage_bps + cfg.spread_bps / 2) / 10000
    return price * (1 + cost if side == "buy" else 1 - cost)


def entry_fill(bar: pd.Series, trigger: float, mode: str, cfg: Execution):
    if mode == "breakout":
        if bar.high < trigger:
            return None
        return market_fill(max(float(bar.open), trigger), "buy", cfg), bool(bar.open >= trigger)
    if bar.low > trigger:
        return None
    return min(trigger, market_fill(float(bar.open), "buy", cfg)), bool(bar.open <= trigger)


def exit_fill(
    bar: pd.Series,
    stop: float,
    target: float,
    cfg: Execution,
    *,
    due=False,
    at_open=True,
    entry_day=False,
):
    if due:
        return market_fill(float(bar.open), "sell", cfg), "time", False
    if not entry_day or at_open:
        if bar.open <= stop:
            return market_fill(float(bar.open), "sell", cfg), "stop", False
        if bar.open >= target:
            return max(target, market_fill(float(bar.open), "sell", cfg)), "target", False
    stop_hit, target_hit = bar.low <= stop, bar.high >= target
    ambiguous = bool(
        (stop_hit and target_hit) or (entry_day and not at_open and (stop_hit or target_hit))
    )
    if stop_hit:
        return market_fill(stop, "sell", cfg), "stop", ambiguous
    # Intraday entries must not claim a target that may predate the fill.
    if target_hit and (not entry_day or at_open or bar.close >= target):
        return target, "target", ambiguous
    return None, "", ambiguous


def position_size(cash: float, entry: float, stop: float, cfg: Execution) -> int:
    budget = min(cash, cash * cfg.allocation_pct)
    if cfg.fixed_amount > 0:
        budget = min(budget, cfg.fixed_amount)
    affordable = math.floor(max(0, budget - cfg.commission) / entry)
    risk_per_share = entry - market_fill(stop, "sell", cfg)
    risk_budget = max(0, cash * cfg.risk_pct - 2 * cfg.commission)
    return (
        max(0, min(affordable, math.floor(risk_budget / risk_per_share)))
        if risk_per_share > 0
        else 0
    )


def summarise(equity: pd.Series, trades: pd.DataFrame) -> dict:
    returns = equity.pct_change().dropna()
    sd = returns.std(ddof=1)
    pnl = trades.net_pnl if not trades.empty else pd.Series(dtype=float)
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "trades": len(trades),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "net_pnl": float(equity.iloc[-1] - equity.iloc[0]),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "sharpe": float(returns.mean() / sd * np.sqrt(252))
        if len(returns) > 1 and sd > 0
        else np.nan,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "expectancy": float(pnl.mean()) if len(pnl) else np.nan,
        "profit_factor": float(gains / losses) if losses > 0 else (np.inf if gains > 0 else np.nan),
    }


def run_backtest(
    df: pd.DataFrame, strategy: Strategy, execution: Execution, *, start=200, end=None
) -> Backtest:
    """Input is validated bars with causal indicators; end is exclusive."""
    end = len(df) if end is None else end
    if not 1 <= start < end <= len(df):
        raise ValueError("Need a prior signal bar and at least one evaluation bar.")
    cash, position = float(execution.capital), None
    records, values, dates = [], [cash], [df.index[start - 1]]
    diagnostics = {"ambiguous_bars": 0, "unaffordable_or_risk_limited": 0, "forced_final_exits": 0}

    def close_position(price, reason, j):
        nonlocal position, cash
        p = position
        gross = (price - p["entry_price"]) * p["quantity"]
        net = gross - 2 * execution.commission
        cash += price * p["quantity"] - execution.commission
        records.append(
            {
                "entry_date": df.index[p["entry_index"]],
                "exit_date": df.index[j],
                "entry_price": p["entry_price"],
                "exit_price": price,
                "stop": p["stop"],
                "target": p["target"],
                "quantity": p["quantity"],
                "sessions_held": j - p["entry_index"],
                "reason": reason,
                "gross_pnl": gross,
                "fees": 2 * execution.commission,
                "net_pnl": net,
                "net_r": net / p["planned_risk"],
            }
        )
        position = None

    for j in range(start, end):
        bar, exited = df.iloc[j], False
        if position is not None:
            price, reason, ambiguous = exit_fill(
                bar,
                position["stop"],
                position["target"],
                execution,
                due=j - position["entry_index"] >= strategy.horizon,
            )
            diagnostics["ambiguous_bars"] += int(ambiguous)
            if price is not None:
                close_position(price, reason, j)
                exited = True
        if position is None and not exited and j < end - 1:
            previous = df.iloc[j - 1]
            setup = assess(previous, strategy)
            if setup.eligible:
                fill = entry_fill(bar, setup.entry, strategy.mode, execution)
                if fill is not None:
                    entry, at_open = fill
                    stop, target = (
                        entry - strategy.atr_stop * previous.atr14,
                        entry + strategy.atr_target * previous.atr14,
                    )
                    qty = position_size(cash, entry, stop, execution) if stop > 0 else 0
                    if qty == 0:
                        diagnostics["unaffordable_or_risk_limited"] += 1
                    else:
                        cash -= entry * qty + execution.commission
                        position = {
                            "entry_index": j,
                            "entry_price": entry,
                            "stop": stop,
                            "target": target,
                            "quantity": qty,
                            "planned_risk": (entry - market_fill(stop, "sell", execution)) * qty
                            + 2 * execution.commission,
                        }
                        price, reason, ambiguous = exit_fill(
                            bar, stop, target, execution, at_open=at_open, entry_day=True
                        )
                        diagnostics["ambiguous_bars"] += int(ambiguous)
                        if price is not None:
                            close_position(price, reason, j)
        if j == end - 1 and position is not None:
            close_position(market_fill(float(bar.close), "sell", execution), "end_of_window", j)
            diagnostics["forced_final_exits"] += 1
        values.append(
            float(cash + (position["quantity"] * bar.close if position is not None else 0))
        )
        dates.append(df.index[j])
    equity = pd.Series(values, index=pd.DatetimeIndex(dates), name="Strategy")
    columns = [
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "stop",
        "target",
        "quantity",
        "sessions_held",
        "reason",
        "gross_pnl",
        "fees",
        "net_pnl",
        "net_r",
    ]
    trades = pd.DataFrame(records, columns=columns)
    return Backtest(equity, trades, summarise(equity, trades), diagnostics)


def buy_and_hold(df: pd.DataFrame, cfg: Execution, start: int, end: int) -> pd.Series:
    entry = market_fill(float(df.open.iloc[start]), "buy", cfg)
    qty = max(0, math.floor((cfg.capital - cfg.commission) / entry))
    if qty == 0:
        return pd.Series(cfg.capital, index=df.index[start - 1 : end], name="Buy & hold")
    cash = cfg.capital - qty * entry - cfg.commission
    values = [cfg.capital] + (cash + qty * df.close.iloc[start:end]).tolist()
    values[-1] = (
        cash + qty * market_fill(float(df.close.iloc[end - 1]), "sell", cfg) - cfg.commission
    )
    return pd.Series(values, index=df.index[start - 1 : end], name="Buy & hold")
