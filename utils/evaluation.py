"""Chronological holdout evaluation; no fitted or claimed profit probabilities."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .backtester import Backtest, buy_and_hold, run_backtest
from .config import Execution, Strategy


@dataclass
class Evaluation:
    development: Backtest
    holdout: Backtest
    stress: Backtest
    benchmark: pd.Series
    windows: pd.DataFrame
    interval: tuple[float, float]
    label: str
    reasons: list[str]
    split_date: str


def expectancy_interval(pnl: pd.Series, *, samples=2000) -> tuple[float, float]:
    x = np.asarray(pnl, dtype=float)
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    block = max(2, int(np.sqrt(n)))
    rng = np.random.default_rng(42)
    starts = rng.integers(0, n - block + 1, size=(samples, int(np.ceil(n / block))))
    indices = (starts[..., None] + np.arange(block)).reshape(samples, -1)[:, :n]
    return tuple(float(v) for v in np.quantile(x[indices].mean(axis=1), [0.025, 0.975]))


def evaluate(df: pd.DataFrame, strategy: Strategy, execution: Execution) -> Evaluation:
    start = 200
    if len(df) - start < 252:
        raise ValueError(
            "Need at least 452 completed daily bars: 200 warmup plus 252 evaluation sessions."
        )
    split = start + int((len(df) - start) * 0.70)
    development = run_backtest(df, strategy, execution, start=start, end=split)
    holdout = run_backtest(df, strategy, execution, start=split)
    stress_cfg = replace(
        execution,
        slippage_bps=min(500, execution.slippage_bps * 2),
        spread_bps=min(1000, execution.spread_bps * 2),
        commission=min(1000, execution.commission * 2),
    )
    stress = run_backtest(df, strategy, stress_cfg, start=split)
    benchmark = buy_and_hold(df, execution, split, len(df))
    windows = []
    boundaries = np.linspace(split, len(df), 4, dtype=int)
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        r = run_backtest(df, strategy, execution, start=int(a), end=int(b))
        bh = buy_and_hold(df, execution, int(a), int(b))
        windows.append(
            {
                "From": str(df.index[a].date()),
                "To": str(df.index[b - 1].date()),
                "Trades": r.metrics["trades"],
                "Strategy return": r.metrics["total_return"],
                "Buy & hold return": float(bh.iloc[-1] / bh.iloc[0] - 1),
            }
        )
    windows = pd.DataFrame(windows)
    interval = expectancy_interval(holdout.trades.net_pnl)
    reasons = []
    if holdout.metrics["trades"] < 30:
        reasons.append(
            "Fewer than 30 holdout trades; the sample is too small for a strong conclusion."
        )
    if len(df) - split < 252:
        reasons.append("The holdout covers less than one trading year.")
    if not np.isfinite(interval[0]) or interval[0] <= 0:
        reasons.append(
            "The exploratory expectancy interval does not establish positive average net trade profit."
        )
    if holdout.metrics["total_return"] <= float(benchmark.iloc[-1] / benchmark.iloc[0] - 1):
        reasons.append(
            "The strategy did not outperform fully invested buy & hold over the holdout."
        )
    if stress.metrics["total_return"] <= 0:
        reasons.append("The holdout was not profitable under execution-cost stress.")
    if int((windows["Strategy return"] > 0).sum()) < 2:
        reasons.append(
            "Positive returns were not present in at least two of the three separate holdout windows."
        )
    label = "Promising historical evidence" if not reasons else "Mixed historical evidence"
    if holdout.metrics["trades"] < 30 or len(df) - split < 252:
        label = "Insufficient evidence"
    if not reasons:
        reasons.append(
            "The historical checks passed. Confirm with forward paper trading; this is not a profit guarantee."
        )
    return Evaluation(
        development,
        holdout,
        stress,
        benchmark,
        windows,
        interval,
        label,
        reasons,
        str(df.index[split].date()),
    )
