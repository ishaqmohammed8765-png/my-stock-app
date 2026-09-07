"""Monthly, unlevered portfolio research with prior-close targets and next-open fills."""

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_loader import MarketData

CANDIDATES = ("trend_200", "momentum_126", "momentum_252")
LAB_VERSION = "1.0"
WARMUP = 274


@dataclass(frozen=True)
class PortfolioConfig:
    capital: float = 10000.0
    slippage_bps: float = 10.0
    spread_bps: float = 10.0
    commission: float = 1.0
    max_weight: float = 0.50
    drawdown_stop: float = 0.20

    def __post_init__(self):
        values = vars(self)
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in values.values()
        ):
            raise ValueError("Portfolio inputs must be finite numbers.")
        if not 100 <= self.capital <= 1e9 or not 0 <= self.commission <= 1000:
            raise ValueError("Capital must be at least USD 100 and commission 0–1000.")
        if not 0 <= self.slippage_bps <= 500 or not 0 <= self.spread_bps <= 1000:
            raise ValueError("Unsupported spread or slippage.")
        if not 0.1 <= self.max_weight <= 1 or not 0.05 <= self.drawdown_stop <= 0.50:
            raise ValueError("Use a 10–100% position cap and a 5–50% drawdown stop.")


@dataclass
class Panel:
    opens: pd.DataFrame
    closes: pd.DataFrame
    sources: dict
    digest: str
    synthetic: bool


def make_panel(datasets: list[MarketData]) -> Panel:
    if not 2 <= len(datasets) <= 10:
        raise ValueError("Use 2–10 instruments for a portfolio study.")
    symbols = [d.symbol for d in datasets]
    if len(set(symbols)) != len(symbols):
        raise ValueError("Duplicate symbols are not allowed.")
    if any(d.currency != "USD" for d in datasets):
        raise ValueError("Every instrument must have USD quote currency.")
    if len({d.demo for d in datasets}) != 1:
        raise ValueError("Synthetic and market data cannot be mixed.")
    left = max(d.bars.index.min() for d in datasets)
    right = min(d.bars.index.max() for d in datasets)
    if left >= right:
        raise ValueError("No common historical period.")
    series, opens, sources = {}, {}, {}
    expected = None
    for data in sorted(datasets, key=lambda d: d.symbol):
        b = data.bars.loc[left:right]
        if not b.index.is_monotonic_increasing or b.index.has_duplicates:
            raise ValueError("Sessions must be sorted and unique.")
        if expected is None:
            expected = b.index
        if not b.index.equals(expected):
            raise ValueError(
                "Missing sessions within the common period; no silent forward-filling is permitted."
            )
        if (
            not np.isfinite(b[["open", "close"]]).all().all()
            or (b[["open", "close"]] <= 0).any().any()
        ):
            raise ValueError("Prices must be finite and positive.")
        series[data.symbol], opens[data.symbol] = b.close, b.open
        sources[data.symbol] = {
            "provider": data.source,
            "fetched_at": data.fetched_at,
            "warnings": data.warnings,
        }
    close, opening = pd.DataFrame(series), pd.DataFrame(opens)
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(
            pd.concat({"open": opening, "close": close}, axis=1), index=True
        ).values.tobytes()
        + json.dumps(
            {"symbols": list(close), "sources": sources, "synthetic": datasets[0].demo},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return Panel(opening, close, sources, digest, datasets[0].demo)


def target_weights(closes, index, candidate, max_weight=0.5):
    """index is the last observable close, never the execution session."""
    weights = pd.Series(0.0, index=closes.columns)
    if candidate == "cash":
        return weights
    if candidate == "equal_weight":
        return weights + 1 / len(weights)
    if candidate not in CANDIDATES:
        raise ValueError("Unknown candidate.")
    history = closes.iloc[: index + 1]
    if len(history) < WARMUP:
        return weights
    trend = history.iloc[-1] > history.tail(200).mean()
    if candidate == "trend_200":
        weights.loc[trend] = min(max_weight, 1 / len(weights))
        return weights
    lookback = 126 if candidate == "momentum_126" else 252
    momentum = history.iloc[-22] / history.iloc[-22 - lookback] - 1
    # Stable tie-breaking by the alphabetically ordered panel columns.
    eligible = (
        momentum.loc[trend & (momentum > 0)].sort_values(ascending=False, kind="stable").head(2)
    )
    weights.loc[eligible.index] = min(max_weight, 0.5)
    return weights


@dataclass
class PortfolioRun:
    equity: pd.Series
    orders: pd.DataFrame
    exposure: pd.Series
    halted: bool
    peak: float


def run_portfolio(
    panel,
    candidate,
    cfg,
    *,
    start,
    end=None,
    cost_multiplier=1.0,
    inherited_peak=None,
    halted=False,
):
    end = len(panel.closes) if end is None else end
    if not WARMUP <= start < end <= len(panel.closes):
        raise ValueError("Invalid portfolio simulation window.")
    if not 1 <= cost_multiplier <= 3:
        raise ValueError("Cost multiplier must be 1–3.")
    if candidate not in (*CANDIDATES, "equal_weight", "cash"):
        raise ValueError("Unknown portfolio candidate.")
    cash = float(cfg.capital)
    holdings = pd.Series(0, index=panel.closes.columns, dtype=int)
    peak = max(cash, inherited_peak or cash)
    navs, exposures, dates, orders = [cash], [0.0], [panel.closes.index[start - 1]], []
    fee = cfg.commission * cost_multiplier
    bps = (cfg.slippage_bps + cfg.spread_bps / 2) * cost_multiplier / 10000
    last_nav = cash
    pending_halt = halted

    def execute(symbol, qty, side, raw_price, date, reason):
        nonlocal cash
        price = float(raw_price) * (1 + bps if side == "buy" else 1 - bps)
        if side == "buy":
            qty = min(qty, max(0, int(math.floor((cash - fee - fee * len(holdings)) / price))))
        if qty <= 0:
            return
        if side == "sell" and qty < holdings[symbol]:
            reserve = fee * int((holdings > 0).sum())
            if cash + qty * price - fee < reserve:
                return
        signed = qty if side == "buy" else -qty
        cash -= signed * price + fee
        holdings[symbol] += signed
        orders.append(
            {
                "date": date,
                "symbol": symbol,
                "side": side,
                "quantity": qty,
                "fill_price": price,
                "commission": fee,
                "cash_after": cash,
                "reason": reason,
            }
        )

    for j in range(start, end):
        date = panel.closes.index[j]
        op, close = panel.opens.iloc[j], panel.closes.iloc[j]
        if candidate != "equal_weight" and last_nav / peak - 1 <= -cfg.drawdown_stop:
            pending_halt = True
        rebalance = j == start or date.month != panel.closes.index[j - 1].month
        if (rebalance and j < end - 1) or pending_halt:
            weights = target_weights(
                panel.closes, j - 1, "cash" if pending_halt else candidate, cfg.max_weight
            )
            nav_open = cash + float((holdings * op).sum())
            target = np.floor(weights * max(0, nav_open) / (op * (1 + bps))).astype(int)
            for symbol in holdings.index:
                if holdings[symbol] > target[symbol]:
                    execute(
                        symbol,
                        int(holdings[symbol] - target[symbol]),
                        "sell",
                        op[symbol],
                        date,
                        "risk_stop" if pending_halt else "rebalance",
                    )
            for symbol in holdings.index:
                if holdings[symbol] < target[symbol]:
                    execute(
                        symbol,
                        int(target[symbol] - holdings[symbol]),
                        "buy",
                        op[symbol],
                        date,
                        "rebalance",
                    )
        if j == end - 1:
            for symbol in holdings.index:
                execute(symbol, int(holdings[symbol]), "sell", close[symbol], date, "window_end")
        value = float((holdings * close).sum())
        last_nav = cash + value
        peak = max(peak, last_nav)
        navs.append(last_nav)
        exposures.append(value / last_nav if last_nav > 0 else 0.0)
        dates.append(date)
    idx = pd.DatetimeIndex(dates)
    return PortfolioRun(
        pd.Series(navs, index=idx, name=candidate),
        pd.DataFrame(
            orders,
            columns=[
                "date",
                "symbol",
                "side",
                "quantity",
                "fill_price",
                "commission",
                "cash_after",
                "reason",
            ],
        ),
        pd.Series(exposures, index=idx),
        pending_halt,
        peak,
    )


def portfolio_metrics(run):
    e = run.equity
    sessions = len(e) - 1
    returns = e.pct_change().dropna()
    sd = returns.std()
    return {
        "net_return": float(e.iloc[-1] / e.iloc[0] - 1),
        "cagr": float((e.iloc[-1] / e.iloc[0]) ** (252 / sessions) - 1) if e.iloc[-1] > 0 else -1.0,
        "max_drawdown": float((e / e.cummax() - 1).min()),
        "sharpe": float(returns.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        "orders": len(run.orders),
        "fees": float(run.orders.commission.sum()),
        "mean_exposure": float(run.exposure.mean()),
        "halted": run.halted,
    }
