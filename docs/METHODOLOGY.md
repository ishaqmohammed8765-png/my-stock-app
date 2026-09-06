# Methodology

## Execution chronology

An existing position's scheduled maximum-hold exit happens at the open before
examining the day's high/low. Otherwise opening gaps through a stop/target precede
intraday ranges. A target gap cannot be overridden by a later stop. If both levels
are hit in an unknowable intraday path, stop-first is used and counted as ambiguous.

When flat, yesterday's completed indicators determine a next-session order. Quantity
is limited by cash, allocation, optional fixed investment cap and planned risk,
including both commissions and estimated stop costs. Rejected orders do not charge
fees. A filled entry consumes cash and commission before its entry-day protective
levels are checked. An intraday entry does not receive an unproven target fill:
entry at the open or a final close above target establishes a later target crossing.
Otherwise a potential stop is honoured conservatively.

There is no same-session re-entry after an existing position exits. A maximum hold
of 20 means an entry at session j exits at the open at j+20 unless stopped/targeted
sooner. Stops and targets use effective entry and prior-session ATR, so opening gaps
can change them from the indicative plan. The final session cannot open a new trade;
remaining holdings are closed at its close with costs.

Slippage plus half the full spread applies to each market/stop fill. Limits respect
their price and can improve on opening gaps. Partial fills, queue priority, liquidity,
halts and market impact are not modelled. Daily fills are approximations.

## Accounting

Cash changes only for filled orders. Each trade records gross P&L, both commissions,
net P&L and net R. Net trade P&L sums to ending equity minus starting capital. Every
open position is valued at the daily close; returns and drawdowns therefore include
unrealised movements. Win rate, expectancy and profit factor use net P&L. Sharpe
annualises daily account returns with 252 sessions and zero risk-free/cash interest.

All cash is USD. No tax or FX effects are included. Risk budgets size orders; they
cannot cap losses through gaps. Adjusted prices provide a corporate-action-adjusted
research proxy, not a historical broker share/split/dividend cash ledger.

## Evidence

Warmup history is available to causal indicators, but neither positions nor outcomes
cross the development/holdout boundary. No future outcomes fit the rules. Holdout
results are reported separately instead of blended into development performance.
The three additional windows are fixed-rule stability checks, not model retraining;
they start with independent capital, so their returns do not sum to the full holdout.

Buy and hold uses the same stock, dates, costs and starting account, fully invested
to whole-share affordability. The strategy usually retains more cash. Read the
comparison alongside drawdown and exposure: it is not a risk-matched benchmark.

Cost stress scales costs up to 2×, with caps of 500 bps slippage, 1,000 bps spread
and USD 1,000 commission per filled order. Zero stays zero; this is not a worst case.

The moving-block bootstrap resamples contiguous net trade P&L blocks, with seed 42
and block length approximately the square root of the trade count. Its exploratory
95% interval can underestimate regime, sizing and selection uncertainty. It is not
a prediction interval for future account returns. Repeated tuning against the
holdout makes it another development sample.

Insufficient evidence means fewer than 30 holdout trades or 252 sessions. Mixed
historical evidence means one or more documented checks fail. Promising historical
evidence means all checks pass, but still requires forward paper observation and
does not establish reliable future profit.

The current screener universe can introduce survivorship bias. Fundamentals are
current, not point-in-time historical inputs. News is contextual. Present-day Zoya
compliance cannot validate historical-period compliance.
