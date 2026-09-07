# Strategy Lab research protocol

The goal is to discover whether a predefined strategy has useful evidence after costs. No profitable strategy has yet been established by this implementation. Real-data validation remains outstanding: Yahoo returned HTTP 429 in the development environment; the Financial Datasets connection was confirmed but no callable market-data tools were exposed to the session. Synthetic tests validate software only.

## Fixed candidates

- trend_200: each instrument receives its equal universe weight, capped at 50%, only when its previous close exceeds its 200-session average.
- momentum_126 and momentum_252: select up to two instruments with positive trailing momentum and the same trend filter. Skip the latest 21 sessions when computing momentum; allocate at most 50% to each. Empty slots stay in cash.

These are research hypotheses inspired by trend and momentum literature, not replications or recommendations. The default ETF universe is illustrative and chosen retrospectively. Results are conditional on that universe and available instruments; survivorship and selection bias remain.

References: https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum and https://mebfaber.com/white-papers/ . Repeated experiments increase overfitting risk: https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf .

## Chronological selection

Reserve the last 20% of aligned history, with a minimum of 252 sessions. Earlier development folds train on 756 sessions and test on the next 126 (short final folds of at least 21 sessions are allowed). Every candidate needs 274 prior warmup sessions. Folds are independent, start with equal capital, liquidate at their ends, and are not stitched into a claimed continuous track record.

Training selection maximizes CAGR / max(5%, absolute maximum drawdown), requiring positive net return and drawdown no worse than the configured threshold. Ties retain the published candidate order. If none qualify, select cash. The final candidate is chosen using the 756 sessions immediately before the final period. Data provenance, configuration, cutoff, candidate and plan hashes are exported before reveal. The final period is used once for the frozen candidate, an equal-weight monthly benchmark, and doubled costs. Changing rules after reveal contaminates that period; browser counters are informational, not durable experiment enforcement.

## Execution and interpretation

Signals use prior closes; monthly rebalances fill at next opens. Whole shares, a shared cash ledger, commissions, half-spread and slippage apply. Rebalancing sells precede buys in alphabetical order. Final positions liquidate at the last close with costs; no new position opens on the final session. Target caps apply at rebalances, not continuously between them. Cash earns zero; no margin, tax model, market impact, liquidity model or borrow trades. Adjusted OHLC is a total-return approximation, not a literal historical share ledger.

A 20% closing drawdown triggers liquidation at the following open and permanently halts that simulation window. It is not a guaranteed loss cap. The equal-weight benchmark remains fully invested and does not use this filter. Each independent fold and final experiment starts with a fresh risk state.

Only real-data final results with positive net return, benchmark outperformance and positive doubled-cost return pass the displayed historical screen. This screen does not estimate significance or future profitability. Seek forward paper observations using current verified data and a recorded portfolio risk state before assessing deployment. Exported weights are observation targets, not broker orders; historical halt state is not automatically a forward portfolio state.

## Data

Use 10 years through existing Yahoo/Alpaca loaders or import CSV with symbol,date,open,high,low,close,volume,currency. The uploader must confirm a consistent split/dividend-adjusted OHLC basis. USD instruments only; 2–10 unique symbols; positive finite prices; consistent OHLC; completed sessions only. Common start/end dates are used, but differing sessions within that overlap fail rather than being filled. Provider warnings remain in the plan. CSV claims are user assertions, not independently verified provenance. Do not mix synthetic and observed data.

Connecting a ChatGPT data plugin does not provision API credentials inside a deployed Streamlit app. No Financial Datasets app adapter is claimed in this change. The CSV path supports validated data exports without embedding credentials or licensed datasets in this public repository.
