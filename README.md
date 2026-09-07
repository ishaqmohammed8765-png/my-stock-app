# Stock Research

A Streamlit workspace for investigating US stocks and ETFs with transparent rules,
consistent USD accounting and chronological historical evaluation. It does not place
orders or promise profitable trades.

## Run locally

Use Python 3.12. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Choose a ticker and click **Load / refresh data**. Yahoo requires no credentials.
**Explore synthetic demo** runs without network requests; generated prices are
explicitly labelled and cannot establish an investment edge.

## Workflow

- **Overview:** completed-session indicators, reasons a setup qualifies or fails,
  indicative entry/stop/target, and available company fundamentals.
- **Charts:** adjusted candlesticks, volume, moving averages, Bollinger Bands, RSI,
  MACD and ADX.
- **Evaluate:** separate development and holdout periods, a same-stock buy-and-hold
  benchmark, execution-cost stress and independent holdout windows.
- **Screener:** the same entry rules for up to 20 tickers; unknown market caps cannot
  pass a market-cap ceiling. Unavailable symbols are reported.
- **News:** source-linked headlines, without an invented sentiment score.
- **Paper planning:** download a conditional plan, then record subsequent paper
  fills and costs separately to compare forward observations with assumptions.

All cash and profit values are **USD**. Trade planning and backtesting require
verified USD quote currency. There is no implicit GBP conversion, leverage, shorting
or fractional-share model. Positions are bounded by cash, an allocation cap, an
optional fixed investment cap and a risk budget including fees. Gaps can exceed the
planned loss. Small accounts may not support even one share.

## Research design

`assess()` supplies the overview, screener and backtester with one rule set. A setup
requires 200-bar readiness, price above MA50 above MA200, and the configured RSI,
relative-volume, volatility and ADX filters. Defaults are exploratory, not optimised
recommendations. Breakout/pullback orders are valid for the next session only.

Evaluation reserves 200 warmup bars, then splits the remainder chronologically:
70% development and 30% holdout. Each run starts flat and closes at its own end.
Headline metrics use only the holdout and include fees. Three additional holdout
windows start independently to expose instability. Cost stress increases spread,
slippage and commissions up to twice their values, capped at supported limits; zero
remains zero. An exploratory 95% moving-block bootstrap interval describes observed
net trade expectancy, not future-return probability.

At least 452 completed bars are needed to run evaluation. "Promising historical
evidence" additionally requires 30 holdout trades, 252 holdout sessions, a positive
lower expectancy bound, outperformance of buy & hold, positive cost-stress return,
and positive results in two of three independent windows. These criteria do not
prove dependable future profit. Repeated tuning against the holdout destroys its
independence. No model fitting or automated parameter optimisation is performed.

See [methodology and limitations](docs/METHODOLOGY.md).

## Data and result integrity

Yahoo uses adjusted daily OHLC prices; Alpaca requests `adjustment=all` and the IEX
feed. These are corporate-action-adjusted research proxies, not a broker's historical
share/dividend ledger. IEX volume can differ from consolidated volume.

Current-UTC-date bars are always excluded, including after an exchange closes.
Current setups are blocked when history is over five calendar days stale. Historical
evaluation remains available with explicit dates. Missing/nonpositive/inconsistent
prices block analysis; duplicate sessions are removed with a note, and long gaps
are flagged.

Provider caches last 15 minutes. Refresh bypasses them with a new identity. Result
identity includes ticker, source, currency, demo status, every price/volume row,
engine version and all parameters. Changing inputs hides old results and exports;
refresh explicitly clears results. Fundamentals are attached to each ticker's data.
Missing values stay unknown.

## Optional credentials

Create `.streamlit/secrets.toml` locally, or configure deployment secrets:

```toml
ALPACA_KEY = "YOUR_KEY"
ALPACA_SECRET = "YOUR_SECRET"
ZOYA_API_KEY = "live-YOUR_KEY"
```

The file is git-ignored. Environment variables with these names also work. Never
commit credentials. Alpaca failures/empty responses fall back to Yahoo with a note;
the Alpaca SDK is no longer required.

Zoya is connected to an explicit overview check and an optional compliant-only
screener filter. It uses the documented `basicCompliance.report` query and raw API
key in Authorization. Unknown/error reports are excluded when that filter is used.
Sandbox keys are rejected because their reports are randomized. Current compliance
does not establish historical compliance. Real reports require a live subscription.

## Exports and persistence

Version 2 settings exports contain validated JSON and no credentials. Old unversioned
settings are rejected; re-enter them using current controls. Export holdout trades
as CSV and assumptions/provenance as a JSON research record. Synthetic exports are
prefixed `DEMO-`. State lasts for the Streamlit session; download records to retain
them. There is no user database or brokerage execution service.

## Maintenance

```sh
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest -q
```

Tests cover execution chronology, fees, cash reconciliation, indicator causality,
training boundaries, settings, caches, provider contracts and Streamlit state changes.
Provider responses are mocked. These tests do not establish profitability or live
provider availability. GitHub Actions runs the same checks with Python 3.12.

To update the committed locks deliberately, install `pip-tools` and run:

```sh
pip-compile --strip-extras --output-file requirements.txt requirements.in
pip-compile --strip-extras --output-file requirements-dev.txt requirements-dev.in
```

| Area | Files |
|---|---|
| Interface | `app.py`, `ui/` |
| Rules and validation | `utils/config.py`, `utils/strategy.py` |
| Indicators and simulation | `utils/indicators.py`, `utils/backtester.py` |
| Evidence evaluation | `utils/evaluation.py` |
| Providers and cache | `utils/data_loader.py`, `utils/services.py` |
| Identity and screening | `utils/state.py`, `utils/screener.py` |
| Compliance | `utils/zoya_api.py` |

Version 2 removes the old probability gate, arbitrary opportunity scores, manual
news-tone bonuses, random stock picks, predicted opportunity timeframes, alternative
non-cash ledger and option to hide unrealised losses. It also removes obsolete
configuration and the standalone encoding-repair script.

## API references

- [Streamlit testing](https://docs.streamlit.io/develop/api-reference/app-testing)
- [Yahoo price history](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html)
- [Alpaca historical bars](https://docs.alpaca.markets/us/reference/stockbars)
- [Zoya API](https://developer.zoya.finance/docs)
