# my-stock-app

Streamlit stock analysis app with:
- Alpaca historical bars (if you provide API keys)
- Yahoo Finance fallback (history, basic fundamentals, news)
- Indicators + charts
- A beginner-friendly backtester with execution cost assumptions

## Quickstart (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run .\app.py
```

## Alpaca Keys (Optional)

Create `.streamlit/secrets.toml` (this file is git-ignored):

```toml
ALPACA_KEY = "YOUR_KEY"
ALPACA_SECRET = "YOUR_SECRET"
```

If keys are missing/invalid, the app falls back to Yahoo for historical data.

## Zoya Sharia Compliance (Optional)

If you have a Zoya API key, add it to `.streamlit/secrets.toml`:

```toml
ZOYA_API_KEY = "sandbox-... or live-..."
```

The app will display Zoya Shariah compliance status per ticker (and can filter opportunity scans to compliant names).

## Notes

- The backtest uses daily OHLC bars; stop/target order within a day is unknowable, so results depend on assumptions.
- Install dependencies before running; the repo does not vendor packages.
