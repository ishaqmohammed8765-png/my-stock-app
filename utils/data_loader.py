"""Provider adapters, completed-session policy and strict bar validation."""

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    symbol: str
    bars: pd.DataFrame
    source: str
    currency: str | None
    fetched_at: str
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    demo: bool = False


def valid_symbol(symbol):
    return bool(re.fullmatch(r"[A-Z][A-Z0-9]{0,9}(?:[.-][A-Z0-9]{1,3})?", symbol))


def parse_symbols(raw):
    symbols = list(dict.fromkeys(re.split(r"[,\s]+", raw.upper().strip())))
    if not symbols or any(not valid_symbol(s) for s in symbols):
        raise ValueError(
            "Enter valid ticker symbols separated by commas, such as AAPL, MSFT, BRK-B."
        )
    if len(symbols) > 20:
        raise ValueError("Please scan at most 20 symbols at a time.")
    return symbols


def normalise_bars(raw, *, now=None):
    if raw is None or raw.empty:
        raise ValueError("The provider returned no historical bars.")
    df = raw.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" in df:
        df.index = pd.to_datetime(df.pop("timestamp"), utc=True, errors="coerce")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Historical bars do not have valid session dates.")
    if df.index.isna().any():
        raise ValueError("Historical data contains invalid dates.")
    df.index = pd.DatetimeIndex([str(t.date()) for t in df.index], tz="UTC", name="date")
    required = ["open", "high", "low", "close", "volume"]
    if set(required) - set(df):
        raise ValueError("Historical data is missing price or volume columns.")
    df = df[required].apply(pd.to_numeric, errors="coerce").sort_index()
    warnings = []
    if df.index.duplicated().any():
        warnings.append("Duplicate sessions were removed; the last provider value was retained.")
        df = df.loc[~df.index.duplicated(keep="last")]
    today = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    today = today.tz_localize("UTC") if today.tzinfo is None else today.tz_convert("UTC")
    incomplete = df.index >= today.normalize()
    if incomplete.any():
        warnings.append(
            "Current-day or future bars excluded. Signals use completed prior sessions only."
        )
        df = df.loc[~incomplete]
    if df.empty:
        raise ValueError("No completed historical sessions were returned.")
    if (
        not np.isfinite(df.to_numpy()).all()
        or (df[["open", "high", "low", "close"]] <= 0).any().any()
        or (df.volume < 0).any()
    ):
        raise ValueError(
            "Invalid or missing prices/volume detected; analysis is blocked rather than silently repaired."
        )
    if (
        (df.high < df[["open", "close", "low"]].max(axis=1))
        | (df.low > df[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError("Inconsistent daily high/low prices detected; analysis is blocked.")
    if (today.normalize() - df.index[-1]).days > 5:
        warnings.append(
            "The last available session is more than five calendar days old; current setups are blocked."
        )
    if (df.index.to_series().diff().dt.days > 7).any():
        warnings.append(
            "History has gaps longer than seven days; halts or missing data may affect results."
        )
    return df, warnings


def is_stale(data, now=None):
    if data.demo:
        return False
    today = pd.Timestamp(now or datetime.now(timezone.utc))
    today = today.tz_localize("UTC") if today.tzinfo is None else today.tz_convert("UTC")
    return (today.normalize() - data.bars.index[-1]).days > 5


def _yahoo(symbol, years):
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    raw = ticker.history(
        period=f"{years}y", interval="1d", auto_adjust=True, actions=False, timeout=12
    )
    warnings, info = [], {}
    try:
        info = ticker.get_info() or {}
    except Exception:
        warnings.append(
            "Company metadata could not be verified; unavailable fields remain unknown."
        )
    currency = info.get("currency")
    if info.get("quoteType") not in {None, "EQUITY", "ETF"}:
        warnings.append("Only stocks and ETFs are supported; this asset type cannot be evaluated.")
        return raw, None, info, warnings
    if not currency:
        try:
            currency = (ticker.get_history_metadata() or {}).get("currency")
        except Exception:
            warnings.append("Quote currency could not be verified.")
    return raw, currency, info, warnings


def _alpaca(symbol, years, key, secret):
    if not key or not secret:
        raise ValueError("Alpaca credentials are missing.")
    now = datetime.now(timezone.utc)
    params = {
        "timeframe": "1Day",
        "start": (now - timedelta(days=365 * years + 10)).isoformat(),
        "end": now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat(),
        "adjustment": "all",
        "feed": "iex",
        "limit": 10000,
        "sort": "asc",
    }
    rows = []
    while True:
        url = f"https://data.alpaca.markets/v2/stocks/{urllib.parse.quote(symbol, safe='')}/bars?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            payload = json.load(response)
        rows.extend(payload.get("bars") or [])
        token = payload.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    return pd.DataFrame(rows).rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )


def load_market(symbol, years=5, provider="Yahoo", key="", secret=""):
    if not valid_symbol(symbol) or years not in {2, 5, 10}:
        raise ValueError("Unsupported ticker or historical period.")
    warnings, info, currency, raw, source = [], {}, None, None, "Yahoo Finance"
    if provider == "Alpaca":
        try:
            raw = _alpaca(symbol, years, key, secret)
            currency, source = "USD", "Alpaca IEX"
            warnings.append(
                "IEX is a single exchange feed; volume signals can differ from consolidated data."
            )
        except Exception:
            warnings.append("Alpaca was unavailable. Yahoo Finance was used instead.")
    if raw is None or raw.empty:
        if provider == "Alpaca" and raw is not None:
            warnings.append("Alpaca returned no bars. Yahoo Finance was used instead.")
        raw, currency, info, extra = _yahoo(symbol, years)
        source = "Yahoo Finance"
        warnings.extend(extra)
    bars, checks = normalise_bars(raw)
    return MarketData(
        symbol,
        bars,
        source,
        currency,
        datetime.now(timezone.utc).isoformat(),
        info,
        warnings + checks,
    )


def load_news(symbol):
    from xml.etree import ElementTree

    import yfinance as yf

    try:
        result = []
        for row in yf.Ticker(symbol).get_news(count=12) or []:
            item = row.get("content", row)
            url = (item.get("canonicalUrl") or {}).get("url") or item.get("link", "")
            if url.startswith("https://"):
                result.append(
                    {
                        "title": item.get("title", "News"),
                        "url": url,
                        "publisher": (item.get("provider") or {}).get(
                            "displayName", item.get("publisher", "")
                        ),
                        "date": str(item.get("pubDate", item.get("providerPublishTime", ""))),
                    }
                )
        if result:
            return result
    except Exception:
        pass  # The separate RSS request below supplies the documented fallback.
    req = urllib.request.Request(
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(symbol)}&region=US&lang=en-US",
        headers={"User-Agent": "stock-research/2.0"},
    )
    with urllib.request.urlopen(req, timeout=12) as response:
        root = ElementTree.fromstring(response.read())
    return [
        {
            "title": i.findtext("title", "News"),
            "url": i.findtext("link", ""),
            "publisher": "Yahoo RSS",
            "date": i.findtext("pubDate", ""),
        }
        for i in root.findall(".//item")
        if i.findtext("link", "").startswith("https://")
    ][:12]


def demo_market(symbol="DEMO"):
    rng = np.random.default_rng(17)
    dates = pd.bdate_range(
        end=(datetime.now(timezone.utc) - timedelta(days=1)).date(), periods=1500, tz="UTC"
    )
    close = 60 * np.exp(np.cumsum(rng.normal(0.00045, 0.016, len(dates))))
    opens = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.003, len(dates)))
    bars = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, close) * 1.012,
            "low": np.minimum(opens, close) * 0.988,
            "close": close,
            "volume": rng.integers(1000000, 6000000, len(dates)),
        },
        index=dates,
    )
    return MarketData(
        symbol, bars, "Synthetic demo", "USD", datetime.now(timezone.utc).isoformat(), demo=True
    )
