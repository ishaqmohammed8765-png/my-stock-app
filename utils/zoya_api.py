"""Documented Zoya basic report adapter; sandbox data cannot verify compliance."""

import json
import urllib.request
from dataclasses import dataclass

from .data_loader import valid_symbol


@dataclass(frozen=True)
class Compliance:
    symbol: str
    status: str
    report_date: str | None = None
    error: str | None = None


def fetch_compliance(symbol, api_key):
    if not valid_symbol(symbol):
        return Compliance(symbol, "UNKNOWN", error="Invalid ticker.")
    if not api_key.startswith("live-"):
        return Compliance(
            symbol,
            "UNKNOWN",
            error="A live Zoya key is required. Sandbox data is randomized and cannot verify compliance.",
        )
    query = "query StockReport($symbol: String!) { basicCompliance { report(symbol: $symbol) { symbol status reportDate } } }"
    req = urllib.request.Request(
        "https://api.zoya.finance/graphql",
        data=json.dumps({"query": query, "variables": {"symbol": symbol}}).encode(),
        headers={"Content-Type": "application/json", "Authorization": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            payload = json.load(response)
        if payload.get("errors"):
            return Compliance(
                symbol,
                "UNKNOWN",
                error="Zoya returned an API error; verify your subscription and symbol.",
            )
        report = ((payload.get("data") or {}).get("basicCompliance") or {}).get("report") or {}
        if report.get("symbol") != symbol:
            return Compliance(symbol, "UNKNOWN", error="No matching report returned.")
        status = report.get("status", "UNKNOWN")
        if status not in {"COMPLIANT", "NON_COMPLIANT", "QUESTIONABLE"}:
            status = "UNKNOWN"
        return Compliance(symbol, status, report.get("reportDate"))
    except Exception:
        return Compliance(
            symbol, "UNKNOWN", error="Zoya could not be reached or the response was invalid."
        )
