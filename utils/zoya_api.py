from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional


ZOYA_LIVE_ENDPOINT = "https://api.zoya.finance/graphql"
ZOYA_SANDBOX_ENDPOINT = "https://sandbox-api.zoya.finance/graphql"


def infer_endpoint(api_key: str) -> str:
    k = (api_key or "").strip().lower()
    # Docs: keys are prefixed by environment (e.g. "sandbox-...", "live-...").
    if k.startswith("sandbox-"):
        return ZOYA_SANDBOX_ENDPOINT
    return ZOYA_LIVE_ENDPOINT


@dataclass(frozen=True, slots=True)
class ZoyaCompliance:
    symbol: str
    status: str  # COMPLIANT | NON_COMPLIANT | QUESTIONABLE | UNKNOWN | ERROR
    company_name: Optional[str] = None
    exchange: Optional[str] = None
    report_date: Optional[str] = None
    provider: str = "Zoya"
    error: Optional[str] = None


def _post_graphql(
    endpoint: str,
    *,
    api_key: str,
    query: str,
    variables: dict[str, Any],
    timeout_sec: float = 12.0,
    retries: int = 2,
) -> dict[str, Any]:
    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "my-stock-app/1.0",
    }

    last_err: Exception | None = None
    for i in range(max(1, int(retries) + 1)):
        try:
            req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=float(timeout_sec)) as resp:
                raw = resp.read()
            out = json.loads(raw.decode("utf-8", errors="replace"))
            if not isinstance(out, dict):
                raise RuntimeError("Invalid JSON response.")
            return out
        except urllib.error.HTTPError as e:
            last_err = e
            # Handle rate limiting with a small backoff.
            if int(getattr(e, "code", 0)) == 429 and i < retries:
                time.sleep(0.8 * (i + 1))
                continue
            raise
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(0.4 * (i + 1))
                continue
            raise RuntimeError(f"Zoya request failed: {type(last_err).__name__}: {last_err}") from last_err

    raise RuntimeError(f"Zoya request failed: {type(last_err).__name__}: {last_err}")


def fetch_compliance(symbol: str, *, api_key: str) -> ZoyaCompliance:
    """
    Fetch Shariah compliance status from Zoya.
    Requires a Zoya API key.
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return ZoyaCompliance(symbol="", status="UNKNOWN", error="Empty symbol")

    endpoint = infer_endpoint(api_key)

    # Based on Zoya GraphQL schema: Query -> stock(symbol: String!) { ... }
    # Keep this minimal to reduce breakage if the schema evolves.
    query = """
      query StockCompliance($symbol: String!) {
        stock(symbol: $symbol) {
          symbol
          exchange
          name
          shariahCompliance {
            status
            reportDate
          }
        }
      }
    """
    try:
        payload = _post_graphql(endpoint, api_key=api_key, query=query, variables={"symbol": sym})
        if "errors" in payload and payload["errors"]:
            msg = str(payload["errors"][0].get("message") if isinstance(payload["errors"], list) else payload["errors"])
            return ZoyaCompliance(symbol=sym, status="ERROR", error=msg)

        data = payload.get("data", {})
        stock = data.get("stock") if isinstance(data, dict) else None
        if not isinstance(stock, dict):
            return ZoyaCompliance(symbol=sym, status="UNKNOWN", error="No stock data returned")

        comp = stock.get("shariahCompliance") or {}
        status = (comp.get("status") or "UNKNOWN").upper()
        report_date = comp.get("reportDate")

        # Normalize known values
        if status not in {"COMPLIANT", "NON_COMPLIANT", "QUESTIONABLE", "UNKNOWN"}:
            status = "UNKNOWN"

        return ZoyaCompliance(
            symbol=str(stock.get("symbol") or sym).upper(),
            status=status,
            company_name=str(stock.get("name") or "") or None,
            exchange=str(stock.get("exchange") or "") or None,
            report_date=str(report_date) if report_date else None,
        )
    except Exception as e:
        return ZoyaCompliance(symbol=sym, status="ERROR", error=f"{type(e).__name__}: {e}")

