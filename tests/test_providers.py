import io
import json
from dataclasses import replace

import pytest

from utils import data_loader, zoya_api
from utils.data_loader import demo_market, load_market
from utils.services import market


def test_alpaca_fallback_is_labelled_and_normalised(monkeypatch):
    def fail(*args):
        raise OSError("Provider unavailable")

    monkeypatch.setattr(data_loader, "_alpaca", fail)
    monkeypatch.setattr(
        data_loader, "_yahoo", lambda *args: (demo_market().bars, "USD", {"marketCap": 123}, [])
    )
    result = load_market("AAPL", provider="Alpaca", key="example", secret="example")
    assert result.source == "Yahoo Finance" and result.currency == "USD"
    assert any("unavailable" in w for w in result.warnings)


def test_metadata_remains_bound_to_symbol(monkeypatch):
    monkeypatch.setattr(
        data_loader,
        "_yahoo",
        lambda symbol, years: (
            demo_market().bars,
            "USD",
            {"marketCap": 1 if symbol == "AAPL" else 2},
            [],
        ),
    )
    assert load_market("AAPL").metadata["marketCap"] == 1
    assert load_market("MSFT").metadata["marketCap"] == 2


def test_cache_refresh_token_forces_reload(monkeypatch):
    from utils import services

    calls = []

    def fake(symbol, *args):
        calls.append(symbol)
        return replace(demo_market(), symbol=symbol)

    monkeypatch.setattr(services, "load_market", fake)
    market.clear()
    for symbol, refresh in [("AAPL", 0), ("AAPL", 0), ("AAPL", 1), ("MSFT", 1)]:
        market(symbol, 5, "Yahoo", "", "", refresh)
    assert calls == ["AAPL", "AAPL", "MSFT"]
    market.clear()


def test_zoya_uses_documented_auth_and_report_shape(monkeypatch):
    def response(request, timeout):
        assert request.get_header("Authorization") == "live-test"
        body = json.loads(request.data)
        assert "basicCompliance" in body["query"] and body["variables"] == {"symbol": "AAPL"}
        return io.BytesIO(
            json.dumps(
                {
                    "data": {
                        "basicCompliance": {
                            "report": {
                                "symbol": "AAPL",
                                "status": "COMPLIANT",
                                "reportDate": "2026-08-01",
                            }
                        }
                    }
                }
            ).encode()
        )

    monkeypatch.setattr(zoya_api.urllib.request, "urlopen", response)
    result = zoya_api.fetch_compliance("AAPL", "live-test")
    assert result.status == "COMPLIANT" and result.report_date == "2026-08-01"


def test_sandbox_compliance_never_calls_api(monkeypatch):
    monkeypatch.setattr(
        zoya_api.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("Sandbox must not be queried"),
    )
    assert zoya_api.fetch_compliance("AAPL", "sandbox-test").status == "UNKNOWN"


def test_zoya_mismatched_symbol_is_unknown(monkeypatch):
    monkeypatch.setattr(
        zoya_api.urllib.request,
        "urlopen",
        lambda *a, **k: io.BytesIO(
            json.dumps(
                {
                    "data": {
                        "basicCompliance": {"report": {"symbol": "OTHER", "status": "COMPLIANT"}}
                    }
                }
            ).encode()
        ),
    )
    assert zoya_api.fetch_compliance("AAPL", "live-test").status == "UNKNOWN"
