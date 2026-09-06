import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from utils.config import Execution, Strategy, export_settings, import_settings
from utils.data_loader import demo_market, normalise_bars, parse_symbols
from utils.evaluation import evaluate, expectancy_interval
from utils.indicators import add_indicators, wilder
from utils.screener import screen_row
from utils.state import result_key
from utils.strategy import assess


def test_indicator_values_do_not_change_when_future_data_changes():
    df = demo_market().bars
    a = add_indicators(df)
    changed = df.copy()
    changed.iloc[1000:, :4] *= 3
    pd.testing.assert_frame_equal(a.iloc[:1000], add_indicators(changed).iloc[:1000])


def test_flat_rsi_is_neutral_and_wilder_has_sma_seed():
    df = demo_market().bars.iloc[:250].copy()
    df[["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.0]
    assert add_indicators(df).rsi14.iloc[-1] == 50
    smoothed = wilder(pd.Series([1.0, 2.0, 3.0, 8.0]), 3)
    assert np.isnan(smoothed.iloc[1]) and smoothed.iloc[2] == 2 and smoothed.iloc[3] == 4


def test_readiness_and_shared_screen_rule():
    data = demo_market()
    assert not add_indicators(data.bars.iloc[:199]).ind_ready.iloc[-1]
    assert (screen_row(data, Strategy())["Setup"] == "Qualifies") == assess(
        add_indicators(data.bars).iloc[-1], Strategy()
    ).eligible


def test_result_identity_changes_for_ticker_data_currency_or_settings():
    data, s, e = demo_market(), Strategy(), Execution()
    initial = result_key(data, s, e)
    assert initial != result_key(replace(data, symbol="OTHER"), s, e)
    assert initial != result_key(replace(data, currency="GBP"), s, e)
    assert initial != result_key(data, replace(s, atr_stop=3), e)
    assert initial != result_key(data, s, replace(e, commission=1))
    changed = data.bars.copy()
    changed.iloc[-1, changed.columns.get_loc("volume")] += 1
    assert initial != result_key(replace(data, bars=changed), s, e)


def test_cap_filter_rejects_unknown_metadata():
    row = screen_row(demo_market(), Strategy(), max_cap_b=5)
    assert row["Setup"] == "Wait" and row["Market cap (USD B)"] is None
    assert "unknown" in row["Reason"]


def test_completed_bars_and_invalid_prices():
    dates = pd.date_range("2025-01-01", periods=3, tz="UTC")
    raw = pd.DataFrame(
        {
            "open": [100] * 3,
            "high": [101] * 3,
            "low": [99] * 3,
            "close": [100] * 3,
            "volume": [1000] * 3,
        },
        index=dates,
    )
    clean, notes = normalise_bars(raw, now="2025-01-03T18:00:00Z")
    assert len(clean) == 2 and any("Current-day" in w for w in notes)
    raw.loc[dates[0], "high"] = 90
    with pytest.raises(ValueError, match="Inconsistent"):
        normalise_bars(raw, now="2025-01-03T18:00:00Z")


def test_unknown_currency_cannot_qualify():
    row = screen_row(replace(demo_market(), currency=None), Strategy())
    assert row["Setup"] == "Wait" and "currency" in row["Reason"]
    assert row["Close (USD)"] is None


def test_settings_roundtrip_and_reject_untrusted_fields():
    s, e = Strategy(), Execution()
    assert import_settings(export_settings(s, e)) == (s, e)
    bad = json.loads(export_settings(s, e))
    bad["strategy"]["rsi_min"] = 99
    with pytest.raises(ValueError):
        import_settings(json.dumps(bad))
    bad["strategy"]["secret"] = "not-allowed"
    with pytest.raises(ValueError):
        import_settings(json.dumps(bad))
    with pytest.raises(ValueError):
        Strategy(horizon=True)
    with pytest.raises(ValueError):
        Execution(capital=float("nan"))


def test_symbol_input_is_bounded_and_deduplicated():
    assert parse_symbols("aapl, MSFT aapl") == ["AAPL", "MSFT"]
    for raw in ("", "../../secrets", "AAPL, <script>"):
        with pytest.raises(ValueError):
            parse_symbols(raw)


def test_evaluation_uses_later_period_and_independent_windows():
    r = evaluate(add_indicators(demo_market().bars), Strategy(), Execution())
    assert r.development.equity.index[-1] == r.holdout.equity.index[0]
    assert r.holdout.equity.index.equals(r.benchmark.index) and len(r.windows) == 3
    assert r.label == "Insufficient evidence"
    assert r.holdout.metrics["net_pnl"] == pytest.approx(r.holdout.trades.net_pnl.sum())
    if not r.development.trades.empty:
        assert r.development.trades.exit_date.max() < pd.Timestamp(r.split_date, tz="UTC")


def test_short_history_is_not_claimed_as_evidence():
    with pytest.raises(ValueError, match="452"):
        evaluate(add_indicators(demo_market().bars.iloc[:400]), Strategy(), Execution())
    lo, hi = expectancy_interval(pd.Series([1, 2, 3]))
    assert np.isnan(lo) and np.isnan(hi)


def test_all_losing_trade_interval_is_negative_and_repeatable():
    x = pd.Series([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    assert expectancy_interval(x) == expectancy_interval(x) and expectancy_interval(x)[1] < 0


def test_numeric_integer_imports_are_normalised_for_widgets():
    s, e = import_settings(export_settings(Strategy(rsi_min=40), Execution(capital=1000)))
    assert type(s.rsi_min) is float and type(e.capital) is float and type(s.horizon) is int


def test_stale_data_blocks_current_setup():
    data = demo_market()
    old = data.bars.copy()
    old.index = old.index - pd.Timedelta(days=100)
    row = screen_row(replace(data, demo=False, bars=old), Strategy())
    assert row["Setup"] == "Wait" and "Stale" in row["Reason"]
