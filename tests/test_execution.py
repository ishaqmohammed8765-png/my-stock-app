from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from utils.backtester import buy_and_hold, entry_fill, exit_fill, run_backtest
from utils.config import Execution, Strategy


def bars(rows):
    df = pd.DataFrame(
        rows,
        columns=["open", "high", "low", "close"],
        index=pd.date_range("2025-01-01", periods=len(rows), tz="UTC"),
    )
    for key, value in {
        "volume": 1000,
        "rsi14": 50.0,
        "rvol": 2.0,
        "vol_ann": 0.2,
        "atr14": 1.0,
        "ma50": 99.0,
        "ma200": 98.0,
        "adx14": 30.0,
        "ind_ready": True,
    }.items():
        df[key] = value
    return df


S = Strategy(atr_entry=0, horizon=2)
E = Execution(capital=1000, risk_pct=0.10, allocation_pct=1, slippage_bps=0, spread_bps=0)


def test_entry_day_stop_is_not_missed():
    df = bars([(100, 101, 99, 100), (100, 101, 95, 100), (100, 101, 99, 100)])
    t = run_backtest(df, S, E, start=1).trades.iloc[0]
    assert t.entry_date == t.exit_date == df.index[1]
    assert t.exit_price == 98 and t.reason == "stop" and t.net_pnl == -20


def test_entry_at_open_can_hit_target_same_day():
    df = bars([(100, 101, 99, 100), (100, 105, 99, 104), (104, 105, 103, 104)])
    t = run_backtest(df, S, E, start=1).trades.iloc[0]
    assert t.exit_price == 103 and t.reason == "target"


def test_stop_first_for_ambiguous_entry_day():
    df = bars([(100, 101, 99, 100), (100, 105, 95, 100), (100, 101, 99, 100)])
    r = run_backtest(df, S, E, start=1)
    assert r.trades.iloc[0].reason == "stop" and r.diagnostics["ambiguous_bars"] == 1


def test_opening_gap_target_precedes_later_stop():
    price, reason, _ = exit_fill(pd.Series(dict(open=105, high=106, low=95, close=100)), 98, 103, E)
    assert (price, reason) == (105, "target")


def test_opening_gap_stop_fills_at_gap_not_stop():
    price, reason, _ = exit_fill(pd.Series(dict(open=94, high=100, low=90, close=97)), 98, 103, E)
    assert (price, reason) == (94, "stop")


def test_scheduled_open_exit_precedes_later_stop():
    price, reason, _ = exit_fill(
        pd.Series(dict(open=100, high=101, low=95, close=99)), 98, 103, E, due=True
    )
    assert (price, reason) == (100, "time")


def test_intraday_pullback_does_not_claim_pre_entry_high():
    bar = pd.Series(dict(open=105, high=107, low=99, close=101))
    assert entry_fill(bar, 100, "pullback", E) == (100, False)
    price, _, ambiguous = exit_fill(bar, 98, 103, E, entry_day=True, at_open=False)
    assert price is None and ambiguous


def test_limit_prices_respected_with_costs():
    cfg = replace(E, slippage_bps=100, spread_bps=100)
    price, _ = entry_fill(pd.Series(dict(open=99.9, low=99, high=101)), 100, "pullback", cfg)
    assert price <= 100


def test_rejected_order_does_not_charge_commission():
    r = run_backtest(
        bars([(100, 101, 99, 100)] * 5),
        S,
        replace(E, capital=100, fixed_amount=100, commission=1),
        start=1,
    )
    assert r.trades.empty and (r.equity == 100).all() and r.metrics["total_return"] == 0


def test_trade_pnl_and_equity_reconcile_after_all_costs():
    df = bars(
        [(100, 101, 99, 100), (100, 104, 99, 103), (103, 104, 102, 103), (103, 104, 102, 103)]
    )
    cfg = replace(E, commission=2, slippage_bps=10, spread_bps=10)
    r = run_backtest(df, S, cfg, start=1)
    assert r.trades.net_pnl.sum() == pytest.approx(r.equity.iloc[-1] - cfg.capital)
    assert (r.trades.net_pnl == r.trades.gross_pnl - r.trades.fees).all()
    assert r.metrics["expectancy"] == r.trades.net_pnl.mean()


def test_fees_can_turn_gross_winner_into_net_loser():
    df = bars([(100, 101, 99, 100), (100, 100.2, 99, 100.1), (100.1, 100.2, 99, 100.1)])
    r = run_backtest(df, replace(S, atr_target=0.1), replace(E, commission=2), start=1)
    assert r.trades.iloc[0].gross_pnl > 0 and r.trades.iloc[0].net_pnl < 0
    assert r.metrics["win_rate"] == 0


def test_mark_to_market_reflects_open_losses():
    r = run_backtest(
        bars([(100, 101, 99, 100), (100, 100, 99, 99), (99, 100, 99, 99)]), S, E, start=1
    )
    assert r.equity.iloc[1] == 990
    assert r.metrics["max_drawdown"] == pytest.approx(-0.01)


def test_trend_rule_is_required_by_simulator():
    df = bars([(100, 101, 99, 100)] * 5)
    df["ma200"] = 101
    assert run_backtest(df, S, E, start=1).trades.empty


def test_training_never_reads_beyond_end_boundary():
    df = bars([(100, 101, 99, 100)] * 12)
    a = run_backtest(df, S, E, start=1, end=6)
    df.loc[df.index[6:], ["open", "high", "low", "close"]] *= 10
    b = run_backtest(df, S, E, start=1, end=6)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    pd.testing.assert_series_equal(a.equity, b.equity)
    assert (a.trades.exit_date < df.index[6]).all()


def test_no_final_bar_entry_and_forced_close_uses_close():
    df = bars([(100, 101, 99, 100), (100, 101, 99, 100), (101, 102, 100, 101)])
    r = run_backtest(df, replace(S, horizon=20), E, start=1)
    assert len(r.trades) == 1 and r.trades.iloc[0].reason == "end_of_window"
    assert r.trades.iloc[0].exit_price == 101 and r.diagnostics["forced_final_exits"] == 1


def test_benchmark_matches_hand_calculated_cash_ledger():
    df = bars([(100, 101, 99, 100), (100, 101, 99, 100), (110, 111, 109, 110)])
    equity = buy_and_hold(df, replace(E, commission=1), 1, 3)
    assert equity.iloc[-1] == 1088 and equity.index.equals(df.index)


def test_random_paths_reconcile_without_margin():
    close = 100 + np.cumsum(np.random.default_rng(8).normal(0, 1, 120))
    df = bars([(float(c), float(c + 3), float(c - 3), float(c)) for c in close])
    df.ma50, df.ma200 = df.close - 1, df.close - 2
    r = run_backtest(df, S, replace(E, commission=0.1, slippage_bps=15, spread_bps=10), start=1)
    assert len(r.trades) > 5 and (r.equity >= 0).all()
    assert r.metrics["net_pnl"] == pytest.approx(r.trades.net_pnl.sum())


def test_entry_costs_cannot_create_a_favourable_immediate_stop_fill():
    bar = pd.Series(dict(open=100, high=106, low=99, close=102))
    cfg = replace(E, slippage_bps=500)
    entry, at_open = entry_fill(bar, 100, "breakout", cfg)
    assert entry == 105
    price, reason, _ = exit_fill(bar, 103, 108, cfg, entry_day=True, at_open=at_open)
    assert price == 95 and reason == "stop"
