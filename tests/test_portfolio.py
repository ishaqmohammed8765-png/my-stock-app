from dataclasses import replace

import numpy as np
import pytest

from utils.data_loader import demo_market
from utils.portfolio import PortfolioConfig, make_panel, run_portfolio, target_weights
from utils.strategy_lab import prepare_study
from utils.strategy_lab import test_frozen as reveal


def panel():
    return make_panel([demo_market("AAA"), demo_market("BBB")])


def test_targets_cannot_see_future():
    p = panel()
    before = target_weights(p.closes, 1000, "momentum_252")
    p.closes.iloc[1001:] *= 100
    assert before.equals(target_weights(p.closes, 1000, "momentum_252"))


def test_cash_ledger_reconciles_and_no_leverage():
    p = panel()
    cfg = PortfolioConfig()
    run = run_portfolio(p, "equal_weight", cfg, start=274)
    cash = cfg.capital
    shares = dict.fromkeys(p.closes.columns, 0)
    for row in run.orders.itertuples():
        direction = 1 if row.side == "buy" else -1
        cash -= direction * row.quantity * row.fill_price + row.commission
        shares[row.symbol] += direction * row.quantity
        assert shares[row.symbol] >= 0
        assert cash >= -1e-8
        assert cash == pytest.approx(row.cash_after)
    assert all(v == 0 for v in shares.values())
    assert cash == pytest.approx(run.equity.iloc[-1])
    assert run.exposure.max() <= 1


def test_final_prices_do_not_change_selection():
    p = panel()
    cfg = PortfolioConfig()
    first = prepare_study(p, cfg)
    cut = first["manifest"]["cutoff"]
    p.closes.iloc[cut:] *= 0.01
    p.opens.iloc[cut:] *= 0.01
    second = prepare_study(p, cfg)
    assert first["manifest"]["candidate"] == second["manifest"]["candidate"]
    assert first["ranking"].equals(second["ranking"])
    assert first["folds"].equals(second["folds"])


def test_changed_config_rejected():
    p = panel()
    cfg = PortfolioConfig()
    plan = prepare_study(p, cfg)["manifest"]
    with pytest.raises(ValueError, match="changed"):
        reveal(p, replace(cfg, commission=2), plan)
    plan["candidate"] = "cash"
    plan["cutoff"] -= 1
    with pytest.raises(ValueError, match="changed"):
        reveal(p, cfg, plan)


def test_missing_sessions_rejected():
    a, b = demo_market("AAA"), demo_market("BBB")
    b.bars = b.bars.drop(b.bars.index[500])
    with pytest.raises(ValueError, match="Missing sessions"):
        make_panel([a, b])


def test_risk_stop_executes_after_observed_close():
    p = panel()
    p.opens.iloc[:] = 100.0
    p.closes.iloc[:] = 100.0
    p.closes.iloc[300:] = 60.0
    p.opens.iloc[301:] = 60.0
    run = run_portfolio(p, "trend_200", PortfolioConfig(), start=274)
    # Flat prices do not qualify; create a prior upward trend.
    p.closes.iloc[:274] = np.linspace(80, 100, 274)[:, None] * np.ones((1, 2))
    run = run_portfolio(p, "trend_200", PortfolioConfig(), start=274)
    stops = run.orders.loc[run.orders.reason.eq("risk_stop")]
    assert len(stops) == 2
    assert stops.date.eq(p.closes.index[301]).all()
    assert run.halted


def test_last_session_does_not_open_new_positions():
    run = run_portfolio(panel(), "equal_weight", PortfolioConfig(), start=1499)
    assert run.orders.empty


def test_large_commissions_do_not_borrow_cash():
    p = panel()
    p.closes.iloc[400:] *= 0.001
    p.opens.iloc[400:] *= 0.001
    run = run_portfolio(
        p, "equal_weight", PortfolioConfig(capital=10000, commission=1000), start=274
    )
    assert run.orders.cash_after.min() >= -1e-8
    assert run.equity.min() >= 0
