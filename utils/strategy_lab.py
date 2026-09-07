"""Predeclared selection protocol; final-period prices never enter selection."""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from .portfolio import CANDIDATES, LAB_VERSION, WARMUP, portfolio_metrics, run_portfolio

TRAIN = 756
FORWARD = 126


def select_candidate(panel, cfg, end):
    rows = []
    for candidate in CANDIDATES:
        metrics = portfolio_metrics(
            run_portfolio(panel, candidate, cfg, start=end - TRAIN, end=end)
        )
        eligible = metrics["net_return"] > 0 and abs(metrics["max_drawdown"]) <= cfg.drawdown_stop
        score = (
            metrics["cagr"] / max(0.05, abs(metrics["max_drawdown"])) if eligible else -float("inf")
        )
        rows.append(dict(candidate=candidate, score=score, **metrics))
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
    return (ranked[0]["candidate"] if ranked[0]["score"] > 0 else "cash"), pd.DataFrame(rows)


def prepare_study(panel, cfg):
    cutoff = len(panel.closes) - max(252, len(panel.closes) // 5)
    if cutoff < WARMUP + TRAIN + FORWARD:
        raise ValueError(
            "At least 1,445 aligned daily sessions are required; load 10 years when possible."
        )
    folds = []
    for start in range(WARMUP + TRAIN, cutoff, FORWARD):
        end = min(start + FORWARD, cutoff)
        if end - start < 21:
            continue
        chosen, _ = select_candidate(panel, cfg, start)
        result = run_portfolio(panel, chosen, cfg, start=start, end=end)
        benchmark = run_portfolio(panel, "equal_weight", cfg, start=start, end=end)
        folds.append(
            dict(
                start=str(panel.closes.index[start].date()),
                end=str(panel.closes.index[end - 1].date()),
                candidate=chosen,
                **portfolio_metrics(result),
                benchmark_return=portfolio_metrics(benchmark)["net_return"],
            )
        )
    chosen, ranking = select_candidate(panel, cfg, cutoff)
    manifest = dict(
        version=LAB_VERSION,
        dataset_sha256=panel.digest,
        config=asdict(cfg),
        candidate=chosen,
        candidates=list(CANDIDATES),
        cutoff=cutoff,
        final_start=str(panel.closes.index[cutoff].date()),
        final_end=str(panel.closes.index[-1].date()),
        created_at=datetime.now(timezone.utc).isoformat(),
        synthetic=panel.synthetic,
        sources=panel.sources,
    )
    manifest["plan_sha256"] = fingerprint(manifest)
    return dict(manifest=manifest, ranking=ranking, folds=pd.DataFrame(folds))


def fingerprint(manifest):
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "plan_sha256"}, sort_keys=True
        ).encode()
    ).hexdigest()


def test_frozen(panel, cfg, manifest):
    if (
        manifest["plan_sha256"] != fingerprint(manifest)
        or manifest["dataset_sha256"] != panel.digest
        or manifest["config"] != asdict(cfg)
        or manifest["version"] != LAB_VERSION
    ):
        raise ValueError(
            "Data or configuration changed. This frozen plan cannot be tested against them."
        )
    start = manifest["cutoff"]
    selected = manifest["candidate"]
    return {
        label: run_portfolio(panel, candidate, cfg, start=start, cost_multiplier=cost)
        for label, candidate, cost in [
            ("Selected", selected, 1),
            ("Equal-weight benchmark", "equal_weight", 1),
            ("Selected, double costs", selected, 2),
        ]
    }


def paper_targets(panel, cfg, manifest):
    """Export weights only. No broker orders or historical profit assertion."""
    from .portfolio import target_weights

    if manifest["dataset_sha256"] != panel.digest or manifest["config"] != asdict(cfg):
        raise ValueError("Plan does not match current inputs.")
    weights = target_weights(
        panel.closes, len(panel.closes) - 1, manifest["candidate"], cfg.max_weight
    )
    return pd.DataFrame(
        {
            "symbol": weights.index,
            "target_weight": weights.values,
            "as_of": str(panel.closes.index[-1].date()),
            "mode": "paper observation only",
        }
    )
