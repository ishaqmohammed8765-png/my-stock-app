"""Validated immutable research settings; cash is USD throughout."""

import json
import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class Strategy:
    mode: str = "breakout"
    rsi_min: float = 40.0
    rsi_max: float = 70.0
    rvol_min: float = 1.2
    vol_max: float = 1.0
    adx_min: float = 20.0
    atr_entry: float = 0.5
    atr_stop: float = 2.0
    atr_target: float = 3.0
    horizon: int = 20

    def __post_init__(self):
        if self.mode not in {"breakout", "pullback"}:
            raise ValueError("Choose breakout or pullback.")
        for name, value in asdict(self).items():
            if name != "mode" and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number.")
            if name not in {"mode", "horizon"}:
                object.__setattr__(self, name, float(value))
        if not 0 <= self.rsi_min < self.rsi_max <= 100:
            raise ValueError("RSI minimum must be below maximum, within 0–100.")
        if (
            not 0 <= self.rvol_min <= 10
            or not 0.01 <= self.vol_max <= 5
            or not 0 <= self.adx_min <= 100
        ):
            raise ValueError("Volume, volatility or ADX setting is outside its supported range.")
        if (
            not 0 <= self.atr_entry <= 5
            or not 0.1 <= self.atr_stop <= 20
            or not 0.1 <= self.atr_target <= 50
        ):
            raise ValueError("ATR entry must be 0–5; stop 0.1–20 and target 0.1–50.")
        if type(self.horizon) is not int or not 1 <= self.horizon <= 200:
            raise ValueError("Maximum holding period must be 1–200 whole sessions.")


@dataclass(frozen=True)
class Execution:
    capital: float = 10000.0
    risk_pct: float = 0.01
    allocation_pct: float = 0.20
    fixed_amount: float = 0.0
    slippage_bps: float = 5.0
    spread_bps: float = 5.0
    commission: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number.")
            object.__setattr__(self, name, float(value))
        if (
            not 1 <= self.capital <= 1e8
            or not 0.001 <= self.risk_pct <= 0.10
            or not 0.01 <= self.allocation_pct <= 1
        ):
            raise ValueError("Use positive capital, risk 0.1–10%, and allocation 1–100%.")
        if not 0 <= self.fixed_amount <= 1e8 or not 0 <= self.commission <= 1000:
            raise ValueError("Investment amount or commission is outside its supported range.")
        if not 0 <= self.slippage_bps <= 500 or not 0 <= self.spread_bps <= 1000:
            raise ValueError("Execution costs are outside their supported range.")


def export_settings(strategy: Strategy, execution: Execution) -> str:
    return json.dumps(
        {"version": 2, "strategy": asdict(strategy), "execution": asdict(execution)}, indent=2
    )


def import_settings(raw: str) -> tuple[Strategy, Execution]:
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"version", "strategy", "execution"}
        or payload["version"] != 2
    ):
        raise ValueError("Use a version 2 settings export from this app.")
    result = []
    for name, cls in (("strategy", Strategy), ("execution", Execution)):
        values = payload[name]
        if not isinstance(values, dict) or set(values) != {f.name for f in fields(cls)}:
            raise ValueError(f"The {name} settings contain missing or unknown fields.")
        result.append(cls(**values))
    return tuple(result)
