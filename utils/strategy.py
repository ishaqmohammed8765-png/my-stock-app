"""One rule set shared by the current setup, screener and simulator."""
from dataclasses import dataclass
import math
import pandas as pd
from .config import Strategy


@dataclass(frozen=True)
class Setup:
    eligible: bool
    reasons: tuple[str, ...]
    entry: float | None = None
    stop: float | None = None
    target: float | None = None


def assess(row: pd.Series, cfg: Strategy) -> Setup:
    if not bool(row.get("ind_ready", False)):
        return Setup(False, ("Need at least 200 completed daily bars and valid indicators.",))
    needed = ("close", "ma50", "ma200", "rsi14", "rvol", "vol_ann", "adx14", "atr14")
    if not all(math.isfinite(float(row.get(k, float("nan")))) for k in needed):
        return Setup(False, ("Indicators are incomplete.",))
    checks = [
        (row.close > row.ma50 > row.ma200, "Price must be above MA50, with MA50 above MA200."),
        (cfg.rsi_min <= row.rsi14 <= cfg.rsi_max, f"RSI must be {cfg.rsi_min:g}–{cfg.rsi_max:g}."),
        (row.rvol >= cfg.rvol_min, f"Relative volume must be at least {cfg.rvol_min:g}."),
        (row.vol_ann <= cfg.vol_max, f"Annualised volatility must be at most {cfg.vol_max:.0%}."),
        (row.adx14 >= cfg.adx_min, f"ADX must be at least {cfg.adx_min:g}."),
    ]
    direction = 1 if cfg.mode == "breakout" else -1
    entry = float(row.close + direction*cfg.atr_entry*row.atr14)
    stop, target = float(entry-cfg.atr_stop*row.atr14), float(entry+cfg.atr_target*row.atr14)
    reasons = tuple(message for passed, message in checks if not passed)
    if stop <= 0 or entry <= 0:
        return Setup(False, reasons + ("The proposed stop or entry is not positive.",))
    return Setup(not reasons, reasons or ("All configured rules pass. This is a conditional setup, not a profit forecast.",), entry, stop, target)
