"""Descriptive screening without probability scores or missing-data bonuses."""

import numpy as np

from .data_loader import is_stale
from .indicators import add_indicators
from .strategy import assess


def screen_row(data, strategy, max_cap_b=None):
    bars = add_indicators(data.bars)
    last = bars.iloc[-1]
    setup = assess(last, strategy)
    cap = data.metadata.get("marketCap")
    known = isinstance(cap, (int, float)) and np.isfinite(cap) and cap > 0
    eligible, reason = setup.eligible, "; ".join(setup.reasons)
    if data.currency != "USD":
        eligible, reason = False, "USD quote currency could not be verified."
    elif is_stale(data):
        eligible, reason = False, "Stale historical bars."
    elif max_cap_b is not None and (not known or cap > max_cap_b * 1e9):
        eligible, reason = False, "Market cap unknown or above the selected limit."
    resistance = float(bars.high.iloc[-61:-1].max()) if len(bars) >= 61 else np.nan
    distance = (resistance / last.close - 1) * 100
    proximity = (
        "Unknown"
        if not np.isfinite(distance)
        else (
            "Above prior resistance"
            if distance < 0
            else ("Within 3% below resistance" if distance <= 3 else "Below resistance")
        )
    )
    return {
        "Symbol": data.symbol,
        "Setup": "Qualifies" if eligible else "Wait",
        "Close (USD)": float(last.close) if data.currency == "USD" else None,
        "RSI": float(last.rsi14),
        "RVOL": float(last.rvol),
        "ADX": float(last.adx14),
        "Market cap (USD B)": cap / 1e9 if known and data.currency == "USD" else None,
        "Resistance context": proximity,
        "Reason": reason,
    }
