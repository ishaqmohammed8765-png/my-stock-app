from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

@dataclass(frozen=True)
class Config:
    # Market Constants
    TRADING_DAYS: int = 252
    RISK_FREE: float = 0.045

    # Volatility Clamps
    VOL_FLOOR: float = 0.10
    VOL_CAP: float = 1.80
    VOL_DEFAULT: float = 0.30

    # Sizing Caps
    KELLY_MIN: float = 0.01
    KELLY_MAX: float = 0.15
    MAX_ALLOC_PCT: float = 0.10  # 10% max position allocation
    MAX_RISK_PCT: float = 0.02   # 2% max risk per trade

    # Execution Costs
    BASE_SLIPPAGE_BPS: float = 5.0
    COMMISSION_PER_TRADE: float = 0.00
    WIDE_SPREAD_BPS_WARN: float = 25.0

    # Strategy & Backtest
    MIN_HIST_DAYS: int = 240
    MAX_BT_ITERS: int = 800
    BT_OOS_FRAC: float = 0.30
    BT_MIN_TRADES_FOR_KELLY: int = 40

    # Score Thresholds
    SCORE_STRONG_BUY: int = 80
    SCORE_BUY: int = 65
    SCORE_CONDITIONAL: int = 50
    SCORE_HOLD: int = 35

# Initialize the config object for use in other files
CFG = Config()

# Essential Helper Functions to include here
def validate_keys(k: str, s: str) -> bool:
    """Checks if Alpaca keys meet the minimum length requirements."""
    return bool(k and s and len(k) >= 10 and len(s) >= 10)

def parse_ts(ts: Any) -> datetime:
    """Standardizes various timestamp formats into UTC datetime."""
    try:
        if ts is None:
            return datetime.now(timezone.utc)
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        if hasattr(ts, "to_pydatetime"):
            dt = ts.to_pydatetime()
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)

def clamp(x: float, lo: float, hi: float) -> float:
    """Ensures a value stays within a specific min/max range."""
    return float(max(lo, min(hi, x)))
