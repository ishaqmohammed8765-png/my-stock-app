from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
import math
import re


@dataclass(frozen=True, slots=True)
class Config:
    # Market Constants
    TRADING_DAYS: int = 252
    RISK_FREE: float = 0.045  # annualized (overrideable if you want)

    # Volatility Clamps (annualized)
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


CFG = Config()


# Alpaca key IDs are typically 20 chars, often start with PK... (paper) or AK... (live).
# Secrets are typically much longer.
_KEY_ID_RE = re.compile(r"^[A-Z0-9]{16,32}$")


def validate_keys(api_key: str | None, secret_key: str | None) -> bool:
    """
    Best-effort validation for Alpaca credentials.
    - Avoids false positives from short strings.
    - Still doesn't *prove* the keys are valid (only an API call can).
    """
    if not api_key or not secret_key:
        return False

    k = api_key.strip()
    s = secret_key.strip()

    # Quick length sanity checks
    if len(k) < 16 or len(s) < 32:
        return False

    # Basic pattern check for key id (Alpaca often uses uppercase alnum key IDs)
    if not _KEY_ID_RE.match(k):
        return False

    return True


def parse_ts(ts: Any, *, default: Optional[datetime] = None, strict: bool = False) -> datetime:
    """
    Standardizes various timestamp formats into a timezone-aware UTC datetime.

    Supported:
    - None -> default (or now UTC)
    - datetime (naive assumed UTC)
    - pandas Timestamp (via to_pydatetime)
    - ISO 8601 strings (handles trailing 'Z')
    """
    if default is None:
        default = datetime.now(timezone.utc)

    try:
        if ts is None:
            return default

        # pandas Timestamp / similar
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        if isinstance(ts, datetime):
            dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        if isinstance(ts, str):
            s = ts.strip()
            # "Z" -> UTC offset for fromisoformat
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

    except Exception as e:
        if strict:
            raise ValueError(f"Could not parse timestamp: {ts!r}") from e

    return default


def clamp(x: float, lo: float, hi: float, *, default: Optional[float] = None) -> float:
    """
    Ensures a value stays within [lo, hi].

    - Handles NaN/inf safely (returns `default` if provided, else clamps to lo).
    - Guarantees float output.
    """
    try:
        xf = float(x)
    except Exception:
        return float(lo if default is None else default)

    if not math.isfinite(xf):
        return float(lo if default is None else default)

    if lo > hi:
        lo, hi = hi, lo  # defensive: swap if misconfigured

    return float(max(lo, min(hi, xf)))
