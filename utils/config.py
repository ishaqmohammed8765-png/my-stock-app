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
    RISK_FREE: float = 0.045  # annualized

    # Volatility Clamps (annualized)
    VOL_FLOOR: float = 0.10
    VOL_CAP: float = 1.80
    VOL_DEFAULT: float = 0.30

    # Sizing Caps (implications: smaller = safer but slower growth)
    KELLY_MIN: float = 0.01
    KELLY_MAX: float = 0.15
    MAX_ALLOC_PCT: float = 0.10  # max % of equity per position
    MAX_RISK_PCT: float = 0.02   # max % of equity risked per trade

    # Execution Costs (bps = 1/100 of 1%)
    BASE_SLIPPAGE_BPS: float = 5.0
    COMMISSION_PER_TRADE: float = 0.00
    WIDE_SPREAD_BPS_WARN: float = 25.0

    # Strategy & Backtest
    MIN_HIST_DAYS: int = 150  # more beginner-friendly default than 240
    MAX_BT_ITERS: int = 800
    BT_OOS_FRAC: float = 0.30
    BT_MIN_TRADES_FOR_KELLY: int = 40

    # Score Thresholds
    SCORE_STRONG_BUY: int = 80
    SCORE_BUY: int = 65
    SCORE_CONDITIONAL: int = 50
    SCORE_HOLD: int = 35

CFG = Config()

_KEY_ID_RE = re.compile(r"^[A-Z0-9]{12,40}$")  # softer range

def validate_keys(api_key: str | None, secret_key: str | None) -> bool:
    """Best-effort format validation only (not an entitlement check)."""
    if not api_key or not secret_key:
        return False
    k = api_key.strip()
    s = secret_key.strip()
    if len(k) < 12 or len(s) < 20:
        return False
    if not _KEY_ID_RE.match(k):
        return False
    return True

def validate_config(cfg: Config = CFG) -> None:
    """Raises ValueError if the config is internally inconsistent."""
    if cfg.TRADING_DAYS <= 0:
        raise ValueError("TRADING_DAYS must be > 0")
    if cfg.VOL_FLOOR <= 0 or cfg.VOL_CAP <= 0:
        raise ValueError("VOL_FLOOR/VOL_CAP must be > 0")
    if cfg.VOL_FLOOR > cfg.VOL_CAP:
        raise ValueError("VOL_FLOOR cannot exceed VOL_CAP")
    if not (0 < cfg.MAX_RISK_PCT <= 1):
        raise ValueError("MAX_RISK_PCT must be in (0, 1]")
    if not (0 < cfg.MAX_ALLOC_PCT <= 1):
        raise ValueError("MAX_ALLOC_PCT must be in (0, 1]")
    if not (0 <= cfg.KELLY_MIN <= cfg.KELLY_MAX <= 1):
        raise ValueError("KELLY_MIN/KELLY_MAX must be within [0, 1] and MIN <= MAX")
    if not (0 <= cfg.BT_OOS_FRAC < 1):
        raise ValueError("BT_OOS_FRAC must be in [0, 1)")
    if not (cfg.SCORE_HOLD <= cfg.SCORE_CONDITIONAL <= cfg.SCORE_BUY <= cfg.SCORE_STRONG_BUY <= 100):
        raise ValueError("Score thresholds must be ordered and <= 100")

def parse_ts(ts: Any, *, default: Optional[datetime] = None, strict: bool = False) -> datetime:
    if default is None:
        default = datetime.now(timezone.utc)

    try:
        if ts is None:
            return default

        # epoch seconds (int/float)
        if isinstance(ts, (int, float)) and math.isfinite(float(ts)):
            # treat as seconds
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)

        # pandas Timestamp / similar
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        if isinstance(ts, datetime):
            dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        if isinstance(ts, str):
            s = ts.strip()
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
    try:
        xf = float(x)
    except Exception:
        return float(lo if default is None else default)

    if not math.isfinite(xf):
        return float(lo if default is None else default)

    if lo > hi:
        lo, hi = hi, lo

    return float(max(lo, min(hi, xf)))
