from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import logging
import math
import os
import re

logger = logging.getLogger(__name__)


# ---------------------------
# Helpers: env parsing
# ---------------------------

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def _env_get(name: str) -> str | None:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v if v != "" else None


def _parse_int(v: str, *, default: int) -> int:
    try:
        return int(v)
    except Exception:
        logger.warning("Invalid int for env var (using default=%s).", default)
        return default


def _parse_float(v: str, *, default: float) -> float:
    try:
        x = float(v)
        if not math.isfinite(x):
            raise ValueError("non-finite")
        return x
    except Exception:
        logger.warning("Invalid float for env var (using default=%s).", default)
        return default


def _parse_bool(v: str, *, default: bool) -> bool:
    s = v.strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    logger.warning("Invalid bool for env var (using default=%s).", default)
    return default


# ---------------------------
# Config
# ---------------------------

@dataclass(frozen=True, slots=True)
class Config:
    # Market Constants
    TRADING_DAYS: int = 252
    RISK_FREE: float = 0.045  # annualized

    # Volatility Clamps (annualized)
    VOL_FLOOR: float = 0.10
    VOL_CAP: float = 1.80
    VOL_DEFAULT: float = 0.30

    # Sizing Caps
    KELLY_MIN: float = 0.01
    KELLY_MAX: float = 0.15
    MAX_ALLOC_PCT: float = 0.10
    MAX_RISK_PCT: float = 0.02

    # Execution Costs (bps = 1/100 of 1%)
    BASE_SLIPPAGE_BPS: float = 5.0
    COMMISSION_PER_TRADE: float = 0.00
    WIDE_SPREAD_BPS_WARN: float = 25.0

    # Strategy & Backtest
    MIN_HIST_DAYS: int = 150
    MAX_BT_ITERS: int = 800
    BT_OOS_FRAC: float = 0.30
    BT_MIN_TRADES_FOR_KELLY: int = 40

    # Score Thresholds
    SCORE_STRONG_BUY: int = 80
    SCORE_BUY: int = 65
    SCORE_CONDITIONAL: int = 50
    SCORE_HOLD: int = 35

    @classmethod
    def from_env(cls, *, prefix: str = "TRADER_") -> "Config":
        """
        Build a Config using environment variables, falling back to defaults.
        Example: TRADER_TRADING_DAYS=252, TRADER_MAX_RISK_PCT=0.02, ...
        """
        d = cls()  # defaults

        def E(name: str) -> str | None:
            return _env_get(prefix + name)

        return cls(
            TRADING_DAYS=_parse_int(E("TRADING_DAYS") or "", default=d.TRADING_DAYS)
            if E("TRADING_DAYS") is not None else d.TRADING_DAYS,

            RISK_FREE=_parse_float(E("RISK_FREE") or "", default=d.RISK_FREE)
            if E("RISK_FREE") is not None else d.RISK_FREE,

            VOL_FLOOR=_parse_float(E("VOL_FLOOR") or "", default=d.VOL_FLOOR)
            if E("VOL_FLOOR") is not None else d.VOL_FLOOR,

            VOL_CAP=_parse_float(E("VOL_CAP") or "", default=d.VOL_CAP)
            if E("VOL_CAP") is not None else d.VOL_CAP,

            VOL_DEFAULT=_parse_float(E("VOL_DEFAULT") or "", default=d.VOL_DEFAULT)
            if E("VOL_DEFAULT") is not None else d.VOL_DEFAULT,

            KELLY_MIN=_parse_float(E("KELLY_MIN") or "", default=d.KELLY_MIN)
            if E("KELLY_MIN") is not None else d.KELLY_MIN,

            KELLY_MAX=_parse_float(E("KELLY_MAX") or "", default=d.KELLY_MAX)
            if E("KELLY_MAX") is not None else d.KELLY_MAX,

            MAX_ALLOC_PCT=_parse_float(E("MAX_ALLOC_PCT") or "", default=d.MAX_ALLOC_PCT)
            if E("MAX_ALLOC_PCT") is not None else d.MAX_ALLOC_PCT,

            MAX_RISK_PCT=_parse_float(E("MAX_RISK_PCT") or "", default=d.MAX_RISK_PCT)
            if E("MAX_RISK_PCT") is not None else d.MAX_RISK_PCT,

            BASE_SLIPPAGE_BPS=_parse_float(E("BASE_SLIPPAGE_BPS") or "", default=d.BASE_SLIPPAGE_BPS)
            if E("BASE_SLIPPAGE_BPS") is not None else d.BASE_SLIPPAGE_BPS,

            COMMISSION_PER_TRADE=_parse_float(E("COMMISSION_PER_TRADE") or "", default=d.COMMISSION_PER_TRADE)
            if E("COMMISSION_PER_TRADE") is not None else d.COMMISSION_PER_TRADE,

            WIDE_SPREAD_BPS_WARN=_parse_float(E("WIDE_SPREAD_BPS_WARN") or "", default=d.WIDE_SPREAD_BPS_WARN)
            if E("WIDE_SPREAD_BPS_WARN") is not None else d.WIDE_SPREAD_BPS_WARN,

            MIN_HIST_DAYS=_parse_int(E("MIN_HIST_DAYS") or "", default=d.MIN_HIST_DAYS)
            if E("MIN_HIST_DAYS") is not None else d.MIN_HIST_DAYS,

            MAX_BT_ITERS=_parse_int(E("MAX_BT_ITERS") or "", default=d.MAX_BT_ITERS)
            if E("MAX_BT_ITERS") is not None else d.MAX_BT_ITERS,

            BT_OOS_FRAC=_parse_float(E("BT_OOS_FRAC") or "", default=d.BT_OOS_FRAC)
            if E("BT_OOS_FRAC") is not None else d.BT_OOS_FRAC,

            BT_MIN_TRADES_FOR_KELLY=_parse_int(E("BT_MIN_TRADES_FOR_KELLY") or "", default=d.BT_MIN_TRADES_FOR_KELLY)
            if E("BT_MIN_TRADES_FOR_KELLY") is not None else d.BT_MIN_TRADES_FOR_KELLY,

            SCORE_STRONG_BUY=_parse_int(E("SCORE_STRONG_BUY") or "", default=d.SCORE_STRONG_BUY)
            if E("SCORE_STRONG_BUY") is not None else d.SCORE_STRONG_BUY,

            SCORE_BUY=_parse_int(E("SCORE_BUY") or "", default=d.SCORE_BUY)
            if E("SCORE_BUY") is not None else d.SCORE_BUY,

            SCORE_CONDITIONAL=_parse_int(E("SCORE_CONDITIONAL") or "", default=d.SCORE_CONDITIONAL)
            if E("SCORE_CONDITIONAL") is not None else d.SCORE_CONDITIONAL,

            SCORE_HOLD=_parse_int(E("SCORE_HOLD") or "", default=d.SCORE_HOLD)
            if E("SCORE_HOLD") is not None else d.SCORE_HOLD,
        )


CFG = Config()  # or: CFG = Config.from_env()


# ---------------------------
# Key validation
# ---------------------------

_ILLEGAL_KEY_CHARS = re.compile(r"[\s\x00-\x1f\x7f]")  # whitespace + control chars


def validate_keys(api_key: str | None, secret_key: str | None) -> bool:
    """
    Best-effort sanity validation only (not entitlement/subscription check).
    - rejects None/empty
    - rejects whitespace/control characters
    - enforces minimum lengths (loosely)
    """
    if not api_key or not secret_key:
        return False

    k = api_key.strip()
    s = secret_key.strip()

    if len(k) < 8 or len(s) < 12:
        return False

    if _ILLEGAL_KEY_CHARS.search(k) or _ILLEGAL_KEY_CHARS.search(s):
        return False

    return True


# ---------------------------
# Config validation
# ---------------------------

def validate_config(cfg: Config = CFG) -> None:
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


# ---------------------------
# Timestamp parsing
# ---------------------------

def _iso_normalize(s: str) -> str:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    # Trim fractional seconds to 6 digits if present (fromisoformat supports up to microseconds)
    # Example: 2025-01-01T00:00:00.123456789+00:00 -> 2025-01-01T00:00:00.123456+00:00
    if "." in s:
        # split off timezone part if present
        tz_part = ""
        main = s
        if "+" in s[10:] or "-" in s[10:]:
            # find last + or - after the date portion
            plus = s.rfind("+")
            minus = s.rfind("-")
            idx = max(plus, minus)
            if idx > 10:
                main, tz_part = s[:idx], s[idx:]

        if "." in main:
            left, frac = main.split(".", 1)
            frac_digits = "".join(ch for ch in frac if ch.isdigit())
            if len(frac_digits) > 6:
                frac_digits = frac_digits[:6]
            main = left + "." + frac_digits

        s = main + tz_part

    return s


def parse_ts(ts: Any, *, default: datetime | None = None, strict: bool = False) -> datetime:
    if default is None:
        default = datetime.now(timezone.utc)

    try:
        if ts is None:
            return default

        # epoch seconds/millis/micros (int/float)
        if isinstance(ts, (int, float)):
            x = float(ts)
            if not math.isfinite(x):
                raise ValueError("non-finite epoch")

            ax = abs(x)
            # Heuristics:
            # - seconds ~ 1e9
            # - millis  ~ 1e12
            # - micros  ~ 1e15
            if ax >= 1e14:
                x = x / 1_000_000.0
            elif ax >= 1e11:
                x = x / 1_000.0

            return datetime.fromtimestamp(x, tz=timezone.utc)

        # pandas Timestamp / similar
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        if isinstance(ts, datetime):
            dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        if isinstance(ts, str):
            s = _iso_normalize(ts)
            dt = datetime.fromisoformat(s)
            dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        raise TypeError(f"Unsupported timestamp type: {type(ts)!r}")

    except Exception as e:
        if strict:
            raise ValueError(f"Could not parse timestamp: {ts!r}") from e
        logger.warning("Could not parse timestamp %r; using default.", ts, exc_info=False)
        return default


# ---------------------------
# Clamp
# ---------------------------

def clamp(x: float, lo: float, hi: float, *, default: float | None = None) -> float:
    try:
        xf = float(x)
        if not math.isfinite(xf):
            raise ValueError("non-finite")
    except Exception:
        val = float(lo if default is None else default)
        logger.warning("Invalid clamp input %r; using %s.", x, val)
        return val

    if lo > hi:
        lo, hi = hi, lo

    return float(max(lo, min(hi, xf)))
