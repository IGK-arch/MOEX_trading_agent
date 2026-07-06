"""Multi-timeframe (MTF) trend confluence.

v0.0.38 — confirm short-timeframe (10m) signals with higher timeframe
context (60m, daily). Industry standard: don't fight the trend.

Trend per timeframe is +1/0/-1, derived from:
    - EMA(EMA_FAST) vs EMA(EMA_SLOW): direction
    - ADX (period=ADX_PERIOD): magnitude (must exceed adx_min to count as trending)

Confluence score combines a signal's direction against {10m, 60m, daily} trends:
    - 3/3 agree → +1.0 (full credit)
    -  2/3 agree → +0.7
    -  1/3 agree → +0.3
    -  0/3 agree → -0.5 (counter-trend veto candidate)

A flat (0) trend never contributes for or against — it just doesn't add to
the agreement count, and is removed from the denominator.

Usage (aggregator):
    trends = compute_mtf_trend(df_10m, df_60m, df_24h)
    score = mtf_confluence_score(signal_direction, trends)
    if score < 0: VETO else combined_magnitude *= score

v0.0.39 — added MTFConfluence (class) for pattern-level validation. The
class consumes higher-TF and lower-TF candle frames and decides whether a
reversal / continuation pattern has the structural support to fire:

    * reversal+BUY  needs RSI<35 on higher TF AND bullish lower-TF bars
    * reversal+SELL needs RSI>65 on higher TF AND bearish lower-TF bars
    * continuation+BUY  needs EMA20>EMA50 on higher TF + bullish lower-TF
    * continuation+SELL needs EMA20<EMA50 on higher TF + bearish lower-TF

ta_trader.py invokes it AFTER the confluence_filters check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import app.config as cfg
from app.agents.ta_indicators import compute_adx, compute_ema, compute_rsi
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

def _classify_trend(
    df: Any,
    ema_fast: int = cfg.EMA_FAST,
    ema_slow: int = cfg.EMA_SLOW,
    adx_period: int = cfg.ADX_PERIOD,
    adx_min: float = 20.0,
) -> int:
    """Return trend label for a single timeframe.

    Returns:
        +1 (uptrend), 0 (flat / not enough data), -1 (downtrend)

    Trend = EMA_fast vs EMA_slow direction, **gated** by ADX > adx_min.
    If ADX is below the threshold the market is choppy → return 0.
    Any error (missing data, exception in indicators) returns 0 — safe default.
    """
    if not _HAS_PANDAS:
        return 0
    if df is None or not hasattr(df, "empty") or df.empty:
        return 0
    if len(df) < max(ema_slow, adx_period * 2) + 1:
        return 0

    try:
        ema_f = compute_ema(df, period=ema_fast)
        ema_s = compute_ema(df, period=ema_slow)
        adx_df = compute_adx(df, period=adx_period)
    except Exception as exc:
        logger.debug("MTF _classify_trend indicator failure", extra={"error": str(exc)})
        return 0

    if ema_f is None or ema_s is None or adx_df is None:
        return 0
    if getattr(ema_f, "empty", True) or getattr(ema_s, "empty", True):
        return 0
    if not hasattr(adx_df, "columns") or "ADX" not in adx_df.columns:
        return 0

    try:
        last_fast = float(ema_f.iloc[-1])
        last_slow = float(ema_s.iloc[-1])
        last_adx = float(adx_df["ADX"].iloc[-1])
    except (IndexError, ValueError, TypeError):
        return 0

    if last_fast != last_fast or last_slow != last_slow or last_adx != last_adx:
        return 0

    if last_adx < adx_min:
        return 0

    if last_fast > last_slow:
        return 1
    if last_fast < last_slow:
        return -1
    return 0

def compute_mtf_trend(
    df_10m: Any | None,
    df_60m: Any | None,
    df_24h: Any | None,
    adx_min: float = 20.0,
) -> dict[str, int]:
    """Return {trend_10m, trend_60m, trend_daily} ∈ {-1, 0, 1} for each timeframe.

    Missing / short DataFrames produce a flat (0) trend on that timeframe —
    callers should treat 0 as "no opinion" rather than "neutral against".
    """
    return {
        "trend_10m": _classify_trend(df_10m, adx_min=adx_min),
        "trend_60m": _classify_trend(df_60m, adx_min=adx_min),
        "trend_daily": _classify_trend(df_24h, adx_min=adx_min),
    }

def mtf_confluence_score(direction: str, trends: dict[str, int]) -> float:
    """Score a signal's direction against the three MTF trends.

    Args:
        direction: "BUY", "SELL", or "NEUTRAL" (case-insensitive; also accepts
            +1 / -1 / 0 numerically).
        trends: dict from `compute_mtf_trend`.

    Returns:
        Float in {-0.5, 0.3, 0.7, 1.0}:
            3 agree  →  1.0
            2 agree  →  0.7
            1 agree  →  0.3
            0 agree  → -0.5 (counter-trend)

        Special cases:
            - If all 3 trends are flat (0): returns 1.0 (no opinion → neutral,
              caller's signal stands on its own).
            - NEUTRAL direction: returns 1.0 (multiplier left untouched).

    The score is intended to multiply `combined_magnitude`; aggregator should
    VETO on negative score.
    """
    dir_sign = _direction_to_sign(direction)
    if dir_sign == 0:
        return 1.0

    triad = [
        trends.get("trend_10m", 0),
        trends.get("trend_60m", 0),
        trends.get("trend_daily", 0),
    ]
    opinions = [t for t in triad if t != 0]
    if not opinions:
        return 1.0

    agreed = sum(1 for t in opinions if t == dir_sign)
    n = len(opinions)

    if n == 3:
        if agreed == 3:
            return 1.0
        if agreed == 2:
            return 0.7
        if agreed == 1:
            return 0.3
        return -0.5

    if n == 2:
        if agreed == 2:
            return 1.0
        if agreed == 1:
            return 0.3
        return -0.5

    return 1.0 if agreed == 1 else -0.5

def resample_ohlcv(df: Any, rule: str) -> Any:
    """Resample an OHLCV DataFrame to a higher timeframe.

    `rule` is a pandas offset alias ("60min", "1D", etc.). The DataFrame must
    have a DatetimeIndex OR a `timestamp`/`begin` column we can use as one.
    Returns the resampled DataFrame, or the input untouched if pandas is
    unavailable / df is invalid.
    """
    if not _HAS_PANDAS or df is None:
        return df
    if not hasattr(df, "columns") or not hasattr(df, "empty") or df.empty:
        return df
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        return df

    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        ts_col = None
        for cand in ("timestamp", "begin", "ts"):
            if cand in work.columns:
                ts_col = cand
                break
        if ts_col is None:
            return df
        try:
            work[ts_col] = pd.to_datetime(work[ts_col])
            work = work.set_index(ts_col)
        except Exception:
            return df

    agg: dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in work.columns:
        agg["volume"] = "sum"
    try:
        out = work.resample(rule).agg(agg).dropna(subset=["open", "close"])
    except Exception as exc:
        logger.debug("MTF resample failed", extra={"rule": rule, "error": str(exc)})
        return df
    return out

def _direction_to_sign(direction: Any) -> int:
    """Coerce direction (str/int/enum) to {-1, 0, +1}."""
    if isinstance(direction, int):
        if direction > 0:
            return 1
        if direction < 0:
            return -1
        return 0
    if hasattr(direction, "value"):
        direction = direction.value
    if not isinstance(direction, str):
        return 0
    d = direction.upper()
    if d == "BUY" or d == "LONG":
        return 1
    if d == "SELL" or d == "SHORT":
        return -1
    return 0

@dataclass
class MTFContext:
    """Snapshot of higher / lower timeframe state used to score a pattern."""

    higher_trend: str
    higher_exhaustion: bool
    lower_confirmation: bool

class MTFConfluence:
    """Pattern-level multi-timeframe confluence validator.

    Used by `app/agents/ta_trader.py` to gate reversal / continuation
    patterns on their higher-TF context (RSI for reversal exhaustion, EMA
    cross for continuation trend) AND a short-window lower-TF body-color
    confirmation (last 2-3 bars agree with the trade direction).
    """

    def __init__(
        self,
        higher_tf: int = 60,
        lower_tf: int = 5,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        confirm_bars: int = 3,
        ema_fast: int = 20,
        ema_slow: int = 50,
    ) -> None:
        """Init."""
        self.higher_tf = higher_tf
        self.lower_tf = lower_tf
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.confirm_bars = max(2, int(confirm_bars))
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def build_context(
        self,
        df_higher: Any,
        df_lower: Any,
        direction: str,
    ) -> MTFContext:
        """Compute the MTF state snapshot for a single trade decision.

        Returns:
            MTFContext: higher_trend / higher_exhaustion / lower_confirmation.
        """
        higher_trend = self._higher_trend(df_higher)
        higher_exhaustion = self._higher_exhaustion(df_higher, direction)
        lower_confirmation = self._lower_confirmation(df_lower, direction)
        return MTFContext(
            higher_trend=higher_trend,
            higher_exhaustion=higher_exhaustion,
            lower_confirmation=lower_confirmation,
        )

    def validate(
        self,
        family: str,
        direction: str,
        df_higher: Any,
        df_lower: Any,
    ) -> tuple[bool, str]:
        """Decide whether a pattern of (family, direction) is supported.

        Returns:
            (ok, reason) — `ok` is True when the higher- and lower-TF context
            both support the trade. `reason` is a short tag explaining the
            outcome (empty when ok=True).
        """
        family = (family or "").lower()
        sign = _direction_to_sign(direction)
        if sign == 0:
            return True, ""
        ctx = self.build_context(df_higher, df_lower, direction)

        if family == "reversal":
            if not ctx.higher_exhaustion:
                return False, "no_higher_exhaustion"
            if not ctx.lower_confirmation:
                return False, "no_lower_confirmation"
            return True, ""

        if family == "continuation":
            expected = "up" if sign > 0 else "down"
            if ctx.higher_trend != expected:
                return False, f"higher_trend_{ctx.higher_trend}!={expected}"
            if not ctx.lower_confirmation:
                return False, "no_lower_confirmation"
            return True, ""

        return True, ""

    def _higher_trend(self, df_higher: Any) -> str:
        """Higher trend."""
        if not _HAS_PANDAS or df_higher is None:
            return "neutral"
        if not hasattr(df_higher, "columns") or df_higher.empty:
            return "neutral"
        if len(df_higher) < self.ema_slow + 1:
            return "neutral"
        try:
            ema_f = compute_ema(df_higher, period=self.ema_fast)
            ema_s = compute_ema(df_higher, period=self.ema_slow)
        except Exception as exc:
            logger.debug("MTFConfluence._higher_trend ema failure", extra={"error": str(exc)})
            return "neutral"
        if ema_f is None or ema_s is None:
            return "neutral"
        if getattr(ema_f, "empty", True) or getattr(ema_s, "empty", True):
            return "neutral"
        try:
            f_last = float(ema_f.iloc[-1])
            s_last = float(ema_s.iloc[-1])
        except (IndexError, ValueError, TypeError):
            return "neutral"
        if f_last != f_last or s_last != s_last:
            return "neutral"
        if f_last > s_last:
            return "up"
        if f_last < s_last:
            return "down"
        return "neutral"

    def _higher_exhaustion(self, df_higher: Any, direction: str) -> bool:
        """RSI-based exhaustion gate for reversal patterns."""
        if not _HAS_PANDAS or df_higher is None:
            return False
        if not hasattr(df_higher, "columns") or df_higher.empty:
            return False
        try:
            rsi = compute_rsi(df_higher, period=14)
        except Exception as exc:
            logger.debug("MTFConfluence._higher_exhaustion rsi failure", extra={"error": str(exc)})
            return False
        if rsi is None or getattr(rsi, "empty", True):
            return False
        try:
            rsi_last = float(rsi.iloc[-1])
        except (IndexError, ValueError, TypeError):
            return False
        if rsi_last != rsi_last:
            return False
        sign = _direction_to_sign(direction)
        if sign > 0:
            return rsi_last < self.rsi_oversold
        if sign < 0:
            return rsi_last > self.rsi_overbought
        return False

    def _lower_confirmation(self, df_lower: Any, direction: str) -> bool:
        """Confirmation = last `confirm_bars` lower-TF bars trend in the
        signal direction (green for BUY, red for SELL) with non-zero volume.

        Volume confirmation is opportunistic: missing volume column ⇒
        body-color only. Stops false confirmations from a single huge candle
        followed by reversal — we want a sequence.
        """
        if not _HAS_PANDAS or df_lower is None:
            return False
        if not hasattr(df_lower, "columns") or df_lower.empty:
            return False
        if len(df_lower) < self.confirm_bars:
            return False
        sign = _direction_to_sign(direction)
        if sign == 0:
            return False
        tail = df_lower.tail(self.confirm_bars)
        try:
            opens = tail["open"].astype(float).values
            closes = tail["close"].astype(float).values
        except (KeyError, ValueError, TypeError):
            return False
        bodies = closes - opens
        aligned = sum(1 for b in bodies if b > 0) if sign > 0 else sum(1 for b in bodies if b < 0)
        if aligned < self.confirm_bars - 1:
            return False
        if "volume" in tail.columns:
            try:
                vols = tail["volume"].astype(float).values
                return bool(any(v > 0 for v in vols))
            except (ValueError, TypeError):
                return True
        return True

_mtf_confluence_singleton: MTFConfluence | None = None

def get_mtf_confluence() -> MTFConfluence:
    """Module-level singleton helper (mirrors get_hmm_detector / get_ta_trader)."""
    global _mtf_confluence_singleton
    if _mtf_confluence_singleton is None:
        _mtf_confluence_singleton = MTFConfluence(
            higher_tf=int(getattr(cfg, "MTF_HIGHER_TF_MIN", 60)),
            lower_tf=int(getattr(cfg, "MTF_LOWER_TF_MIN", 5)),
        )
    return _mtf_confluence_singleton
