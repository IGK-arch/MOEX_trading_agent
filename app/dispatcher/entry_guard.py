"""Late-stage entry guard.

v1.0.4 — Cycle #4. The aggregator already filters with confluence / meta /
tier; the risk_manager sizes and caps. By the time `OrderManager.submit()`
is called, the bot is essentially committed. Empirically a non-trivial slice
of losing trades happens when microstructure flips between signal-creation
(~10–20 s ago, intra-cycle) and the actual order submit:

  * spread blows out (illiquid bar, news micro-shock);
  * OFI reverses sharply against our direction;
  * an instantaneous price spike pushes the entry far beyond the level
    the strategy assumed.

`EntryGuard.confirm_entry()` runs a quick "look once more before you jump"
check on the SuperCandles slice the dispatcher already fetched for this
cycle. It is intentionally:

  - **read-only** (no I/O, no broker calls) — keeps the latency budget;
  - **fail-open** — if any input is missing it returns ``(True, "no_data")``
    so we never block the funnel on a data outage;
  - **toggleable** via ``cfg.ENTRY_GUARD_ENABLED``.

Three checks:

  1. **Spread widening** — current bar's relative spread > ``MAX_SPREAD_MULT``
     × average spread over the previous N bars → reject.
  2. **OFI reversal** — OFI computed over the most recent window has
     flipped against ``direction`` and exceeds
     ``OFI_REVERSE_THRESHOLD`` in absolute value → reject.
  3. **Price spike** — last-bar close return exceeds
     ``PRICE_SPIKE_MAX`` (e.g. 0.5 %) → reject.

A rejection bumps ``cycle_rejections["late_stage_check"]`` in the
dispatcher and the decision is *not* submitted. The decision is persisted
with ``rationale`` prefixed by ``REJECTED_LATE_STAGE_CHECK``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import app.config as cfg
from app.agents.microstructure import compute_ofi_series
from app.dispatcher.signal import Decision, Direction
from app.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass(frozen=True)
class GuardResult:
    """One late-stage check verdict."""

    ok: bool
    reason: str
    spread_mult: float = 1.0
    ofi_now: float = 0.0
    price_return: float = 0.0

def _safe_get_series(df: Any, col: str) -> Any | None:
    """Return df[col] iff df has it and it's non-empty, else None."""
    try:
        if df is None or len(df) == 0:
            return None
        if col not in getattr(df, "columns", []):
            return None
        return df[col]
    except Exception:
        return None

def _compute_relative_spread(df: Any, lookback: int = 10) -> tuple[float, float] | None:
    """Return (last_spread_pct, avg_spread_pct) over ``lookback`` bars.

    Spread is taken from ``spread_bbo_bps`` first (preferred), falling back
    to ``(high - low) / close`` if quote-level spread is missing.
    """
    try:
        if df is None or len(df) < max(2, lookback):
            return None
        bbo = _safe_get_series(df, "spread_bbo_bps")
        if bbo is not None:
            tail = bbo.tail(lookback).astype(float).fillna(0.0)
            last = float(tail.iloc[-1])
            prev = tail.iloc[:-1]
            avg = float(prev.mean()) if len(prev) > 0 else last
            if avg <= 0:
                return None
            return last, avg

        high = _safe_get_series(df, "high")
        low = _safe_get_series(df, "low")
        close = _safe_get_series(df, "close")
        if high is None or low is None or close is None:
            return None
        tail_hi = high.tail(lookback).astype(float).fillna(0.0)
        tail_lo = low.tail(lookback).astype(float).fillna(0.0)
        tail_cl = close.tail(lookback).astype(float).fillna(0.0)
        spread_pct = ((tail_hi - tail_lo) / tail_cl.replace(0.0, 1e-9)).fillna(0.0)
        last = float(spread_pct.iloc[-1])
        prev = spread_pct.iloc[:-1]
        avg = float(prev.mean()) if len(prev) > 0 else last
        if avg <= 0:
            return None
        return last, avg
    except Exception:
        return None

def _compute_last_return(df: Any) -> float | None:
    """Return |close_t / close_{t-1} - 1| over the last bar."""
    try:
        close = _safe_get_series(df, "close")
        if close is None or len(close) < 2:
            return None
        c1 = float(close.iloc[-1])
        c0 = float(close.iloc[-2])
        if c0 <= 0:
            return None
        return abs(c1 / c0 - 1.0)
    except Exception:
        return None

class EntryGuard:
    """Read-only sanity check applied right before broker submit."""

    def __init__(self) -> None:
        """Init."""
        self._checks_total = 0
        self._rejects_total = 0
        self._reject_reasons: dict[str, int] = {}

    async def confirm_entry(
        self,
        decision: Decision,
        supercandles: Any | None,
    ) -> tuple[bool, str]:
        """Late-stage check before submit.

        Args:
            decision: dispatcher decision about to be sent to the broker.
            supercandles: most-recent supercandles slice (may be None).
        Returns:
            ``(can_execute, reason)`` — ``can_execute=False`` aborts the
            submit. The ``reason`` field is stamped onto ``decision.rationale``
            by the caller.
        """
        self._checks_total += 1
        if not getattr(cfg, "ENTRY_GUARD_ENABLED", True):
            return True, "guard_disabled"

        if supercandles is None:
            return True, "no_data"

        try:
            n = len(supercandles)
        except Exception:
            n = 0
        if n < 3:
            return True, "no_data"

        direction = decision.direction
        dir_u = getattr(direction, "value", str(direction)).upper()
        if dir_u not in ("BUY", "SELL"):
            return True, "neutral_direction"

        max_spread_mult = float(getattr(cfg, "ENTRY_GUARD_MAX_SPREAD_MULT", 1.5))
        ofi_threshold = float(getattr(cfg, "ENTRY_GUARD_OFI_REVERSE_THRESHOLD", 0.4))
        price_spike_max = float(getattr(cfg, "ENTRY_GUARD_PRICE_SPIKE_MAX", 0.005))

        spread_pair = _compute_relative_spread(supercandles, lookback=10)
        spread_mult = 1.0
        if spread_pair is not None:
            last_sp, avg_sp = spread_pair
            if avg_sp > 0:
                spread_mult = last_sp / avg_sp
            if spread_mult > max_spread_mult:
                return self._reject(
                    decision,
                    f"spread_blown_{spread_mult:.2f}x",
                    spread_mult=spread_mult,
                )

        try:
            ofi_now = float(compute_ofi_series(supercandles, window=cfg.OFI_WINDOW_BARS))
        except Exception:
            ofi_now = 0.0

        opposes_buy = dir_u == "BUY" and ofi_now < -ofi_threshold
        opposes_sell = dir_u == "SELL" and ofi_now > ofi_threshold
        if opposes_buy or opposes_sell:
            return self._reject(
                decision,
                f"ofi_reversed_{ofi_now:+.2f}",
                spread_mult=spread_mult,
                ofi_now=ofi_now,
            )

        last_ret = _compute_last_return(supercandles)
        price_return = float(last_ret) if last_ret is not None else 0.0
        if last_ret is not None and last_ret > price_spike_max:
            return self._reject(
                decision,
                f"price_spike_{price_return:.4f}",
                spread_mult=spread_mult,
                ofi_now=ofi_now,
                price_return=price_return,
            )

        return True, "ok"

    def _reject(
        self,
        decision: Decision,
        reason: str,
        *,
        spread_mult: float = 1.0,
        ofi_now: float = 0.0,
        price_return: float = 0.0,
    ) -> tuple[bool, str]:
        """Record a rejection and return ``(False, reason)``."""
        self._rejects_total += 1
        bucket = reason.split("_")[0] if "_" in reason else reason
        self._reject_reasons[bucket] = self._reject_reasons.get(bucket, 0) + 1
        logger.info(
            "EntryGuard REJECT",
            extra={
                "decision_id": getattr(decision, "decision_id", "?"),
                "ticker": getattr(decision, "ticker", "?"),
                "direction": getattr(decision.direction, "value", str(decision.direction)),
                "reason": reason,
                "spread_mult": round(spread_mult, 3),
                "ofi_now": round(ofi_now, 3),
                "price_return": round(price_return, 5),
            },
        )
        return False, reason

    @property
    def stats(self) -> dict[str, Any]:
        """Cumulative counters for dashboards."""
        return {
            "checks_total": self._checks_total,
            "rejects_total": self._rejects_total,
            "reject_reasons": dict(self._reject_reasons),
        }

_entry_guard: EntryGuard | None = None

def get_entry_guard() -> EntryGuard:
    """Return process-wide :class:`EntryGuard` singleton."""
    global _entry_guard
    if _entry_guard is None:
        _entry_guard = EntryGuard()
    return _entry_guard

__all__ = [
    "EntryGuard",
    "GuardResult",
    "get_entry_guard",
    "Direction",
]
