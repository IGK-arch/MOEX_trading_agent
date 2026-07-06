"""Safe wrapper for chart-pattern detectors.

The dispatcher in `app/agents/ta_trader.py` calls roughly 30 detector
functions per ticker per cycle. Any single detector raising an unexpected
exception on degenerate inputs (empty df, all-NaN OHLCV, missing column,
ATR series of zeros, garbage pivots) would historically be caught by an
ad-hoc ``try/except`` block scattered through the dispatcher — but those
blocks logged at ``WARNING`` / ``ERROR`` level, polluting the runtime log
with what are in fact *expected* edge cases (intraday warm-up bars,
illiquid tickers, MOEX outages that return short / NaN-padded series).

This module gives the dispatcher a single, uniform, idempotent entry
point — :func:`safe_detect` — which:

  * catches **every** exception (BaseException is intentionally not
    caught — KeyboardInterrupt / SystemExit must still propagate);
  * returns the detector's documented "no signal" sentinel (an empty
    list) on failure;
  * logs at :data:`logging.DEBUG` so production logs stay clean while
    still leaving an audit trail for backtest researchers who tail with
    ``--log-cli-level=DEBUG``;
  * is generic enough to wrap reversal / continuation / harmonic / SMC /
    research / Dasha / VPVR / candle detectors regardless of their
    individual call signatures.

Usage in the dispatcher::

    from app.agents.ta_patterns.safe_runner import safe_detect

    for name, detector in REVERSAL_DETECTORS:
        patterns = safe_detect(detector, df, pivots, atr_series,
                               _detector_name=name, _ticker=ticker)
        ...

A successful call returns the detector's own list; a failed call returns
``[]`` — callers can safely ``extend`` the result without checking.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from app.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

def safe_detect(
    detector_func: Callable[..., list[T]],
    *args: Any,
    _detector_name: str | None = None,
    _ticker: str | None = None,
    **kwargs: Any,
) -> list[T]:
    """Run ``detector_func`` and never raise.

    Args:
        detector_func: Any chart-pattern detector that returns ``list[T]``.
        *args: Positional args forwarded to ``detector_func``.
        _detector_name: Optional human-readable name used in DEBUG logs.
            Falls back to ``detector_func.__name__`` if missing.
        _ticker: Optional ticker context for the DEBUG log payload.
        **kwargs: Keyword args forwarded to ``detector_func``.

    Returns:
        The list returned by the detector on success, or ``[]`` on **any**
        failure — including:

          * empty / undersized DataFrames (handled inside detectors but
            occasionally surface ``IndexError`` / ``ValueError`` on
            internal pandas operations);
          * missing OHLCV columns (``KeyError``);
          * ``TypeError`` from string-typed close columns or other
            schema corruption upstream of us;
          * NaN-only / Inf-only ATR series leading to divide-by-zero or
            ``ZeroDivisionError`` in geometry computations;
          * ``np.linalg.LinAlgError`` from polyfit on degenerate pivot
            sequences (already caught locally in some detectors — this
            is the defense-in-depth fallback);
          * any other ``Exception`` subclass.

        ``KeyboardInterrupt``, ``SystemExit`` and other
        :class:`BaseException` subclasses are **not** caught, so manual
        shutdown signals still work as expected.

    Notes:
        The detector contract is "return list" — this wrapper enforces
        the contract even when a buggy detector accidentally returns
        ``None`` (which would otherwise blow up the dispatcher's
        ``.extend()`` call downstream). A non-list, non-None return is
        coerced to ``[]`` with a DEBUG log entry.
    """
    name = _detector_name or getattr(detector_func, "__name__", "<anonymous>")
    try:
        result = detector_func(*args, **kwargs)
    except Exception as exc:
        logger.debug(
            "safe_detect: detector raised — swallowed",
            extra={
                "detector": name,
                "ticker": _ticker,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return []

    if result is None:
        logger.debug(
            "safe_detect: detector returned None — coerced to []",
            extra={"detector": name, "ticker": _ticker},
        )
        return []
    if not isinstance(result, list):
        logger.debug(
            "safe_detect: detector returned non-list — coerced to []",
            extra={
                "detector": name,
                "ticker": _ticker,
                "return_type": type(result).__name__,
            },
        )
        if isinstance(result, (dict, str, bytes)):
            return []
        try:
            return list(result)  # type: ignore[arg-type]
        except Exception:
            return []
    return result

__all__ = ["safe_detect"]
