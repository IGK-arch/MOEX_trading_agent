"""Phase 27.9 — Session-aware sizing helpers.

Thin functions that callers (risk_manager, evening_pipeline, integrators)
can use to adjust a candidate trade's notional / strategy / ticker
membership against the current MOEX sub-session.

This module does NOT mutate any state on its own — it returns numbers
and booleans. ``risk_manager.py`` is intentionally untouched in this
phase; integrators will wire it up in a follow-up step.
"""

from __future__ import annotations

from datetime import datetime

from app.utils.session_profile import (
    SessionLabel,
    current_session,
    get_profile,
    is_trading_session,
)

def apply_session_multiplier(
    base_notional: float,
    ts_utc: datetime | None = None,
) -> tuple[float, str]:
    """Scale a candidate notional by the active session's size multiplier.

    During non-trading windows (premarket, evening_gap, night) the function
    returns ``(0.0, "non_trading")`` so callers can early-exit.

    Args:
        base_notional: notional in RUB before sizing.
        ts_utc: optional UTC ts; defaults to ``utcnow``.
    Returns:
        tuple[float, str]: ``(adjusted_notional, session_label)``.
    """
    label = current_session(ts_utc)
    if not is_trading_session(label):
        return 0.0, "non_trading"
    profile = get_profile(label)
    return float(base_notional) * profile.size_multiplier, label.value

def is_strategy_allowed(
    source: str,
    ts_utc: datetime | None = None,
) -> bool:
    """Whether the given strategy source is allowed in the current session.

    Args:
        source: signal source value such as ``"TA"``, ``"NEWS"``,
            ``"ANOMALY"``, ``"PAIR"``, ``"MEAN_REV"`` (case-insensitive).
        ts_utc: optional UTC ts; defaults to ``utcnow``.
    Returns:
        bool: True if allowed; False otherwise.
    """
    if not source:
        return False
    label = current_session(ts_utc)
    if not is_trading_session(label):
        return False
    profile = get_profile(label)
    return source.upper() in profile.allowed_strategies

def is_ticker_allowed(
    ticker: str,
    ts_utc: datetime | None = None,
) -> bool:
    """Whether the given ticker may trade in the current session.

    Args:
        ticker: instrument code (case-insensitive).
        ts_utc: optional UTC ts; defaults to ``utcnow``.
    Returns:
        bool: True if allowed; False otherwise.
    """
    if not ticker:
        return False
    label = current_session(ts_utc)
    if not is_trading_session(label):
        return False
    profile = get_profile(label)
    subset = profile.allowed_tickers_subset
    if subset is None:
        return True
    return ticker.upper() in subset

def session_magnitude_floor(ts_utc: datetime | None = None) -> float:
    """Return the minimum ``combined_magnitude`` required in this session.

    Non-trading sessions return ``1.0`` (impossible to satisfy), so any
    naive comparison automatically gates them out.
    """
    label = current_session(ts_utc)
    profile = get_profile(label)
    return profile.magnitude_floor

def session_min_meta_score(ts_utc: datetime | None = None) -> float:
    """Return the minimum meta-classifier probability for the current session."""
    label = current_session(ts_utc)
    profile = get_profile(label)
    return profile.min_meta_score

def should_skip_last_minutes(
    seconds_to_close: float,
    ts_utc: datetime | None = None,
) -> bool:
    """True if ``seconds_to_close`` falls inside the session's skip window.

    Some sessions (e.g. closing, evening) specify a ``skip_last_min`` value
    that callers can use to avoid opening trades into the last N minutes
    of that session.

    Args:
        seconds_to_close: seconds until the session boundary.
        ts_utc: optional UTC ts; defaults to ``utcnow``.
    Returns:
        bool: True if we are within ``skip_last_min`` minutes of close.
    """
    if seconds_to_close < 0:
        return True
    label = current_session(ts_utc)
    profile = get_profile(label)
    if profile.skip_last_min <= 0:
        return False
    return seconds_to_close <= profile.skip_last_min * 60

def session_passes_filters(
    *,
    source: str,
    ticker: str,
    combined_magnitude: float,
    ts_utc: datetime | None = None,
) -> tuple[bool, str]:
    """Composite gate combining strategy/ticker/magnitude checks.

    Returns ``(True, "")`` when the trade is allowed by the session
    profile, or ``(False, reason)`` otherwise. Pure function — no side
    effects.
    """
    label = current_session(ts_utc)
    if not is_trading_session(label):
        return False, f"non_trading_session={label.value}"
    profile = get_profile(label)
    if source.upper() not in profile.allowed_strategies:
        return False, f"strategy_not_allowed={source.upper()}@{label.value}"
    if (
        profile.allowed_tickers_subset is not None
        and ticker.upper() not in profile.allowed_tickers_subset
    ):
        return False, f"ticker_not_allowed={ticker.upper()}@{label.value}"
    if combined_magnitude < profile.magnitude_floor:
        return (
            False,
            f"magnitude<{profile.magnitude_floor:.2f}@{label.value}",
        )
    return True, ""

__all__ = [
    "apply_session_multiplier",
    "is_strategy_allowed",
    "is_ticker_allowed",
    "session_magnitude_floor",
    "session_min_meta_score",
    "should_skip_last_minutes",
    "session_passes_filters",
    "SessionLabel",
]
