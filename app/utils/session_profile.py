"""Phase 27.9 — Session-aware trading profile.

Encodes the 5 MOEX intraday sub-sessions (plus 3 non-trading windows) and
returns a :class:`SessionProfile` per session with the sizing /
strategy / floor parameters to apply.

The module is *purely declarative* — no imports from risk_manager,
dispatcher or executor. Helper modules wrap it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta, timezone
from enum import StrEnum

MSK_OFFSET = timezone(timedelta(hours=3))

class SessionLabel(StrEnum):
    """Labels for every minute of the MOEX trading day."""

    PREMARKET = "premarket"
    MORNING_OPEN = "morning_open"
    MORNING = "morning"
    MIDDAY = "midday"
    CLOSING = "closing"
    EVENING_GAP = "evening_gap"
    EVENING = "evening"
    NIGHT = "night"
    WEEKEND = "weekend"

_SESSION_BOUNDS: dict[SessionLabel, tuple[time, time]] = {
    SessionLabel.PREMARKET: (time(0, 0), time(10, 0)),
    SessionLabel.MORNING_OPEN: (time(10, 0), time(10, 30)),
    SessionLabel.MORNING: (time(10, 30), time(12, 0)),
    SessionLabel.MIDDAY: (time(12, 0), time(17, 0)),
    SessionLabel.CLOSING: (time(17, 0), time(18, 50)),
    SessionLabel.EVENING_GAP: (time(18, 50), time(19, 5)),
    SessionLabel.EVENING: (time(19, 5), time(23, 50)),
    SessionLabel.NIGHT: (time(23, 50), time(23, 59, 59, 999999)),
    SessionLabel.WEEKEND: (time(10, 0), time(19, 0)),
}

@dataclass(frozen=True)
class SessionProfile:
    """Trading parameters that apply during one sub-session."""

    label: SessionLabel
    size_multiplier: float
    magnitude_floor: float
    min_meta_score: float
    allowed_strategies: frozenset[str]
    allowed_tickers_subset: frozenset[str] | None = None
    skip_microstructure_gates: bool = False
    skip_last_min: int = 0
    rationale: str = ""

    def is_trading(self) -> bool:
        """True if this profile permits trading at all."""
        return self.size_multiplier > 0.0 and bool(self.allowed_strategies)

SESSION_PROFILES: dict[SessionLabel, SessionProfile] = {
    SessionLabel.PREMARKET: SessionProfile(
        label=SessionLabel.PREMARKET,
        size_multiplier=0.0,
        magnitude_floor=1.0,
        min_meta_score=1.0,
        allowed_strategies=frozenset(),
        skip_microstructure_gates=False,
        skip_last_min=0,
        rationale="Premarket — no MOEX trading window.",
    ),
    SessionLabel.MORNING_OPEN: SessionProfile(
        label=SessionLabel.MORNING_OPEN,
        size_multiplier=0.7,
        magnitude_floor=0.45,
        min_meta_score=0.45,
        allowed_strategies=frozenset({"NEWS", "MEAN_REV"}),
        skip_microstructure_gates=False,
        skip_last_min=0,
        rationale="Opening volatility — keep only strong news/reversion.",
    ),
    SessionLabel.MORNING: SessionProfile(
        label=SessionLabel.MORNING,
        size_multiplier=0.9,
        magnitude_floor=0.35,
        min_meta_score=0.35,
        allowed_strategies=frozenset({"TA", "ANOMALY", "NEWS", "MEAN_REV"}),
        skip_microstructure_gates=False,
        skip_last_min=0,
        rationale="High liquidity, normal trading.",
    ),
    SessionLabel.MIDDAY: SessionProfile(
        label=SessionLabel.MIDDAY,
        size_multiplier=1.2,
        magnitude_floor=0.30,
        min_meta_score=0.30,
        allowed_strategies=frozenset({"TA", "ANOMALY", "NEWS", "MEAN_REV"}),
        skip_microstructure_gates=False,
        skip_last_min=0,
        rationale="Sweet spot — best R:R, low noise.",
    ),
    SessionLabel.CLOSING: SessionProfile(
        label=SessionLabel.CLOSING,
        size_multiplier=0.8,
        magnitude_floor=0.40,
        min_meta_score=0.40,
        allowed_strategies=frozenset({"TA", "ANOMALY"}),
        skip_microstructure_gates=False,
        skip_last_min=30,
        rationale="Squaring + continuation, news muted.",
    ),
    SessionLabel.EVENING_GAP: SessionProfile(
        label=SessionLabel.EVENING_GAP,
        size_multiplier=0.0,
        magnitude_floor=1.0,
        min_meta_score=1.0,
        allowed_strategies=frozenset(),
        skip_microstructure_gates=False,
        skip_last_min=0,
        rationale="Evening gap — 18:50–19:05 МСК, MOEX closed.",
    ),
    SessionLabel.EVENING: SessionProfile(
        label=SessionLabel.EVENING,
        size_multiplier=0.5,
        magnitude_floor=0.50,
        min_meta_score=0.45,
        allowed_strategies=frozenset({"NEWS"}),
        allowed_tickers_subset=frozenset({"SBER", "GAZP", "LKOH", "ROSN", "VTBR", "PLZL", "NVTK"}),
        skip_microstructure_gates=False,
        skip_last_min=10,
        rationale="Thin book — blue chips + US/global news only.",
    ),
    SessionLabel.NIGHT: SessionProfile(
        label=SessionLabel.NIGHT,
        size_multiplier=0.0,
        magnitude_floor=1.0,
        min_meta_score=1.0,
        allowed_strategies=frozenset(),
        skip_microstructure_gates=False,
        skip_last_min=0,
        rationale="Overnight — no trading.",
    ),
    SessionLabel.WEEKEND: SessionProfile(
        label=SessionLabel.WEEKEND,
        size_multiplier=0.5,
        magnitude_floor=0.45,
        min_meta_score=0.40,
        allowed_strategies=frozenset({"TA", "MEAN_REV"}),
        allowed_tickers_subset=frozenset(
            {"SBER", "GAZP", "LKOH", "ROSN", "VTBR", "GMKN", "NVTK", "MOEX", "PLZL"}
        ),
        skip_microstructure_gates=False,
        skip_last_min=15,
        rationale="Низкая ликвидность выходных — консервативно: blue chips + TA/mean-rev.",
    ),
}

_NON_TRADING_LABELS: frozenset[SessionLabel] = frozenset(
    {
        SessionLabel.PREMARKET,
        SessionLabel.EVENING_GAP,
        SessionLabel.NIGHT,
    }
)

def _to_msk(ts_utc: datetime | None) -> datetime:
    """Coerce a UTC datetime to МСК; default = now."""
    if ts_utc is None:
        ts_utc = datetime.now(tz=UTC)
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=UTC)
    return ts_utc.astimezone(MSK_OFFSET)

def current_session(ts_utc: datetime | None = None) -> SessionLabel:
    """Return the :class:`SessionLabel` containing ``ts_utc``.

    v1.0.7 — Saturday/Sunday between 10:00-19:00 МСК now route to
    :attr:`SessionLabel.WEEKEND` (ArenaGo opens a synthetic weekend window).
    Outside that band on weekends we fall through to the regular weekday
    labels, which all evaluate to non-trading on the upstream
    :func:`is_trading_open` guard.

    Args:
        ts_utc: optional UTC datetime; defaults to ``utcnow``.
    Returns:
        SessionLabel: the sub-session label, including non-trading windows.
    """
    msk = _to_msk(ts_utc)
    t = msk.time()

    if msk.weekday() >= 5 and time(10, 0) <= t < time(19, 0):
        return SessionLabel.WEEKEND

    if time(0, 0) <= t < time(10, 0):
        return SessionLabel.PREMARKET
    if time(10, 0) <= t < time(10, 30):
        return SessionLabel.MORNING_OPEN
    if time(10, 30) <= t < time(12, 0):
        return SessionLabel.MORNING
    if time(12, 0) <= t < time(17, 0):
        return SessionLabel.MIDDAY
    if time(17, 0) <= t < time(18, 50):
        return SessionLabel.CLOSING
    if time(18, 50) <= t < time(19, 5):
        return SessionLabel.EVENING_GAP
    if time(19, 5) <= t < time(23, 50):
        return SessionLabel.EVENING
    return SessionLabel.NIGHT

def get_profile(label: SessionLabel | None = None) -> SessionProfile:
    """Return the profile for ``label`` (default: the current session).

    Args:
        label: explicit session label or ``None`` for the current one.
    Returns:
        SessionProfile: trading parameters.
    """
    if label is None:
        label = current_session()
    return SESSION_PROFILES[label]

def is_trading_session(label: SessionLabel | None = None) -> bool:
    """Whether ``label`` represents a window where any trading is allowed."""
    if label is None:
        label = current_session()
    return label not in _NON_TRADING_LABELS

def _session_start_msk(label: SessionLabel, day_msk: datetime) -> datetime:
    """Return МСК datetime of the start of ``label`` on ``day_msk``."""
    start, _ = _SESSION_BOUNDS[label]
    return day_msk.replace(
        hour=start.hour,
        minute=start.minute,
        second=0,
        microsecond=0,
    )

def _session_end_msk(label: SessionLabel, day_msk: datetime) -> datetime:
    """Return МСК datetime of the end of ``label`` on ``day_msk``."""
    _, end = _SESSION_BOUNDS[label]
    return day_msk.replace(
        hour=end.hour,
        minute=end.minute,
        second=end.second,
        microsecond=end.microsecond,
    )

def time_until_next_session(
    ts_utc: datetime | None,
    target_label: SessionLabel,
) -> timedelta:
    """Time from ``ts_utc`` until the next occurrence of ``target_label`` start.

    Args:
        ts_utc: current UTC ts (default = now).
        target_label: which session we want to reach.
    Returns:
        timedelta: 0 if already inside the target, otherwise positive delta.
    """
    msk = _to_msk(ts_utc)
    if current_session(ts_utc) == target_label:
        return timedelta(0)

    for offset in range(0, 8):
        candidate_day = msk + timedelta(days=offset)
        candidate = _session_start_msk(target_label, candidate_day)
        if candidate > msk:
            return candidate - msk
    return timedelta(0)

def session_progress_pct(ts_utc: datetime | None = None) -> float:
    """Fraction of the current session elapsed, in ``[0.0, 1.0]``.

    Non-trading windows still return a valid progress fraction so callers
    can use this for telemetry; you should additionally check
    :func:`is_trading_session` before relying on the result for sizing.
    """
    msk = _to_msk(ts_utc)
    label = current_session(ts_utc)
    start = _session_start_msk(label, msk)
    end = _session_end_msk(label, msk)
    total = (end - start).total_seconds()
    if total <= 0:
        return 0.0
    elapsed = (msk - start).total_seconds()
    return max(0.0, min(1.0, elapsed / total))

def session_end_msk(
    label: SessionLabel | None = None,
    ts_utc: datetime | None = None,
) -> datetime:
    """Return МСК datetime of the end of the given session for ``ts_utc``."""
    msk = _to_msk(ts_utc)
    if label is None:
        label = current_session(ts_utc)
    return _session_end_msk(label, msk)

__all__ = [
    "SessionLabel",
    "SessionProfile",
    "SESSION_PROFILES",
    "MSK_OFFSET",
    "current_session",
    "get_profile",
    "is_trading_session",
    "time_until_next_session",
    "session_progress_pct",
    "session_end_msk",
]
