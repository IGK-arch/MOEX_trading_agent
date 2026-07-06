"""Расписание торговых сессий MOEX."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, timezone

import app.config as cfg

MSK_OFFSET = timezone(timedelta(hours=3))

MOEX_HOLIDAYS_2026: frozenset[date] = frozenset(
    {
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 7),
        date(2026, 1, 8),
        date(2026, 2, 23),
        date(2026, 3, 9),
        date(2026, 5, 1),
        date(2026, 5, 11),
        date(2026, 6, 12),
        date(2026, 11, 4),
        date(2026, 12, 31),
    }
)

def _now_msk() -> datetime:
    """Now msk."""
    return datetime.now(tz=MSK_OFFSET)

def _time_in_range(t: time, start: time, end: time) -> bool:
    """True if t is in [start, end] inclusive."""
    return start <= t <= end

def is_moex_holiday(d: date) -> bool:
    """True if `d` is a MOEX-declared holiday (markets closed)."""
    return d in MOEX_HOLIDAYS_2026

def is_trading_open(now_utc: datetime | None = None) -> bool:
    """True if ArenaGo is currently in a trading session.

    Расписание ArenaGo (v1.0.6, синхронизировано с панелью соревнования):
    - Будни: 07:00-23:50 МСК (одна непрерывная сессия)
    - Выходные: 10:00-19:00 МСК
    - Праздники MOEX: закрыто

    Args:
        now_utc: optional UTC datetime; defaults to now.
    Returns:
        bool: True if session active.
    """
    if now_utc is None:
        now_utc = datetime.now(tz=UTC)

    now_msk = now_utc.astimezone(MSK_OFFSET)

    if is_moex_holiday(now_msk.date()):
        return False

    t = now_msk.time()
    is_weekend = now_msk.weekday() >= 5

    if is_weekend:
        if not getattr(cfg, "WEEKEND_TRADING_ENABLED", True):
            return False
        return _time_in_range(
            t,
            time(*cfg.WEEKEND_SESSION_OPEN_MSK),
            time(*cfg.WEEKEND_SESSION_CLOSE_MSK),
        )

    return _time_in_range(
        t,
        time(*cfg.MAIN_SESSION_OPEN_MSK),
        time(*cfg.MAIN_SESSION_CLOSE_MSK),
    )

def is_main_session(now_utc: datetime | None = None) -> bool:
    """True only during main session (10:00–18:50 MSK, weekdays)."""
    if now_utc is None:
        now_utc = datetime.now(tz=UTC)
    now_msk = now_utc.astimezone(MSK_OFFSET)
    if now_msk.weekday() >= 5:
        return False
    t = now_msk.time()
    return _time_in_range(
        t,
        time(*cfg.MAIN_SESSION_OPEN_MSK),
        time(*cfg.MAIN_SESSION_CLOSE_MSK),
    )

def is_evening_session(now_utc: datetime | None = None) -> bool:
    """True only during evening session (19:05–23:50 MSK, weekdays)."""
    if now_utc is None:
        now_utc = datetime.now(tz=UTC)
    now_msk = now_utc.astimezone(MSK_OFFSET)
    if now_msk.weekday() >= 5:
        return False
    t = now_msk.time()
    return _time_in_range(
        t,
        time(*cfg.EVENING_SESSION_OPEN_MSK),
        time(*cfg.EVENING_SESSION_CLOSE_MSK),
    )

def seconds_to_close(now_utc: datetime | None = None) -> float:
    """Seconds until current session closes; 0 if not in session."""
    if now_utc is None:
        now_utc = datetime.now(tz=UTC)
    now_msk = now_utc.astimezone(MSK_OFFSET)

    if not is_trading_open(now_utc):
        return 0.0

    today = now_msk.date()

    if is_main_session(now_utc):
        close_msk = datetime.combine(
            today,
            time(*cfg.MAIN_SESSION_CLOSE_MSK),
            tzinfo=MSK_OFFSET,
        )
    else:
        close_msk = datetime.combine(
            today,
            time(*cfg.EVENING_SESSION_CLOSE_MSK),
            tzinfo=MSK_OFFSET,
        )

    return max(0.0, (close_msk - now_msk).total_seconds())

def next_session_open(now_utc: datetime | None = None) -> datetime:
    """UTC datetime of the next session open (skips weekends)."""
    if now_utc is None:
        now_utc = datetime.now(tz=UTC)
    now_msk = now_utc.astimezone(MSK_OFFSET)

    t = now_msk.time()
    main_close = time(*cfg.MAIN_SESSION_CLOSE_MSK)
    eve_open = time(*cfg.EVENING_SESSION_OPEN_MSK)

    if main_close < t < eve_open and now_msk.weekday() < 5:
        next_open_msk = datetime.combine(
            now_msk.date(), time(*cfg.EVENING_SESSION_OPEN_MSK), tzinfo=MSK_OFFSET
        )
        return next_open_msk.astimezone(UTC)

    candidate = now_msk.date() + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)

    next_open_msk = datetime.combine(candidate, time(*cfg.MAIN_SESSION_OPEN_MSK), tzinfo=MSK_OFFSET)
    return next_open_msk.astimezone(UTC)

def current_session_name(now_utc: datetime | None = None) -> str:
    """Returns: 'main', 'evening', or 'closed'."""
    if is_main_session(now_utc):
        return "main"
    if is_evening_session(now_utc):
        return "evening"
    return "closed"

def current_session_label(now_utc: datetime | None = None) -> str:
    """Phase 27.9 — return the fine-grained MOEX sub-session label.

    Wraps :func:`app.utils.session_profile.current_session` and returns the
    string value of the resulting :class:`SessionLabel` (e.g. ``"midday"``,
    ``"evening"``). The function is lazy-imported so legacy callers that do
    not need Phase 27.9 do not pay the import cost.

    Args:
        now_utc: optional UTC datetime; defaults to ``utcnow``.
    Returns:
        str: session label such as ``"morning_open"``, ``"morning"``,
            ``"midday"``, ``"closing"``, ``"evening"``, ``"premarket"``,
            ``"evening_gap"`` or ``"night"``.
    """
    from app.utils.session_profile import current_session

    return current_session(now_utc).value
