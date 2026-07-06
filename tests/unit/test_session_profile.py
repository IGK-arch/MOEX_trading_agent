"""Phase 27.9 — session profile module tests.

Covers SessionLabel mapping, default profiles, trading vs non-trading
classification, DST-safe UTC→МСК conversion, helper math.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.utils.session_profile import (
    MSK_OFFSET,
    SESSION_PROFILES,
    SessionLabel,
    SessionProfile,
    current_session,
    get_profile,
    is_trading_session,
    session_end_msk,
    session_progress_pct,
    time_until_next_session,
)


def _msk_to_utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """Build a UTC datetime from an МСК wall clock."""
    return datetime(year, month, day, hour, minute, tzinfo=MSK_OFFSET).astimezone(UTC)


@pytest.mark.parametrize(
    "msk_hour,msk_min,expected",
    [
        (10, 15, SessionLabel.MORNING_OPEN),
        (10, 0, SessionLabel.MORNING_OPEN),
        (10, 29, SessionLabel.MORNING_OPEN),
        (10, 30, SessionLabel.MORNING),
        (11, 30, SessionLabel.MORNING),
        (12, 0, SessionLabel.MIDDAY),
        (13, 30, SessionLabel.MIDDAY),
        (16, 59, SessionLabel.MIDDAY),
        (17, 0, SessionLabel.CLOSING),
        (18, 49, SessionLabel.CLOSING),
        (18, 55, SessionLabel.EVENING_GAP),
        (19, 5, SessionLabel.EVENING),
        (19, 30, SessionLabel.EVENING),
        (23, 49, SessionLabel.EVENING),
        (23, 55, SessionLabel.NIGHT),
        (2, 0, SessionLabel.PREMARKET),
        (9, 59, SessionLabel.PREMARKET),
    ],
)
def test_current_session_msk_boundaries(
    msk_hour: int, msk_min: int, expected: SessionLabel
) -> None:
    """Exact 5-session + 3 non-trading session boundaries."""
    ts = _msk_to_utc(2026, 5, 26, msk_hour, msk_min)
    assert current_session(ts) == expected


def test_current_session_defaults_to_utcnow() -> None:
    """When ts_utc is None it must still return a valid label."""
    label = current_session(None)
    assert isinstance(label, SessionLabel)


def test_current_session_handles_naive_datetime() -> None:
    """Naive datetimes are treated as UTC."""
    naive = datetime(2026, 5, 26, 10, 0)
    assert current_session(naive) == SessionLabel.MIDDAY


def test_no_dst_in_msk() -> None:
    """МСК is UTC+3 year-round; verify mid-summer and mid-winter map equally."""
    winter = _msk_to_utc(2026, 1, 15, 13, 0)
    summer = _msk_to_utc(2026, 7, 15, 13, 0)
    assert current_session(winter) == SessionLabel.MIDDAY
    assert current_session(summer) == SessionLabel.MIDDAY


def test_all_sessions_have_profiles() -> None:
    """Every SessionLabel must have a SESSION_PROFILES entry."""
    for label in SessionLabel:
        assert label in SESSION_PROFILES
        prof = SESSION_PROFILES[label]
        assert isinstance(prof, SessionProfile)
        assert prof.label == label


def test_non_trading_sessions_are_disabled() -> None:
    """Non-trading windows must have size_multiplier=0 and empty allowed."""
    for label in (SessionLabel.PREMARKET, SessionLabel.EVENING_GAP, SessionLabel.NIGHT):
        prof = SESSION_PROFILES[label]
        assert prof.size_multiplier == 0.0
        assert prof.allowed_strategies == frozenset()
        assert not prof.is_trading()


def test_midday_is_strongest() -> None:
    """Midday must be the highest-sized, lowest-floor session."""
    midday = SESSION_PROFILES[SessionLabel.MIDDAY]
    others = [
        SESSION_PROFILES[s]
        for s in SessionLabel
        if s not in (SessionLabel.MIDDAY,) and SESSION_PROFILES[s].is_trading()
    ]
    for o in others:
        assert midday.size_multiplier >= o.size_multiplier
        assert midday.magnitude_floor <= o.magnitude_floor


def test_evening_has_ticker_subset() -> None:
    """Evening must restrict to blue chips."""
    prof = SESSION_PROFILES[SessionLabel.EVENING]
    assert prof.allowed_tickers_subset is not None
    assert "SBER" in prof.allowed_tickers_subset
    assert "AFLT" not in prof.allowed_tickers_subset


def test_morning_open_allows_only_news_meanrev() -> None:
    """morning_open allowed_strategies must equal {NEWS, MEAN_REV}."""
    prof = SESSION_PROFILES[SessionLabel.MORNING_OPEN]
    assert prof.allowed_strategies == frozenset({"NEWS", "MEAN_REV"})


def test_closing_has_skip_last_min() -> None:
    """Closing has skip_last_min > 0 to avoid the final minutes."""
    prof = SESSION_PROFILES[SessionLabel.CLOSING]
    assert prof.skip_last_min > 0


def test_get_profile_with_label_and_default() -> None:
    """get_profile should accept a label and fallback to current_session."""
    p = get_profile(SessionLabel.MIDDAY)
    assert p.label == SessionLabel.MIDDAY
    p2 = get_profile(None)
    assert isinstance(p2, SessionProfile)


def test_is_trading_session() -> None:
    """All five active labels are trading; the three gap labels are not."""
    for s in (
        SessionLabel.MORNING_OPEN,
        SessionLabel.MORNING,
        SessionLabel.MIDDAY,
        SessionLabel.CLOSING,
        SessionLabel.EVENING,
    ):
        assert is_trading_session(s) is True
    for s in (SessionLabel.PREMARKET, SessionLabel.EVENING_GAP, SessionLabel.NIGHT):
        assert is_trading_session(s) is False


def test_session_progress_pct_bounds() -> None:
    """Progress is in [0, 1] for any timestamp."""
    for h in range(24):
        for m in (0, 15, 30, 45):
            ts = _msk_to_utc(2026, 5, 26, h, m)
            p = session_progress_pct(ts)
            assert 0.0 <= p <= 1.0


def test_session_progress_pct_midday_start() -> None:
    """At 12:00 МСК (midday start) progress is ~0."""
    ts = _msk_to_utc(2026, 5, 26, 12, 0)
    assert session_progress_pct(ts) < 0.01


def test_session_progress_pct_midday_mid() -> None:
    """At 14:30 МСК (midday is 12:00–17:00) progress is ~0.5."""
    ts = _msk_to_utc(2026, 5, 26, 14, 30)
    p = session_progress_pct(ts)
    assert 0.4 <= p <= 0.6


def test_time_until_next_session_inside_target() -> None:
    """If we are already in target session, delta is 0."""
    midday_ts = _msk_to_utc(2026, 5, 26, 14, 0)
    delta = time_until_next_session(midday_ts, SessionLabel.MIDDAY)
    assert delta == timedelta(0)


def test_time_until_next_session_forward() -> None:
    """Premarket at 09:00 МСК → next midday is in 3h."""
    pre_ts = _msk_to_utc(2026, 5, 26, 9, 0)
    delta = time_until_next_session(pre_ts, SessionLabel.MIDDAY)
    assert delta == timedelta(hours=3)


def test_time_until_next_session_next_day() -> None:
    """Evening 22:00 МСК → next midday is the following day."""
    ev_ts = _msk_to_utc(2026, 5, 26, 22, 0)
    delta = time_until_next_session(ev_ts, SessionLabel.MIDDAY)
    assert timedelta(hours=12) < delta < timedelta(hours=20)


def test_session_end_msk_midday() -> None:
    """end of midday is 17:00 МСК."""
    ts = _msk_to_utc(2026, 5, 26, 13, 0)
    end = session_end_msk(SessionLabel.MIDDAY, ts)
    assert end.hour == 17 and end.minute == 0
