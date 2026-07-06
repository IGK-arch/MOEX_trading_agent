"""Phase 27.9 — session_aware_sizing helper tests."""

from __future__ import annotations

from datetime import UTC, datetime

from app.risk.session_aware_sizing import (
    apply_session_multiplier,
    is_strategy_allowed,
    is_ticker_allowed,
    session_magnitude_floor,
    session_min_meta_score,
    session_passes_filters,
    should_skip_last_minutes,
)
from app.utils.session_profile import MSK_OFFSET


def _msk_ts(hour: int, minute: int = 0) -> datetime:
    """Msk ts."""
    return datetime(2026, 5, 26, hour, minute, tzinfo=MSK_OFFSET).astimezone(UTC)


def test_midday_multiplier_is_1_2() -> None:
    """Midday at 13:00 МСК scales by 1.2."""
    n, lbl = apply_session_multiplier(100_000, _msk_ts(13, 0))
    assert n == 120_000.0
    assert lbl == "midday"


def test_evening_multiplier_is_0_5() -> None:
    """Evening at 20:00 МСК scales by 0.5."""
    n, lbl = apply_session_multiplier(100_000, _msk_ts(20, 0))
    assert n == 50_000.0
    assert lbl == "evening"


def test_morning_open_multiplier_is_0_7() -> None:
    """Morning_open at 10:15 МСК scales by 0.7."""
    n, lbl = apply_session_multiplier(100_000, _msk_ts(10, 15))
    assert n == 70_000.0
    assert lbl == "morning_open"


def test_non_trading_returns_zero() -> None:
    """Premarket / evening_gap / night → notional 0."""
    for h, m in [(8, 0), (18, 55), (23, 55)]:
        n, lbl = apply_session_multiplier(100_000, _msk_ts(h, m))
        assert n == 0.0
        assert lbl == "non_trading"


def test_strategy_news_allowed_evening() -> None:
    """NEWS is allowed in evening."""
    assert is_strategy_allowed("NEWS", _msk_ts(20, 0)) is True


def test_strategy_ta_blocked_evening() -> None:
    """TA is blocked in evening."""
    assert is_strategy_allowed("TA", _msk_ts(20, 0)) is False


def test_strategy_ta_allowed_midday() -> None:
    """TA is allowed in midday."""
    assert is_strategy_allowed("TA", _msk_ts(14, 0)) is True


def test_strategy_blocked_in_non_trading() -> None:
    """No strategy is allowed during non-trading windows.

    MSK boundaries:
        08:00 → premarket
        18:55 → evening_gap
        23:55 → night (after evening cutoff 23:50)
    """
    for h, m in [(8, 0), (18, 55), (23, 55)]:
        ts = _msk_ts(h, m)
        assert is_strategy_allowed("TA", ts) is False
        assert is_strategy_allowed("NEWS", ts) is False


def test_strategy_case_insensitive() -> None:
    """source argument is case-insensitive."""
    assert is_strategy_allowed("ta", _msk_ts(14, 0)) is True
    assert is_strategy_allowed("News", _msk_ts(20, 0)) is True


def test_ticker_sber_allowed_evening() -> None:
    """SBER passes the evening subset."""
    assert is_ticker_allowed("SBER", _msk_ts(20, 0)) is True


def test_ticker_nlmk_blocked_evening() -> None:
    """NLMK is not on the evening blue-chip subset."""
    assert is_ticker_allowed("NLMK", _msk_ts(20, 0)) is False


def test_ticker_any_allowed_midday() -> None:
    """Midday has no subset → any ticker passes."""
    assert is_ticker_allowed("NLMK", _msk_ts(14, 0)) is True
    assert is_ticker_allowed("SBER", _msk_ts(14, 0)) is True


def test_magnitude_floor_midday() -> None:
    """Test magnitude floor midday."""
    assert session_magnitude_floor(_msk_ts(14, 0)) == 0.30


def test_magnitude_floor_evening() -> None:
    """Test magnitude floor evening."""
    assert session_magnitude_floor(_msk_ts(20, 0)) == 0.50


def test_min_meta_score_midday() -> None:
    """Test min meta score midday."""
    assert session_min_meta_score(_msk_ts(14, 0)) == 0.30


def test_should_skip_last_minutes_evening_close() -> None:
    """Evening skip_last_min=10 → 8 minutes before close should skip."""
    assert should_skip_last_minutes(8 * 60, _msk_ts(23, 0)) is True
    assert should_skip_last_minutes(15 * 60, _msk_ts(23, 0)) is False


def test_should_skip_last_minutes_midday() -> None:
    """Midday has no skip → always False."""
    assert should_skip_last_minutes(10 * 60, _msk_ts(14, 0)) is False


def test_session_passes_filters_midday_ok() -> None:
    """TA + SBER at midday + mag>floor passes."""
    ok, reason = session_passes_filters(
        source="TA",
        ticker="SBER",
        combined_magnitude=0.5,
        ts_utc=_msk_ts(14, 0),
    )
    assert ok is True
    assert reason == ""


def test_session_passes_filters_evening_ta_blocked() -> None:
    """TA blocked in evening."""
    ok, reason = session_passes_filters(
        source="TA",
        ticker="SBER",
        combined_magnitude=0.6,
        ts_utc=_msk_ts(20, 0),
    )
    assert ok is False
    assert "strategy_not_allowed" in reason


def test_session_passes_filters_evening_news_nlmk_blocked() -> None:
    """NEWS + NLMK in evening → ticker not on subset."""
    ok, reason = session_passes_filters(
        source="NEWS",
        ticker="NLMK",
        combined_magnitude=0.6,
        ts_utc=_msk_ts(20, 0),
    )
    assert ok is False
    assert "ticker_not_allowed" in reason


def test_session_passes_filters_magnitude_below_floor() -> None:
    """Below the session floor → rejected."""
    ok, reason = session_passes_filters(
        source="TA",
        ticker="SBER",
        combined_magnitude=0.10,
        ts_utc=_msk_ts(14, 0),
    )
    assert ok is False
    assert "magnitude<" in reason


def test_session_passes_filters_non_trading() -> None:
    """No trading at 03:00 МСК."""
    ok, reason = session_passes_filters(
        source="TA",
        ticker="SBER",
        combined_magnitude=0.9,
        ts_utc=_msk_ts(3, 0),
    )
    assert ok is False
    assert "non_trading_session" in reason
