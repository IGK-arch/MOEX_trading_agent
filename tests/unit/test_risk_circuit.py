"""Test risk manager sizing + circuit breakers."""

from datetime import UTC

from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    DecisionTier,
    Direction,
    SignalSource,
    UnifiedSignal,
)
from app.dispatcher.tier_classifier import classify_tier, tier_size_pct
from app.risk.circuit_breakers import CircuitState


def _build_decision(mag, rr, ticker="SBER"):
    """Build decision."""
    sig = UnifiedSignal(
        source=SignalSource.TA,
        detector="test",
        ticker=ticker,
        direction=Direction.BUY,
        magnitude=mag,
        raw_confidence=mag,
        horizon_min=60,
        price=100.0,
        entry_level=100.0,
        stop_level=98.0,
        target_level=104.0,
        expected_rr=rr,
        atr=1.0,
    )
    return Decision(
        decision_id="test_dec",
        cycle_id="c1",
        ticker=ticker,
        action=DecisionAction.EXECUTE,
        direction=Direction.BUY,
        combined_magnitude=mag,
        signals=[sig],
        expected_rr=rr,
    )


def test_tier1_classification():
    """Test tier1 classification."""
    d = _build_decision(mag=0.80, rr=2.5)
    tier = classify_tier(d)
    assert tier == DecisionTier.TIER1


def test_tier2_classification():
    """Test tier2 classification."""
    d = _build_decision(mag=0.60, rr=1.2)
    tier = classify_tier(d)
    assert tier == DecisionTier.TIER2


def test_tier3_classification():
    """Test tier3 classification."""
    d = _build_decision(mag=0.30, rr=0.9)
    tier = classify_tier(d)
    assert tier == DecisionTier.TIER3


def test_no_tier_below_thresholds():
    """Test no tier below thresholds."""
    d = _build_decision(mag=0.15, rr=0.3)
    tier = classify_tier(d)
    assert tier == DecisionTier.NONE


def test_pair_trade_always_tier2():
    """Test pair trade always tier2."""
    sig = UnifiedSignal(
        source=SignalSource.PAIR,
        detector="pair_x_y",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=0.40,
        raw_confidence=0.40,
        horizon_min=180,
        price=100.0,
        expected_rr=1.0,
    )
    d = Decision(
        decision_id="t",
        cycle_id="c1",
        ticker="SBER",
        action=DecisionAction.EXECUTE,
        direction=Direction.BUY,
        combined_magnitude=0.40,
        signals=[sig],
        expected_rr=1.0,
    )
    assert classify_tier(d) == DecisionTier.TIER2


def test_tier_sizing_modes():
    """v0.0.32: defaults flipped to LIVE_SIZING=True. Validate semantic ordering."""
    import app.config as cfg

    pct1 = tier_size_pct(DecisionTier.TIER1)
    pct2 = tier_size_pct(DecisionTier.TIER2)
    pct3 = tier_size_pct(DecisionTier.TIER3)

    if cfg.LIVE_SIZING:
        assert pct1 >= 0.02
        assert pct1 > pct2 > pct3 > 0.0
    else:
        assert pct1 == 0.005
        assert pct2 == 0.003
        assert pct3 == 0.0015
    assert tier_size_pct(DecisionTier.NONE) == 0.0


def test_circuit_state_default():
    """Test circuit state default."""
    s = CircuitState()
    assert s.daily_pnl_rub == 0.0
    assert s.sizing_multiplier == 1.0
    assert not s.is_blocked


def test_circuit_streak_multiplier():
    """Test circuit streak multiplier."""
    s = CircuitState(losing_streak=3)
    assert s.sizing_multiplier == 0.5
    s = CircuitState(winning_streak=5)
    assert s.sizing_multiplier == 1.5
    s = CircuitState(losing_streak=2, winning_streak=0)
    assert s.sizing_multiplier == 1.0


def test_circuit_is_blocked_with_future_unblock():
    """Test circuit is blocked with future unblock."""
    from datetime import datetime, timedelta

    future = (datetime.now(tz=UTC) + timedelta(hours=1)).isoformat()
    s = CircuitState(blocked_until_iso=future)
    assert s.is_blocked


def test_circuit_is_blocked_with_past_unblock():
    """Test circuit is blocked with past unblock."""
    from datetime import datetime, timedelta

    past = (datetime.now(tz=UTC) - timedelta(hours=1)).isoformat()
    s = CircuitState(blocked_until_iso=past)
    assert not s.is_blocked
