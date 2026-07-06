"""Phase 27.9 — tier_classifier session_label parameter tests."""

from __future__ import annotations

import pytest

import app.config as cfg
from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    DecisionTier,
    Direction,
    SignalSource,
    UnifiedSignal,
)
from app.dispatcher.tier_classifier import apply_tier


def _build_decision(mag: float, rr: float = 2.5) -> Decision:
    """Build decision."""
    sig = UnifiedSignal(
        source=SignalSource.TA,
        detector="x",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=mag,
        raw_confidence=mag,
        horizon_min=60,
        price=300.0,
        expected_rr=rr,
    )
    return Decision(
        decision_id="d1",
        cycle_id="c1",
        ticker="SBER",
        action=DecisionAction.EXECUTE,
        tier=DecisionTier.NONE,
        direction=Direction.BUY,
        combined_magnitude=mag,
        expected_rr=rr,
        signals=[sig],
        rationale="seed",
    )


def test_apply_tier_default_path_unchanged() -> None:
    """Calling apply_tier with no session_label must match pre-Phase-27.9 behaviour."""
    d = _build_decision(mag=0.55, rr=2.5)
    apply_tier(d)
    assert d.action == DecisionAction.EXECUTE
    assert d.tier != DecisionTier.NONE


def test_apply_tier_default_no_trade_below_threshold() -> None:
    """Pure pre-Phase-27.9 NO_TRADE path still works."""
    d = _build_decision(mag=0.05, rr=1.0)
    apply_tier(d)
    assert d.action == DecisionAction.NO_TRADE
    assert d.tier == DecisionTier.NONE


def test_session_label_ignored_when_feature_disabled() -> None:
    """When SESSION_PROFILE_ENABLED=False the session_label is a no-op."""
    saved = cfg.SESSION_PROFILE_ENABLED
    try:
        cfg.SESSION_PROFILE_ENABLED = False
        d = _build_decision(mag=0.40, rr=2.5)
        apply_tier(d, session_label="evening")
        assert "session_floor" not in d.rationale
    finally:
        cfg.SESSION_PROFILE_ENABLED = saved


@pytest.fixture
def feature_on():
    """Feature on."""
    saved = cfg.SESSION_PROFILE_ENABLED
    cfg.SESSION_PROFILE_ENABLED = True
    yield
    cfg.SESSION_PROFILE_ENABLED = saved


def test_session_floor_blocks_evening_low_magnitude(feature_on) -> None:
    """Magnitude 0.40 < evening floor 0.50 → NO_TRADE with session rationale."""
    d = _build_decision(mag=0.40, rr=2.5)
    apply_tier(d, session_label="evening")
    assert d.action == DecisionAction.NO_TRADE
    assert d.tier == DecisionTier.NONE
    assert "session_floor" in d.rationale
    assert "evening" in d.rationale


def test_session_floor_blocks_midday_low_magnitude(feature_on) -> None:
    """Magnitude 0.10 < midday floor 0.30 → NO_TRADE."""
    d = _build_decision(mag=0.10, rr=2.5)
    apply_tier(d, session_label="midday")
    assert d.action == DecisionAction.NO_TRADE
    assert "session_floor" in d.rationale


def test_session_floor_passes_when_above(feature_on) -> None:
    """Magnitude 0.65 > evening floor 0.50 → continues to tier classification."""
    d = _build_decision(mag=0.65, rr=2.5)
    apply_tier(d, session_label="evening")
    assert "session_floor" not in d.rationale
    assert d.action in (DecisionAction.EXECUTE, DecisionAction.NO_TRADE)


def test_unknown_session_label_falls_back(feature_on) -> None:
    """Unknown labels do not raise; fall back to default tier path."""
    d = _build_decision(mag=0.10, rr=1.0)
    apply_tier(d, session_label="lunar_eclipse")
    assert d.action == DecisionAction.NO_TRADE
    assert "session_floor" not in d.rationale


def test_session_floor_preserves_existing_rationale(feature_on) -> None:
    """When the session floor blocks, the prior rationale is preserved as suffix."""
    d = _build_decision(mag=0.10, rr=2.5)
    d.rationale = "anomaly_high_vol"
    apply_tier(d, session_label="midday")
    assert "session_floor" in d.rationale
    assert "anomaly_high_vol" in d.rationale
