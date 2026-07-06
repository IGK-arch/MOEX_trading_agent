"""
tests/integration/test_meta_in_dispatcher.py — Aggregator + Meta integration.

Verifies that:
  - When meta_score >= threshold → decision stays EXECUTE.
  - When meta_score < threshold → decision becomes NO_TRADE with meta-reason.
  - Meta-score is recorded on the Decision either way.
"""

from __future__ import annotations

import pytest

from app.agents.meta_classifier import MetaClassifier, MetaContext
from app.dispatcher.aggregator import SignalAggregator
from app.dispatcher.signal import (
    DecisionAction,
    Direction,
    SignalSource,
    UnifiedSignal,
)


def _make_signal(
    source: SignalSource, direction: Direction, magnitude: float = 0.7
) -> UnifiedSignal:
    """Make signal."""
    return UnifiedSignal(
        source=source,
        detector="test",
        ticker="SBER",
        direction=direction,
        magnitude=magnitude,
        raw_confidence=magnitude,
        horizon_min=60,
        price=100.0,
        entry_level=100.0,
        stop_level=98.0,
        target_level=104.0,
        expected_rr=2.0,
        atr=2.0,
    )


class _StubMetaPass(MetaClassifier):
    """Always returns a HIGH score (above default threshold 0.55)."""

    def __init__(self) -> None:
        """Init."""
        super().__init__(model_path=None)

    def score(self, decision, context) -> float:
        """Score."""
        return 0.80


class _StubMetaBlock(MetaClassifier):
    """Always returns a LOW score (below default threshold 0.55)."""

    def __init__(self) -> None:
        """Init."""
        super().__init__(model_path=None)

    def score(self, decision, context) -> float:
        """Score."""
        return 0.20


@pytest.mark.asyncio
async def test_meta_pass_keeps_execute():
    """Test meta pass keeps execute."""
    sigs = [
        _make_signal(SignalSource.TA, Direction.BUY),
        _make_signal(SignalSource.NEWS, Direction.BUY),
    ]
    agg = SignalAggregator(meta_classifier=_StubMetaPass())
    dec = await agg.aggregate("SBER", "cyc1", sigs, meta_context=MetaContext())
    assert dec.action == DecisionAction.EXECUTE
    assert dec.meta_score == pytest.approx(0.80)
    assert dec.meta_threshold == 0.35


@pytest.mark.asyncio
async def test_meta_block_becomes_no_trade():
    """Test meta block becomes no trade."""
    sigs = [
        _make_signal(SignalSource.TA, Direction.BUY),
        _make_signal(SignalSource.NEWS, Direction.BUY),
    ]
    agg = SignalAggregator(meta_classifier=_StubMetaBlock())
    dec = await agg.aggregate("SBER", "cyc2", sigs, meta_context=MetaContext())
    assert dec.action == DecisionAction.NO_TRADE
    assert "meta_score" in dec.rationale.lower() or "no_trade" in dec.rationale.lower()


@pytest.mark.asyncio
async def test_meta_skipped_when_no_context():
    """If meta_context is None, meta-gate is bypassed."""
    sigs = [
        _make_signal(SignalSource.TA, Direction.BUY),
        _make_signal(SignalSource.NEWS, Direction.BUY),
    ]
    agg = SignalAggregator(meta_classifier=_StubMetaBlock())
    dec = await agg.aggregate("SBER", "cyc3", sigs, meta_context=None)

    assert dec.action == DecisionAction.EXECUTE
    assert dec.meta_score is None


@pytest.mark.asyncio
async def test_meta_skipped_when_no_classifier():
    """Test meta skipped when no classifier."""
    sigs = [
        _make_signal(SignalSource.TA, Direction.BUY),
        _make_signal(SignalSource.NEWS, Direction.BUY),
    ]
    agg = SignalAggregator(meta_classifier=None)
    dec = await agg.aggregate("SBER", "cyc4", sigs, meta_context=MetaContext())
    assert dec.action == DecisionAction.EXECUTE
    assert dec.meta_score is None
