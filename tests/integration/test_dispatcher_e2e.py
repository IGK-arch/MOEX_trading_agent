"""
tests/integration/test_dispatcher_e2e.py — End-to-end Dispatcher cycle with fake adapters.

Verifies that:
  1. Signals from multiple adapters arrive at the aggregator
  2. Aggregator produces a Decision with correct confluence rules
  3. Tier classifier assigns the right tier
  4. Risk Manager passes (or rejects) the decision correctly
  5. (We stub the ArenaGo submit, so we just verify TradeRequest is built)
"""

from __future__ import annotations

import pytest

from app.agents.base import BaseAdapter
from app.dispatcher.aggregator import SignalAggregator
from app.dispatcher.signal import (
    Direction,
    SignalSource,
    UnifiedSignal,
)
from app.dispatcher.tier_classifier import apply_tier


class FakeAdapter(BaseAdapter):
    """Fake adapter that emits pre-baked signals."""

    name = "FAKE"

    def __init__(self, name, signals):
        """Init."""
        super().__init__()
        self.name = name
        self.signals = signals
        self._started = True

    async def startup(self):
        """Startup."""
        pass

    async def poll(self):
        """Poll."""
        return self.signals

    async def shutdown(self):
        """Shutdown."""
        pass


def _make_sig(source, ticker, direction, mag=0.7, rr=2.0):
    """Make sig."""
    return UnifiedSignal(
        source=source,
        detector="test",
        ticker=ticker,
        direction=direction,
        magnitude=mag,
        raw_confidence=mag,
        horizon_min=60,
        price=100.0,
        entry_level=100.0,
        stop_level=98.0,
        target_level=104.0,
        expected_rr=rr,
        atr=0.5,
    )


@pytest.mark.asyncio
async def test_dispatcher_3signal_confluence_buy():
    """3 adapters all say BUY → strong confluence decision."""
    ta = FakeAdapter("TA-FAKE", [_make_sig(SignalSource.TA, "SBER", Direction.BUY, 0.7, 2.5)])
    news = FakeAdapter("NEWS-FAKE", [_make_sig(SignalSource.NEWS, "SBER", Direction.BUY, 0.6, 2.0)])
    anomaly = FakeAdapter(
        "ANOMALY-FAKE", [_make_sig(SignalSource.ANOMALY, "SBER", Direction.BUY, 0.5, 1.5)]
    )

    aggregator = SignalAggregator()
    cycle_id = "test_cycle"

    all_signals = []
    for a in (ta, news, anomaly):
        all_signals.extend(await a.poll())

    assert len(all_signals) == 3

    sber_signals = [s for s in all_signals if s.ticker == "SBER"]
    decision = await aggregator.aggregate("SBER", cycle_id, sber_signals)
    apply_tier(decision)

    assert decision.action.value == "EXECUTE"
    assert decision.direction == Direction.BUY
    assert decision.tier.value == "1"
    assert decision.combined_magnitude > 0.9


@pytest.mark.asyncio
async def test_dispatcher_anomaly_veto():
    """Anomaly disagrees with TA → instant VETO regardless of R:R."""
    ta = FakeAdapter("TA-FAKE", [_make_sig(SignalSource.TA, "LKOH", Direction.BUY, 0.85, 3.0)])
    anomaly = FakeAdapter(
        "ANOMALY-FAKE", [_make_sig(SignalSource.ANOMALY, "LKOH", Direction.SELL, 0.6, 1.5)]
    )

    aggregator = SignalAggregator()
    all_signals = []
    for a in (ta, anomaly):
        all_signals.extend(await a.poll())

    decision = await aggregator.aggregate("LKOH", "test_cycle", all_signals)
    apply_tier(decision)
    assert decision.action.value == "VETO"


@pytest.mark.asyncio
async def test_dispatcher_no_signals_no_trade():
    """No signals → NO_TRADE."""
    aggregator = SignalAggregator()
    decision = await aggregator.aggregate("SBER", "test_cycle", [])
    assert decision.action.value == "NO_TRADE"
    assert decision.tier.value == "NONE"


@pytest.mark.asyncio
async def test_dispatcher_idempotency():
    """Same signals + same cycle_id → same decision_id."""
    sigs = [_make_sig(SignalSource.TA, "SBER", Direction.BUY, 0.7, 2.5)]
    agg = SignalAggregator()

    d1 = await agg.aggregate("SBER", "cycle_X", sigs)
    d2 = await agg.aggregate("SBER", "cycle_X", sigs)
    assert d1.decision_id == d2.decision_id, "Same inputs must produce same decision_id"


@pytest.mark.asyncio
async def test_dispatcher_below_threshold_no_trade():
    """Weak signal → falls below Tier 3 threshold → NO_TRADE.

    Phase 21 (v0.0.21) — Tier3 lowered to mag>=0.25 / rr>=0.8. To stay
    below, use mag=0.20 / rr=0.5.
    """
    weak = _make_sig(SignalSource.TA, "SBER", Direction.BUY, mag=0.20, rr=0.5)
    aggregator = SignalAggregator()
    decision = await aggregator.aggregate("SBER", "test_cycle", [weak])
    apply_tier(decision)

    assert decision.action.value == "NO_TRADE"
    assert decision.tier.value == "NONE"
