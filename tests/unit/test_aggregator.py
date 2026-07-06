"""Test SignalAggregator confluence/veto rules."""

import pytest

from app.dispatcher.aggregator import SignalAggregator
from app.dispatcher.signal import (
    DecisionAction,
    Direction,
    SignalSource,
    UnifiedSignal,
)
from app.dispatcher.tier_classifier import apply_tier


def _make_sig(source, direction, mag=0.7, rr=2.0, ticker="SBER"):
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
async def test_ta_news_buy_confluence():
    """Test ta news buy confluence."""
    agg = SignalAggregator()
    sigs = [
        _make_sig(SignalSource.TA, Direction.BUY, mag=0.7),
        _make_sig(SignalSource.NEWS, Direction.BUY, mag=0.6),
    ]
    d = await agg.aggregate("SBER", "c1", sigs)
    apply_tier(d)
    assert d.action == DecisionAction.EXECUTE
    assert d.direction == Direction.BUY

    assert d.combined_magnitude >= 0.7


@pytest.mark.asyncio
async def test_ta_buy_anomaly_sell_veto():
    """Anomaly disagrees with TA → VETO immediately."""
    agg = SignalAggregator()
    sigs = [
        _make_sig(SignalSource.TA, Direction.BUY, mag=0.8),
        _make_sig(SignalSource.ANOMALY, Direction.SELL, mag=0.5),
    ]
    d = await agg.aggregate("SBER", "c1", sigs)
    apply_tier(d)
    assert d.action == DecisionAction.VETO
    assert d.tier.value == "NONE"


@pytest.mark.asyncio
async def test_anomaly_only_gets_reduced_weight():
    """Anomaly-only standalone signals get x0.7 multiplier."""
    agg = SignalAggregator()
    sigs = [_make_sig(SignalSource.ANOMALY, Direction.BUY, mag=1.0, rr=2.0)]
    d = await agg.aggregate("SBER", "c1", sigs)
    apply_tier(d)

    assert d.combined_magnitude <= 0.71
    assert d.action == DecisionAction.EXECUTE


@pytest.mark.asyncio
async def test_three_source_strong_confluence():
    """3 sources agreeing → tiered multiplier (1.7x, was 2.0x flat).

    Avg mag = (0.6 + 0.5 + 0.5) / 3 ≈ 0.533. With CONFLUENCE_TIERED_BOOST
    (default True) the 3-source bucket multiplier is 1.7 → 0.533 × 1.7 ≈ 0.907,
    so the bound stays at ≥ 0.9.
    """
    agg = SignalAggregator()
    sigs = [
        _make_sig(SignalSource.TA, Direction.BUY, mag=0.6),
        _make_sig(SignalSource.NEWS, Direction.BUY, mag=0.5),
        _make_sig(SignalSource.ANOMALY, Direction.BUY, mag=0.5),
    ]
    d = await agg.aggregate("SBER", "c1", sigs)
    apply_tier(d)
    assert d.action == DecisionAction.EXECUTE
    assert d.combined_magnitude >= 0.9


@pytest.mark.asyncio
async def test_empty_signals_no_trade():
    """Test empty signals no trade."""
    agg = SignalAggregator()
    d = await agg.aggregate("SBER", "c1", [])
    assert d.action == DecisionAction.NO_TRADE


@pytest.mark.asyncio
async def test_neutral_only_no_trade():
    """Test neutral only no trade."""
    agg = SignalAggregator()
    sigs = [_make_sig(SignalSource.ANOMALY, Direction.NEUTRAL, mag=0.5)]
    d = await agg.aggregate("SBER", "c1", sigs)
    assert d.action == DecisionAction.NO_TRADE
