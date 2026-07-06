"""Test UnifiedSignal, Decision, TradeRequest contracts."""

from app.dispatcher.signal import (
    Decision,
    Direction,
    SignalSource,
    TradeRequest,
    UnifiedSignal,
)


def test_unified_signal_ticker_uppercased():
    """Test unified signal ticker uppercased."""
    s = UnifiedSignal(
        source=SignalSource.TA,
        detector="x",
        ticker="sber",
        direction=Direction.BUY,
        magnitude=0.5,
        raw_confidence=0.5,
        horizon_min=60,
        price=100.0,
    )
    assert s.ticker == "SBER"


def test_unified_signal_magnitude_clamped():
    """Test unified signal magnitude clamped."""
    s = UnifiedSignal(
        source=SignalSource.TA,
        detector="x",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=1.5,
        raw_confidence=2.0,
        horizon_min=60,
        price=100.0,
    )
    assert s.magnitude == 1.0
    assert s.raw_confidence == 1.0

    s2 = UnifiedSignal(
        source=SignalSource.TA,
        detector="x",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=-0.2,
        raw_confidence=-0.5,
        horizon_min=60,
        price=100.0,
    )
    assert s2.magnitude == 0.0
    assert s2.raw_confidence == 0.0


def test_decision_id_deterministic():
    """Test decision id deterministic."""
    id1 = Decision.make_id("cycleA", "SBER", ["s1", "s2", "s3"])
    id2 = Decision.make_id("cycleA", "SBER", ["s3", "s2", "s1"])
    assert id1 == id2, "decision_id must be order-independent (sorted)"


def test_decision_id_unique_per_cycle():
    """Test decision id unique per cycle."""
    id1 = Decision.make_id("cycleA", "SBER", ["s1"])
    id2 = Decision.make_id("cycleB", "SBER", ["s1"])
    assert id1 != id2, "different cycle → different decision_id"


def test_trade_request_arena_mapping():
    """Test trade request arena mapping."""
    tr = TradeRequest(
        decision_id="abc",
        ticker="SBER",
        direction=Direction.BUY,
        quantity=10,
        bot="MyBot",
        price_at_signal=100.0,
    )
    order = tr.to_arena_order()
    assert order["direction"] == "B"
    assert order["secid"] == "SBER"
    assert order["quantity"] == 10
    assert order["bot"] == "MyBot"

    tr2 = TradeRequest(
        decision_id="abc",
        ticker="LKOH",
        direction=Direction.SELL,
        quantity=5,
        bot="MyBot",
        price_at_signal=5000.0,
    )
    assert tr2.to_arena_order()["direction"] == "S"


def test_arena_error_parsing():
    """Test arena error parsing."""
    from app.dispatcher.signal import ArenaGoError
    from app.execution.arenago_client import SubmitResult

    r1 = SubmitResult(success=False, message="MARKET CLOSED")
    assert r1.arena_error == ArenaGoError.MARKET_CLOSED

    r2 = SubmitResult(success=False, message="INSUFFICIENT CASH for trade")
    assert r2.arena_error == ArenaGoError.INSUFFICIENT_CASH

    r3 = SubmitResult(success=False, message="BOT MyBot HAS REACHED DAILY TRADE LIMIT")
    assert r3.arena_error == ArenaGoError.DAILY_TRADE_LIMIT

    r4 = SubmitResult(success=False, message="NOT VALID SECID XYZ")
    assert r4.arena_error == ArenaGoError.NOT_VALID_SECID

    r5 = SubmitResult(success=True, message="OK")
    assert r5.arena_error is None
