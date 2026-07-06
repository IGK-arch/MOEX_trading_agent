"""Test TA pivots, indicators, and pattern detection."""

from app.agents.ta_indicators import compute_atr, compute_bollinger, compute_rsi
from app.agents.ta_patterns.pivots import find_pivots, trend_direction


def test_compute_atr_shape(synthetic_double_top):
    """Test compute atr shape."""
    atr = compute_atr(synthetic_double_top, period=14)
    assert len(atr) == len(synthetic_double_top)

    assert atr.iloc[-1] > 0


def test_compute_rsi_in_range(synthetic_double_top):
    """Test compute rsi in range."""
    rsi = compute_rsi(synthetic_double_top, period=14)

    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_compute_bollinger_columns(synthetic_double_top):
    """Test compute bollinger columns."""
    bb = compute_bollinger(synthetic_double_top, period=20, std_dev=2.0)
    assert "BBM" in bb.columns
    assert "BBU" in bb.columns
    assert "BBL" in bb.columns

    valid = bb.dropna()
    if len(valid) > 0:
        assert (valid["BBU"] >= valid["BBM"]).all()
        assert (valid["BBM"] >= valid["BBL"]).all()


def test_pivots_found_in_double_top(synthetic_double_top):
    """Test pivots found in double top."""
    atr = compute_atr(synthetic_double_top)
    pivots = find_pivots(synthetic_double_top, order=5, atr=atr)

    assert len(pivots) >= 3
    highs = [p for p in pivots if p.kind == "H"]
    lows = [p for p in pivots if p.kind == "L"]
    assert len(highs) >= 1
    assert len(lows) >= 1


def test_trend_direction_uptrend(synthetic_uptrend):
    """Test trend direction uptrend."""
    atr = compute_atr(synthetic_uptrend)
    pivots = find_pivots(synthetic_uptrend, order=5, atr=atr)
    if len(pivots) >= 4:
        trend = trend_direction(pivots)
        assert trend in ("UP", "SIDEWAYS", "UNDEFINED")


def test_trend_direction_downtrend(synthetic_downtrend):
    """Test trend direction downtrend."""
    atr = compute_atr(synthetic_downtrend)
    pivots = find_pivots(synthetic_downtrend, order=5, atr=atr)
    if len(pivots) >= 4:
        trend = trend_direction(pivots)
        assert trend in ("DOWN", "SIDEWAYS", "UNDEFINED")


def test_pattern_signal_has_valid_rr():
    """Any PatternSignal must have entry/stop/target consistent with direction."""
    from app.agents.ta_patterns.reversal import PatternSignal

    s = PatternSignal(
        pattern="double_top",
        direction="SELL",
        confidence=0.7,
        bar_idx=50,
        entry=100.0,
        stop=102.0,
        target=96.0,
        expected_rr=2.0,
        atr_at_entry=1.0,
    )
    assert s.confidence == 0.7
    assert s.expected_rr >= 0

    s2 = PatternSignal(
        pattern="x",
        direction="SELL",
        confidence=1.5,
        bar_idx=0,
        entry=0,
        stop=0,
        target=0,
        expected_rr=-0.5,
        atr_at_entry=0,
    )
    assert s2.confidence == 1.0
    assert s2.expected_rr == 0.0
