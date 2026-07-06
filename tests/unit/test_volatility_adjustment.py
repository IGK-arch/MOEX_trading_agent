"""Volatility-adjusted sizing — high/low/normal regime scenarios."""

from __future__ import annotations

import pytest

from app.risk.risk_manager import (
    VOL_HIGH_MULT,
    VOL_HIGH_RATIO,
    VOL_LOW_MULT,
    VOL_LOW_RATIO,
    RiskManager,
)


def _rm_with_median(median_atr: float | None) -> RiskManager:
    """Build a RiskManager whose vol-cache loader returns `median_atr`."""

    async def _fake_loader(ticker: str, lookback_days: int) -> float | None:
        """Fake loader."""
        return median_atr

    rm = RiskManager()
    rm._vol_cache_loader = _fake_loader
    return rm


async def test_high_volatility_shrinks_sizing():
    """Test high volatility shrinks sizing."""
    rm = _rm_with_median(median_atr=1.0)
    mult = await rm._volatility_adjustment("SBER", current_atr=1.6)
    assert mult == pytest.approx(VOL_HIGH_MULT)
    assert mult < 1.0


async def test_high_volatility_extreme_still_clamped():
    """Test high volatility extreme still clamped."""
    rm = _rm_with_median(median_atr=1.0)
    mult = await rm._volatility_adjustment("SBER", current_atr=5.0)
    assert mult == pytest.approx(VOL_HIGH_MULT)


async def test_low_volatility_upsizes_sizing():
    """Test low volatility upsizes sizing."""
    rm = _rm_with_median(median_atr=1.0)
    mult = await rm._volatility_adjustment("SBER", current_atr=0.5)
    assert mult == pytest.approx(VOL_LOW_MULT)
    assert mult > 1.0


async def test_normal_volatility_no_adjustment():
    """Test normal volatility no adjustment."""
    rm = _rm_with_median(median_atr=1.0)
    mult = await rm._volatility_adjustment("SBER", current_atr=1.0)
    assert mult == pytest.approx(1.0)

    mult = await rm._volatility_adjustment("SBER", current_atr=1.49)
    assert mult == pytest.approx(1.0)

    mult = await rm._volatility_adjustment("SBER", current_atr=0.71)
    assert mult == pytest.approx(1.0)


async def test_missing_median_returns_one():
    """Test missing median returns one."""
    rm = _rm_with_median(median_atr=None)
    mult = await rm._volatility_adjustment("SBER", current_atr=2.0)
    assert mult == 1.0


async def test_zero_median_returns_one():
    """Test zero median returns one."""
    rm = _rm_with_median(median_atr=0.0)
    mult = await rm._volatility_adjustment("SBER", current_atr=2.0)
    assert mult == 1.0


async def test_zero_atr_returns_one():
    """Test zero atr returns one."""
    rm = _rm_with_median(median_atr=1.0)
    mult = await rm._volatility_adjustment("SBER", current_atr=0.0)
    assert mult == 1.0


async def test_empty_ticker_returns_one():
    """Test empty ticker returns one."""
    rm = _rm_with_median(median_atr=1.0)
    mult = await rm._volatility_adjustment("", current_atr=2.0)
    assert mult == 1.0


async def test_boundary_at_high_ratio_exact():
    """Exactly at 1.5 → still normal (> is strict)."""
    rm = _rm_with_median(median_atr=1.0)
    mult = await rm._volatility_adjustment("SBER", current_atr=VOL_HIGH_RATIO)
    assert mult == pytest.approx(1.0)
    mult = await rm._volatility_adjustment("SBER", current_atr=VOL_HIGH_RATIO + 0.001)
    assert mult == pytest.approx(VOL_HIGH_MULT)


async def test_boundary_at_low_ratio_exact():
    """Exactly at 0.7 → still normal."""
    rm = _rm_with_median(median_atr=1.0)
    mult = await rm._volatility_adjustment("SBER", current_atr=VOL_LOW_RATIO)
    assert mult == pytest.approx(1.0)
    mult = await rm._volatility_adjustment("SBER", current_atr=VOL_LOW_RATIO - 0.001)
    assert mult == pytest.approx(VOL_LOW_MULT)


async def test_cache_avoids_repeated_loader_calls():
    """Test cache avoids repeated loader calls."""
    calls: list[str] = []

    async def _counting_loader(ticker: str, lookback_days: int) -> float | None:
        """Counting loader."""
        calls.append(ticker)
        return 1.0

    rm = RiskManager()
    rm._vol_cache_loader = _counting_loader

    m1 = await rm._volatility_adjustment("SBER", current_atr=2.0)
    m2 = await rm._volatility_adjustment("SBER", current_atr=2.0)
    assert m1 == pytest.approx(VOL_HIGH_MULT)
    assert m2 == pytest.approx(VOL_HIGH_MULT)
    assert calls == ["SBER"]


async def test_cache_separates_per_ticker():
    """Different tickers must each fetch their own median."""
    seen: list[str] = []

    async def _per_ticker(ticker: str, lookback_days: int) -> float | None:
        """Per ticker."""
        seen.append(ticker)
        return 1.0 if ticker == "SBER" else 2.0

    rm = RiskManager()
    rm._vol_cache_loader = _per_ticker

    sber_mult = await rm._volatility_adjustment("SBER", current_atr=2.0)
    gazp_mult = await rm._volatility_adjustment("GAZP", current_atr=2.0)
    assert sber_mult == pytest.approx(VOL_HIGH_MULT)
    assert gazp_mult == pytest.approx(1.0)
    assert set(seen) == {"SBER", "GAZP"}


def test_compute_quantity_high_vol_reduces_qty():
    """vol_mult=0.7 must produce <= the qty of vol_mult=1.0 (other inputs equal)."""
    from app.dispatcher.signal import (
        Decision,
        DecisionAction,
        DecisionTier,
        Direction,
        SignalSource,
        UnifiedSignal,
    )

    sig = UnifiedSignal(
        source=SignalSource.TA,
        detector="t",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=0.80,
        raw_confidence=0.80,
        horizon_min=60,
        price=100.0,
        entry_level=100.0,
        stop_level=98.0,
        target_level=104.0,
        expected_rr=2.0,
        atr=1.0,
    )
    decision = Decision(
        decision_id="t",
        cycle_id="c",
        ticker="SBER",
        action=DecisionAction.EXECUTE,
        direction=Direction.BUY,
        tier=DecisionTier.TIER1,
        combined_magnitude=0.80,
        signals=[sig],
        expected_rr=2.0,
    )
    rm = RiskManager()
    qty_normal = rm._compute_quantity(decision, price=100.0, atr=1.0, vol_mult=1.0)
    qty_high_vol = rm._compute_quantity(decision, price=100.0, atr=1.0, vol_mult=0.7)
    assert qty_high_vol <= qty_normal


def test_compute_quantity_low_vol_boosts_qty():
    """vol_mult=1.2 must produce >= the qty of vol_mult=1.0 (when not hard-capped)."""
    from app.dispatcher.signal import (
        Decision,
        DecisionAction,
        DecisionTier,
        Direction,
        SignalSource,
        UnifiedSignal,
    )

    sig = UnifiedSignal(
        source=SignalSource.TA,
        detector="t",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=0.30,
        raw_confidence=0.30,
        horizon_min=60,
        price=100.0,
        entry_level=100.0,
        stop_level=98.0,
        target_level=104.0,
        expected_rr=1.0,
        atr=1.0,
    )
    decision = Decision(
        decision_id="t",
        cycle_id="c",
        ticker="SBER",
        action=DecisionAction.EXECUTE,
        direction=Direction.BUY,
        tier=DecisionTier.TIER3,
        combined_magnitude=0.30,
        signals=[sig],
        expected_rr=1.0,
    )
    rm = RiskManager()
    qty_normal = rm._compute_quantity(decision, price=100.0, atr=1.0, vol_mult=1.0)
    qty_low_vol = rm._compute_quantity(decision, price=100.0, atr=1.0, vol_mult=1.2)
    assert qty_low_vol >= qty_normal
