"""
tests/unit/test_drawdown_kelly.py — Drawdown-Kelly multiplier correctness.
"""

from __future__ import annotations

import pytest

from app.risk.circuit_breakers import CircuitState


def test_no_dd_no_adjustment():
    """Test no dd no adjustment."""
    s = CircuitState(current_drawdown_pct=0.0)
    assert s.drawdown_kelly_multiplier == 1.0


def test_below_activation_no_adjustment():
    """Test below activation no adjustment."""

    s = CircuitState(current_drawdown_pct=0.01)
    assert s.drawdown_kelly_multiplier == 1.0


def test_half_max_dd_gives_half():
    """Test half max dd gives half."""

    s = CircuitState(current_drawdown_pct=0.04)
    assert s.drawdown_kelly_multiplier == pytest.approx(0.5, abs=0.01)


def test_near_max_dd_floors_at_01():
    """Test near max dd floors at 01."""

    s = CircuitState(current_drawdown_pct=0.08)
    assert s.drawdown_kelly_multiplier == pytest.approx(0.1, abs=0.001)


def test_beyond_max_dd_still_at_floor():
    """Test beyond max dd still at floor."""
    s = CircuitState(current_drawdown_pct=0.20)
    assert s.drawdown_kelly_multiplier == pytest.approx(0.1, abs=0.001)


def test_streak_multiplier_combines():
    """streak × drawdown × regime are independent multiplicative factors."""
    s = CircuitState(losing_streak=3, current_drawdown_pct=0.04)

    assert s.sizing_multiplier == 0.5
    assert s.drawdown_kelly_multiplier == pytest.approx(0.5, abs=0.01)


def test_winning_streak_streak_mult():
    """Test winning streak streak mult."""
    s = CircuitState(winning_streak=5)
    assert s.sizing_multiplier == 1.5
