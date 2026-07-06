"""Partial profit-taking ladder — quantity split + trigger semantics."""

from __future__ import annotations

import pytest

from app.risk.partial_exit import (
    PartialExitPlan,
    plan_partial_exit,
    tp1_hit,
    tp2_hit,
)


def test_explicit_tp1_tp2_even_qty():
    """Test explicit tp1 tp2 even qty."""
    plan = plan_partial_exit(
        quantity=10,
        take_profit=110.0,
        take_profit_1=105.0,
        take_profit_2=110.0,
    )
    assert plan.qty_tp1 == 5
    assert plan.qty_tp2 == 5
    assert plan.tp1_price == 105.0
    assert plan.tp2_price == 110.0
    assert plan.has_partial


def test_explicit_tp1_tp2_odd_qty_no_lots_lost():
    """A 5-lot position must close 2 at TP1 + 3 at TP2 (or 3+2). Never lose lots."""
    plan = plan_partial_exit(
        quantity=5,
        take_profit=110.0,
        take_profit_1=105.0,
    )
    assert plan.qty_tp1 + plan.qty_tp2 == 5
    assert plan.qty_tp1 >= 1
    assert plan.qty_tp2 >= 1


def test_single_lot_no_partial():
    """qty=1 → no partial split possible (a 0.5 lot doesn't exist)."""
    plan = plan_partial_exit(
        quantity=1,
        take_profit=110.0,
        take_profit_1=105.0,
    )
    assert plan.qty_tp1 == 0
    assert plan.qty_tp2 == 1
    assert not plan.has_partial


def test_derive_tp1_midpoint_buy():
    """Entry=100, TP=110 → midpoint TP1=105."""
    plan = plan_partial_exit(
        quantity=10,
        take_profit=110.0,
        entry_price=100.0,
    )
    assert plan.tp1_price == pytest.approx(105.0)
    assert plan.tp2_price == pytest.approx(110.0)
    assert plan.has_partial
    assert plan.qty_tp1 == 5


def test_derive_tp1_midpoint_sell():
    """SELL: entry=100, TP=90 → midpoint TP1=95."""
    plan = plan_partial_exit(
        quantity=10,
        take_profit=90.0,
        entry_price=100.0,
    )
    assert plan.tp1_price == pytest.approx(95.0)
    assert plan.tp2_price == pytest.approx(90.0)


def test_no_tp_or_entry_means_single_exit():
    """No TP1, no entry → fall back to all-or-nothing at TP2."""
    plan = plan_partial_exit(quantity=10, take_profit=110.0)
    assert plan.qty_tp1 == 0
    assert plan.qty_tp2 == 10
    assert not plan.has_partial


def test_zero_quantity_returns_empty():
    """Test zero quantity returns empty."""
    plan = plan_partial_exit(quantity=0, take_profit=110.0, take_profit_1=105.0)
    assert plan.qty_total == 0
    assert plan.qty_tp1 == 0
    assert plan.qty_tp2 == 0
    assert not plan.has_partial


def test_tp1_buy_triggers_at_or_above_price():
    """Test tp1 buy triggers at or above price."""
    assert tp1_hit("BUY", price=105.0, tp1=105.0)
    assert tp1_hit("BUY", price=105.5, tp1=105.0)
    assert not tp1_hit("BUY", price=104.9, tp1=105.0)


def test_tp1_sell_triggers_at_or_below_price():
    """Test tp1 sell triggers at or below price."""
    assert tp1_hit("SELL", price=95.0, tp1=95.0)
    assert tp1_hit("SELL", price=94.5, tp1=95.0)
    assert not tp1_hit("SELL", price=95.1, tp1=95.0)


def test_tp_hits_handle_none():
    """Test tp hits handle none."""
    assert not tp1_hit("BUY", price=110.0, tp1=None)
    assert not tp2_hit("BUY", price=110.0, tp2=None)


def test_tp_hits_handle_zero_price():
    """Test tp hits handle zero price."""
    assert not tp1_hit("BUY", price=0.0, tp1=105.0)


def test_50pct_close_at_tp1_then_remainder_at_tp2_buy():
    """A 10-lot BUY closes 5 at TP1=105, remaining 5 at TP2=110."""
    plan = plan_partial_exit(
        quantity=10,
        take_profit=110.0,
        take_profit_1=105.0,
        take_profit_2=110.0,
    )
    assert not tp1_hit("BUY", 104.0, plan.tp1_price)
    assert tp1_hit("BUY", 105.5, plan.tp1_price)
    assert not tp2_hit("BUY", 108.0, plan.tp2_price)
    assert tp2_hit("BUY", 110.0, plan.tp2_price)
    assert plan.qty_tp1 + plan.qty_tp2 == 10
    assert plan.qty_tp1 == 5


def test_partial_exit_plan_is_immutable():
    """Frozen dataclass — callers can't mutate the qty split."""
    plan = plan_partial_exit(quantity=4, take_profit=110.0, take_profit_1=105.0)
    with pytest.raises((AttributeError, Exception)):
        plan.qty_tp1 = 99  # type: ignore[misc]


def test_partial_exit_plan_type():
    """Public type is a frozen dataclass with expected fields."""
    plan = plan_partial_exit(quantity=4, take_profit=110.0, take_profit_1=105.0)
    assert isinstance(plan, PartialExitPlan)
    assert hasattr(plan, "qty_tp1")
    assert hasattr(plan, "qty_tp2")
    assert hasattr(plan, "tp1_price")
    assert hasattr(plan, "tp2_price")
