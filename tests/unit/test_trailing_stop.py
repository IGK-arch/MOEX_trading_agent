"""ATR-laddered trailing stop — synthetic price-series verification."""

from __future__ import annotations

import pytest

from app.risk.trailing_stop import (
    DEFAULT_LADDER,
    TrailingRung,
    compute_trailing_stop,
    effective_stop,
)


def test_no_trail_before_first_rung_buy():
    """Profit < 1 ATR → trail stays None."""
    trail = compute_trailing_stop(
        direction="BUY",
        entry_price=100.0,
        current_price=100.5,
        atr=1.0,
    )
    assert trail is None


def test_breakeven_trail_at_1_atr_buy():
    """+1 ATR → trail to entry price (lock_atr=0)."""
    trail = compute_trailing_stop(
        direction="BUY",
        entry_price=100.0,
        current_price=101.0,
        atr=1.0,
    )
    assert trail == pytest.approx(100.0)


def test_lock_half_at_2_atr_buy():
    """+2 ATR → trail to entry + 1 ATR."""
    trail = compute_trailing_stop(
        direction="BUY",
        entry_price=100.0,
        current_price=102.0,
        atr=1.0,
    )
    assert trail == pytest.approx(101.0)


def test_lock_more_at_3_atr_buy():
    """+3 ATR → trail to entry + 2 ATR."""
    trail = compute_trailing_stop(
        direction="BUY",
        entry_price=100.0,
        current_price=103.0,
        atr=1.0,
    )
    assert trail == pytest.approx(102.0)


def test_above_3_atr_clamps_to_last_rung_buy():
    """+5 ATR still uses the 3rd-rung lock; no extra rungs."""
    trail = compute_trailing_stop(
        direction="BUY",
        entry_price=100.0,
        current_price=105.0,
        atr=1.0,
    )
    assert trail == pytest.approx(102.0)


def test_trail_never_moves_backwards_buy():
    """After locking +1 ATR, a price drop back to +1.5 ATR keeps the lock."""
    trail1 = compute_trailing_stop(
        direction="BUY",
        entry_price=100.0,
        current_price=102.0,
        atr=1.0,
    )
    assert trail1 == pytest.approx(101.0)
    trail2 = compute_trailing_stop(
        direction="BUY",
        entry_price=100.0,
        current_price=101.5,
        atr=1.0,
        prev_trailing=trail1,
    )
    assert trail2 == pytest.approx(101.0)


def test_trail_monotonic_across_synthetic_series_buy():
    """Walk a synthetic price path; trail must be non-decreasing."""
    prices = [100.5, 101.0, 101.8, 102.5, 102.1, 103.5, 102.0, 101.0]
    trail = None
    seq: list[float] = []
    for p in prices:
        trail = compute_trailing_stop(
            direction="BUY",
            entry_price=100.0,
            current_price=p,
            atr=1.0,
            prev_trailing=trail,
        )
        if trail is not None:
            seq.append(trail)
    assert all(seq[i] >= seq[i - 1] for i in range(1, len(seq)))
    assert seq[-1] >= 101.0


def test_breakeven_trail_at_1_atr_sell():
    """Test breakeven trail at 1 atr sell."""
    trail = compute_trailing_stop(
        direction="SELL",
        entry_price=100.0,
        current_price=99.0,
        atr=1.0,
    )
    assert trail == pytest.approx(100.0)


def test_lock_half_at_2_atr_sell():
    """Test lock half at 2 atr sell."""
    trail = compute_trailing_stop(
        direction="SELL",
        entry_price=100.0,
        current_price=98.0,
        atr=1.0,
    )
    assert trail == pytest.approx(99.0)


def test_lock_more_at_3_atr_sell():
    """Test lock more at 3 atr sell."""
    trail = compute_trailing_stop(
        direction="SELL",
        entry_price=100.0,
        current_price=97.0,
        atr=1.0,
    )
    assert trail == pytest.approx(98.0)


def test_trail_never_moves_backwards_sell():
    """Test trail never moves backwards sell."""
    trail1 = compute_trailing_stop(
        direction="SELL",
        entry_price=100.0,
        current_price=98.0,
        atr=1.0,
    )
    assert trail1 == pytest.approx(99.0)
    trail2 = compute_trailing_stop(
        direction="SELL",
        entry_price=100.0,
        current_price=98.5,
        atr=1.0,
        prev_trailing=trail1,
    )
    assert trail2 == pytest.approx(99.0)


def test_zero_atr_returns_prev():
    """ATR=0 (data anomaly) → return prev_trailing unchanged."""
    trail = compute_trailing_stop(
        direction="BUY",
        entry_price=100.0,
        current_price=110.0,
        atr=0.0,
        prev_trailing=99.0,
    )
    assert trail == 99.0


def test_loss_returns_prev_buy():
    """BUY position underwater → trail does NOT activate."""
    trail = compute_trailing_stop(
        direction="BUY",
        entry_price=100.0,
        current_price=98.0,
        atr=1.0,
        prev_trailing=None,
    )
    assert trail is None


def test_unknown_direction_returns_prev():
    """Test unknown direction returns prev."""
    trail = compute_trailing_stop(
        direction="FLAT",
        entry_price=100.0,
        current_price=110.0,
        atr=1.0,
        prev_trailing=42.0,
    )
    assert trail == 42.0


def test_ladder_is_three_rungs():
    """Default ladder shape: 3 rungs at 1/2/3 ATR with 0/1/2 lock."""
    assert len(DEFAULT_LADDER) == 3
    assert DEFAULT_LADDER[0] == TrailingRung(profit_atr=1.0, lock_atr=0.0)
    assert DEFAULT_LADDER[1] == TrailingRung(profit_atr=2.0, lock_atr=1.0)
    assert DEFAULT_LADDER[2] == TrailingRung(profit_atr=3.0, lock_atr=2.0)


def test_custom_ladder_overrides_default():
    """Risk team can swap the ladder for tuning experiments."""
    ladder = (
        TrailingRung(profit_atr=0.5, lock_atr=0.0),
        TrailingRung(profit_atr=1.0, lock_atr=0.5),
    )
    trail = compute_trailing_stop(
        direction="BUY",
        entry_price=100.0,
        current_price=101.0,
        atr=1.0,
        ladder=ladder,
    )
    assert trail == pytest.approx(100.5)


def test_effective_stop_buy_uses_max():
    """BUY: combine initial+trail → higher (closer-below) wins."""
    assert effective_stop("BUY", initial_stop=98.0, trailing_stop=100.0) == 100.0
    assert effective_stop("BUY", initial_stop=99.0, trailing_stop=98.5) == 99.0


def test_effective_stop_sell_uses_min():
    """SELL: combine initial+trail → lower (closer-above) wins."""
    assert effective_stop("SELL", initial_stop=102.0, trailing_stop=100.0) == 100.0
    assert effective_stop("SELL", initial_stop=101.0, trailing_stop=101.5) == 101.0


def test_effective_stop_handles_nones():
    """Test effective stop handles nones."""
    assert effective_stop("BUY", initial_stop=None, trailing_stop=None) is None
    assert effective_stop("BUY", initial_stop=None, trailing_stop=99.0) == 99.0
    assert effective_stop("BUY", initial_stop=98.0, trailing_stop=None) == 98.0
