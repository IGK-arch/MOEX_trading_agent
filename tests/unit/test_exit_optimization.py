"""Phase 29 (v0.3.1) — exit-logic optimisation tests.

Covers:
  - High-WR ticker early-lock ladder (SBER / GAZP @ +0.5 ATR break-even).
  - HMM-regime overlay on trailing ladder (trending widens, MR/crisis tighten).
  - Crisis disables the final "let it run" rung.
  - 3-tier partial-exit plan for high-PF setups (research family / GAZP).
  - Regime-aware SL/TP overlay (`derive_sl_tp_with_regime`).
  - End-of-day force-close window (30-min before 18:50 MSK).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import app.config as cfg
from app.risk.partial_exit import (
    PartialExitPlan3,
    plan_partial_exit_3tier,
)
from app.risk.sl_tp_rules import (
    derive_sl_tp,
    derive_sl_tp_with_regime,
    regime_exit_adjustment,
)
from app.risk.trailing_stop import (
    DEFAULT_LADDER,
    HIGH_WR_LADDER,
    compute_trailing_stop,
    ladder_for_regime,
    should_force_close_eod,
)

MSK = timezone(timedelta(hours=3))


def test_high_wr_ladder_locks_break_even_earlier_for_sber():
    """SBER at +0.5 ATR profit must already trail to break-even — the
    default ladder would still be `None` at that profit level."""
    ladder = ladder_for_regime(family="research", ticker="SBER", hmm_regime="trending")
    assert ladder[0].profit_atr <= 0.7
    assert ladder[0].lock_atr == pytest.approx(0.0)


def test_high_wr_ladder_used_for_gazp_regardless_of_family():
    """GAZP is in HIGH_WR_TICKERS — ticker overrides family."""
    ladder_smc = ladder_for_regime(family="smc", ticker="GAZP", hmm_regime="unknown")
    assert ladder_smc[0].profit_atr == pytest.approx(HIGH_WR_LADDER[0].profit_atr)


def test_default_ladder_used_for_non_high_wr_ticker():
    """Non-priviledged ticker → falls back to family default."""
    ladder = ladder_for_regime(family="research", ticker="VTBR", hmm_regime="unknown")
    assert ladder == DEFAULT_LADDER


def test_trending_widens_trailing_ladder():
    """Trending regime → rungs trigger later (×1.20)."""
    base = ladder_for_regime(family="research", hmm_regime="unknown")
    trending = ladder_for_regime(family="research", hmm_regime="trending")
    assert trending[0].profit_atr > base[0].profit_atr
    assert trending[-1].lock_atr > base[-1].lock_atr


def test_mean_reverting_tightens_trailing_ladder():
    """Mean-reverting → rungs trigger sooner (×0.80)."""
    base = ladder_for_regime(family="research", hmm_regime="unknown")
    mr = ladder_for_regime(family="research", hmm_regime="mean_reverting")
    assert mr[0].profit_atr < base[0].profit_atr
    assert mr[0].profit_atr == pytest.approx(base[0].profit_atr * 0.80)


def test_crisis_drops_final_rung_and_tightens():
    """Crisis regime tightens AND removes the "let it run" tier."""
    base = ladder_for_regime(family="research", hmm_regime="unknown")
    crisis = ladder_for_regime(family="research", hmm_regime="crisis")
    assert len(crisis) == len(base) - 1
    assert crisis[0].profit_atr == pytest.approx(base[0].profit_atr * 0.60)


def test_trailing_stop_uses_regime_ladder_for_sber_trending():
    """End-to-end: SBER (high-WR) + trending regime → at +0.7 ATR we
    already have a trailing stop at break-even (default would be None)."""
    ladder = ladder_for_regime(family="research", ticker="SBER", hmm_regime="trending")
    trail = compute_trailing_stop(
        direction="BUY",
        entry_price=100.0,
        current_price=100.7,
        atr=1.0,
        ladder=ladder,
    )
    assert trail == pytest.approx(100.0)

    no_trail = compute_trailing_stop(
        direction="BUY",
        entry_price=100.0,
        current_price=100.7,
        atr=1.0,
    )
    assert no_trail is None


def test_3tier_partial_exit_splits_thirds_buy():
    """qty=9 → 3/3/3 with TP1=+1ATR, TP2=+2ATR, TP3=trailing (None)."""
    plan = plan_partial_exit_3tier(quantity=9, entry_price=100.0, atr=2.0, direction="BUY")
    assert isinstance(plan, PartialExitPlan3)
    assert plan.qty_tp1 == 3
    assert plan.qty_tp2 == 3
    assert plan.qty_tp3 == 3
    assert plan.qty_tp1 + plan.qty_tp2 + plan.qty_tp3 == 9
    assert plan.tp1_price == pytest.approx(102.0)
    assert plan.tp2_price == pytest.approx(104.0)
    assert plan.tp3_price is None
    assert plan.has_three_tier is True


def test_3tier_partial_exit_handles_odd_lots():
    """qty=10 → 3/3/4 (remainder rides the trail). All qty preserved."""
    plan = plan_partial_exit_3tier(quantity=10, entry_price=100.0, atr=1.0, direction="BUY")
    assert plan.qty_tp1 + plan.qty_tp2 + plan.qty_tp3 == 10
    assert plan.qty_tp3 >= plan.qty_tp1
    assert plan.qty_tp3 >= plan.qty_tp2


def test_3tier_partial_exit_small_qty_degrades_gracefully():
    """qty=1 → 100% on the trail (no fixed TPs fired)."""
    plan = plan_partial_exit_3tier(quantity=1, entry_price=100.0, atr=1.0, direction="BUY")
    assert plan.qty_tp1 == 0
    assert plan.qty_tp2 == 0
    assert plan.qty_tp3 == 1


def test_3tier_partial_exit_sell_direction():
    """SELL: TPs are entry - 1/2 ATR (in profit direction)."""
    plan = plan_partial_exit_3tier(quantity=6, entry_price=200.0, atr=3.0, direction="SELL")
    assert plan.tp1_price == pytest.approx(197.0)
    assert plan.tp2_price == pytest.approx(194.0)
    assert plan.tp3_price is None


def test_regime_adjustment_lookup_table():
    """Each regime label maps to the expected multiplier shape."""
    trending = regime_exit_adjustment("trending")
    mr = regime_exit_adjustment("mean_reverting")
    crisis = regime_exit_adjustment("crisis")
    unknown = regime_exit_adjustment("unknown")

    assert trending.sl_mult > 1.0
    assert trending.tp_rr_mult > 1.0

    assert mr.sl_mult < 1.0
    assert mr.tp_rr_mult < 1.0

    assert crisis.sl_mult < mr.sl_mult
    assert crisis.disable_trailing is True

    assert unknown.sl_mult == 1.0
    assert unknown.tp_rr_mult == 1.0


def test_crisis_overlay_tightens_sl_distance_buy():
    """Crisis tightens SL closer to entry than the family baseline."""
    sl_base, tp_base = derive_sl_tp(
        pattern="vcp_breakout",
        direction="BUY",
        entry=100.0,
        atr=1.0,
    )
    sl_crisis, tp_crisis = derive_sl_tp_with_regime(
        pattern="vcp_breakout",
        direction="BUY",
        entry=100.0,
        atr=1.0,
        hmm_regime="crisis",
    )
    assert sl_crisis > sl_base
    assert tp_crisis < tp_base


def test_trending_overlay_widens_sl_and_extends_tp_buy():
    """Trending widens SL (let setup breathe) and pushes TP further out."""
    sl_base, tp_base = derive_sl_tp(
        pattern="vcp_breakout",
        direction="BUY",
        entry=100.0,
        atr=1.0,
    )
    sl_t, tp_t = derive_sl_tp_with_regime(
        pattern="vcp_breakout",
        direction="BUY",
        entry=100.0,
        atr=1.0,
        hmm_regime="trending",
    )
    assert sl_t < sl_base
    assert tp_t > tp_base


def test_unknown_regime_is_noop():
    """Missing / unknown regime must NOT alter the baseline."""
    sl_base, tp_base = derive_sl_tp(
        pattern="vcp_breakout",
        direction="BUY",
        entry=100.0,
        atr=1.0,
    )
    sl_u, tp_u = derive_sl_tp_with_regime(
        pattern="vcp_breakout",
        direction="BUY",
        entry=100.0,
        atr=1.0,
        hmm_regime=None,
    )
    assert sl_u == pytest.approx(sl_base)
    assert tp_u == pytest.approx(tp_base)


def test_force_close_30min_before_main_session_close():
    """30 min before 18:50 MSK is 18:20 → should fire."""
    now = datetime(2026, 5, 26, 18, 20, 0, tzinfo=MSK)
    assert should_force_close_eod(now) is True


def test_no_force_close_well_before_window():
    """At 14:00 MSK we are NOT in the force-close window."""
    now = datetime(2026, 5, 26, 14, 0, 0, tzinfo=MSK)
    assert should_force_close_eod(now) is False


def test_force_close_at_exact_18_20_boundary():
    """Exact boundary (close - N minutes) must trigger (inclusive)."""
    now = datetime(2026, 5, 26, 18, 19, 59, tzinfo=MSK)
    assert should_force_close_eod(now) is False
    now2 = now.replace(minute=20, second=0)
    assert should_force_close_eod(now2) is True


def test_force_close_disabled_when_minutes_zero():
    """Setting the window to 0 disables the force-close entirely."""
    now = datetime(2026, 5, 26, 18, 49, 0, tzinfo=MSK)
    assert should_force_close_eod(now, minutes_before_close=0) is False


def test_force_close_uses_config_default(monkeypatch):
    """Function defaults to cfg.FORCE_CLOSE_BEFORE_CLOSE_MIN."""
    monkeypatch.setattr(cfg, "FORCE_CLOSE_BEFORE_CLOSE_MIN", 60)
    now = datetime(2026, 5, 26, 17, 50, 0, tzinfo=MSK)
    assert should_force_close_eod(now) is True
    now_pre = datetime(2026, 5, 26, 17, 49, 0, tzinfo=MSK)
    assert should_force_close_eod(now_pre) is False


def test_force_close_outside_session_hours_returns_false():
    """Before market open (e.g. 03:00 MSK) → no positions to close intraday."""
    now = datetime(2026, 5, 26, 3, 0, 0, tzinfo=MSK)
    assert should_force_close_eod(now) is False


def test_high_wr_tickers_config_default_contains_sber_and_gazp():
    """Sanity: the high-WR allow-list ships with SBER + GAZP enabled."""
    assert "SBER" in cfg.HIGH_WR_TICKERS
    assert "GAZP" in cfg.HIGH_WR_TICKERS
