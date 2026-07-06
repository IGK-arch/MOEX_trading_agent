"""
tests/unit/test_turnover_escalation.py — v0.14.0 dynamic turnover ladder.

Covers:
  - escalation level transitions (0 → 1 → 2) based on (day_index, cum_turnover)
  - de-escalation latching (level can rise but not fall mid-stage)
  - PER_TICKER_POLICY mutation correctness at each level
  - get_size_mult_for_escalation() level 2 → 0.5
  - get_volume_weighted_size_mult() — SBER > GAZP > CHMF, all > 1.0
  - intraday meta boost: trades-today gate, hour gate, abs floor, idempotency
"""

from __future__ import annotations

import pytest

import app.config as cfg
from app.execution.turnover_tracker import TurnoverTracker


@pytest.fixture(autouse=True)
def _reset_escalation_state():
    """Each test starts from the baseline (level 0) policy."""
    cfg.apply_turnover_escalation(0)
    yield
    cfg.apply_turnover_escalation(0)


def test_level_0_baseline_only_4_active_tickers():
    """Test level 0 baseline only 4 active tickers."""
    cfg.apply_turnover_escalation(0)
    active = {t for t, p in cfg.PER_TICKER_POLICY.items() if p != "DISABLED"}
    assert active == {"SBER", "GAZP", "CHMF", "PIKK"}
    assert cfg.TURNOVER_ESCALATION_LEVEL == 0
    assert cfg.get_size_mult_for_escalation() == 1.0


def test_level_1_promotes_whitelist_tickers_to_whitelist_only():
    """Test level 1 promotes whitelist tickers to whitelist only."""
    cfg.apply_turnover_escalation(1)
    assert cfg.PER_TICKER_POLICY["SBER"] == "GOLD"
    assert cfg.PER_TICKER_POLICY["GAZP"] == "GOLD"
    assert cfg.PER_TICKER_POLICY["CHMF"] == "GOLD"
    assert cfg.PER_TICKER_POLICY["PIKK"] == "WHITELIST_ONLY"
    for tk in cfg.TURNOVER_ESCALATION_WHITELIST:
        assert cfg.PER_TICKER_POLICY[tk] == "WHITELIST_ONLY", tk
    assert cfg.PER_TICKER_POLICY["NLMK"] == "DISABLED"
    assert cfg.TURNOVER_ESCALATION_LEVEL == 1
    assert cfg.get_size_mult_for_escalation() == 1.0


def test_level_2_enables_all_tickers_with_half_sizing():
    """Test level 2 enables all tickers with half sizing."""
    cfg.apply_turnover_escalation(2)
    disabled = [t for t, p in cfg.PER_TICKER_POLICY.items() if p == "DISABLED"]
    assert disabled == [], f"Expected no DISABLED at level 2, got: {disabled}"
    assert cfg.PER_TICKER_POLICY["SBER"] == "GOLD"
    assert cfg.PER_TICKER_POLICY["GAZP"] == "GOLD"
    assert cfg.PER_TICKER_POLICY["CHMF"] == "GOLD"
    assert cfg.TURNOVER_ESCALATION_LEVEL == 2
    assert cfg.get_size_mult_for_escalation() == 0.5


def test_evaluate_ladder_latches_upward_only():
    """Once we escalate, we don't drop back even if turnover later recovers."""
    tt = TurnoverTracker()
    lvl = tt._evaluate_escalation_ladder(day_index=7, cum_turnover_rub=2_000_000.0)
    assert lvl == 1
    assert cfg.TURNOVER_ESCALATION_LEVEL == 1
    lvl = tt._evaluate_escalation_ladder(day_index=8, cum_turnover_rub=4_500_000.0)
    assert lvl == 1
    lvl = tt._evaluate_escalation_ladder(day_index=10, cum_turnover_rub=6_000_000.0)
    assert lvl == 2


def test_evaluate_ladder_no_escalation_when_on_track():
    """Healthy turnover keeps us at level 0."""
    tt = TurnoverTracker()
    lvl = tt._evaluate_escalation_ladder(day_index=7, cum_turnover_rub=5_000_000.0)
    assert lvl == 0
    lvl = tt._evaluate_escalation_ladder(day_index=10, cum_turnover_rub=8_500_000.0)
    assert lvl == 0
    assert cfg.TURNOVER_ESCALATION_LEVEL == 0


def test_volume_weighted_size_mult_ranks_by_liquidity():
    """Test volume weighted size mult ranks by liquidity."""
    sber = cfg.get_volume_weighted_size_mult("SBER")
    gazp = cfg.get_volume_weighted_size_mult("GAZP")
    chmf = cfg.get_volume_weighted_size_mult("CHMF")
    pikk = cfg.get_volume_weighted_size_mult("PIKK")
    unknown = cfg.get_volume_weighted_size_mult("ZZZZ")

    assert sber == pytest.approx(1.5, abs=0.01)
    assert 1.0 < gazp < sber
    assert chmf >= 1.0
    assert pikk >= 1.0
    assert chmf <= gazp
    assert unknown == 1.0


def test_intraday_boost_not_triggered_before_cutoff_hour(monkeypatch):
    """Test intraday boost not triggered before cutoff hour."""
    tt = TurnoverTracker()
    monkeypatch.setattr(cfg, "META_MIN_PROBA", 0.50)
    monkeypatch.setattr(cfg, "TURNOVER_INTRADAY_BOOST_CUTOFF_HOUR_MSK", 14)
    new = tt.apply_intraday_meta_boost(now_msk_hour=12)
    assert new == 0.50
    assert not tt._intraday_boost_active


def test_intraday_boost_triggers_when_trades_lag_after_cutoff(monkeypatch):
    """Test intraday boost triggers when trades lag after cutoff."""
    tt = TurnoverTracker()
    monkeypatch.setattr(cfg, "META_MIN_PROBA", 0.50)
    monkeypatch.setattr(cfg, "TURNOVER_INTRADAY_BOOST_CUTOFF_HOUR_MSK", 14)
    monkeypatch.setattr(cfg, "TURNOVER_INTRADAY_BOOST_MIN_TRADES", 5)
    monkeypatch.setattr(cfg, "TURNOVER_INTRADAY_BOOST_DELTA", 0.05)
    monkeypatch.setattr(cfg, "TURNOVER_INTRADAY_BOOST_ABS_FLOOR", 0.25)

    for _ in range(3):
        tt.on_trade_opened()
    new = tt.apply_intraday_meta_boost(now_msk_hour=14)
    assert new == pytest.approx(0.45, abs=1e-6)
    assert pytest.approx(0.45, abs=1e-6) == cfg.META_MIN_PROBA
    assert tt._intraday_boost_active

    new2 = tt.apply_intraday_meta_boost(now_msk_hour=15)
    assert new2 == pytest.approx(0.45, abs=1e-6)


def test_intraday_boost_skipped_when_trades_sufficient(monkeypatch):
    """Test intraday boost skipped when trades sufficient."""
    tt = TurnoverTracker()
    monkeypatch.setattr(cfg, "META_MIN_PROBA", 0.50)
    monkeypatch.setattr(cfg, "TURNOVER_INTRADAY_BOOST_CUTOFF_HOUR_MSK", 14)
    monkeypatch.setattr(cfg, "TURNOVER_INTRADAY_BOOST_MIN_TRADES", 5)
    for _ in range(5):
        tt.on_trade_opened()
    new = tt.apply_intraday_meta_boost(now_msk_hour=14)
    assert new == 0.50
    assert not tt._intraday_boost_active


def test_intraday_boost_respects_abs_floor(monkeypatch):
    """Test intraday boost respects abs floor."""
    tt = TurnoverTracker()
    monkeypatch.setattr(cfg, "META_MIN_PROBA", 0.28)
    monkeypatch.setattr(cfg, "TURNOVER_INTRADAY_BOOST_ABS_FLOOR", 0.25)
    monkeypatch.setattr(cfg, "TURNOVER_INTRADAY_BOOST_DELTA", 0.05)
    monkeypatch.setattr(cfg, "TURNOVER_INTRADAY_BOOST_CUTOFF_HOUR_MSK", 14)
    monkeypatch.setattr(cfg, "TURNOVER_INTRADAY_BOOST_MIN_TRADES", 5)

    new = tt.apply_intraday_meta_boost(now_msk_hour=15)
    assert new == pytest.approx(0.25, abs=1e-6)


def test_reset_daily_counters_restores_meta(monkeypatch):
    """Test reset daily counters restores meta."""
    tt = TurnoverTracker()
    monkeypatch.setattr(cfg, "META_MIN_PROBA", 0.50)
    tt._original_meta_min_proba = 0.50
    for _ in range(2):
        tt.on_trade_opened()
    tt.apply_intraday_meta_boost(now_msk_hour=15)
    assert tt._intraday_boost_active
    assert cfg.META_MIN_PROBA < 0.50
    tt.reset_daily_counters()
    assert not tt._intraday_boost_active
    assert tt._trades_today == 0
    assert cfg.META_MIN_PROBA == 0.50


def test_apply_turnover_escalation_is_idempotent():
    """Calling the same level twice is a no-op."""
    cfg.apply_turnover_escalation(1)
    snapshot = dict(cfg.PER_TICKER_POLICY)
    cfg.apply_turnover_escalation(1)
    assert dict(cfg.PER_TICKER_POLICY) == snapshot
    assert cfg.TURNOVER_ESCALATION_LEVEL == 1


def test_apply_turnover_escalation_clamps_invalid_levels():
    """Negative / oversized levels clamp to [0, 2]."""
    cfg.apply_turnover_escalation(-3)
    assert cfg.TURNOVER_ESCALATION_LEVEL == 0
    cfg.apply_turnover_escalation(99)
    assert cfg.TURNOVER_ESCALATION_LEVEL == 2
