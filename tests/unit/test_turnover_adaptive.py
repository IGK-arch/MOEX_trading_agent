"""
tests/unit/test_turnover_adaptive.py — Phase 11.11 adaptive meta threshold.
"""

from __future__ import annotations

import pytest

import app.config as cfg
from app.execution.turnover_tracker import TurnoverTracker


def test_adaptive_no_change_with_few_trades():
    """Test adaptive no change with few trades."""
    tt = TurnoverTracker()
    orig = cfg.META_MIN_PROBA

    for _ in range(5):
        tt.on_trade_outcome(100.0)
    new = tt.adaptive_meta_step()
    assert new == orig


def test_adaptive_raises_threshold_on_high_winrate(monkeypatch):
    """Test adaptive raises threshold on high winrate."""
    tt = TurnoverTracker()
    monkeypatch.setattr(cfg, "META_MIN_PROBA", 0.55)

    for _ in range(25):
        tt.on_trade_outcome(100.0)
    new = tt.adaptive_meta_step()
    assert new == pytest.approx(0.56, abs=0.001)
    assert pytest.approx(0.56, abs=0.001) == cfg.META_MIN_PROBA


def test_adaptive_lowers_threshold_on_low_winrate(monkeypatch):
    """Test adaptive lowers threshold on low winrate."""
    tt = TurnoverTracker()
    monkeypatch.setattr(cfg, "META_MIN_PROBA", 0.55)

    for i in range(25):
        tt.on_trade_outcome(100.0 if i < 12 else -100.0)
    new = tt.adaptive_meta_step()
    assert new == pytest.approx(0.54, abs=0.001)


def test_adaptive_respects_ceiling(monkeypatch):
    """Test adaptive respects ceiling."""
    tt = TurnoverTracker()
    monkeypatch.setattr(cfg, "META_MIN_PROBA", 0.70)
    monkeypatch.setattr(cfg, "META_MIN_PROBA_CEILING", 0.70)
    for _ in range(25):
        tt.on_trade_outcome(100.0)
    new = tt.adaptive_meta_step()
    assert new == 0.70


def test_adaptive_respects_floor(monkeypatch):
    """Test adaptive respects floor."""
    tt = TurnoverTracker()
    monkeypatch.setattr(cfg, "META_MIN_PROBA", 0.45)
    monkeypatch.setattr(cfg, "META_MIN_PROBA_FLOOR", 0.45)
    for _i in range(25):
        tt.on_trade_outcome(-100.0)
    new = tt.adaptive_meta_step()
    assert new == 0.45


def test_today_volume_exposed():
    """Test today volume exposed."""
    tt = TurnoverTracker()
    tt._daily_actual_rub = 123_456.78
    assert tt.today_volume == 123_456.78
