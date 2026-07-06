"""
tests/unit/test_regime_sizing.py — HMM regime_size_multiplier correctness.

Phase 21 (v0.0.21) — multipliers bumped for test-phase trading. See
app/agents/hmm_regime.py:regime_size_multiplier.
  unknown        0.5 → 0.7
  crisis         0.3 → 0.7
  mean_reverting 0.7 → 0.85
  trending       1.0 → 1.0 (unchanged)
"""

from __future__ import annotations

from app.agents.hmm_regime import HMMRegimeDetector


def test_unknown_default():
    """Test unknown default."""
    det = HMMRegimeDetector()
    assert det.regime_size_multiplier() == 0.7


def test_crisis_multiplier():
    """Test crisis multiplier."""
    det = HMMRegimeDetector()
    det._current_label = "crisis"
    assert det.regime_size_multiplier() == 0.7


def test_mean_reverting_multiplier():
    """Test mean reverting multiplier."""
    det = HMMRegimeDetector()
    det._current_label = "mean_reverting"
    assert det.regime_size_multiplier() == 0.85


def test_trending_full_size():
    """Test trending full size."""
    det = HMMRegimeDetector()
    det._current_label = "trending"
    assert det.regime_size_multiplier() == 1.0
