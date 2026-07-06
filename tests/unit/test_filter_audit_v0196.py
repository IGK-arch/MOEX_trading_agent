"""Unit tests for Phase 30 (v0.19.6) filter quality audit changes.

Covers:
  * VPIN threshold raise 0.45 → 0.60
  * Kyle lambda p95 → p90
  * OFI opposition 0.20 → 0.30
  * Confluence ATR low percentile 30 → 20
  * Adaptive CRISIS keeps trading (size_mult 0.25, can_open_new True)
  * HMM alignment lets continuation through in crisis
"""

from __future__ import annotations

import app.config as cfg
from app.agents.ta_patterns.confluence_filters import passes_hmm_alignment
from app.risk.adaptive_regime import compute_risk_regime


def test_v0196_vpin_threshold_is_raised() -> None:
    """Phase 30: VPIN raised to 0.60 per Easley/de Prado literature."""
    assert cfg.VPIN_BLOCK_THRESHOLD == 0.60


def test_v0196_kyles_lambda_p90() -> None:
    """Phase 30: Kyle's λ tightened p95 → p90."""
    assert cfg.KYLES_LAMBDA_BLOCK_PCT == 0.90


def test_v0196_ofi_threshold_relaxed() -> None:
    """Phase 30: OFI threshold relaxed 0.20 → 0.30."""
    assert cfg.OFI_OPPOSITION_THRESHOLD == 0.30


def test_v0196_atr_low_bound_relaxed() -> None:
    """Phase 30: ATR percentile low bound relaxed 30 → 20."""
    assert cfg.CONFLUENCE_ATR_PCT_LOW == 20.0


def test_v0196_crisis_dd_threshold_widened() -> None:
    """Phase 30: CRISIS DD trigger widened 0.035 → 0.06."""
    assert cfg.ADAPTIVE_CRISIS_DD_PCT == 0.06


def test_v0196_crisis_keeps_tiny_sizing() -> None:
    """Phase 30: CRISIS no longer means full halt — 0.25× sizing."""
    assert cfg.ADAPTIVE_CRISIS_SIZE_MULT == 0.25


def test_v0196_crisis_regime_allows_new_entries() -> None:
    """Confirms compute_risk_regime in CRISIS allows can_open_new."""
    regime = compute_risk_regime(
        current_drawdown_from_peak_pct=0.10,
        losing_streak=0,
        daily_pnl_pct=0.0,
    )
    assert regime.name == "CRISIS"
    assert regime.can_open_new is True
    assert regime.size_multiplier == 0.25


def test_v0196_hmm_alignment_lets_continuation_through_crisis() -> None:
    """Phase 30: continuation patterns survive a crisis regime."""
    assert passes_hmm_alignment("continuation", "crisis") is True


def test_v0196_hmm_alignment_still_vetoes_reversal_in_crisis() -> None:
    """Reversal patterns still vetoed in crisis (don't fade crashes)."""
    assert passes_hmm_alignment("reversal", "crisis") is False


def test_v0196_hmm_alignment_other_families_pass_crisis() -> None:
    """research/dasha/smc/candle/harmonic should pass crisis at smaller size."""
    for family in ("research", "dasha", "smc", "candle", "harmonic"):
        assert passes_hmm_alignment(family, "crisis") is True
