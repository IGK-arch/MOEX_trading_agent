"""
tests/unit/test_wr70_tier_sizing.py — v0.12.x WR-driven tier sizing.

Verifies that risk_manager._compute_quantity routes BASE notional through
cfg.wr70_tier_size_pct(ticker, detector) when a (ticker, detector) combo
appears in cfg.WR_70_WHITELIST, and otherwise falls back to the generic
tier_size_pct(decision.tier).

These tests pin the user-facing capital-growth contract:
  - SBER bear_flag    (WR 85.7 %) → Tier A (4 % live)
  - SBER candle_hammer (WR 76.9 %) → Tier B (3 % live)
  - GAZP bull_flag    (WR 70.0 %) → Tier C (2 % live)
  - GAZP research_family (WR 92.5 %) via family lookup → Tier A
  - Untracked detector → fallback to plain tier_size_pct
  - hard-cap MAX_POSITION_PCT is still respected (sizing never explodes)
"""

from __future__ import annotations

import app.config as cfg
from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    DecisionTier,
    Direction,
    SignalSource,
    UnifiedSignal,
)
from app.dispatcher.tier_classifier import tier_size_pct
from app.risk.risk_manager import HARD_CAP_PCT, RiskManager


def _build_decision(
    *,
    ticker: str,
    detector: str,
    mag: float = 0.60,
    rr: float = 2.0,
    tier: DecisionTier = DecisionTier.TIER2,
    price: float = 100.0,
) -> Decision:
    """Build decision."""
    sig = UnifiedSignal(
        source=SignalSource.TA,
        detector=detector,
        ticker=ticker,
        direction=Direction.BUY,
        magnitude=mag,
        raw_confidence=mag,
        horizon_min=60,
        price=price,
        entry_level=price,
        stop_level=price * 0.98,
        target_level=price * 1.04,
        expected_rr=rr,
        atr=1.0,
    )
    return Decision(
        decision_id=f"wr70_{ticker}_{detector}",
        cycle_id="c1",
        ticker=ticker,
        action=DecisionAction.EXECUTE,
        direction=Direction.BUY,
        tier=tier,
        combined_magnitude=mag,
        signals=[sig],
        expected_rr=rr,
    )


def test_wr70_tier_a_sbermega_megaphone_top():
    """SBER megaphone_top WR 94.4 % → Tier A pct."""
    pct = cfg.wr70_tier_size_pct("SBER", "megaphone_top")
    assert pct == cfg.WR_70_TIER_A_SIZE_PCT


def test_wr70_tier_a_sber_bear_flag():
    """SBER bear_flag WR 85.7 % → Tier A pct (boundary, >= 0.85)."""
    pct = cfg.wr70_tier_size_pct("SBER", "bear_flag")
    assert pct == cfg.WR_70_TIER_A_SIZE_PCT


def test_wr70_tier_b_sber_candle_hammer():
    """SBER candle_hammer WR 76.9 % → Tier B pct."""
    pct = cfg.wr70_tier_size_pct("SBER", "candle_hammer")
    assert pct == cfg.WR_70_TIER_B_SIZE_PCT


def test_wr70_tier_c_gazp_bull_flag():
    """GAZP bull_flag WR 70.0 % → Tier C pct (boundary, >= 0.70)."""
    pct = cfg.wr70_tier_size_pct("GAZP", "bull_flag")
    assert pct == cfg.WR_70_TIER_C_SIZE_PCT


def test_wr70_family_lookup_research():
    """GAZP bb_squeeze_breakout → research family (WR 92.5 %) → Tier A."""
    pct = cfg.wr70_tier_size_pct("GAZP", "bb_squeeze_breakout")
    assert pct == cfg.WR_70_TIER_A_SIZE_PCT


def test_wr70_no_match_returns_none():
    """A ticker/detector not in the whitelist must return None."""
    assert cfg.wr70_tier_size_pct("SBER", "nonexistent_detector_xyz") is None
    assert cfg.wr70_tier_size_pct("UNKNOWN_TICKER", "bear_flag") is None
    assert cfg.wr70_tier_size_pct("", "bear_flag") is None
    assert cfg.wr70_tier_size_pct("SBER", None) is None


def test_wr70_tier_ordering_a_gt_b_gt_c():
    """Sanity: A > B > C size pcts so capital-growth contract is preserved."""
    assert cfg.WR_70_TIER_A_SIZE_PCT > cfg.WR_70_TIER_B_SIZE_PCT > cfg.WR_70_TIER_C_SIZE_PCT > 0.0


def test_wr70_tier_a_within_max_position_cap():
    """The most aggressive tier MUST stay <= MAX_POSITION_PCT to keep the
    absolute hard cap as a meaningful last line of defense."""
    assert cfg.WR_70_TIER_A_SIZE_PCT <= cfg.MAX_POSITION_PCT


def test_compute_quantity_uses_wr70_for_sber_bear_flag(monkeypatch):
    """
    SBER bear_flag is on the WR_70_WHITELIST (Tier A in live).
    Even when the decision is classified as TIER2 (which has a SMALLER
    base pct than Tier A), the WR lookup must lift the base notional.
    """
    rm = RiskManager(deposit_total=1_000_000.0)
    monkeypatch.setattr(RiskManager, "_regime_size_multiplier", staticmethod(lambda: 1.0))

    decision = _build_decision(
        ticker="SBER",
        detector="bear_flag",
        mag=0.60,
        rr=2.0,
        tier=DecisionTier.TIER2,
        price=300.0,
    )
    qty = rm._compute_quantity(decision, price=300.0, atr=2.0, vol_mult=1.0)

    decision_no_wr = _build_decision(
        ticker="SBER",
        detector="nonexistent_detector_xyz",
        mag=0.60,
        rr=2.0,
        tier=DecisionTier.TIER2,
        price=300.0,
    )
    qty_no_wr = rm._compute_quantity(decision_no_wr, price=300.0, atr=2.0, vol_mult=1.0)
    assert qty >= qty_no_wr

    assert qty * 300.0 <= 1_000_000.0 * HARD_CAP_PCT + 1.0


def test_compute_quantity_falls_back_when_no_wr_match(monkeypatch):
    """Plain tier_size_pct path must still be active for detectors not in
    the whitelist; otherwise the rest of the system regresses."""
    rm = RiskManager(deposit_total=1_000_000.0)
    monkeypatch.setattr(RiskManager, "_regime_size_multiplier", staticmethod(lambda: 1.0))

    decision = _build_decision(
        ticker="SBER",
        detector="some_unknown_detector",
        mag=0.80,
        rr=2.5,
        tier=DecisionTier.TIER1,
        price=300.0,
    )
    qty = rm._compute_quantity(decision, price=300.0, atr=2.0, vol_mult=1.0)
    assert qty >= 0
    target_base_notional = 1_000_000.0 * tier_size_pct(DecisionTier.TIER1)
    assert qty * 300.0 <= max(target_base_notional, 1_000_000.0 * HARD_CAP_PCT) + 1.0


def test_wr70_picks_largest_tier_when_multiple_signals(monkeypatch):
    """If a Decision fuses several signals, the strongest matching tier wins
    so we never down-size on the back of an additional weak confirmation."""
    RiskManager(deposit_total=1_000_000.0)
    sig_a = UnifiedSignal(
        source=SignalSource.TA,
        detector="bear_flag",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=0.7,
        raw_confidence=0.7,
        horizon_min=60,
        price=300.0,
        entry_level=300.0,
        stop_level=294.0,
        target_level=312.0,
        expected_rr=2.0,
        atr=1.0,
    )
    sig_b = UnifiedSignal(
        source=SignalSource.TA,
        detector="candle_hammer",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=0.5,
        raw_confidence=0.5,
        horizon_min=60,
        price=300.0,
        entry_level=300.0,
        stop_level=294.0,
        target_level=312.0,
        expected_rr=2.0,
        atr=1.0,
    )
    decision = Decision(
        decision_id="multi",
        cycle_id="c1",
        ticker="SBER",
        action=DecisionAction.EXECUTE,
        direction=Direction.BUY,
        tier=DecisionTier.TIER2,
        combined_magnitude=0.7,
        signals=[sig_b, sig_a],
        expected_rr=2.0,
    )
    pct = RiskManager._wr70_size_pct(decision)
    assert pct in (cfg.WR_70_TIER_A_SIZE_PCT, cfg.WR_90_TIER_S_SIZE_PCT)
