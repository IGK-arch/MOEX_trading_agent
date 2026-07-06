"""Классификация решений по tier 1/2/3."""

from __future__ import annotations

import app.config as cfg
from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    DecisionTier,
    SignalSource,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

def classify_tier(decision: Decision) -> DecisionTier:
    """Assign tier based on combined_magnitude and expected_rr.

    Args:
        decision: decision to classify
    Returns:
        DecisionTier: TIER1 / TIER2 / TIER3 / NONE
    """
    if decision.action != DecisionAction.EXECUTE:
        return DecisionTier.NONE

    mag = decision.combined_magnitude
    rr = decision.expected_rr or 1.0

    is_pair = any(s.source == SignalSource.PAIR for s in decision.signals)
    if is_pair:
        return DecisionTier.TIER2

    if mag >= cfg.TIER1_MIN_MAGNITUDE and rr >= cfg.TIER1_MIN_RR:
        return DecisionTier.TIER1

    if mag >= cfg.TIER2_MIN_MAGNITUDE and rr >= cfg.TIER2_MIN_RR:
        return DecisionTier.TIER2

    if mag >= cfg.TIER3_MIN_MAGNITUDE and rr >= cfg.TIER3_MIN_RR:
        return DecisionTier.TIER3

    return DecisionTier.NONE

def apply_tier(
    decision: Decision,
    session_label: str | None = None,
) -> Decision:
    """Mutate decision: set tier, downgrade to NO_TRADE if Tier=NONE.

    When ``session_label`` is provided AND :data:`cfg.SESSION_PROFILE_ENABLED`
    is True, an additional Phase 27.9 per-session magnitude floor is applied
    BEFORE the normal tier classification. Trades below the session floor
    are downgraded to ``NO_TRADE`` with rationale ``"session_floor: ..."``.
    Default behaviour (``session_label=None`` or feature disabled) is
    identical to the pre-Phase-27.9 implementation.

    Args:
        decision: decision to update.
        session_label: optional sub-session label (e.g. ``"midday"``) — see
            :mod:`app.utils.session_profile`.
    Returns:
        Decision: same decision (mutated).
    """
    if session_label is not None and getattr(cfg, "SESSION_PROFILE_ENABLED", False):
        try:
            from app.utils.session_profile import SessionLabel, get_profile

            profile = get_profile(SessionLabel(session_label))
            if (
                decision.action == DecisionAction.EXECUTE
                and decision.combined_magnitude < profile.magnitude_floor
            ):
                decision.action = DecisionAction.NO_TRADE
                decision.tier = DecisionTier.NONE
                decision.rationale = (
                    f"session_floor: {profile.magnitude_floor:.2f} "
                    f"({session_label}) | " + decision.rationale
                )
                logger.debug(
                    "Session floor blocked",
                    extra={
                        "ticker": decision.ticker,
                        "session": session_label,
                        "mag": decision.combined_magnitude,
                        "floor": profile.magnitude_floor,
                    },
                )
                return decision
        except (ValueError, KeyError):
            pass

    tier = classify_tier(decision)
    decision.tier = tier
    if tier == DecisionTier.NONE and decision.action == DecisionAction.EXECUTE:
        decision.action = DecisionAction.NO_TRADE
        decision.rationale = (
            f"NO_TRADE: below tier threshold "
            f"(mag={decision.combined_magnitude:.2f}, rr={decision.expected_rr:.2f}) | "
            + decision.rationale
        )
        logger.info(
            "Tier=NONE: решение отклонено",
            extra={
                "ticker": decision.ticker,
                "mag": round(decision.combined_magnitude, 3),
                "rr": round(decision.expected_rr or 0.0, 3),
                "t1_min_mag": cfg.TIER1_MIN_MAGNITUDE,
                "t1_min_rr": cfg.TIER1_MIN_RR,
                "t2_min_mag": cfg.TIER2_MIN_MAGNITUDE,
                "t2_min_rr": cfg.TIER2_MIN_RR,
                "t3_min_mag": cfg.TIER3_MIN_MAGNITUDE,
                "t3_min_rr": cfg.TIER3_MIN_RR,
                "n_signals": len(decision.signals),
                "sources": list({str(s.source) for s in decision.signals}),
                "direction": decision.direction.value if decision.direction else None,
            },
        )
    else:
        logger.info(
            "Tier назначен",
            extra={
                "ticker": decision.ticker,
                "tier": tier.value,
                "mag": round(decision.combined_magnitude, 3),
                "rr": round(decision.expected_rr or 0.0, 3),
                "action": decision.action.value,
            },
        )
    return decision

def tier_size_pct(tier: DecisionTier) -> float:
    """Return position sizing as % of deposit for the tier.

    Args:
        tier: decision tier
    Returns:
        float: pct of deposit (0.0 for NONE)
    """
    if tier == DecisionTier.TIER1:
        return cfg.TIER1_SIZE_PCT
    if tier == DecisionTier.TIER2:
        return cfg.TIER2_SIZE_PCT
    if tier == DecisionTier.TIER3:
        return cfg.TIER3_SIZE_PCT
    return 0.0
