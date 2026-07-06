"""Per-pattern-family SL/TP rule defaults."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class FamilyRule:
    """SL / TP / time-stop preset for one pattern family."""

    sl_atr: float
    rr: float
    time_stop_bars: int | None
    regime_high_sl_mult: float = 1.20
    regime_low_sl_mult: float = 0.85

RULES: dict[str, FamilyRule] = {
    "chart": FamilyRule(sl_atr=3.00, rr=1.5, time_stop_bars=32),
    "research": FamilyRule(sl_atr=2.00, rr=3.0, time_stop_bars=16),
    "candle": FamilyRule(sl_atr=1.25, rr=2.5, time_stop_bars=8),
    "smc": FamilyRule(sl_atr=2.00, rr=2.0, time_stop_bars=12),
    "other": FamilyRule(sl_atr=1.75, rr=2.0, time_stop_bars=16),
}

def family_of(pattern: str) -> str:
    """Map detector-emitted pattern name to family bucket.

    Args:
        pattern: detector pattern string
    Returns:
        str: family bucket name
    """
    if not pattern:
        return "other"
    p = pattern.lower()
    if p.startswith("smc_"):
        return "smc"
    if p.startswith("candle_") or any(
        k in p
        for k in (
            "doji",
            "hammer",
            "engulfing",
            "pinbar",
            "marubozu",
            "harami",
            "morningstar",
            "eveningstar",
            "soldiers",
            "crows",
        )
    ):
        return "candle"
    if any(
        k in p
        for k in (
            "flag",
            "pennant",
            "triangle",
            "wedge",
            "rectangle",
            "compression",
            "head_shoulders",
            "rounding",
            "megaphone",
            "double_top",
            "double_bottom",
            "triple_top",
            "triple_bottom",
            "diamond",
            "cup_handle",
        )
    ):
        return "chart"
    if any(k in p for k in ("vcp", "breakout", "inside_bar", "pivot_reversal", "bb_squeeze")):
        return "research"
    return "other"

def rule_for(pattern: str) -> FamilyRule:
    """Return the SL/TP rule for a pattern name.

    Args:
        pattern: detector pattern string
    Returns:
        FamilyRule: SL/TP preset (always non-None)
    """
    return RULES.get(family_of(pattern), RULES["other"])

def adaptive_sl_atr(
    pattern: str,
    *,
    vol_ratio: float | None = None,
) -> float:
    """Compute SL distance in ATR units with optional vol-regime adjustment.

    Args:
        pattern: detector pattern string
        vol_ratio: current_atr / median_atr (60d) or None
    Returns:
        float: SL distance in ATR units
    """
    rule = rule_for(pattern)
    sl = rule.sl_atr
    if vol_ratio is None or vol_ratio <= 0:
        return sl
    if vol_ratio > 1.5:
        return sl * rule.regime_high_sl_mult
    if vol_ratio < 0.7:
        return sl * rule.regime_low_sl_mult
    return sl

def derive_sl_tp(
    pattern: str,
    direction: str,
    entry: float,
    atr: float,
    *,
    detector_stop: float | None = None,
    detector_target: float | None = None,
    vol_ratio: float | None = None,
) -> tuple[float, float]:
    """Compute (stop_loss, take_profit) for a pattern signal.

    Args:
        pattern: detector pattern string
        direction: BUY or SELL
        entry: entry price
        atr: ATR value
        detector_stop: detector-supplied SL or None
        detector_target: detector-supplied TP or None
        vol_ratio: current_atr / median ratio or None
    Returns:
        tuple[float, float]: (stop_loss, take_profit)
    """
    if atr <= 0 or entry <= 0:
        return (
            detector_stop if detector_stop is not None else entry,
            detector_target if detector_target is not None else entry,
        )

    sl_atr = adaptive_sl_atr(pattern, vol_ratio=vol_ratio)
    rule = rule_for(pattern)
    dir_u = (direction or "").upper()

    if dir_u == "BUY":
        family_stop = entry - sl_atr * atr
        if detector_stop is not None:
            tight_bound = entry - (sl_atr / 3.0) * atr
            wide_bound = entry - sl_atr * 1.5 * atr
            bounded = min(max(detector_stop, wide_bound), tight_bound)
            family_stop = max(family_stop, bounded)
        sl = family_stop
        risk = entry - sl
        tp = entry + rule.rr * risk
    elif dir_u == "SELL":
        family_stop = entry + sl_atr * atr
        if detector_stop is not None:
            tight_bound = entry + (sl_atr / 3.0) * atr
            wide_bound = entry + sl_atr * 1.5 * atr
            bounded = max(min(detector_stop, wide_bound), tight_bound)
            family_stop = min(family_stop, bounded)
        sl = family_stop
        risk = sl - entry
        tp = entry - rule.rr * risk
    else:
        return (
            detector_stop if detector_stop is not None else entry,
            detector_target if detector_target is not None else entry,
        )

    return float(sl), float(tp)

@dataclass(frozen=True)
class RegimeExitAdjustment:
    """Per-regime overlay applied on top of family defaults."""

    sl_mult: float = 1.0
    tp_rr_mult: float = 1.0
    trailing_mult: float = 1.0
    disable_trailing: bool = False

_REGIME_ADJUSTMENTS: dict[str, RegimeExitAdjustment] = {
    "trending": RegimeExitAdjustment(sl_mult=1.10, tp_rr_mult=1.30, trailing_mult=1.20),
    "mean_reverting": RegimeExitAdjustment(sl_mult=0.90, tp_rr_mult=0.80, trailing_mult=0.80),
    "crisis": RegimeExitAdjustment(
        sl_mult=0.70, tp_rr_mult=0.70, trailing_mult=0.60, disable_trailing=True
    ),
    "unknown": RegimeExitAdjustment(),
}

def regime_exit_adjustment(hmm_regime: str | None) -> RegimeExitAdjustment:
    """Return (sl, tp, trailing) overlay for current HMM regime.

    Args:
        hmm_regime: regime label or None
    Returns:
        RegimeExitAdjustment: overlay multipliers
    """
    if not hmm_regime:
        return _REGIME_ADJUSTMENTS["unknown"]
    return _REGIME_ADJUSTMENTS.get(hmm_regime.lower(), _REGIME_ADJUSTMENTS["unknown"])

def derive_sl_tp_with_regime(
    pattern: str,
    direction: str,
    entry: float,
    atr: float,
    *,
    detector_stop: float | None = None,
    detector_target: float | None = None,
    vol_ratio: float | None = None,
    hmm_regime: str | None = None,
) -> tuple[float, float]:
    """Compute (SL, TP) with HMM regime overlay.

    Args:
        pattern: detector pattern string
        direction: BUY or SELL
        entry: entry price
        atr: ATR value
        detector_stop: detector SL or None
        detector_target: detector TP or None
        vol_ratio: vol ratio or None
        hmm_regime: regime label or None
    Returns:
        tuple[float, float]: (stop_loss, take_profit)
    """
    sl, tp = derive_sl_tp(
        pattern=pattern,
        direction=direction,
        entry=entry,
        atr=atr,
        detector_stop=detector_stop,
        detector_target=detector_target,
        vol_ratio=vol_ratio,
    )
    if not hmm_regime:
        return sl, tp

    adj = regime_exit_adjustment(hmm_regime)
    if adj.sl_mult == 1.0 and adj.tp_rr_mult == 1.0:
        return sl, tp

    dir_u = (direction or "").upper()
    if dir_u == "BUY":
        risk = entry - sl
        if risk <= 0:
            return sl, tp
        new_sl = entry - risk * adj.sl_mult
        rr_eff = (tp - entry) / risk
        new_tp = entry + (entry - new_sl) * (rr_eff * adj.tp_rr_mult / max(adj.sl_mult, 1e-9))
        return float(new_sl), float(new_tp)
    elif dir_u == "SELL":
        risk = sl - entry
        if risk <= 0:
            return sl, tp
        new_sl = entry + risk * adj.sl_mult
        rr_eff = (entry - tp) / risk
        new_tp = entry - (new_sl - entry) * (rr_eff * adj.tp_rr_mult / max(adj.sl_mult, 1e-9))
        return float(new_sl), float(new_tp)
    return sl, tp

__all__ = [
    "FamilyRule",
    "RULES",
    "RegimeExitAdjustment",
    "family_of",
    "rule_for",
    "adaptive_sl_atr",
    "derive_sl_tp",
    "derive_sl_tp_with_regime",
    "regime_exit_adjustment",
]
