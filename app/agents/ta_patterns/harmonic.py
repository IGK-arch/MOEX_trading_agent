"""
app/agents/ta_patterns/harmonic.py — 6 Fibonacci-based XABCD patterns.

Each detector searches the last `recent_n` pivots for a 5-point sequence
X-A-B-C-D where the price legs match prescribed Fibonacci ratios:

| Pattern   | AB/XA          | BC/AB          | CD/BC          | AD/XA     |
|-----------|----------------|----------------|----------------|-----------|
| Gartley   | 0.618          | 0.382 – 0.886  | 1.272 – 1.618  | 0.786     |
| Bat       | 0.382 – 0.500  | 0.382 – 0.886  | 1.618 – 2.618  | 0.886     |
| Butterfly | 0.786          | 0.382 – 0.886  | 1.618 – 2.240  | 1.272 – 1.618 |
| Crab      | 0.382 – 0.618  | 0.382 – 0.886  | 2.240 – 3.618  | 1.618     |
| Cypher    | 0.382 – 0.618  | 1.130 – 1.414  | 1.272 – 2.000  | 0.786     |
| Shark     | (5-0) any      | 1.130 – 1.618  | 0.886 – 1.130  | (use OX-AB) |

Bullish patterns end at a low (D) and emit BUY; bearish patterns end at a
high (D) and emit SELL.

Entry / Stop / Target rules:
  entry  = D price
  stop   = D ± 0.382 × CD  (just beyond the D pivot)
  target = 38.2% retracement of CD towards C (conservative)

Tolerance is ±5% by default (`HARMONIC_TOLERANCE`).
"""

from __future__ import annotations

from app.agents.ta_patterns.pivots import PivotPoint
from app.agents.ta_patterns.reversal import PatternSignal, _atr_at, _rr
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import numpy as np  # noqa: F401
    import pandas as pd

    _READY = True
except ImportError:
    _READY = False

HARMONIC_TOLERANCE = 0.05

def _in_range(actual: float, lo: float, hi: float, tol: float = HARMONIC_TOLERANCE) -> bool:
    """In range."""
    return (lo * (1 - tol)) <= actual <= (hi * (1 + tol))

def _ratio(diff_num: float, diff_den: float) -> float:
    """Ratio."""
    if diff_den <= 0:
        return 0.0
    return diff_num / diff_den

def _make_signal(
    pattern_name: str,
    direction: str,
    bar_idx: int,
    entry: float,
    leg_cd: float,
    atr_series: pd.Series,
    confidence: float,
) -> PatternSignal:
    """Common builder: stop just past D, target = 38.2% retrace of CD."""
    is_buy = direction == "BUY"
    stop_offset = 0.382 * abs(leg_cd)
    stop = entry - stop_offset if is_buy else entry + stop_offset
    target = entry + 0.382 * abs(leg_cd) if is_buy else entry - 0.382 * abs(leg_cd)

    target = entry + 0.618 * abs(leg_cd) if is_buy else entry - 0.618 * abs(leg_cd)
    return PatternSignal(
        pattern=pattern_name,
        direction=direction,
        confidence=confidence,
        bar_idx=bar_idx,
        entry=entry,
        stop=stop,
        target=target,
        expected_rr=_rr(entry, stop, target),
        atr_at_entry=_atr_at(atr_series, bar_idx),
    )

def _iter_xabcd(pivots: list[PivotPoint]):
    """
    Yield all (X, A, B, C, D) tuples where pivots strictly alternate
    high/low and are listed in chronological order.
    """
    n = len(pivots)
    if n < 5:
        return

    pivots = pivots[-30:]
    n = len(pivots)
    for i in range(n - 4):
        X, A, B, C, D = pivots[i : i + 5]

        ks = (X.kind, A.kind, B.kind, C.kind, D.kind)
        if ks not in (("H", "L", "H", "L", "H"), ("L", "H", "L", "H", "L")):
            continue
        yield X, A, B, C, D

def _detect_harmonic(
    pattern_name: str,
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr_series: pd.Series,
    *,
    ab_min: float,
    ab_max: float,
    bc_min: float,
    bc_max: float,
    cd_min: float,
    cd_max: float,
    ad_min: float,
    ad_max: float,
) -> list[PatternSignal]:
    """
    Generic harmonic detector. Each *pattern* (Gartley/Bat/...) is a thin
    wrapper that supplies its own ratio bounds.
    """
    if not _READY or df is None or len(df) < 10:
        return []

    out: list[PatternSignal] = []
    for X, A, B, C, D in _iter_xabcd(pivots):
        xa = A.price - X.price
        ab = B.price - A.price
        bc = C.price - B.price
        cd = D.price - C.price
        ad = D.price - X.price

        if xa == 0 or ab == 0 or bc == 0:
            continue

        ab_over_xa = abs(ab) / abs(xa)
        bc_over_ab = abs(bc) / abs(ab)
        cd_over_bc = abs(cd) / abs(bc)
        ad_over_xa = abs(ad) / abs(xa)

        if not _in_range(ab_over_xa, ab_min, ab_max):
            continue
        if not _in_range(bc_over_ab, bc_min, bc_max):
            continue
        if not _in_range(cd_over_bc, cd_min, cd_max):
            continue
        if not _in_range(ad_over_xa, ad_min, ad_max):
            continue

        direction = "BUY" if D.kind == "L" else "SELL"

        ratio_devs = [
            abs(ab_over_xa - (ab_min + ab_max) / 2) / max(0.01, (ab_max - ab_min) / 2),
            abs(bc_over_ab - (bc_min + bc_max) / 2) / max(0.01, (bc_max - bc_min) / 2),
            abs(cd_over_bc - (cd_min + cd_max) / 2) / max(0.01, (cd_max - cd_min) / 2),
            abs(ad_over_xa - (ad_min + ad_max) / 2) / max(0.01, (ad_max - ad_min) / 2),
        ]
        confidence = max(0.0, min(1.0, 0.85 - 0.05 * sum(ratio_devs)))

        sig = _make_signal(
            pattern_name=pattern_name,
            direction=direction,
            bar_idx=D.idx,
            entry=D.price,
            leg_cd=cd,
            atr_series=atr_series,
            confidence=confidence,
        )
        sig.metadata = {
            "X_idx": X.idx,
            "A_idx": A.idx,
            "B_idx": B.idx,
            "C_idx": C.idx,
            "ab_over_xa": round(ab_over_xa, 3),
            "bc_over_ab": round(bc_over_ab, 3),
            "cd_over_bc": round(cd_over_bc, 3),
            "ad_over_xa": round(ad_over_xa, 3),
        }
        out.append(sig)

    return out

def detect_gartley(df, pivots, atr_series):
    """Detect gartley."""
    return _detect_harmonic(
        "gartley",
        df,
        pivots,
        atr_series,
        ab_min=0.618,
        ab_max=0.618,
        bc_min=0.382,
        bc_max=0.886,
        cd_min=1.272,
        cd_max=1.618,
        ad_min=0.786,
        ad_max=0.786,
    )

def detect_bat(df, pivots, atr_series):
    """Detect bat."""
    return _detect_harmonic(
        "bat",
        df,
        pivots,
        atr_series,
        ab_min=0.382,
        ab_max=0.500,
        bc_min=0.382,
        bc_max=0.886,
        cd_min=1.618,
        cd_max=2.618,
        ad_min=0.886,
        ad_max=0.886,
    )

def detect_butterfly(df, pivots, atr_series):
    """Detect butterfly."""
    return _detect_harmonic(
        "butterfly",
        df,
        pivots,
        atr_series,
        ab_min=0.786,
        ab_max=0.786,
        bc_min=0.382,
        bc_max=0.886,
        cd_min=1.618,
        cd_max=2.240,
        ad_min=1.272,
        ad_max=1.618,
    )

def detect_crab(df, pivots, atr_series):
    """Detect crab."""
    return _detect_harmonic(
        "crab",
        df,
        pivots,
        atr_series,
        ab_min=0.382,
        ab_max=0.618,
        bc_min=0.382,
        bc_max=0.886,
        cd_min=2.240,
        cd_max=3.618,
        ad_min=1.618,
        ad_max=1.618,
    )

def detect_cypher(df, pivots, atr_series):
    """Detect cypher."""
    return _detect_harmonic(
        "cypher",
        df,
        pivots,
        atr_series,
        ab_min=0.382,
        ab_max=0.618,
        bc_min=1.130,
        bc_max=1.414,
        cd_min=1.272,
        cd_max=2.000,
        ad_min=0.786,
        ad_max=0.786,
    )

def detect_shark(df, pivots, atr_series):
    """Detect shark."""

    return _detect_harmonic(
        "shark",
        df,
        pivots,
        atr_series,
        ab_min=0.500,
        ab_max=1.000,
        bc_min=1.130,
        bc_max=1.618,
        cd_min=0.886,
        cd_max=1.130,
        ad_min=0.886,
        ad_max=1.130,
    )

HARMONIC_DETECTORS = [
    detect_gartley,
    detect_bat,
    detect_butterfly,
    detect_crab,
    detect_cypher,
    detect_shark,
]

PRODUCTION_PATTERNS: set[str] = {
    "gartley",
    "bat",
    "butterfly",
    "crab",
    "cypher",
    "shark",
}

__all__ = [
    "detect_gartley",
    "detect_bat",
    "detect_butterfly",
    "detect_crab",
    "detect_cypher",
    "detect_shark",
    "HARMONIC_DETECTORS",
    "PRODUCTION_PATTERNS",
]
