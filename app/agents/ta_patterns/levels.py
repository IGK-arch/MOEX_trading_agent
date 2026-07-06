"""Support/resistance levels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.utils.logging import get_logger

from .pivots import PivotPoint

logger = get_logger(__name__)

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    _READY = True
except ImportError:
    _READY = False

@dataclass
class Level:
    """A price level with strength score."""

    price: float
    kind: str
    touches: int
    avg_volume: float = 0.0
    last_touch_idx: int = 0
    strength: float = 0.0
    pivot_idxs: list[int] = field(default_factory=list)

    def distance_to(self, price: float) -> float:
        """Distance to."""
        return abs(self.price - price)

    def is_above(self, price: float) -> bool:
        """Is above."""
        return self.price > price

    def is_below(self, price: float) -> bool:
        """Is below."""
        return self.price < price

def find_support_resistance(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr: pd.Series,
    current_price: float | None = None,
    merge_atr: float = 0.5,
    min_touches: int = 2,
    recency_tau: float = 50.0,
) -> list[Level]:
    """
    Cluster pivots into support/resistance levels.

    Args:
        df: OHLCV DataFrame
        pivots: list[PivotPoint]
        atr: ATR series for clustering distance + strength scaling
        current_price: If provided, levels split into "above" (resistance) / "below" (support)
                       regardless of pivot kind. Useful when a former resistance flipped.
        merge_atr: cluster width in ATR multiples
        min_touches: minimum pivots to form a level
        recency_tau: time-decay constant (bars). Older touches contribute less.

    Returns:
        List of Level objects sorted by strength descending.
    """
    if not _READY or not pivots or df is None:
        return []

    n_bars = len(df)
    if n_bars == 0:
        return []

    if len(atr) == 0:
        return []
    avg_atr = float(atr.dropna().tail(50).mean()) if hasattr(atr, "dropna") else 0.0
    if avg_atr <= 0:
        return []

    merge_distance = merge_atr * avg_atr

    sorted_pivs = sorted(pivots, key=lambda p: p.price)

    clusters: list[list[PivotPoint]] = []
    current_cluster: list[PivotPoint] = []

    for p in sorted_pivs:
        if not current_cluster:
            current_cluster.append(p)
            continue
        cluster_price = np.mean([x.price for x in current_cluster])
        if abs(p.price - cluster_price) <= merge_distance:
            current_cluster.append(p)
        else:
            if len(current_cluster) >= min_touches:
                clusters.append(current_cluster)
            current_cluster = [p]
    if len(current_cluster) >= min_touches:
        clusters.append(current_cluster)

    levels: list[Level] = []
    for cluster in clusters:
        avg_price = float(np.mean([p.price for p in cluster]))
        avg_vol = float(np.mean([p.volume for p in cluster]))
        last_idx = max(p.idx for p in cluster)
        ages = [n_bars - p.idx for p in cluster]
        recency_weights = [math.exp(-a / recency_tau) for a in ages]
        recency_score = sum(recency_weights)

        recent_vol_mean = float(df["volume"].tail(50).mean()) if "volume" in df.columns else 1.0
        vol_score = (avg_vol / recent_vol_mean) if recent_vol_mean > 0 else 1.0

        strength = len(cluster) * recency_score * (0.5 + 0.5 * vol_score)

        n_highs = sum(1 for p in cluster if p.kind == "H")
        n_lows = sum(1 for p in cluster if p.kind == "L")
        if current_price is not None:
            kind = "resistance" if avg_price > current_price else "support"
        else:
            kind = "resistance" if n_highs > n_lows else "support"

        levels.append(
            Level(
                price=avg_price,
                kind=kind,
                touches=len(cluster),
                avg_volume=avg_vol,
                last_touch_idx=last_idx,
                strength=strength,
                pivot_idxs=[p.idx for p in cluster],
            )
        )

    levels.sort(key=lambda l: l.strength, reverse=True)
    logger.debug(
        "find_support_resistance",
        extra={"clusters": len(clusters), "min_touches": min_touches},
    )
    return levels

def nearest_levels(
    levels: list[Level],
    price: float,
    k: int = 3,
) -> dict[str, list[Level]]:
    """
    Return {'support': [...], 'resistance': [...]} — k nearest levels below/above.
    """
    below = [l for l in levels if l.price < price]
    above = [l for l in levels if l.price > price]
    below.sort(key=lambda l: abs(price - l.price))
    above.sort(key=lambda l: abs(price - l.price))
    return {"support": below[:k], "resistance": above[:k]}

def distance_to_nearest_atrs(
    levels: list[Level],
    price: float,
    atr_val: float,
) -> dict[str, float]:
    """
    Return {'support_atrs': X, 'resistance_atrs': Y} — distance to nearest
    support and resistance levels measured in ATR units.

    Useful as a CatBoost feature: "how close are we to a strong level?"
    """
    if atr_val <= 0:
        return {"support_atrs": float("inf"), "resistance_atrs": float("inf")}

    near = nearest_levels(levels, price, k=1)
    sup_dist = near["support"][0].distance_to(price) / atr_val if near["support"] else float("inf")
    res_dist = (
        near["resistance"][0].distance_to(price) / atr_val if near["resistance"] else float("inf")
    )
    return {"support_atrs": sup_dist, "resistance_atrs": res_dist}
