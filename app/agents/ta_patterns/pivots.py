"""Детектор пивотов (ZigZag + find_peaks)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    from scipy.signal import argrelextrema, find_peaks  # type: ignore

    _READY = True
except ImportError:
    _READY = False

@dataclass
class PivotPoint:
    """A single price pivot with Dow Theory classification."""

    idx: int
    price: float
    kind: str
    label: str
    ts: Any = None
    volume: float = 0.0

    def is_high(self) -> bool:
        """Is high."""
        return self.kind == "H"

    def is_low(self) -> bool:
        """Is low."""
        return self.kind == "L"

    def is_higher_high(self) -> bool:
        """Is higher high."""
        return self.label == "HH"

    def is_lower_low(self) -> bool:
        """Is lower low."""
        return self.label == "LL"

    def is_higher_low(self) -> bool:
        """Is higher low."""
        return self.label == "HL"

    def is_lower_high(self) -> bool:
        """Is lower high."""
        return self.label == "LH"

def find_pivots(
    df: pd.DataFrame,
    order: int | None = None,
    atr: pd.Series | None = None,
    merge_distance_atr: float | None = None,
    method: str | None = None,
    atr_mult: float | None = None,
    min_bars_between: int | None = None,
    prominence_mult: float | None = None,
) -> list[PivotPoint]:
    """
    Detect swing highs and lows.

    Args:
        df:                 OHLCV DataFrame (must have open/high/low/close).
        order:              Legacy parameter for argrelextrema; ignored when method != 'argrelextrema'.
        atr:                ATR series (pd.Series). Required for ZigZag and prominence.
        merge_distance_atr: Same-kind pivots within this many ATRs are merged. Default 0.2
                            (was 0.5 — too aggressive, killed close pivots in tight setups).
        method:             "zigzag" (default, best) | "prominence" | "argrelextrema" (legacy)
        atr_mult:           For ZigZag — reversal threshold = atr_mult × ATR.
        min_bars_between:   For ZigZag — minimum bars between adjacent pivots.
        prominence_mult:    For prominence method — prominence ≥ prominence_mult × median ATR.

    Returns:
        Sorted list[PivotPoint] with Dow Theory labels applied.
    """
    import app.config as _cfg

    if order is None:
        order = int(getattr(_cfg, "PIVOT_ARGREL_ORDER", 5))
    if merge_distance_atr is None:
        merge_distance_atr = float(getattr(_cfg, "PIVOT_MERGE_DISTANCE_ATR", 0.2))
    if method is None:
        method = "prominence"
    if atr_mult is None:
        atr_mult = float(getattr(_cfg, "PIVOT_ZIGZAG_ATR_MULT", 0.6))
    if min_bars_between is None:
        min_bars_between = int(getattr(_cfg, "PIVOT_MIN_BARS_BETWEEN", 2))
    if prominence_mult is None:
        prominence_mult = float(getattr(_cfg, "PIVOT_PROMINENCE_MULT", 0.3))

    if not _READY or df is None or len(df) < 10:
        return []

    if method == "zigzag" and atr is not None:
        pivots = _zigzag_atr_pivots(df, atr, atr_mult, min_bars_between)
    elif method == "prominence" and atr is not None:
        pivots = _prominence_pivots(df, atr, prominence_mult, distance=max(min_bars_between, 3))
    else:
        pivots = _argrelextrema_pivots(df, order=order)

    if atr is not None and len(atr) > 0 and merge_distance_atr > 0:
        pivots = _merge_clusters(pivots, atr, df, merge_distance_atr)

    pivots = _label_dow_theory(pivots)

    logger.debug(
        "Pivots found",
        extra={
            "method": method,
            "total": len(pivots),
            "highs": sum(1 for p in pivots if p.kind == "H"),
            "lows": sum(1 for p in pivots if p.kind == "L"),
        },
    )
    return pivots

def _zigzag_atr_pivots(
    df: pd.DataFrame,
    atr: pd.Series,
    atr_mult: float = 0.6,
    min_bars_between: int = 2,
) -> list[PivotPoint]:
    """
    Walk through bars; confirm a new pivot whenever price reverses by
    `atr_mult × ATR` from the running extreme. Adapted from
    `TA/reversal.py:zigzag_atr_pivots_hilo`.
    """
    if df is None or len(df) < 5:
        return []

    highs = df["high"].values
    lows = df["low"].values

    ts_col = "begin" if "begin" in df.columns else None
    vol_col = "volume" if "volume" in df.columns else None

    n = len(df)
    pivots: list[PivotPoint] = []

    if hasattr(atr, "first_valid_index") and atr.first_valid_index() is not None:
        start_idx = int(atr.first_valid_index())
    else:
        start_idx = 0
    start_idx = max(start_idx, 0)

    last_pivot_idx = start_idx
    direction = 0
    candidate_high_idx = start_idx
    candidate_high = float(highs[start_idx])
    candidate_low_idx = start_idx
    candidate_low = float(lows[start_idx])

    def _make_pivot(idx: int, price: float, kind: str) -> PivotPoint:
        """Make pivot."""
        return PivotPoint(
            idx=int(idx),
            price=float(price),
            kind=kind,
            label="UNDEFINED",
            ts=df[ts_col].iloc[idx] if ts_col else None,
            volume=float(df[vol_col].iloc[idx]) if vol_col else 0.0,
        )

    for i in range(start_idx + 1, n):
        high = float(highs[i])
        low = float(lows[i])
        atr_val = float(atr.iloc[i]) if i < len(atr) and pd.notna(atr.iloc[i]) else 0.0

        if atr_val <= 0:
            continue

        threshold = atr_mult * atr_val

        if high >= candidate_high:
            candidate_high = high
            candidate_high_idx = i
        if low <= candidate_low:
            candidate_low = low
            candidate_low_idx = i

        if direction == 0:
            if (high - candidate_low) >= threshold and (i - candidate_low_idx) >= min_bars_between:
                pivots.append(_make_pivot(candidate_low_idx, candidate_low, "L"))
                direction = 1
                last_pivot_idx = candidate_low_idx
                candidate_high = high
                candidate_high_idx = i
            elif (candidate_high - low) >= threshold and (
                i - candidate_high_idx
            ) >= min_bars_between:
                pivots.append(_make_pivot(candidate_high_idx, candidate_high, "H"))
                direction = -1
                last_pivot_idx = candidate_high_idx
                candidate_low = low
                candidate_low_idx = i
            continue

        if direction == 1:
            if (candidate_high - low) >= threshold and (
                candidate_high_idx - last_pivot_idx
            ) >= min_bars_between:
                pivots.append(_make_pivot(candidate_high_idx, candidate_high, "H"))
                direction = -1
                last_pivot_idx = candidate_high_idx
                candidate_low = low
                candidate_low_idx = i

        elif direction == -1:
            if (high - candidate_low) >= threshold and (
                candidate_low_idx - last_pivot_idx
            ) >= min_bars_between:
                pivots.append(_make_pivot(candidate_low_idx, candidate_low, "L"))
                direction = 1
                last_pivot_idx = candidate_low_idx
                candidate_high = high
                candidate_high_idx = i

    cleaned: list[PivotPoint] = []
    for p in pivots:
        if not cleaned or cleaned[-1].kind != p.kind:
            cleaned.append(p)
            continue

        if (
            p.kind == "H"
            and p.price > cleaned[-1].price
            or p.kind == "L"
            and p.price < cleaned[-1].price
        ):
            cleaned[-1] = p

    return cleaned

def _prominence_pivots(
    df: pd.DataFrame,
    atr: pd.Series,
    prominence_mult: float = 0.5,
    distance: int = 3,
) -> list[PivotPoint]:
    """
    Use scipy.signal.find_peaks with prominence threshold scaled by median ATR.
    """
    if df is None or len(df) < 10:
        return []

    median_atr = float(atr.dropna().median()) if hasattr(atr, "dropna") else 0.0
    if median_atr <= 0:
        return []
    prominence = prominence_mult * median_atr

    highs = df["high"].values
    lows = df["low"].values

    peak_idx, _ = find_peaks(highs, prominence=prominence, distance=distance)
    trough_idx, _ = find_peaks(-lows, prominence=prominence, distance=distance)

    ts_col = "begin" if "begin" in df.columns else None
    vol_col = "volume" if "volume" in df.columns else None

    pivots: list[PivotPoint] = []
    for i in peak_idx:
        pivots.append(
            PivotPoint(
                idx=int(i),
                price=float(highs[i]),
                kind="H",
                label="UNDEFINED",
                ts=df[ts_col].iloc[i] if ts_col else None,
                volume=float(df[vol_col].iloc[i]) if vol_col else 0.0,
            )
        )
    for i in trough_idx:
        pivots.append(
            PivotPoint(
                idx=int(i),
                price=float(lows[i]),
                kind="L",
                label="UNDEFINED",
                ts=df[ts_col].iloc[i] if ts_col else None,
                volume=float(df[vol_col].iloc[i]) if vol_col else 0.0,
            )
        )

    pivots.sort(key=lambda p: p.idx)
    return pivots

def _argrelextrema_pivots(df: pd.DataFrame, order: int = 5) -> list[PivotPoint]:
    """Original implementation kept for backward compat and unit tests."""
    if df is None or len(df) < order * 2 + 1:
        return []

    highs = df["high"].values
    lows = df["low"].values
    peak_idx = argrelextrema(highs, np.greater, order=order)[0]
    trough_idx = argrelextrema(lows, np.less, order=order)[0]

    ts_col = "begin" if "begin" in df.columns else None
    vol_col = "volume" if "volume" in df.columns else None

    pivots: list[PivotPoint] = []
    for i in peak_idx:
        pivots.append(
            PivotPoint(
                idx=int(i),
                price=float(highs[i]),
                kind="H",
                label="UNDEFINED",
                ts=df[ts_col].iloc[i] if ts_col else None,
                volume=float(df[vol_col].iloc[i]) if vol_col else 0.0,
            )
        )
    for i in trough_idx:
        pivots.append(
            PivotPoint(
                idx=int(i),
                price=float(lows[i]),
                kind="L",
                label="UNDEFINED",
                ts=df[ts_col].iloc[i] if ts_col else None,
                volume=float(df[vol_col].iloc[i]) if vol_col else 0.0,
            )
        )

    pivots.sort(key=lambda p: p.idx)
    return pivots

def _merge_clusters(
    pivots: list[PivotPoint],
    atr: pd.Series,
    df: pd.DataFrame,
    merge_atr: float,
) -> list[PivotPoint]:
    """
    Merge same-kind pivots that are closer than merge_atr × ATR in price.
    Keeps the extreme price and the later index.
    """
    if not pivots:
        return pivots

    merged: list[PivotPoint] = []
    i = 0
    while i < len(pivots):
        p = pivots[i]
        atr_val = float(atr.iloc[p.idx]) if p.idx < len(atr) and pd.notna(atr.iloc[p.idx]) else 0.0
        if atr_val <= 0:
            merged.append(p)
            i += 1
            continue

        threshold = merge_atr * atr_val
        cluster = [p]
        j = i + 1
        while j < len(pivots) and pivots[j].kind == p.kind:
            next_p = pivots[j]
            if abs(next_p.price - p.price) <= threshold:
                cluster.append(next_p)
                j += 1
            else:
                break

        if len(cluster) == 1:
            merged.append(p)
        else:
            best = (
                max(cluster, key=lambda x: x.price)
                if p.kind == "H"
                else min(cluster, key=lambda x: x.price)
            )
            latest = max(cluster, key=lambda x: x.idx)
            best.idx = latest.idx
            best.ts = latest.ts
            merged.append(best)
        i = j

    return merged

def _label_dow_theory(pivots: list[PivotPoint]) -> list[PivotPoint]:
    """Assign HH/HL/LH/LL labels by comparing each pivot to previous of same kind."""
    last_high: PivotPoint | None = None
    last_low: PivotPoint | None = None

    for p in pivots:
        if p.kind == "H":
            if last_high is None or p.price > last_high.price:
                p.label = "HH"
            else:
                p.label = "LH"
            last_high = p
        else:
            if last_low is None or p.price < last_low.price:
                p.label = "LL"
            else:
                p.label = "HL"
            last_low = p
    return pivots

def get_recent_pivots(
    pivots: list[PivotPoint],
    n: int = 10,
    kind: str | None = None,
) -> list[PivotPoint]:
    """Return the n most recent pivots (optionally filtered by kind H/L)."""
    filtered = [p for p in pivots if kind is None or p.kind == kind]
    return filtered[-n:]

def trend_direction(pivots: list[PivotPoint]) -> str:
    """Infer trend from the last 2 highs + 2 lows."""
    highs = [p for p in pivots if p.kind == "H"][-2:]
    lows = [p for p in pivots if p.kind == "L"][-2:]
    if len(highs) < 2 or len(lows) < 2:
        return "UNDEFINED"

    hh = highs[1].price > highs[0].price
    hl = lows[1].price > lows[0].price
    lh = highs[1].price < highs[0].price
    ll = lows[1].price < lows[0].price
    if hh and hl:
        return "UP"
    if lh and ll:
        return "DOWN"
    return "SIDEWAYS"
