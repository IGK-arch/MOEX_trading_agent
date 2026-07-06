"""
app/agents/ta_patterns/extras_chart.py — 4 chart patterns missing from
reversal.py / continuation.py.

  - diamond_top / diamond_bottom — broadening-then-contracting (rare but high R:R)
  - cup_and_handle / inverted_cup — round U + small consolidation handle
  - box_breakout_up / box_breakout_down — narrow horizontal range break
  - wedge_continuation_up / down — continuation wedge in trending markets

All four are written to return `list[PatternSignal]` so they can be plugged
into `REVERSAL_DETECTORS` / `CONTINUATION_DETECTORS` lists in `ta_trader.py`.
"""

from __future__ import annotations

from app.agents.ta_patterns.pivots import PivotPoint
from app.agents.ta_patterns.reversal import PatternSignal, _atr_at, _rr
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import numpy as np
    import pandas as pd

    _READY = True
except ImportError:
    _READY = False

def _prior_impulse(
    df: pd.DataFrame,
    atr_series: pd.Series,
    first_idx: int,
    bars: int = 36,
) -> tuple[float, float]:
    """
    Lightweight prior-impulse measure used to gate continuation breakouts.

    Returns (return_pct, impulse_in_atr) over the `bars` bars preceding
    `first_idx`. Mirrors `continuation._prior_impulse`. Ported from the
    reference notebook (TA/continuation patterns) Phase 11.13.
    """
    if not _READY or df is None or first_idx <= 0 or len(df) == 0:
        return 0.0, 0.0
    first_idx = int(first_idx)
    start = max(0, first_idx - bars)
    if first_idx <= start:
        return 0.0, 0.0
    atr_val = _atr_at(atr_series, first_idx)
    if atr_val <= 0:
        return 0.0, 0.0
    c0 = float(df["close"].iloc[start])
    c1 = float(df["close"].iloc[first_idx])
    if c0 <= 0:
        return 0.0, 0.0
    return c1 / c0 - 1.0, abs(c1 - c0) / atr_val

_PRIOR_IMPULSE_BARS = 36
_MIN_PRIOR_IMPULSE_RETURN = 0.003
_MIN_PRIOR_IMPULSE_ATR = 0.5

def detect_diamond(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr_series: pd.Series,
    *,
    min_pivots: int = 6,
) -> list[PatternSignal]:
    """
    Diamond top / bottom — broadening then contracting range.

    Detection:
      Take 6 most recent alternating pivots P1..P6 (H,L,H,L,H,L or inverse).
      The range (H − L) should widen from P1-P2 to P3-P4 (broadening),
      then narrow from P3-P4 to P5-P6 (contracting). Break of the contracting
      side is the entry.
    """
    if not _READY or df is None or len(df) < 30 or len(pivots) < min_pivots:
        return []

    recent = pivots[-12:]
    out: list[PatternSignal] = []

    for i in range(len(recent) - min_pivots + 1):
        seq = recent[i : i + min_pivots]
        if len(seq) < min_pivots:
            continue

        ks = [p.kind for p in seq]
        if ks not in (["H", "L"] * (min_pivots // 2), ["L", "H"] * (min_pivots // 2)):
            continue

        rng1 = abs(seq[0].price - seq[1].price)
        rng2 = abs(seq[2].price - seq[3].price)
        rng3 = abs(seq[4].price - seq[5].price)

        if not (rng2 > rng1 * 1.1 and rng3 < rng2 * 0.9):
            continue
        last_idx = seq[-1].idx
        if last_idx >= len(df) - 1:
            continue

        centre = (seq[-2].price + seq[-1].price) / 2
        next_close = float(df["close"].iloc[last_idx + 1]) if last_idx + 1 < len(df) else None
        if next_close is None:
            continue
        atr_val = _atr_at(atr_series, last_idx + 1)
        if atr_val <= 0:
            continue

        is_top = ks[0] == "H"
        if is_top and next_close < centre:
            direction = "SELL"
            entry = next_close
            stop = max(p.price for p in seq) + 0.2 * atr_val
            target = entry - 1.5 * atr_val
            pattern = "diamond_top"
        elif (not is_top) and next_close > centre:
            direction = "BUY"
            entry = next_close
            stop = min(p.price for p in seq) - 0.2 * atr_val
            target = entry + 1.5 * atr_val
            pattern = "diamond_bottom"
        else:
            continue
        out.append(
            PatternSignal(
                pattern=pattern,
                direction=direction,
                confidence=0.65,
                bar_idx=last_idx + 1,
                entry=entry,
                stop=stop,
                target=target,
                expected_rr=_rr(entry, stop, target),
                atr_at_entry=atr_val,
                metadata={"pivots": [p.idx for p in seq]},
            )
        )
    return out

def detect_cup_handle(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr_series: pd.Series,
    *,
    min_bars: int = 20,
) -> list[PatternSignal]:
    """
    Cup & Handle (bullish) — round U-shape ('cup') + short consolidation ('handle').

    Algorithm:
      1. In the last ~50 bars find a local minimum 'cup bottom'
      2. Left rim (~highs to the left) and right rim (highs to the right) at similar levels
      3. 'Handle' = small pullback after right rim, less than half of cup depth
      4. Entry = break above handle high
    """
    if not _READY or df is None or len(df) < 50:
        return []

    window = df.tail(60).reset_index(drop=True)
    if len(window) < min_bars:
        return []

    close = window["close"].astype(float).to_numpy()
    high = window["high"].astype(float).to_numpy()
    low = window["low"].astype(float).to_numpy()

    start = int(len(window) * 0.1)
    end = int(len(window) * 0.7)
    if end <= start + 5:
        return []
    bottom_rel = int(np.argmin(low[start:end])) + start
    bottom_price = float(low[bottom_rel])

    left_rim_idx = int(np.argmax(high[:bottom_rel])) if bottom_rel > 0 else 0
    left_rim = float(high[left_rim_idx])

    right_search_end = min(len(window) - 5, bottom_rel + 25)
    if right_search_end <= bottom_rel + 3:
        return []
    right_rim_rel = int(np.argmax(high[bottom_rel + 1 : right_search_end])) + bottom_rel + 1
    right_rim = float(high[right_rim_rel])

    if abs(left_rim - right_rim) / max(left_rim, 1e-6) > 0.03:
        return []
    rim_level = (left_rim + right_rim) / 2
    cup_depth = rim_level - bottom_price
    if cup_depth <= 0 or cup_depth / rim_level < 0.05:
        return []

    handle_low = (
        float(low[right_rim_rel + 1 :].min()) if right_rim_rel + 1 < len(window) else right_rim
    )
    handle_drop = rim_level - handle_low
    if handle_drop <= 0 or handle_drop > cup_depth * 0.5:
        return []

    float(high[right_rim_rel:].max())
    if close[-1] <= rim_level * 1.001:
        return []

    full_idx = len(df) - 1
    atr_val = _atr_at(atr_series, full_idx)
    if atr_val <= 0:
        return []
    entry = float(close[-1])
    stop = handle_low - 0.2 * atr_val
    target = entry + cup_depth
    return [
        PatternSignal(
            pattern="cup_and_handle",
            direction="BUY",
            confidence=0.70,
            bar_idx=full_idx,
            entry=entry,
            stop=stop,
            target=target,
            expected_rr=_rr(entry, stop, target),
            atr_at_entry=atr_val,
            metadata={
                "rim_level": round(rim_level, 4),
                "cup_depth": round(cup_depth, 4),
                "handle_drop": round(handle_drop, 4),
            },
        )
    ]

def detect_box_breakout(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr_series: pd.Series,
    *,
    box_bars: int = 15,
    width_max_atr: float = 1.5,
    require_prior_impulse: bool = False,
    prior_impulse_bars: int = _PRIOR_IMPULSE_BARS,
) -> list[PatternSignal]:
    """
    Box breakout — narrow horizontal range followed by breakout bar.

    Algorithm:
      1. Take last `box_bars` bars (excluding the most recent one)
      2. Check if their (max-min) range is < width_max_atr × ATR
      3. If last close breaks outside the box → BUY/SELL signal

    Phase 11.13 — added optional `require_prior_impulse` gate (default OFF —
    available as opt-in for higher-precision continuation-only entries; matches
    reference notebook semantics where bull-box only fires when prior trend is up).
    """
    if not _READY or df is None or len(df) < box_bars + 2:
        return []

    box = df.iloc[-box_bars - 1 : -1]
    last = df.iloc[-1]
    last_idx = len(df) - 1
    atr_val = _atr_at(atr_series, last_idx)
    if atr_val <= 0:
        return []

    box_high = float(box["high"].max())
    box_low = float(box["low"].min())
    box_range = box_high - box_low
    if box_range > width_max_atr * atr_val:
        return []
    if box_range / max(box_low, 1e-6) < 0.002:
        return []

    if require_prior_impulse:
        box_start = last_idx - box_bars
        impulse_ret, impulse_atr = _prior_impulse(
            df, atr_series, box_start, bars=prior_impulse_bars
        )
        allow_up = (
            impulse_ret >= _MIN_PRIOR_IMPULSE_RETURN and impulse_atr >= _MIN_PRIOR_IMPULSE_ATR
        )
        allow_down = (
            impulse_ret <= -_MIN_PRIOR_IMPULSE_RETURN and impulse_atr >= _MIN_PRIOR_IMPULSE_ATR
        )
    else:
        allow_up = allow_down = True
        impulse_atr = 0.0

    close = float(last["close"])
    if allow_up and close > box_high:
        entry = close
        stop = box_low - 0.2 * atr_val
        target = entry + (box_high - box_low)
        return [
            PatternSignal(
                pattern="box_breakout_up",
                direction="BUY",
                confidence=0.60,
                bar_idx=last_idx,
                entry=entry,
                stop=stop,
                target=target,
                expected_rr=_rr(entry, stop, target),
                atr_at_entry=atr_val,
                metadata={
                    "box_high": round(box_high, 4),
                    "box_low": round(box_low, 4),
                    "prior_impulse_atr": round(impulse_atr, 2) if require_prior_impulse else 0.0,
                },
            )
        ]
    if allow_down and close < box_low:
        entry = close
        stop = box_high + 0.2 * atr_val
        target = entry - (box_high - box_low)
        return [
            PatternSignal(
                pattern="box_breakout_down",
                direction="SELL",
                confidence=0.60,
                bar_idx=last_idx,
                entry=entry,
                stop=stop,
                target=target,
                expected_rr=_rr(entry, stop, target),
                atr_at_entry=atr_val,
                metadata={
                    "box_high": round(box_high, 4),
                    "box_low": round(box_low, 4),
                    "prior_impulse_atr": round(impulse_atr, 2) if require_prior_impulse else 0.0,
                },
            )
        ]
    return []

def detect_wedge_continuation(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr_series: pd.Series,
    *,
    min_pivots: int = 4,
    require_prior_impulse: bool = False,
    prior_impulse_bars: int = _PRIOR_IMPULSE_BARS,
) -> list[PatternSignal]:
    """
    Continuation wedge — small wedge AGAINST the prior trend, eventually
    breaking with the trend.

    Algorithm:
      1. Take last 4 pivots: should be H,L,H,L or L,H,L,H
      2. For BUY (bull continuation): pivots tilt DOWN (falling wedge in uptrend)
         and last close breaks above the upper trendline. Prior impulse must
         be POSITIVE (uptrend the wedge is continuing) when the gate is on.
      3. For SELL: rising wedge in downtrend, last close breaks below lower.
         Prior impulse must be NEGATIVE.

    Phase 11.13 — added `require_prior_impulse` gate (default OFF, opt-in only)
    to enforce the "continuation" semantics: falling wedge is only a buy if
    prior trend is up, rising wedge is only a sell if prior trend is down.
    """
    if not _READY or df is None or len(df) < 20 or len(pivots) < min_pivots:
        return []

    recent = pivots[-6:]
    if len(recent) < min_pivots:
        return []

    out: list[PatternSignal] = []
    for i in range(len(recent) - min_pivots + 1):
        seq = recent[i : i + min_pivots]
        ks = [p.kind for p in seq]
        if ks not in (["H", "L", "H", "L"], ["L", "H", "L", "H"]):
            continue

        highs = [p for p in seq if p.is_high()]
        lows = [p for p in seq if p.is_low()]
        if len(highs) < 2 or len(lows) < 2:
            continue
        upper_slope = (highs[1].price - highs[0].price) / max(1, highs[1].idx - highs[0].idx)
        lower_slope = (lows[1].price - lows[0].price) / max(1, lows[1].idx - lows[0].idx)

        if upper_slope * lower_slope <= 0:
            continue
        is_falling = upper_slope < 0 and lower_slope < 0
        is_rising = upper_slope > 0 and lower_slope > 0
        if not (is_falling or is_rising):
            continue

        last_idx = len(df) - 1
        last_close = float(df["close"].iloc[-1])
        atr_val = _atr_at(atr_series, last_idx)
        if atr_val <= 0:
            continue

        first_pivot_idx = seq[0].idx
        if require_prior_impulse:
            impulse_ret, impulse_atr = _prior_impulse(
                df, atr_series, first_pivot_idx, bars=prior_impulse_bars
            )
            allow_up = (
                impulse_ret >= _MIN_PRIOR_IMPULSE_RETURN and impulse_atr >= _MIN_PRIOR_IMPULSE_ATR
            )
            allow_down = (
                impulse_ret <= -_MIN_PRIOR_IMPULSE_RETURN and impulse_atr >= _MIN_PRIOR_IMPULSE_ATR
            )
        else:
            allow_up = allow_down = True
            impulse_atr = 0.0

        ref_h = highs[-1]
        proj_upper = ref_h.price + upper_slope * (last_idx - ref_h.idx)
        ref_l = lows[-1]
        proj_lower = ref_l.price + lower_slope * (last_idx - ref_l.idx)

        if is_falling and allow_up and last_close > proj_upper:
            entry = last_close
            stop = proj_lower - 0.2 * atr_val
            target = entry + 1.5 * atr_val
            out.append(
                PatternSignal(
                    pattern="wedge_continuation_up",
                    direction="BUY",
                    confidence=0.62,
                    bar_idx=last_idx,
                    entry=entry,
                    stop=stop,
                    target=target,
                    expected_rr=_rr(entry, stop, target),
                    atr_at_entry=atr_val,
                    metadata={"prior_impulse_atr": round(impulse_atr, 2)}
                    if require_prior_impulse
                    else {},
                )
            )

        elif is_rising and allow_down and last_close < proj_lower:
            entry = last_close
            stop = proj_upper + 0.2 * atr_val
            target = entry - 1.5 * atr_val
            out.append(
                PatternSignal(
                    pattern="wedge_continuation_down",
                    direction="SELL",
                    confidence=0.62,
                    bar_idx=last_idx,
                    entry=entry,
                    stop=stop,
                    target=target,
                    expected_rr=_rr(entry, stop, target),
                    atr_at_entry=atr_val,
                    metadata={"prior_impulse_atr": round(impulse_atr, 2)}
                    if require_prior_impulse
                    else {},
                )
            )
    return out

CHART_EXTRA_DETECTORS = [
    detect_diamond,
    detect_cup_handle,
    detect_box_breakout,
    detect_wedge_continuation,
]

PRODUCTION_PATTERNS: set[str] = {
    "cup_and_handle",
    "box_breakout_up",
    "box_breakout_down",
    "wedge_continuation_up",
    "wedge_continuation_down",
}

__all__ = [
    "detect_diamond",
    "detect_cup_handle",
    "detect_box_breakout",
    "detect_wedge_continuation",
    "CHART_EXTRA_DETECTORS",
    "PRODUCTION_PATTERNS",
]
