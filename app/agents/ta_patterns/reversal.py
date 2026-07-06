"""Reversal chart patterns."""

from __future__ import annotations

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
class PatternSignal:
    """Detected chart pattern with trade setup parameters."""

    pattern: str
    direction: str
    confidence: float
    bar_idx: int
    entry: float
    stop: float
    target: float
    expected_rr: float
    atr_at_entry: float
    ticker: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post init."""

        self.confidence = max(0.0, min(1.0, self.confidence))

        self.expected_rr = max(0.0, self.expected_rr)

def _atr_at(atr: pd.Series, idx: int) -> float:
    """Safe ATR lookup at bar index."""
    if atr is None or idx < 0 or idx >= len(atr):
        return 0.0
    val = atr.iloc[idx]
    return float(val) if pd.notna(val) else 0.0

def _rr(entry: float, stop: float, target: float) -> float:
    """Risk-Reward ratio. Returns 0 if stop == entry."""
    risk = abs(stop - entry)
    reward = abs(target - entry)
    return reward / risk if risk > 1e-9 else 0.0

_PRIOR_TREND_BARS = 96
_PRIOR_TREND_MIN_RETURN = 0.003
_MAX_WAIT_BARS_REF = 168
_REF_MIN_RR = 0.8

def _prior_trend(
    df: pd.DataFrame, first_idx: int, atr: pd.Series, bars: int = _PRIOR_TREND_BARS
) -> tuple[float, float]:
    """
    Return (ret, trend_atr): pct-change and abs-move-in-ATR over the last
    `bars` bars ending at first_idx. Used to verify there *was* a trend to
    reverse — kills random pattern matches in choppy sideways markets.
    """
    if first_idx <= 0 or atr is None:
        return 0.0, 0.0
    start = max(0, int(first_idx) - bars)
    if first_idx <= start:
        return 0.0, 0.0
    a = _atr_at(atr, int(first_idx))
    if a <= 0:
        return 0.0, 0.0
    try:
        close_now = float(df["close"].iloc[int(first_idx)])
        close_then = float(df["close"].iloc[start])
        if close_then <= 0:
            return 0.0, 0.0
        ret = close_now / close_then - 1.0
        trend_atr = abs(close_now - close_then) / a
        return float(ret), float(trend_atr)
    except (IndexError, KeyError):
        return 0.0, 0.0

def _prior_trend_ok(
    trend_ret: float, trend_atr: float, direction: str, min_return: float = _PRIOR_TREND_MIN_RETURN
) -> bool:
    """direction='UP' for top-reversals (SELL), 'DOWN' for bottom-reversals (BUY)."""
    if direction == "UP":
        return trend_ret >= min_return
    if direction == "DOWN":
        return trend_ret <= -min_return
    return False

def _find_breakout(
    df: pd.DataFrame,
    end_idx: int,
    direction: str,
    entry_level: float,
    atr: pd.Series,
    max_wait_bars: int = _MAX_WAIT_BARS_REF,
) -> int | None:
    """
    Forward-scan from end_idx+1 looking for a close-based breakout of
    `entry_level`. direction in {'LONG','SHORT','BUY','SELL'}.
    Returns the bar index of the first breakout, or None if no breakout in
    the next `max_wait_bars` bars.
    """
    start = int(end_idx) + 1
    end = min(len(df), start + max_wait_bars)
    if start >= len(df):
        return None
    short = direction in ("SHORT", "SELL")
    for i in range(start, end):
        a = _atr_at(atr, i)
        if a <= 0:
            continue
        close = float(df["close"].iloc[i])
        if short and close < entry_level:
            return int(i)
        if (not short) and close > entry_level:
            return int(i)
    return None

def detect_double_top_bottom(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr: pd.Series,
    price_tolerance_atr: float = 0.8,
    min_bars_between: int = 3,
    max_bars_between: int = 168,
    confirm_window: int = 24,
    min_rr: float = 0.7,
) -> list[PatternSignal]:
    """
    Double Top (SELL): two comparable highs with a valley between them.
    Double Bottom (BUY): two comparable lows with a peak between them.

    Neckline = valley low (for double top) or peak high (for double bottom).
    Entry triggered when price closes beyond neckline by > 0.1×ATR.
    """
    if not _READY or not pivots or df is None:
        return []

    signals: list[PatternSignal] = []
    highs = [p for p in pivots if p.kind == "H"]
    lows = [p for p in pivots if p.kind == "L"]

    for i in range(len(highs) - 1):
        h1, h2 = highs[i], highs[i + 1]
        bars_between = h2.idx - h1.idx
        if not (min_bars_between <= bars_between <= max_bars_between):
            continue

        atr_val = _atr_at(atr, h2.idx)
        if atr_val <= 0:
            continue

        if abs(h2.price - h1.price) > price_tolerance_atr * atr_val:
            continue

        valley_lows = [p for p in lows if h1.idx < p.idx < h2.idx]
        if not valley_lows:
            continue
        valley = min(valley_lows, key=lambda p: p.price)

        neckline = valley.price
        top_price = max(h1.price, h2.price)
        pattern_height = top_price - neckline

        if pattern_height < 0.5 * atr_val:
            continue

        current_idx = h2.idx + 1
        if current_idx >= len(df):
            continue

        for confirm_idx in range(current_idx, min(current_idx + confirm_window, len(df))):
            close = float(df["close"].iloc[confirm_idx])
            if close < neckline - 0.1 * atr_val:
                entry = neckline - 0.1 * atr_val
                stop = top_price + 0.3 * atr_val
                target = neckline - pattern_height
                rr = _rr(entry, stop, target)
                if rr < min_rr:
                    continue

                vol_divergence = h1.volume > 0 and h2.volume < h1.volume * 0.85
                conf = 0.60 + (0.15 if vol_divergence else 0.0) + min(0.15, rr / 20)

                signals.append(
                    PatternSignal(
                        pattern="double_top",
                        direction="SELL",
                        confidence=conf,
                        bar_idx=confirm_idx,
                        entry=entry,
                        stop=stop,
                        target=target,
                        expected_rr=rr,
                        atr_at_entry=atr_val,
                        metadata={
                            "h1_idx": h1.idx,
                            "h2_idx": h2.idx,
                            "neckline": neckline,
                            "valley_idx": valley.idx,
                        },
                    )
                )
                break

    for i in range(len(lows) - 1):
        l1, l2 = lows[i], lows[i + 1]
        bars_between = l2.idx - l1.idx
        if not (min_bars_between <= bars_between <= max_bars_between):
            continue

        atr_val = _atr_at(atr, l2.idx)
        if atr_val <= 0:
            continue

        if abs(l2.price - l1.price) > price_tolerance_atr * atr_val:
            continue

        peak_highs = [p for p in highs if l1.idx < p.idx < l2.idx]
        if not peak_highs:
            continue
        peak = max(peak_highs, key=lambda p: p.price)
        neckline = peak.price
        bottom_price = min(l1.price, l2.price)
        pattern_height = neckline - bottom_price

        if pattern_height < 0.5 * atr_val:
            continue

        current_idx = l2.idx + 1
        if current_idx >= len(df):
            continue

        for confirm_idx in range(current_idx, min(current_idx + confirm_window, len(df))):
            close = float(df["close"].iloc[confirm_idx])
            if close > neckline + 0.1 * atr_val:
                entry = neckline + 0.1 * atr_val
                stop = bottom_price - 0.3 * atr_val
                target = neckline + pattern_height
                rr = _rr(entry, stop, target)
                if rr < min_rr:
                    continue

                vol_divergence = l1.volume > 0 and l2.volume > l1.volume * 1.05
                conf = 0.60 + (0.15 if vol_divergence else 0.0) + min(0.15, rr / 20)

                signals.append(
                    PatternSignal(
                        pattern="double_bottom",
                        direction="BUY",
                        confidence=conf,
                        bar_idx=confirm_idx,
                        entry=entry,
                        stop=stop,
                        target=target,
                        expected_rr=rr,
                        atr_at_entry=atr_val,
                        metadata={
                            "l1_idx": l1.idx,
                            "l2_idx": l2.idx,
                            "neckline": neckline,
                            "peak_idx": peak.idx,
                        },
                    )
                )
                break

    logger.debug("detect_double_top_bottom", extra={"found": len(signals)})
    return signals

def detect_triple_top_bottom(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr: pd.Series,
    price_tolerance_atr: float = 0.9,
    min_total_span: int = 8,
    confirm_window: int = 24,
    min_rr: float = 0.7,
) -> list[PatternSignal]:
    """
    Triple Top (SELL) / Triple Bottom (BUY).
    Three roughly equal peaks (or troughs), declining volume on 3rd = stronger signal.
    """
    if not _READY or not pivots or df is None:
        return []

    signals: list[PatternSignal] = []
    highs = [p for p in pivots if p.kind == "H"]
    lows = [p for p in pivots if p.kind == "L"]

    for i in range(len(highs) - 2):
        h1, h2, h3 = highs[i], highs[i + 1], highs[i + 2]
        atr_val = _atr_at(atr, h3.idx)
        if atr_val <= 0:
            continue

        if (
            max(h1.price, h2.price, h3.price) - min(h1.price, h2.price, h3.price)
            > price_tolerance_atr * atr_val
        ):
            continue
        if h3.idx - h1.idx < min_total_span:
            continue

        v1 = [p for p in lows if h1.idx < p.idx < h2.idx]
        v2 = [p for p in lows if h2.idx < p.idx < h3.idx]
        if not v1 or not v2:
            continue

        neckline = min(min(v1, key=lambda p: p.price).price, min(v2, key=lambda p: p.price).price)
        top_avg = (h1.price + h2.price + h3.price) / 3
        height = top_avg - neckline
        if height < 0.5 * atr_val:
            continue

        confirm_idx = h3.idx + 1
        if confirm_idx >= len(df):
            continue
        for ci in range(confirm_idx, min(confirm_idx + confirm_window, len(df))):
            if float(df["close"].iloc[ci]) < neckline - 0.1 * atr_val:
                entry = neckline - 0.1 * atr_val
                stop = top_avg + 0.3 * atr_val
                target = neckline - height
                rr = _rr(entry, stop, target)
                if rr < min_rr:
                    continue

                vol_decay = (h3.volume < h2.volume < h1.volume) if h1.volume > 0 else False
                conf = 0.65 + (0.15 if vol_decay else 0.0) + min(0.10, rr / 20)
                signals.append(
                    PatternSignal(
                        pattern="triple_top",
                        direction="SELL",
                        confidence=conf,
                        bar_idx=ci,
                        entry=entry,
                        stop=stop,
                        target=target,
                        expected_rr=rr,
                        atr_at_entry=atr_val,
                    )
                )
                break

    for i in range(len(lows) - 2):
        l1, l2, l3 = lows[i], lows[i + 1], lows[i + 2]
        atr_val = _atr_at(atr, l3.idx)
        if atr_val <= 0:
            continue
        if (
            max(l1.price, l2.price, l3.price) - min(l1.price, l2.price, l3.price)
            > price_tolerance_atr * atr_val
        ):
            continue
        if l3.idx - l1.idx < min_total_span:
            continue

        p1 = [pv for pv in highs if l1.idx < pv.idx < l2.idx]
        p2 = [pv for pv in highs if l2.idx < pv.idx < l3.idx]
        if not p1 or not p2:
            continue

        neckline = max(max(p1, key=lambda p: p.price).price, max(p2, key=lambda p: p.price).price)
        bottom_avg = (l1.price + l2.price + l3.price) / 3
        height = neckline - bottom_avg
        if height < 0.5 * atr_val:
            continue

        confirm_idx = l3.idx + 1
        for ci in range(confirm_idx, min(confirm_idx + confirm_window, len(df))):
            if ci >= len(df):
                break
            if float(df["close"].iloc[ci]) > neckline + 0.1 * atr_val:
                entry = neckline + 0.1 * atr_val
                stop = bottom_avg - 0.3 * atr_val
                target = neckline + height
                rr = _rr(entry, stop, target)
                if rr < min_rr:
                    continue
                conf = 0.65 + min(0.20, rr / 15)
                signals.append(
                    PatternSignal(
                        pattern="triple_bottom",
                        direction="BUY",
                        confidence=conf,
                        bar_idx=ci,
                        entry=entry,
                        stop=stop,
                        target=target,
                        expected_rr=rr,
                        atr_at_entry=atr_val,
                    )
                )
                break

    return signals

def detect_head_shoulders(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr: pd.Series,
    shoulder_sym_atr: float = 1.5,
    confirm_window: int = 24,
    min_rr: float = 0.7,
    min_head_height_atr: float = 0.5,
) -> list[PatternSignal]:
    """
    Head & Shoulders (SELL) + Inverse H&S (BUY).

    H&S: left_shoulder HIGH → valley → head HIGH (> shoulders) → valley → right_shoulder HIGH
    Both valleys form the neckline. Shoulder heights within shoulder_sym_atr × ATR.
    """
    if not _READY or not pivots or df is None:
        return []

    signals: list[PatternSignal] = []
    highs = [p for p in pivots if p.kind == "H"]
    lows = [p for p in pivots if p.kind == "L"]

    for i in range(len(highs) - 2):
        ls, head, rs = highs[i], highs[i + 1], highs[i + 2]

        if not (head.price > ls.price and head.price > rs.price):
            continue

        atr_val = _atr_at(atr, rs.idx)
        if atr_val <= 0:
            continue

        if abs(ls.price - rs.price) > shoulder_sym_atr * atr_val:
            continue

        v_left = [p for p in lows if ls.idx < p.idx < head.idx]
        v_right = [p for p in lows if head.idx < p.idx < rs.idx]
        if not v_left or not v_right:
            continue

        nl_left = min(v_left, key=lambda p: p.price).price
        nl_right = min(v_right, key=lambda p: p.price).price
        neckline = (nl_left + nl_right) / 2

        head_height = head.price - neckline
        if head_height < min_head_height_atr * atr_val:
            continue

        confirm_idx = rs.idx + 1
        for ci in range(confirm_idx, min(confirm_idx + confirm_window, len(df))):
            if ci >= len(df):
                break
            close = float(df["close"].iloc[ci])
            if close < neckline - 0.1 * atr_val:
                entry = neckline - 0.1 * atr_val
                stop = rs.price + 0.3 * atr_val
                target = neckline - head_height
                rr = _rr(entry, stop, target)
                if rr < min_rr:
                    continue
                conf = min(0.85, 0.65 + head_height / (4 * atr_val) * 0.20)
                signals.append(
                    PatternSignal(
                        pattern="head_shoulders",
                        direction="SELL",
                        confidence=conf,
                        bar_idx=ci,
                        entry=entry,
                        stop=stop,
                        target=target,
                        expected_rr=rr,
                        atr_at_entry=atr_val,
                        metadata={
                            "ls_idx": ls.idx,
                            "head_idx": head.idx,
                            "rs_idx": rs.idx,
                            "neckline": neckline,
                        },
                    )
                )
                break

    for i in range(len(lows) - 2):
        ls, head, rs = lows[i], lows[i + 1], lows[i + 2]

        if not (head.price < ls.price and head.price < rs.price):
            continue

        atr_val = _atr_at(atr, rs.idx)
        if atr_val <= 0:
            continue

        if abs(ls.price - rs.price) > shoulder_sym_atr * atr_val:
            continue

        p_left = [p for p in highs if ls.idx < p.idx < head.idx]
        p_right = [p for p in highs if head.idx < p.idx < rs.idx]
        if not p_left or not p_right:
            continue

        nl_left = max(p_left, key=lambda p: p.price).price
        nl_right = max(p_right, key=lambda p: p.price).price
        neckline = (nl_left + nl_right) / 2

        head_depth = neckline - head.price
        if head_depth < min_head_height_atr * atr_val:
            continue

        confirm_idx = rs.idx + 1
        for ci in range(confirm_idx, min(confirm_idx + confirm_window, len(df))):
            if ci >= len(df):
                break
            close = float(df["close"].iloc[ci])
            if close > neckline + 0.1 * atr_val:
                entry = neckline + 0.1 * atr_val
                stop = rs.price - 0.3 * atr_val
                target = neckline + head_depth
                rr = _rr(entry, stop, target)
                if rr < min_rr:
                    continue
                conf = min(0.85, 0.65 + head_depth / (4 * atr_val) * 0.20)
                signals.append(
                    PatternSignal(
                        pattern="inv_head_shoulders",
                        direction="BUY",
                        confidence=conf,
                        bar_idx=ci,
                        entry=entry,
                        stop=stop,
                        target=target,
                        expected_rr=rr,
                        atr_at_entry=atr_val,
                        metadata={"neckline": neckline},
                    )
                )
                break

    logger.debug("detect_head_shoulders", extra={"found": len(signals)})
    return signals

def detect_wedge_reversal(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr: pd.Series,
    pivot_window: int = 6,
    min_compression: float = 0.02,
    min_range_atr: float = 0.5,
    slope_atr_mult: float = 0.005,
    stop_buffer_atr: float = 0.20,
    max_wait_bars: int = _MAX_WAIT_BARS_REF,
) -> list[PatternSignal]:
    """
    Rising Wedge (SELL reversal) + Falling Wedge (BUY reversal).

    Reference: notebook `detect_wedge_reversal_signal()`.
    Iterates through the pivot stream — at every window of `pivot_window`
    pivots, fits trendlines through highs and lows. Requires:
      - both upper and lower trendlines slope same direction (real wedge)
      - >=2% compression of range from first 3 to last 3 pivots
      - prior_trend matches the direction expected by the reversal
      - close-based breakout of the OPPOSITE side within max_wait_bars
    Used in place of the previous look-at-last-bar-only logic that produced
    zero signals across all 5 tickers on H1.
    """
    if not _READY or not pivots or df is None:
        return []

    signals: list[PatternSignal] = []
    for j in range(pivot_window - 1, len(pivots)):
        w = pivots[j - pivot_window + 1 : j + 1]
        highs = [p for p in w if p.kind == "H"]
        lows = [p for p in w if p.kind == "L"]
        if len(highs) < 2 or len(lows) < 2:
            continue
        end_idx = w[-1].idx
        start_idx = w[0].idx
        atr_end = _atr_at(atr, end_idx)
        if atr_end <= 0:
            continue

        xh = np.array([p.idx for p in highs], dtype=float)
        yh = np.array([p.price for p in highs], dtype=float)
        xl = np.array([p.idx for p in lows], dtype=float)
        yl = np.array([p.price for p in lows], dtype=float)
        try:
            h_slope = float(np.polyfit(xh, yh, 1)[0])
            l_slope = float(np.polyfit(xl, yl, 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.isfinite(h_slope) or not np.isfinite(l_slope):
            continue

        pri = np.array([p.price for p in w])
        first_range = pri[:3].max() - pri[:3].min()
        last_range = pri[-3:].max() - pri[-3:].min()
        full_range = pri.max() - pri.min()
        if first_range <= 0:
            continue
        compression = 1.0 - last_range / first_range
        range_atr = full_range / atr_end
        if compression < min_compression or range_atr < min_range_atr:
            continue

        resistance = max(p.price for p in highs)
        support = min(p.price for p in lows)
        trend_ret, _ = _prior_trend(df, start_idx, atr)
        slope_thr = slope_atr_mult * atr_end

        if _prior_trend_ok(trend_ret, 0.0, "UP") and h_slope > slope_thr and l_slope > slope_thr:
            bi = _find_breakout(df, end_idx, "SHORT", support, atr, max_wait_bars)
            if bi is None:
                continue
            entry = support
            stop = resistance + stop_buffer_atr * atr_end
            target = entry - (stop - entry)
            rr = _rr(entry, stop, target)
            if rr < _REF_MIN_RR:
                continue
            signals.append(
                PatternSignal(
                    pattern="rising_wedge",
                    direction="SELL",
                    confidence=0.65,
                    bar_idx=bi,
                    entry=entry,
                    stop=stop,
                    target=target,
                    expected_rr=rr,
                    atr_at_entry=_atr_at(atr, bi),
                    metadata={
                        "compression": round(compression, 3),
                        "range_atr": round(range_atr, 2),
                        "resistance": resistance,
                        "support": support,
                    },
                )
            )

        elif (
            _prior_trend_ok(trend_ret, 0.0, "DOWN")
            and h_slope < -slope_thr
            and l_slope < -slope_thr
        ):
            bi = _find_breakout(df, end_idx, "LONG", resistance, atr, max_wait_bars)
            if bi is None:
                continue
            entry = resistance
            stop = support - stop_buffer_atr * atr_end
            target = entry + (entry - stop)
            rr = _rr(entry, stop, target)
            if rr < _REF_MIN_RR:
                continue
            signals.append(
                PatternSignal(
                    pattern="falling_wedge",
                    direction="BUY",
                    confidence=0.65,
                    bar_idx=bi,
                    entry=entry,
                    stop=stop,
                    target=target,
                    expected_rr=rr,
                    atr_at_entry=_atr_at(atr, bi),
                    metadata={
                        "compression": round(compression, 3),
                        "range_atr": round(range_atr, 2),
                        "resistance": resistance,
                        "support": support,
                    },
                )
            )

    logger.debug("detect_wedge_reversal", extra={"found": len(signals)})
    return signals

def detect_megaphone(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr: pd.Series,
    pivot_window: int = 6,
    min_expansion: float = 0.05,
    min_range_atr: float = 0.5,
    stop_buffer_atr: float = 0.20,
    max_wait_bars: int = _MAX_WAIT_BARS_REF,
) -> list[PatternSignal]:
    """
    Megaphone / Broadening Formation (reversal). Reference: notebook
    `detect_megaphone_signal()`.

    Geometry: among the last `pivot_window` pivots — high-slope > 0 (rising
    upper line) and low-slope < 0 (falling lower line); last 3-pivot range
    is >5% wider than first 3 (expansion); full pattern range >0.5×ATR.
    After confirming a prior up/down trend, wait up to `max_wait_bars` for
    a close-based break of the OPPOSITE boundary as the entry trigger.
    """
    if not _READY or not pivots or df is None:
        return []

    signals: list[PatternSignal] = []
    for j in range(pivot_window - 1, len(pivots)):
        w = pivots[j - pivot_window + 1 : j + 1]
        highs = [p for p in w if p.kind == "H"]
        lows = [p for p in w if p.kind == "L"]
        if len(highs) < 2 or len(lows) < 2:
            continue
        end_idx = w[-1].idx
        start_idx = w[0].idx
        atr_end = _atr_at(atr, end_idx)
        if atr_end <= 0:
            continue

        xh = np.array([p.idx for p in highs], dtype=float)
        yh = np.array([p.price for p in highs], dtype=float)
        xl = np.array([p.idx for p in lows], dtype=float)
        yl = np.array([p.price for p in lows], dtype=float)
        try:
            h_slope = float(np.polyfit(xh, yh, 1)[0])
            l_slope = float(np.polyfit(xl, yl, 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not (np.isfinite(h_slope) and np.isfinite(l_slope)):
            continue

        pri = np.array([p.price for p in w])
        first_range = pri[:3].max() - pri[:3].min()
        last_range = pri[-3:].max() - pri[-3:].min()
        full_range = pri.max() - pri.min()
        if first_range <= 0:
            continue
        expansion = last_range / first_range - 1.0
        range_atr = full_range / atr_end
        if expansion < min_expansion or range_atr < min_range_atr:
            continue

        if not (h_slope > 0 and l_slope < 0):
            continue

        resistance = max(p.price for p in highs)
        support = min(p.price for p in lows)
        trend_ret, _ = _prior_trend(df, start_idx, atr)

        if _prior_trend_ok(trend_ret, 0.0, "UP"):
            entry = support
            bi = _find_breakout(df, end_idx, "SHORT", entry, atr, max_wait_bars)
            if bi is None:
                continue
            stop = resistance + stop_buffer_atr * atr_end
            target = entry - (stop - entry)
            rr = _rr(entry, stop, target)
            if rr < _REF_MIN_RR:
                continue
            signals.append(
                PatternSignal(
                    pattern="megaphone_top",
                    direction="SELL",
                    confidence=0.62,
                    bar_idx=bi,
                    entry=entry,
                    stop=stop,
                    target=target,
                    expected_rr=rr,
                    atr_at_entry=_atr_at(atr, bi),
                    metadata={"expansion": round(expansion, 3), "range_atr": round(range_atr, 2)},
                )
            )
        elif _prior_trend_ok(trend_ret, 0.0, "DOWN"):
            entry = resistance
            bi = _find_breakout(df, end_idx, "LONG", entry, atr, max_wait_bars)
            if bi is None:
                continue
            stop = support - stop_buffer_atr * atr_end
            target = entry + (entry - stop)
            rr = _rr(entry, stop, target)
            if rr < _REF_MIN_RR:
                continue
            signals.append(
                PatternSignal(
                    pattern="megaphone_bottom",
                    direction="BUY",
                    confidence=0.62,
                    bar_idx=bi,
                    entry=entry,
                    stop=stop,
                    target=target,
                    expected_rr=rr,
                    atr_at_entry=_atr_at(atr, bi),
                    metadata={"expansion": round(expansion, 3), "range_atr": round(range_atr, 2)},
                )
            )

    logger.debug("detect_megaphone", extra={"found": len(signals)})
    return signals

def detect_rounding(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr: pd.Series,
    pivot_window: int = 7,
    min_range_atr: float = 0.7,
    stop_buffer_atr: float = 0.20,
    max_wait_bars: int = _MAX_WAIT_BARS_REF,
    window: int = 30,
) -> list[PatternSignal]:
    """
    Rounding Top (SELL) / Rounding Bottom (BUY). Reference: notebook
    `detect_rounding_signal()` — polyfit on the last `pivot_window` PIVOT
    PRICES (not close prices). Curvature sign + prior trend direction
    decides the pattern; close-based breakout of the opposite side is the
    entry trigger.

    The previous close-price-fit only fired on the most recent 30 bars,
    yielding 0 signals across all 5 tickers. The pivot-based approach
    matches the reference notebook and gives many more usable signals.

    `window` is kept as a legacy keyword so old callers do not break, but
    is unused in the new logic.
    """
    if not _READY or not pivots or df is None:
        return []

    del window

    signals: list[PatternSignal] = []
    for j in range(pivot_window - 1, len(pivots)):
        w = pivots[j - pivot_window + 1 : j + 1]
        if len(w) < 6:
            continue
        end_idx = w[-1].idx
        start_idx = w[0].idx
        atr_end = _atr_at(atr, end_idx)
        if atr_end <= 0:
            continue

        prices = np.array([p.price for p in w])
        x = np.arange(len(prices))
        try:
            coef = np.polyfit(x, prices, 2)
        except (np.linalg.LinAlgError, ValueError):
            continue
        curvature = float(coef[0])
        if not np.isfinite(curvature):
            continue

        highs_p = [p for p in w if p.kind == "H"]
        lows_p = [p for p in w if p.kind == "L"]
        if not highs_p or not lows_p:
            continue
        resistance = max(p.price for p in highs_p)
        support = min(p.price for p in lows_p)
        full_range = resistance - support
        range_atr = full_range / atr_end
        if range_atr < min_range_atr:
            continue

        trend_ret, _ = _prior_trend(df, start_idx, atr)

        if _prior_trend_ok(trend_ret, 0.0, "UP") and curvature < 0:
            entry = support
            bi = _find_breakout(df, end_idx, "SHORT", entry, atr, max_wait_bars)
            if bi is None:
                continue
            stop = resistance + stop_buffer_atr * atr_end
            target = entry - full_range
            rr = _rr(entry, stop, target)
            if rr < _REF_MIN_RR:
                continue
            signals.append(
                PatternSignal(
                    pattern="rounding_top",
                    direction="SELL",
                    confidence=0.62,
                    bar_idx=bi,
                    entry=entry,
                    stop=stop,
                    target=target,
                    expected_rr=rr,
                    atr_at_entry=_atr_at(atr, bi),
                    metadata={"curvature": round(curvature, 6), "range_atr": round(range_atr, 2)},
                )
            )

        elif _prior_trend_ok(trend_ret, 0.0, "DOWN") and curvature > 0:
            entry = resistance
            bi = _find_breakout(df, end_idx, "LONG", entry, atr, max_wait_bars)
            if bi is None:
                continue
            stop = support - stop_buffer_atr * atr_end
            target = entry + full_range
            rr = _rr(entry, stop, target)
            if rr < _REF_MIN_RR:
                continue
            signals.append(
                PatternSignal(
                    pattern="rounding_bottom",
                    direction="BUY",
                    confidence=0.62,
                    bar_idx=bi,
                    entry=entry,
                    stop=stop,
                    target=target,
                    expected_rr=rr,
                    atr_at_entry=_atr_at(atr, bi),
                    metadata={"curvature": round(curvature, 6), "range_atr": round(range_atr, 2)},
                )
            )

    logger.debug("detect_rounding", extra={"found": len(signals)})
    return signals

PRODUCTION_PATTERNS: set[str] = {
    "head_shoulders",
    "double_top",
    "megaphone_top",
    "triple_top",
    "rounding_top",
    "rising_wedge",
    "triple_bottom",
    "double_bottom",
}
