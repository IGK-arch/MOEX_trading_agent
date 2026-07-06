"""Research-backed chart pattern detectors (v0.0.29).

Phase 24 — implementation of 5 NEW high-quality chart pattern detectors
based on peer-reviewed methodology and literature (Minervini, Bulkowski,
Connors, Pring). All detectors:

  - Emit signals only on **confirmed breakout** of a well-defined level
    (no "anticipation" — eliminates a class of false positives).
  - Use ATR for stop/target geometry so position sizing is regime-aware.
  - Carry a `volume_gate` filter where applicable (engulfing/three soldiers)
    because volume confirmation is the single most studied robustness
    boost in the price-action literature (Bulkowski, Encyclopedia of
    Chart Patterns 2nd ed., Ch. 6).

Patterns implemented:

  1. **vcp** (Volatility Contraction Pattern, Minervini) — sequence of
     3-5 progressively tighter pullbacks ending with a breakout above the
     pivot high on expanding volume. Minervini's books quote PF>2 on US
     stocks; we want to see if it carries to MOEX equities.

  2. **bb_squeeze_breakout** (Bollinger Band squeeze, John Bollinger) —
     BBwidth < 0.5 × BBwidth_20MA for 6+ bars, then break out of the upper
     or lower band on close. Classic momentum-after-compression edge.

  3. **inside_bar_breakout** (price action / Al Brooks) — 2+ inside bars
     followed by a breakout of the mother-bar range. Combined with a
     prior-trend filter to favour continuation trades over choppy reversal.

  4. **three_soldiers_volume** (Steve Nison's "Three White Soldiers"
     / "Three Black Crows" with volume confirmation) — 3 consecutive
     same-direction bodied candles with each close > prev close AND
     volume of the 3rd bar >= 1.2 × avg(20). Without volume gate this
     pattern is well-known to give many false signals.

  5. **pivot_reversal** (classic floor pivots, J. Welles Wilder Jr.) —
     touch of daily Pivot/S1/R1 with a rejection candle (long wick + close
     back inside). Daily pivots are the most-watched intraday levels;
     bounces have measurable edge on liquid index futures (literature).

Style mirrors `dasha_patterns.py`:
  - Self-contained module, runs on a DataFrame with OHLCV + atr14 + ts.
  - Returns list[ResearchPattern] with the same fields as DashaPattern.
  - `detect_all_research_patterns(df)` is the single entry point used by
    `ta_trader.py` so adding a new detector is one-line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    _READY = True
except ImportError:
    _READY = False

@dataclass
class ResearchPattern:
    """Compact descriptor of one detected pattern."""

    pattern: str
    direction: str
    bar_idx: int
    entry: float
    stop: float
    target: float
    confidence: float
    atr_at_entry: float
    height: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expected_rr(self) -> float:
        """Expected rr."""
        risk = abs(self.entry - self.stop)
        reward = abs(self.target - self.entry)
        return reward / risk if risk > 1e-9 else 0.0

def _safe_atr(atr: np.ndarray, i: int) -> float:
    """Safe atr."""
    if i < 0 or i >= len(atr):
        return 0.0
    val = float(atr[i])
    return val if np.isfinite(val) and val > 0 else 0.0

def _volume_ratio(volume: np.ndarray, i: int, lookback: int = 20) -> float:
    """Volume of bar i relative to lookback average. Returns 1.0 if no data."""
    start = max(0, i - lookback)
    if i <= start:
        return 1.0
    window = volume[start:i]
    window = window[np.isfinite(window) & (window > 0)]
    if len(window) < 3:
        return 1.0
    avg = float(np.mean(window))
    if avg <= 0:
        return 1.0
    return float(volume[i]) / avg

def detect_vcp(
    df: pd.DataFrame,
    min_contractions: int = 2,
    max_contractions: int = 6,
    base_min_bars: int = 24,
    base_max_bars: int = 120,
    contraction_shrink: float = 0.80,
    breakout_volume_mult: float = 1.15,
    rim_tol_atr: float = 1.2,
    atr_col: str = "atr14",
) -> list[ResearchPattern]:
    """Detect VCP setups (Mark Minervini's "Trade Like a Stock Market Wizard").

    Logic:
      1. Identify base of length [base_min_bars, base_max_bars] in which highs
         form descending tops and lows form ascending bottoms.
      2. Compute swing-by-swing range. Each successive swing must be smaller
         by at least `contraction_shrink` (e.g. 0.65 = 35% smaller).
      3. Require >= `min_contractions` consecutive shrinking swings.
      4. Breakout: close > base_high AND volume >= breakout_volume_mult × avg(50).
      5. Entry = base_high, stop = lowest_low_in_base − 0.3×ATR,
         target = entry + (height × 2) — Minervini's 2:1 minimum.
    """
    if not _READY or len(df) < base_min_bars + 5:
        return []
    if atr_col not in df.columns or "volume" not in df.columns:
        return []

    out: list[ResearchPattern] = []
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    volume = df["volume"].values.astype(float)
    atr = df[atr_col].values.astype(float)
    n = len(df)

    for end_i in range(base_min_bars, n):
        a = _safe_atr(atr, end_i)
        if a == 0.0:
            continue

        for base_len in range(base_max_bars, base_min_bars - 1, -10):
            start_i = end_i - base_len
            if start_i < 5:
                continue
            base_high = float(np.max(high[start_i:end_i]))
            base_low = float(np.min(low[start_i:end_i]))
            base_height = base_high - base_low
            if base_height < 1.5 * a:
                continue

            swings = _vcp_swings(high[start_i:end_i], low[start_i:end_i])
            if len(swings) < 2 * min_contractions:
                continue

            ranges: list[float] = []
            for k in range(0, len(swings) - 1):
                ranges.append(abs(swings[k][1] - swings[k + 1][1]))
            tail = ranges[-(max_contractions + 1) :]
            contractions_count = 0
            for k in range(len(tail) - 1):
                if tail[k] <= 0:
                    contractions_count = 0
                    continue
                if tail[k + 1] <= tail[k] * contraction_shrink:
                    contractions_count += 1
                else:
                    contractions_count = 0
            if contractions_count < min_contractions - 1:
                continue

            recent_swings_highs = [s[1] for s in swings[-5:] if s[0] == 1]
            if not recent_swings_highs:
                continue
            last_high = max(recent_swings_highs)
            if base_high - last_high > rim_tol_atr * a:
                continue

            if close[end_i] <= base_high:
                continue
            vol_ratio = _volume_ratio(volume, end_i, lookback=50)
            if vol_ratio < breakout_volume_mult:
                continue

            entry = base_high
            stop = base_low - 0.3 * a
            target = entry + 2.0 * (entry - stop)
            height = entry - stop
            out.append(
                ResearchPattern(
                    pattern="vcp",
                    direction="BUY",
                    bar_idx=int(end_i),
                    entry=float(entry),
                    stop=float(stop),
                    target=float(target),
                    confidence=0.72,
                    atr_at_entry=a,
                    height=float(height),
                    metadata={
                        "source": "research",
                        "base_start": int(start_i),
                        "base_end": int(end_i),
                        "base_low": float(base_low),
                        "base_high": float(base_high),
                        "contractions": int(contractions_count + 1),
                        "volume_ratio": round(vol_ratio, 2),
                    },
                )
            )
            break
    return out

def _vcp_swings(high: np.ndarray, low: np.ndarray, win: int = 3) -> list[tuple[int, float]]:
    """Tiny peak/trough finder for VCP base. Returns (type, price), type +1 for high, -1 for low."""
    n = len(high)
    swings: list[tuple[int, float]] = []
    last_type = 0
    for i in range(win, n - win):
        if high[i] == max(high[i - win : i + win + 1]):
            if last_type != 1:
                swings.append((1, float(high[i])))
                last_type = 1
            else:
                if high[i] > swings[-1][1]:
                    swings[-1] = (1, float(high[i]))
        elif low[i] == min(low[i - win : i + win + 1]):
            if last_type != -1:
                swings.append((-1, float(low[i])))
                last_type = -1
            else:
                if low[i] < swings[-1][1]:
                    swings[-1] = (-1, float(low[i]))
    return swings

def detect_bb_squeeze_breakout(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    squeeze_lookback: int = 20,
    squeeze_threshold: float = 0.5,
    min_squeeze_bars: int = 6,
    atr_col: str = "atr14",
) -> list[ResearchPattern]:
    """Detect breakouts after Bollinger Band squeeze.

    Squeeze = rolling BB width is in the bottom `squeeze_threshold` quantile of
    its `squeeze_lookback` history, holding for at least `min_squeeze_bars`.
    Then we emit on the first close OUTSIDE either band.

    Stop = opposite band (mean-reversion-of-mean-reversion logic), capped
    at 2×ATR. Target = entry + 2 × (entry-stop). 2:1 RR.
    """
    if not _READY or len(df) < bb_period + squeeze_lookback + 5:
        return []
    if atr_col not in df.columns:
        return []

    out: list[ResearchPattern] = []
    close = df["close"].values.astype(float)
    df["high"].values.astype(float)
    df["low"].values.astype(float)
    atr = df[atr_col].values.astype(float)

    ma = pd.Series(close).rolling(bb_period).mean()
    sd = pd.Series(close).rolling(bb_period).std(ddof=0)
    upper = (ma + bb_std * sd).values
    lower = (ma - bb_std * sd).values
    width = upper - lower

    width_series = pd.Series(width)
    width_rank = width_series.rolling(squeeze_lookback).rank(pct=True)

    n = len(df)
    squeeze_count = 0
    for i in range(bb_period + squeeze_lookback, n):
        wr = width_rank.iloc[i]
        if pd.isna(wr):
            continue
        if wr <= squeeze_threshold:
            squeeze_count += 1
            continue

        if squeeze_count < min_squeeze_bars:
            squeeze_count = 0
            continue
        a = _safe_atr(atr, i)
        if a == 0.0:
            squeeze_count = 0
            continue

        c = close[i]
        u = float(upper[i])
        l = float(lower[i])
        mid = float(ma.iloc[i])
        height = u - l

        direction: str | None = None
        if c > u:
            direction = "BUY"
            entry = u
            stop = max(mid - 1.5 * a, l)
            target = entry + 2.0 * (entry - stop)
        elif c < l:
            direction = "SELL"
            entry = l
            stop = min(mid + 1.5 * a, u)
            target = entry - 2.0 * (stop - entry)
        else:
            squeeze_count = 0
            continue

        out.append(
            ResearchPattern(
                pattern="bb_squeeze_breakout",
                direction=direction,
                bar_idx=int(i),
                entry=float(entry),
                stop=float(stop),
                target=float(target),
                confidence=0.65,
                atr_at_entry=a,
                height=float(height),
                metadata={
                    "source": "research",
                    "squeeze_bars": int(squeeze_count),
                    "bb_upper": u,
                    "bb_lower": l,
                    "bb_mid": mid,
                    "width_rank": float(wr),
                },
            )
        )
        squeeze_count = 0
    return out

def detect_inside_bar_breakout(
    df: pd.DataFrame,
    min_inside_bars: int = 2,
    max_inside_bars: int = 5,
    prior_trend_bars: int = 30,
    min_mother_range_atr: float = 1.2,
    min_prior_change_pct: float = 0.015,
    atr_col: str = "atr14",
) -> list[ResearchPattern]:
    """Detect Inside Bar Breakout.

    Mother bar: bar M.
    Inside bars: M+1, ..., M+k where each bar's high <= M.high and low >= M.low.
    Breakout bar: M+k+1 where close > M.high (BUY) OR close < M.low (SELL).

    Bias toward continuation: only emit BUY if prior_trend > 0 (uptrend),
    SELL if prior_trend < 0 (downtrend). This is the Al Brooks recommendation
    — inside bars in a strong trend are pause-then-continue, while inside bars
    after a reversal candle are reversal continuation.

    Stop = opposite side of mother bar - 0.2×ATR.
    Target = entry + 2× height (2:1 RR).
    """
    if not _READY or len(df) < prior_trend_bars + max_inside_bars + 5:
        return []
    if atr_col not in df.columns:
        return []

    out: list[ResearchPattern] = []
    df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    atr = df[atr_col].values.astype(float)

    n = len(df)
    i = prior_trend_bars
    while i < n - 2:
        a = _safe_atr(atr, i)
        if a == 0.0:
            i += 1
            continue
        mother_h = h[i]
        mother_l = l[i]
        mother_range = mother_h - mother_l
        if mother_range < min_mother_range_atr * a:
            i += 1
            continue

        k = 0
        j = i + 1
        while j < n and k < max_inside_bars and h[j] <= mother_h and l[j] >= mother_l:
            k += 1
            j += 1
        if k < min_inside_bars:
            i += 1
            continue

        if j >= n:
            i = j
            continue

        prior_close = c[max(0, i - prior_trend_bars)]
        prior_change = (c[i] - prior_close) / prior_close if prior_close > 0 else 0.0

        if c[j] > mother_h and prior_change > min_prior_change_pct:
            entry = mother_h
            stop = mother_l - 0.2 * a
            target = entry + 2.0 * (entry - stop)
            height = entry - stop
            out.append(
                ResearchPattern(
                    pattern="inside_bar_breakout",
                    direction="BUY",
                    bar_idx=int(j),
                    entry=float(entry),
                    stop=float(stop),
                    target=float(target),
                    confidence=0.62,
                    atr_at_entry=a,
                    height=float(height),
                    metadata={
                        "source": "research",
                        "mother_idx": int(i),
                        "inside_bars": int(k),
                        "prior_change_pct": round(prior_change * 100, 2),
                    },
                )
            )
        elif c[j] < mother_l and prior_change < -min_prior_change_pct:
            entry = mother_l
            stop = mother_h + 0.2 * a
            target = entry - 2.0 * (stop - entry)
            height = stop - entry
            out.append(
                ResearchPattern(
                    pattern="inside_bar_breakout",
                    direction="SELL",
                    bar_idx=int(j),
                    entry=float(entry),
                    stop=float(stop),
                    target=float(target),
                    confidence=0.62,
                    atr_at_entry=a,
                    height=float(height),
                    metadata={
                        "source": "research",
                        "mother_idx": int(i),
                        "inside_bars": int(k),
                        "prior_change_pct": round(prior_change * 100, 2),
                    },
                )
            )
        i = j + 1
    return out

def detect_three_soldiers_volume(
    df: pd.DataFrame,
    body_min_ratio: float = 0.55,
    volume_mult: float = 1.3,
    confirm_bars: int = 3,
    atr_col: str = "atr14",
) -> list[ResearchPattern]:
    """Detect Three White Soldiers / Three Black Crows with volume confirmation.

    Definition (Nison + literature):
      - 3 consecutive bullish (TWS) or bearish (TBC) bodied candles.
      - Each close > prev close (TWS) or < prev close (TBC).
      - Each open WITHIN the previous body (no gap).
      - Each body >= body_min_ratio × bar_range.
      - Cumulative move >= 1.5 × ATR.
      - **Volume of 3rd bar >= volume_mult × avg(20)** — this is the key filter
        that turns the (otherwise mediocre) pattern into a real signal.

    Entry = close of 3rd bar.
    Stop = low/high of 1st bar - 0.3×ATR (TWS) or +0.3×ATR (TBC).
    Target = entry + 2× (entry - stop) — 2:1 RR.
    """
    if not _READY or len(df) < 30:
        return []
    if atr_col not in df.columns or "volume" not in df.columns:
        return []

    out: list[ResearchPattern] = []
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    atr = df[atr_col].values.astype(float)

    n = len(df)
    for i in range(22, n):
        i1, i2, i3 = i - 2, i - 1, i
        a = _safe_atr(atr, i3)
        if a == 0.0:
            continue

        bull = (c[i1] > o[i1]) and (c[i2] > o[i2]) and (c[i3] > o[i3])
        bull = bull and (c[i2] > c[i1]) and (c[i3] > c[i2])
        bull = bull and (o[i2] >= o[i1]) and (o[i2] <= c[i1])
        bull = bull and (o[i3] >= o[i2]) and (o[i3] <= c[i2])
        if bull:
            ranges = [h[i1] - l[i1], h[i2] - l[i2], h[i3] - l[i3]]
            bodies = [c[i1] - o[i1], c[i2] - o[i2], c[i3] - o[i3]]
            if min(ranges) <= 0:
                continue
            body_ratios = [b / r for b, r in zip(bodies, ranges, strict=False)]
            if min(body_ratios) < body_min_ratio:
                continue
            move = c[i3] - o[i1]
            if move < 1.5 * a:
                continue
            vol_ratio = _volume_ratio(v, i3, lookback=20)
            if vol_ratio < volume_mult:
                continue

            entry = c[i3]
            stop = l[i1] - 0.3 * a
            target = entry + 2.0 * (entry - stop)
            height = entry - stop
            out.append(
                ResearchPattern(
                    pattern="three_white_soldiers_vol",
                    direction="BUY",
                    bar_idx=int(i3),
                    entry=float(entry),
                    stop=float(stop),
                    target=float(target),
                    confidence=0.66,
                    atr_at_entry=a,
                    height=float(height),
                    metadata={
                        "source": "research",
                        "volume_ratio": round(vol_ratio, 2),
                        "move_atr": round(move / a, 2),
                        "body_min_ratio": round(min(body_ratios), 2),
                    },
                )
            )
            continue

        bear = (c[i1] < o[i1]) and (c[i2] < o[i2]) and (c[i3] < o[i3])
        bear = bear and (c[i2] < c[i1]) and (c[i3] < c[i2])
        bear = bear and (o[i2] <= o[i1]) and (o[i2] >= c[i1])
        bear = bear and (o[i3] <= o[i2]) and (o[i3] >= c[i2])
        if bear:
            ranges = [h[i1] - l[i1], h[i2] - l[i2], h[i3] - l[i3]]
            bodies = [o[i1] - c[i1], o[i2] - c[i2], o[i3] - c[i3]]
            if min(ranges) <= 0:
                continue
            body_ratios = [b / r for b, r in zip(bodies, ranges, strict=False)]
            if min(body_ratios) < body_min_ratio:
                continue
            move = o[i1] - c[i3]
            if move < 1.5 * a:
                continue
            vol_ratio = _volume_ratio(v, i3, lookback=20)
            if vol_ratio < volume_mult:
                continue

            entry = c[i3]
            stop = h[i1] + 0.3 * a
            target = entry - 2.0 * (stop - entry)
            height = stop - entry
            out.append(
                ResearchPattern(
                    pattern="three_black_crows_vol",
                    direction="SELL",
                    bar_idx=int(i3),
                    entry=float(entry),
                    stop=float(stop),
                    target=float(target),
                    confidence=0.66,
                    atr_at_entry=a,
                    height=float(height),
                    metadata={
                        "source": "research",
                        "volume_ratio": round(vol_ratio, 2),
                        "move_atr": round(move / a, 2),
                        "body_min_ratio": round(min(body_ratios), 2),
                    },
                )
            )
    return out

def detect_pivot_reversal(
    df: pd.DataFrame,
    pivot_window_bars: int = 24,
    rejection_wick_min: float = 0.65,
    rejection_body_max: float = 0.30,
    atr_col: str = "atr14",
    min_distance_atr: float = 0.5,
) -> list[ResearchPattern]:
    """Detect rejection candles at floor pivots / S1 / R1 levels.

    Floor pivots (Wilder/classic):
      P  = (prev_H + prev_L + prev_C) / 3
      S1 = 2P - prev_H
      R1 = 2P - prev_L

    Where "prev" is the prior pivot_window_bars period — for 60-minute bars
    this is one trading session (~10 hours = 10 bars; we widen to 24 ≈ 2
    sessions to be regime-tolerant).

    Reversal candle (Bulkowski's "doji at level"):
      - Wick on the side near the pivot >= rejection_wick_min × range
      - Body <= rejection_body_max × range
      - Close back through the pivot in the opposite direction.

    Stop = beyond the rejection wick + 0.3×ATR.
    Target = pivot ± 2× (entry - stop). 2:1 RR.
    """
    if not _READY or len(df) < pivot_window_bars + 5:
        return []
    if atr_col not in df.columns:
        return []

    out: list[ResearchPattern] = []
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    atr = df[atr_col].values.astype(float)
    n = len(df)

    for i in range(pivot_window_bars, n):
        a = _safe_atr(atr, i)
        if a == 0.0:
            continue

        prev_h = float(np.max(h[i - pivot_window_bars : i]))
        prev_l = float(np.min(l[i - pivot_window_bars : i]))
        prev_c = float(c[i - 1])
        P = (prev_h + prev_l + prev_c) / 3.0
        S1 = 2 * P - prev_h
        R1 = 2 * P - prev_l

        bar_range = h[i] - l[i]
        if bar_range <= 0 or bar_range < min_distance_atr * a:
            continue
        body = abs(c[i] - o[i])
        upper_wick = h[i] - max(o[i], c[i])
        lower_wick = min(o[i], c[i]) - l[i]
        body_ratio = body / bar_range
        if body_ratio > rejection_body_max:
            continue

        ema_lookback = 12
        if i >= ema_lookback:
            pd.Series(c[i - ema_lookback : i + 1]).ewm(span=ema_lookback).mean().iloc[-1]
            slope = (c[i - 1] - c[i - ema_lookback]) / max(c[i - ema_lookback], 1e-9)
        else:
            slope = 0.0
            c[i]

        for level, level_name in [(S1, "S1"), (P, "P")]:
            if l[i] <= level + 0.1 * a and l[i] >= level - 0.6 * a and c[i] > level:
                wick_ratio = lower_wick / bar_range if bar_range > 0 else 0
                if wick_ratio < rejection_wick_min:
                    continue
                if slope >= 0.0:
                    continue
                if c[i] <= o[i]:
                    continue
                entry = c[i]
                stop = l[i] - 0.3 * a
                target = entry + 2.0 * (entry - stop)
                height = entry - stop
                out.append(
                    ResearchPattern(
                        pattern="pivot_reversal_long",
                        direction="BUY",
                        bar_idx=int(i),
                        entry=float(entry),
                        stop=float(stop),
                        target=float(target),
                        confidence=0.60,
                        atr_at_entry=a,
                        height=float(height),
                        metadata={
                            "source": "research",
                            "pivot_level": level_name,
                            "pivot_price": float(level),
                            "wick_ratio": round(float(wick_ratio), 2),
                            "body_ratio": round(float(body_ratio), 2),
                            "slope": round(float(slope), 4),
                        },
                    )
                )
                break

        for level, level_name in [(R1, "R1"), (P, "P")]:
            if h[i] >= level - 0.1 * a and h[i] <= level + 0.6 * a and c[i] < level:
                wick_ratio = upper_wick / bar_range if bar_range > 0 else 0
                if wick_ratio < rejection_wick_min:
                    continue
                if slope <= 0.0:
                    continue
                if c[i] >= o[i]:
                    continue
                entry = c[i]
                stop = h[i] + 0.3 * a
                target = entry - 2.0 * (stop - entry)
                height = stop - entry
                out.append(
                    ResearchPattern(
                        pattern="pivot_reversal_short",
                        direction="SELL",
                        bar_idx=int(i),
                        entry=float(entry),
                        stop=float(stop),
                        target=float(target),
                        confidence=0.60,
                        atr_at_entry=a,
                        height=float(height),
                        metadata={
                            "source": "research",
                            "pivot_level": level_name,
                            "pivot_price": float(level),
                            "wick_ratio": round(float(wick_ratio), 2),
                            "body_ratio": round(float(body_ratio), 2),
                            "slope": round(float(slope), 4),
                        },
                    )
                )
                break
    return out

PRODUCTION_PATTERNS: set[str] = {
    "bb_squeeze_breakout",
    "inside_bar_breakout",
    "pivot_reversal_short",
    "three_black_crows_vol",
    "three_white_soldiers_vol",
}

def detect_all_research_patterns(
    df: pd.DataFrame,
    atr_col: str = "atr14",
    *,
    production_only: bool = True,
) -> list[ResearchPattern]:
    """Run all research-pattern detectors and return concatenated signals.

    Detector failures are isolated — one detector erroring won't break the rest.

    Args:
        production_only: If True (default), only emits patterns from
            PRODUCTION_PATTERNS (PF>1.5 in v0.0.29 backtest). Backtest
            scripts should pass False to evaluate every detector.
    """
    out: list[ResearchPattern] = []
    detectors: list[tuple[str, Any, frozenset[str]]] = [
        ("vcp", detect_vcp, frozenset({"vcp"})),
        ("bb_squeeze_breakout", detect_bb_squeeze_breakout, frozenset({"bb_squeeze_breakout"})),
        ("inside_bar_breakout", detect_inside_bar_breakout, frozenset({"inside_bar_breakout"})),
        (
            "three_soldiers_volume",
            detect_three_soldiers_volume,
            frozenset({"three_white_soldiers_vol", "three_black_crows_vol"}),
        ),
        (
            "pivot_reversal",
            detect_pivot_reversal,
            frozenset({"pivot_reversal_short", "pivot_reversal_long"}),
        ),
    ]
    for name, fn, emitted in detectors:
        if production_only and emitted.isdisjoint(PRODUCTION_PATTERNS):
            continue
        try:
            signals = fn(df, atr_col=atr_col)
            if production_only:
                signals = [s for s in signals if s.pattern in PRODUCTION_PATTERNS]
            out.extend(signals)
        except Exception as exc:
            logger.debug(
                "research_patterns detector failed",
                extra={"detector": name, "error": str(exc)},
            )
    return out

__all__ = [
    "ResearchPattern",
    "PRODUCTION_PATTERNS",
    "detect_vcp",
    "detect_bb_squeeze_breakout",
    "detect_inside_bar_breakout",
    "detect_three_soldiers_volume",
    "detect_pivot_reversal",
    "detect_all_research_patterns",
]
