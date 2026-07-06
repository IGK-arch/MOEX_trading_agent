"""High-quality chart pattern detectors from teammate Dasha (mlDasha.zip).

Phase 23 (v0.0.22) — integration of Dasha's notebook code that finds the
"best" (cleanest) chart patterns by enforcing strict filters:

  - ZigZag pivots use **hi/lo** prices (not close), so they catch wicks
    that mark real swing points.
  - Double Top/Bottom require a confirmed breakout of the neckline within
    a 50-bar window (was 24 in our existing detector → many setups
    missed their breakout window).
  - Head & Shoulders require a **trend pre-context** (uptrend before HS,
    downtrend before IHS) — without this, the (1,-1,1,-1,1) pivot shape
    matches a lot of noise in sideways markets.

Approach: keep Dasha's logic verbatim (sound, peer-reviewed by her).
Wrap the events into our `PatternSignal` dataclass so the rest of the
TA pipeline (ta_trader → aggregator → risk_manager) can consume them
without changes. Stop/target are derived from the pattern geometry
following her backtest.py convention:

  Double Top:    entry=neckline, stop=max(p1,p3)+0.2*ATR, target=entry-height
  Double Bottom: entry=neckline, stop=min(p1,p3)-0.2*ATR, target=entry+height
  H&S:           entry=neckline, stop=head+0.3*ATR,       target=entry-height
  Inv H&S:       entry=neckline, stop=head-0.3*ATR,       target=entry+height

Used by `app/agents/ta_trader.py` as a parallel source of signals.
Confluence with our existing detector → boosted confidence in aggregator.
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

def zigzag_atr_pivots_hilo(
    df: pd.DataFrame,
    k: float = 5.0,
    min_bars: int = 12,
    atr_col: str = "atr14",
) -> pd.DataFrame:
    """ZigZag pivots using high/low (not close) + ATR-scaled thresholds.

    Returns DataFrame with columns: pivot_i (int), time, price (float), type (+1=high, -1=low).
    """
    if not _READY or len(df) == 0:
        return pd.DataFrame(columns=["pivot_i", "time", "price", "type"])

    if atr_col not in df.columns:
        return pd.DataFrame(columns=["pivot_i", "time", "price", "type"])

    d = df.reset_index(drop=True)
    high = d["high"].values.astype(float)
    low = d["low"].values.astype(float)
    close = d["close"].values.astype(float)
    atr = d[atr_col].values.astype(float)
    ts = d["timestamp"].values if "timestamp" in d.columns else d.index.values

    pivots: list[tuple] = []
    n = len(d)
    trend = 0
    last_pivot_i = 0
    extreme_i = 0
    extreme_price = close[0]
    extreme_atr = atr[0] if np.isfinite(atr[0]) else 0.0

    for i in range(1, n):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue

        if trend == 0:
            thr = k * atr[i]
            if close[i] >= close[0] + thr:
                trend = +1
                extreme_i = i
                extreme_price = high[i]
                extreme_atr = atr[i]
            elif close[i] <= close[0] - thr:
                trend = -1
                extreme_i = i
                extreme_price = low[i]
                extreme_atr = atr[i]
            continue

        if trend == +1:
            if high[i] > extreme_price:
                extreme_i = i
                extreme_price = high[i]
                extreme_atr = atr[i]
            if low[i] <= extreme_price - k * extreme_atr:
                if extreme_i - last_pivot_i >= min_bars:
                    pivots.append((extreme_i, ts[extreme_i], extreme_price, +1))
                    last_pivot_i = extreme_i
                trend = -1
                extreme_i = i
                extreme_price = low[i]
                extreme_atr = atr[i]
        else:
            if low[i] < extreme_price:
                extreme_i = i
                extreme_price = low[i]
                extreme_atr = atr[i]
            if high[i] >= extreme_price + k * extreme_atr:
                if extreme_i - last_pivot_i >= min_bars:
                    pivots.append((extreme_i, ts[extreme_i], extreme_price, -1))
                    last_pivot_i = extreme_i
                trend = +1
                extreme_i = i
                extreme_price = high[i]
                extreme_atr = atr[i]

    return pd.DataFrame(pivots, columns=["pivot_i", "time", "price", "type"])

@dataclass
class DashaPattern:
    """Compact descriptor of one detected pattern. ta_trader.py converts to
    our canonical PatternSignal before passing to aggregator."""

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

def detect_double_patterns_dasha(
    df: pd.DataFrame,
    piv: pd.DataFrame,
    tol_atr: float = 2.5,
    depth_atr: float = 0.3,
    confirm_window: int = 60,
    atr_col: str = "atr14",
    enable_double_bottom: bool = False,
) -> list[DashaPattern]:
    """Detect Double Top (1-(-1)-1 sequence) and Double Bottom ((-1)-1-(-1)).

    Filters (Dasha's defaults — wider than our existing detector):
      - |A.price - C.price| <= tol_atr * ATR  → cosmetic asymmetry OK
      - height >= depth_atr * ATR             → pattern is visible (not noise)
      - breakout of B (neckline) within confirm_window bars after C

    Stop = beyond outer pivot ± 0.2*ATR (Dasha's backtest convention).
    Target = neckline ± height (1:1 risk-symmetry of pattern).

    Defaults tuned in Phase 28 (v0.0.41) via 90d × 20 ticker grid search
    over (depth_atr ∈ {0.3-0.6}, tol_atr ∈ {2.0-3.0}, confirm_window ∈
    {30-60}). Best double_top: PF=8.35 at depth=0.3, tol=2.5, cw=60
    (was PF=7.08 with old defaults 0.4/2.5/50). Grid showed depth_atr
    is non-binding (ZigZag k=5 already enforces large swings); kept at
    lower end for consistency.

    `enable_double_bottom=False` by default — the grid showed double
    bottom maxes out at PF=0.63 (16 trades), nowhere near our 1.5
    cutoff. Pattern is structurally biased here (asymmetric stop/target
    in the down-up-down geometry produces wider-than-expected risk on
    every breakout). We DETECT them implicitly via pivots but skip
    emitting trades to spare the aggregator from negative-expectancy
    setups. See `models_audit.md` for the full grid log.
    """
    if not _READY or len(piv) < 3 or len(df) == 0:
        return []
    if atr_col not in df.columns:
        return []

    out: list[DashaPattern] = []
    close = df["close"].values
    df["high"].values
    df["low"].values
    atr = df[atr_col].values

    p_i = piv["pivot_i"].values
    p_p = piv["price"].values
    p_t = piv["type"].values

    for i in range(len(piv) - 2):
        A_i = int(p_i[i])
        B_i = int(p_i[i + 1])
        C_i = int(p_i[i + 2])
        if (
            C_i >= len(close)
            or A_i < 0
            or B_i < 0
            or C_i < 0
            or A_i >= len(close)
            or B_i >= len(close)
        ):
            continue

        A_p = float(p_p[i])
        B_p = float(p_p[i + 1])
        C_p = float(p_p[i + 2])
        if not (np.isfinite(A_p) and np.isfinite(B_p) and np.isfinite(C_p)):
            continue

        A_t = int(p_t[i])
        B_t = int(p_t[i + 1])
        C_t = int(p_t[i + 2])

        a = float(atr[C_i])
        if not np.isfinite(a) or a <= 0:
            continue

        tol = tol_atr * a
        depth = depth_atr * a

        if enable_double_bottom and A_t == -1 and B_t == 1 and C_t == -1:
            if abs(A_p - C_p) > tol:
                continue
            height = B_p - min(A_p, C_p)
            if height < depth:
                continue
            end = min(len(close) - 1, C_i + confirm_window)
            for j in range(C_i, end + 1):
                if close[j] > B_p:
                    entry = B_p
                    stop = min(A_p, C_p) - 0.2 * a
                    target = B_p + height
                    out.append(
                        DashaPattern(
                            pattern="double_bottom",
                            direction="BUY",
                            bar_idx=j,
                            entry=entry,
                            stop=stop,
                            target=target,
                            confidence=0.70,
                            atr_at_entry=a,
                            height=height,
                            metadata={
                                "source": "dasha",
                                "p1_idx": A_i,
                                "p2_idx": B_i,
                                "p3_idx": C_i,
                                "p1_price": A_p,
                                "p2_price": B_p,
                                "p3_price": C_p,
                                "breakout_bar_offset": j - C_i,
                            },
                        )
                    )
                    break

        if A_t == 1 and B_t == -1 and C_t == 1:
            if abs(A_p - C_p) > tol:
                continue
            height = max(A_p, C_p) - B_p
            if height < depth:
                continue
            end = min(len(close) - 1, C_i + confirm_window)
            for j in range(C_i, end + 1):
                if close[j] < B_p:
                    entry = B_p
                    stop = max(A_p, C_p) + 0.2 * a
                    target = B_p - height
                    out.append(
                        DashaPattern(
                            pattern="double_top",
                            direction="SELL",
                            bar_idx=j,
                            entry=entry,
                            stop=stop,
                            target=target,
                            confidence=0.70,
                            atr_at_entry=a,
                            height=height,
                            metadata={
                                "source": "dasha",
                                "p1_idx": A_i,
                                "p2_idx": B_i,
                                "p3_idx": C_i,
                                "p1_price": A_p,
                                "p2_price": B_p,
                                "p3_price": C_p,
                                "breakout_bar_offset": j - C_i,
                            },
                        )
                    )
                    break

    return out

def detect_hs_patterns_dasha(
    df: pd.DataFrame,
    piv: pd.DataFrame,
    shoulder_tol_atr: float = 2.5,
    head_margin_atr: float = 0.5,
    confirm_window: int = 30,
    min_sep: int = 12,
    atr_col: str = "atr14",
) -> list[DashaPattern]:
    """Detect Head & Shoulders (1,-1,1,-1,1) and Inverted H&S (-1,1,-1,1,-1).

    Dasha's improvements over textbook HS:
      - **Trend pre-context**: HS requires uptrend (last 2 highs ↑, last 2 lows ↑)
        in the 4 pivots BEFORE the pattern. IHS requires downtrend.
        This kills ~70% of false positives that match (1,-1,1,-1,1) shape
        but actually sit inside sideways/down trends.
      - Shoulder symmetry: |A.price - E.price| <= shoulder_tol_atr * ATR
      - Head margin: C.price > A,E + head_margin_atr * ATR  (HS) or vice-versa
      - Time-symmetry: 0.4 <= (C-A)/(E-C) <= 2.5
      - Neckline slope from B to D, breakout of close < neckline (HS) confirms

    Stop = head ± 0.3*ATR, target = neckline ± height (height = head - shoulder).
    """
    if not _READY or len(piv) < 9 or len(df) == 0:
        return []
    if atr_col not in df.columns:
        return []

    out: list[DashaPattern] = []
    close = df["close"].values
    atr = df[atr_col].values
    p_i = piv["pivot_i"].values
    p_p = piv["price"].values
    p_t = piv["type"].values

    for i in range(4, len(piv) - 4):
        A_i, B_i, C_i, D_i, E_i = (int(p_i[i + k]) for k in range(5))
        A_p, B_p, C_p, D_p, E_p = (float(p_p[i + k]) for k in range(5))
        A_t, B_t, C_t, D_t, E_t = (int(p_t[i + k]) for k in range(5))

        idxs = (A_i, B_i, C_i, D_i, E_i)
        if any(ix < 0 or ix >= len(close) for ix in idxs):
            continue
        if min(B_i - A_i, C_i - B_i, D_i - C_i, E_i - D_i) < min_sep:
            continue
        if not all(np.isfinite(p) for p in (A_p, B_p, C_p, D_p, E_p)):
            continue

        a = float(atr[E_i])
        if not np.isfinite(a) or a <= 0:
            continue

        shoulder_tol = shoulder_tol_atr * a
        head_margin = head_margin_atr * a

        prev = piv.iloc[i - 4 : i]
        highs = prev[prev["type"] == 1]
        lows = prev[prev["type"] == -1]
        if len(highs) < 2 or len(lows) < 2:
            continue

        uptrend = float(highs["price"].iloc[-1]) > float(highs["price"].iloc[-2]) and float(
            lows["price"].iloc[-1]
        ) > float(lows["price"].iloc[-2])
        downtrend = float(highs["price"].iloc[-1]) < float(highs["price"].iloc[-2]) and float(
            lows["price"].iloc[-1]
        ) < float(lows["price"].iloc[-2])

        if (A_t, B_t, C_t, D_t, E_t) == (1, -1, 1, -1, 1):
            if not uptrend:
                continue
            if abs(A_p - E_p) > shoulder_tol:
                continue
            if not (C_p > A_p + head_margin and C_p > E_p + head_margin):
                continue

            left = C_i - A_i
            right = E_i - C_i
            if right == 0:
                continue
            ratio = left / right
            if ratio < 0.4 or ratio > 2.5:
                continue
            if abs(B_p - D_p) > 3.0 * a:
                continue

            m = (D_p - B_p) / (D_i - B_i)
            b = B_p - m * B_i
            end = min(len(close) - 1, E_i + confirm_window)
            for j in range(E_i, end + 1):
                neck = m * j + b
                if close[j] < neck:
                    head = C_p
                    height = head - max(B_p, D_p)
                    entry = float(neck)
                    stop = head + 0.3 * a
                    target = entry - height
                    out.append(
                        DashaPattern(
                            pattern="head_shoulders",
                            direction="SELL",
                            bar_idx=j,
                            entry=entry,
                            stop=stop,
                            target=target,
                            confidence=0.75,
                            atr_at_entry=a,
                            height=height,
                            metadata={
                                "source": "dasha",
                                "p1_idx": A_i,
                                "p2_idx": B_i,
                                "p3_idx": C_i,
                                "p4_idx": D_i,
                                "p5_idx": E_i,
                                "neckline_at_breakout": float(neck),
                                "head_price": head,
                            },
                        )
                    )
                    break

        if (A_t, B_t, C_t, D_t, E_t) == (-1, 1, -1, 1, -1):
            if not downtrend:
                continue
            if abs(A_p - E_p) > shoulder_tol:
                continue
            if not (C_p < A_p - head_margin and C_p < E_p - head_margin):
                continue

            left = C_i - A_i
            right = E_i - C_i
            if right == 0:
                continue
            ratio = left / right
            if ratio < 0.4 or ratio > 2.5:
                continue
            if abs(B_p - D_p) > 3.0 * a:
                continue

            m = (D_p - B_p) / (D_i - B_i)
            b = B_p - m * B_i
            end = min(len(close) - 1, E_i + confirm_window)
            for j in range(E_i, end + 1):
                neck = m * j + b
                if close[j] > neck:
                    head = C_p
                    height = min(B_p, D_p) - head
                    entry = float(neck)
                    stop = head - 0.3 * a
                    target = entry + height
                    out.append(
                        DashaPattern(
                            pattern="inv_head_shoulders",
                            direction="BUY",
                            bar_idx=j,
                            entry=entry,
                            stop=stop,
                            target=target,
                            confidence=0.75,
                            atr_at_entry=a,
                            height=height,
                            metadata={
                                "source": "dasha",
                                "p1_idx": A_i,
                                "p2_idx": B_i,
                                "p3_idx": C_i,
                                "p4_idx": D_i,
                                "p5_idx": E_i,
                                "neckline_at_breakout": float(neck),
                                "head_price": head,
                            },
                        )
                    )
                    break

    return out

def detect_all_dasha_patterns(
    df: pd.DataFrame,
    atr_col: str = "atr14",
    zigzag_k: float = 5.0,
    zigzag_min_bars: int = 12,
) -> list[DashaPattern]:
    """One-shot helper. Computes pivots once, runs DT/DB + HS/IHS."""
    piv = zigzag_atr_pivots_hilo(df, k=zigzag_k, min_bars=zigzag_min_bars, atr_col=atr_col)
    if len(piv) < 3:
        return []
    dt_db = detect_double_patterns_dasha(df, piv, atr_col=atr_col)
    hs = detect_hs_patterns_dasha(df, piv, atr_col=atr_col)
    return dt_db + hs

PRODUCTION_PATTERNS: set[str] = {
    "double_top",
    "head_shoulders",
}

__all__ = [
    "DashaPattern",
    "zigzag_atr_pivots_hilo",
    "detect_double_patterns_dasha",
    "detect_hs_patterns_dasha",
    "detect_all_dasha_patterns",
    "PRODUCTION_PATTERNS",
]
