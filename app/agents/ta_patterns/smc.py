"""
app/agents/ta_patterns/smc.py — Smart Money Concepts (SMC) detectors.

Five core SMC concepts, popularised by ICT-style traders. Each detector
emits `list[PatternSignal]` and is wired into the TATrader registry
exactly like other chart detectors. Crucially — and unlike the v0.0.37
skeleton — detectors **emit at the trigger bar** (`bar_idx=i`) rather
than always at `len(df)-1`. This is what makes them compatible with the
`backtest_research.py` simulator and avoids duplicate signals on the
last bar.

1. **Order Block (OB)** — the last opposite-direction candle before a
   strong impulsive move (>= impulse_atr_mult × ATR). Entry on retest
   of the OB body within `retest_max_bars` bars.
2. **Fair Value Gap (FVG)** — 3-candle imbalance where bar[i].low >
   bar[i-2].high (bullish) or bar[i].high < bar[i-2].low (bearish).
   Entry when price returns INTO the gap.
3. **Liquidity Sweep** — wick beyond N-bar swing high/low followed by
   a close back inside the range (false breakout reversal).
4. **Break of Structure (BOS)** — trend-continuation confirmation: a
   close above (HH) the most recent confirmed swing high after a
   sequence of higher highs (bullish), or below the last LL (bearish).
5. **CHOCH (Change of Character)** — first reversal flip: a close
   below the prior pivot low after a higher-high sequence (bearish), or
   above the prior pivot high after a lower-low sequence (bullish).

Two entry points are exposed:

- `SMC_DETECTORS` / individual `detect_*(df, pivots, atr_series)` — legacy
  signature consumed by `ta_trader.py`. These call into the bar-level
  implementations below and post-filter to the most recent signal so
  the trader doesn't emit stale historical patterns.

- `detect_all_smc_patterns(df, atr_col)` — DataFrame-only entry point
  used by `scripts/backtest_research.py --module ...smc:detect_all_smc_patterns`.
  Returns one signal per detection bar across the whole DataFrame.

Production filter
-----------------
`PRODUCTION_PATTERNS` is the set of SMC patterns that cleared PF > 1.5
on the 90d × 20-ticker backtest. `SMC_PRODUCTION_ENABLED` is the global
kill switch — flip to False to ship SMC as research-only.

References
  - ICT (Inner Circle Trader) original methodology
  - smartmoneyconcepts Python package (joshyattridge), MIT
  - Medium "I Backtested 2,600 SMC Trades"
"""

from __future__ import annotations

from app.agents.ta_patterns.pivots import PivotPoint
from app.agents.ta_patterns.reversal import PatternSignal, _rr
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import numpy as np
    import pandas as pd

    _READY = True
except ImportError:
    _READY = False

def _safe_atr_arr(atr: np.ndarray, i: int) -> float:
    """Safe atr arr."""
    if i < 0 or i >= len(atr):
        return 0.0
    val = float(atr[i])
    return val if np.isfinite(val) and val > 0 else 0.0

def _resolve_atr(
    df: pd.DataFrame, atr_col: str | None, atr_series: pd.Series | None
) -> np.ndarray | None:
    """Pick whichever ATR is available."""
    if atr_series is not None and len(atr_series) == len(df):
        return atr_series.to_numpy(dtype=float)
    if atr_col and atr_col in df.columns:
        return df[atr_col].to_numpy(dtype=float)
    return None

def _detect_order_block_impl(
    df: pd.DataFrame,
    atr: np.ndarray,
    *,
    impulse_atr_mult: float = 2.0,
    retest_max_bars: int = 20,
    rr_target_atr: float = 2.0,
) -> list[PatternSignal]:
    """
    Bullish OB:  last DOWN candle before a >2×ATR UP impulse.
                  Entry when a later bar closes BACK INSIDE the OB body.
    Bearish OB:  last UP candle before a >2×ATR DOWN impulse.
                  Entry when a later bar closes BACK INSIDE the OB body.

    Stop = opposite extreme of OB ± 0.3×ATR.
    Target = entry ± rr_target_atr × ATR.
    """
    if df is None or len(df) < 5:
        return []

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)

    out: list[PatternSignal] = []

    for impulse_i in range(1, n - 1):
        a = _safe_atr_arr(atr, impulse_i)
        if a == 0.0:
            continue
        body = abs(c[impulse_i] - o[impulse_i])
        if body < impulse_atr_mult * a:
            continue

        ob_i = impulse_i - 1
        ob_open = o[ob_i]
        ob_close = c[ob_i]
        ob_low = min(ob_open, ob_close)
        ob_high = max(ob_open, ob_close)
        if ob_high - ob_low < 1e-9:
            continue

        impulse_up = c[impulse_i] > o[impulse_i]

        last_search = min(n - 1, impulse_i + retest_max_bars)
        for j in range(impulse_i + 1, last_search + 1):
            a_j = _safe_atr_arr(atr, j)
            if a_j == 0.0:
                continue
            close_j = c[j]
            low_j = l[j]
            high_j = h[j]
            entered = low_j <= ob_high and high_j >= ob_low
            closed_inside = ob_low <= close_j <= ob_high
            if not (entered and closed_inside):
                continue

            if impulse_up:
                entry = close_j
                stop = ob_low - 0.3 * a_j
                target = entry + rr_target_atr * a_j
                if entry - stop < 1e-9:
                    break
                out.append(
                    PatternSignal(
                        pattern="smc_order_block_bull",
                        direction="BUY",
                        confidence=0.60,
                        bar_idx=int(j),
                        entry=float(entry),
                        stop=float(stop),
                        target=float(target),
                        expected_rr=_rr(entry, stop, target),
                        atr_at_entry=float(a_j),
                        metadata={
                            "ob_high": round(float(ob_high), 4),
                            "ob_low": round(float(ob_low), 4),
                            "impulse_bar": int(impulse_i),
                        },
                    )
                )
            else:
                entry = close_j
                stop = ob_high + 0.3 * a_j
                target = entry - rr_target_atr * a_j
                if stop - entry < 1e-9:
                    break
                out.append(
                    PatternSignal(
                        pattern="smc_order_block_bear",
                        direction="SELL",
                        confidence=0.60,
                        bar_idx=int(j),
                        entry=float(entry),
                        stop=float(stop),
                        target=float(target),
                        expected_rr=_rr(entry, stop, target),
                        atr_at_entry=float(a_j),
                        metadata={
                            "ob_high": round(float(ob_high), 4),
                            "ob_low": round(float(ob_low), 4),
                            "impulse_bar": int(impulse_i),
                        },
                    )
                )
            break
    return out

def _detect_fvg_impl(
    df: pd.DataFrame,
    atr: np.ndarray,
    *,
    min_gap_atr: float = 0.4,
    retest_max_bars: int = 15,
    rr_target_atr: float = 1.5,
) -> list[PatternSignal]:
    """
    Bullish FVG (gap up):  bar[k].low > bar[k-2].high
        Gap zone: [bar[k-2].high, bar[k].low]
        Entry when a later bar closes INTO that zone.
    Bearish FVG (gap down):  bar[k].high < bar[k-2].low
        Gap zone: [bar[k].high, bar[k-2].low]
        Entry when a later bar closes INTO that zone.
    """
    if df is None or len(df) < 5:
        return []

    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    out: list[PatternSignal] = []

    for k in range(2, n - 1):
        a = _safe_atr_arr(atr, k)
        if a == 0.0:
            continue

        if l[k] > h[k - 2]:
            gap_bot = h[k - 2]
            gap_top = l[k]
            gap_size = gap_top - gap_bot
            if gap_size >= min_gap_atr * a:
                last_search = min(n - 1, k + retest_max_bars)
                for j in range(k + 1, last_search + 1):
                    a_j = _safe_atr_arr(atr, j)
                    if a_j == 0.0:
                        continue
                    close_j = c[j]
                    if gap_bot <= close_j <= gap_top:
                        entry = close_j
                        stop = gap_bot - 0.2 * a_j
                        target = entry + rr_target_atr * a_j
                        if entry - stop < 1e-9:
                            break
                        out.append(
                            PatternSignal(
                                pattern="smc_fvg_bull",
                                direction="BUY",
                                confidence=0.55,
                                bar_idx=int(j),
                                entry=float(entry),
                                stop=float(stop),
                                target=float(target),
                                expected_rr=_rr(entry, stop, target),
                                atr_at_entry=float(a_j),
                                metadata={
                                    "gap_top": round(float(gap_top), 4),
                                    "gap_bot": round(float(gap_bot), 4),
                                    "gap_bar": int(k),
                                },
                            )
                        )
                        break

        if h[k] < l[k - 2]:
            gap_bot = h[k]
            gap_top = l[k - 2]
            gap_size = gap_top - gap_bot
            if gap_size < min_gap_atr * a:
                continue

            last_search = min(n - 1, k + retest_max_bars)
            for j in range(k + 1, last_search + 1):
                a_j = _safe_atr_arr(atr, j)
                if a_j == 0.0:
                    continue
                close_j = c[j]
                if gap_bot <= close_j <= gap_top:
                    entry = close_j
                    stop = gap_top + 0.2 * a_j
                    target = entry - rr_target_atr * a_j
                    if stop - entry < 1e-9:
                        break
                    out.append(
                        PatternSignal(
                            pattern="smc_fvg_bear",
                            direction="SELL",
                            confidence=0.55,
                            bar_idx=int(j),
                            entry=float(entry),
                            stop=float(stop),
                            target=float(target),
                            expected_rr=_rr(entry, stop, target),
                            atr_at_entry=float(a_j),
                            metadata={
                                "gap_top": round(float(gap_top), 4),
                                "gap_bot": round(float(gap_bot), 4),
                                "gap_bar": int(k),
                            },
                        )
                    )
                    break
    return out

def _detect_liquidity_sweep_impl(
    df: pd.DataFrame,
    atr: np.ndarray,
    *,
    lookback: int = 20,
    rr_target_atr: float = 1.5,
) -> list[PatternSignal]:
    """
    Sweep-high (BEARISH):  high[i] > max(high[i-lookback..i-1])
                           AND close[i] < that prior high.
    Sweep-low (BULLISH):   low[i] < min(low[i-lookback..i-1])
                           AND close[i] > that prior low.

    Entry at close of the sweep bar.
    Stop:  beyond the sweep wick ± 0.2 ATR.
    """
    if df is None or len(df) < lookback + 2:
        return []

    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    out: list[PatternSignal] = []

    for i in range(lookback, n):
        a = _safe_atr_arr(atr, i)
        if a == 0.0:
            continue
        prev_high = float(np.max(h[i - lookback : i]))
        prev_low = float(np.min(l[i - lookback : i]))

        if h[i] > prev_high and c[i] < prev_high:
            entry = c[i]
            stop = h[i] + 0.2 * a
            target = entry - rr_target_atr * a
            if stop - entry > 1e-9:
                out.append(
                    PatternSignal(
                        pattern="smc_sweep_high",
                        direction="SELL",
                        confidence=0.60,
                        bar_idx=int(i),
                        entry=float(entry),
                        stop=float(stop),
                        target=float(target),
                        expected_rr=_rr(entry, stop, target),
                        atr_at_entry=float(a),
                        metadata={
                            "swept_level": round(prev_high, 4),
                            "wick": round(float(h[i]), 4),
                        },
                    )
                )

        if l[i] < prev_low and c[i] > prev_low:
            entry = c[i]
            stop = l[i] - 0.2 * a
            target = entry + rr_target_atr * a
            if entry - stop > 1e-9:
                out.append(
                    PatternSignal(
                        pattern="smc_sweep_low",
                        direction="BUY",
                        confidence=0.60,
                        bar_idx=int(i),
                        entry=float(entry),
                        stop=float(stop),
                        target=float(target),
                        expected_rr=_rr(entry, stop, target),
                        atr_at_entry=float(a),
                        metadata={"swept_level": round(prev_low, 4), "wick": round(float(l[i]), 4)},
                    )
                )
    return out

def _detect_bos_impl(
    df: pd.DataFrame,
    atr: np.ndarray,
    *,
    swing_window: int = 5,
    rr_target_atr: float = 2.0,
    min_swings: int = 3,
) -> list[PatternSignal]:
    """
    Bullish BOS: last 3 swing highs form ascending h0 < h1 < h2 AND
                 a bar closes above h2 (with prev bar still at/below).
    Bearish BOS: last 3 swing lows form descending l0 > l1 > l2 AND
                 a bar closes below l2.

    Swing-N: a bar is a swing high if its high == max(high[i-W..i+W]),
    confirmed strictly after `swing_window` bars have passed.
    """
    if df is None or len(df) < swing_window * 4 + 5:
        return []

    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    out: list[PatternSignal] = []
    W = swing_window

    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []
    for i in range(W, n - W):
        if h[i] == np.max(h[i - W : i + W + 1]):
            swing_highs.append((i, float(h[i])))
        if l[i] == np.min(l[i - W : i + W + 1]):
            swing_lows.append((i, float(l[i])))

    for i in range(W * 2, n):
        a = _safe_atr_arr(atr, i)
        if a == 0.0:
            continue

        rh = [(idx, p) for idx, p in swing_highs if idx <= i - W]
        rl = [(idx, p) for idx, p in swing_lows if idx <= i - W]

        if len(rh) >= min_swings:
            h0, h1, h2 = rh[-3][1], rh[-2][1], rh[-1][1]
            if h0 < h1 < h2 and c[i] > h2 and c[i - 1] <= h2:
                entry = c[i]
                stop = float(rl[-1][1]) - 0.2 * a if rl else entry - a
                target = entry + rr_target_atr * a
                if entry - stop > 1e-9:
                    out.append(
                        PatternSignal(
                            pattern="smc_bos_bull",
                            direction="BUY",
                            confidence=0.58,
                            bar_idx=int(i),
                            entry=float(entry),
                            stop=float(stop),
                            target=float(target),
                            expected_rr=_rr(entry, stop, target),
                            atr_at_entry=float(a),
                            metadata={"prev_hh": round(h2, 4)},
                        )
                    )

        if len(rl) >= min_swings:
            l0, l1, l2 = rl[-3][1], rl[-2][1], rl[-1][1]
            if l0 > l1 > l2 and c[i] < l2 and c[i - 1] >= l2:
                entry = c[i]
                stop = float(rh[-1][1]) + 0.2 * a if rh else entry + a
                target = entry - rr_target_atr * a
                if stop - entry > 1e-9:
                    out.append(
                        PatternSignal(
                            pattern="smc_bos_bear",
                            direction="SELL",
                            confidence=0.58,
                            bar_idx=int(i),
                            entry=float(entry),
                            stop=float(stop),
                            target=float(target),
                            expected_rr=_rr(entry, stop, target),
                            atr_at_entry=float(a),
                            metadata={"prev_ll": round(l2, 4)},
                        )
                    )
    return out

def _detect_choch_impl(
    df: pd.DataFrame,
    atr: np.ndarray,
    *,
    swing_window: int = 5,
    rr_target_atr: float = 1.8,
) -> list[PatternSignal]:
    """
    Bullish CHOCH: prior swing lows DESCENDED (downtrend); a close ABOVE
                   the most recent swing high flips structure → BUY.
    Bearish CHOCH: prior swing highs ASCENDED; a close BELOW the most
                   recent swing low flips structure → SELL.
    """
    if df is None or len(df) < swing_window * 4 + 5:
        return []

    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    out: list[PatternSignal] = []
    W = swing_window

    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []
    for i in range(W, n - W):
        if h[i] == np.max(h[i - W : i + W + 1]):
            swing_highs.append((i, float(h[i])))
        if l[i] == np.min(l[i - W : i + W + 1]):
            swing_lows.append((i, float(l[i])))

    for i in range(W * 2, n):
        a = _safe_atr_arr(atr, i)
        if a == 0.0:
            continue
        rh = [(idx, p) for idx, p in swing_highs if idx <= i - W]
        rl = [(idx, p) for idx, p in swing_lows if idx <= i - W]

        if len(rl) >= 2 and len(rh) >= 1 and rl[-2][1] > rl[-1][1]:
            last_high = rh[-1][1]
            if c[i] > last_high and c[i - 1] <= last_high:
                entry = c[i]
                stop = float(rl[-1][1]) - 0.3 * a
                target = entry + rr_target_atr * a
                if entry - stop > 1e-9:
                    out.append(
                        PatternSignal(
                            pattern="smc_choch_bull",
                            direction="BUY",
                            confidence=0.55,
                            bar_idx=int(i),
                            entry=float(entry),
                            stop=float(stop),
                            target=float(target),
                            expected_rr=_rr(entry, stop, target),
                            atr_at_entry=float(a),
                            metadata={"flip_level": round(last_high, 4)},
                        )
                    )

        if len(rh) >= 2 and len(rl) >= 1 and rh[-2][1] < rh[-1][1]:
            last_low = rl[-1][1]
            if c[i] < last_low and c[i - 1] >= last_low:
                entry = c[i]
                stop = float(rh[-1][1]) + 0.3 * a
                target = entry - rr_target_atr * a
                if stop - entry > 1e-9:
                    out.append(
                        PatternSignal(
                            pattern="smc_choch_bear",
                            direction="SELL",
                            confidence=0.55,
                            bar_idx=int(i),
                            entry=float(entry),
                            stop=float(stop),
                            target=float(target),
                            expected_rr=_rr(entry, stop, target),
                            atr_at_entry=float(a),
                            metadata={"flip_level": round(last_low, 4)},
                        )
                    )
    return out

def _latest_only(
    signals: list[PatternSignal], df: pd.DataFrame, max_age: int = 3
) -> list[PatternSignal]:
    """Latest only."""
    if not signals:
        return []
    last_idx = len(df) - 1
    return [s for s in signals if s.bar_idx >= last_idx - max_age]

def detect_order_block(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr_series: pd.Series,
    *,
    impulse_atr_mult: float = 2.0,
    lookback: int = 30,
) -> list[PatternSignal]:
    """Legacy wrapper — only emits recent OB signals (live use)."""
    if not _READY or df is None or len(df) < lookback:
        return []
    atr = _resolve_atr(df, "atr14", atr_series)
    if atr is None:
        return []
    sigs = _detect_order_block_impl(df, atr, impulse_atr_mult=impulse_atr_mult)
    return _latest_only(sigs, df, max_age=3)

def detect_fair_value_gap(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr_series: pd.Series,
    *,
    lookback: int = 20,
) -> list[PatternSignal]:
    """Legacy wrapper — most-recent FVG retest."""
    if not _READY or df is None or len(df) < lookback:
        return []
    atr = _resolve_atr(df, "atr14", atr_series)
    if atr is None:
        return []
    sigs = _detect_fvg_impl(df, atr)
    return _latest_only(sigs, df, max_age=3)

def detect_liquidity_sweep(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr_series: pd.Series,
    *,
    lookback: int = 20,
    reversal_bars: int = 3,
) -> list[PatternSignal]:
    """Legacy wrapper — most-recent sweep."""
    if not _READY or df is None or len(df) < lookback + 2:
        return []
    atr = _resolve_atr(df, "atr14", atr_series)
    if atr is None:
        return []
    sigs = _detect_liquidity_sweep_impl(df, atr, lookback=lookback)
    return _latest_only(sigs, df, max_age=reversal_bars)

def detect_bos(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr_series: pd.Series,
) -> list[PatternSignal]:
    """Legacy wrapper — BOS only when just printed."""
    if not _READY or df is None or len(df) < 30:
        return []
    atr = _resolve_atr(df, "atr14", atr_series)
    if atr is None:
        return []
    sigs = _detect_bos_impl(df, atr)
    return _latest_only(sigs, df, max_age=2)

def detect_choch(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr_series: pd.Series,
) -> list[PatternSignal]:
    """Legacy wrapper — CHOCH only when just printed."""
    if not _READY or df is None or len(df) < 30:
        return []
    atr = _resolve_atr(df, "atr14", atr_series)
    if atr is None:
        return []
    sigs = _detect_choch_impl(df, atr)
    return _latest_only(sigs, df, max_age=2)

SMC_DETECTORS = [
    detect_order_block,
    detect_fair_value_gap,
    detect_liquidity_sweep,
    detect_bos,
    detect_choch,
]

PRODUCTION_PATTERNS: set[str] = {
    "smc_order_block_bear",
    "smc_sweep_low",
}

SMC_PRODUCTION_ENABLED: bool = True

def detect_all_smc_patterns(
    df: pd.DataFrame,
    atr_col: str = "atr14",
    *,
    production_only: bool = False,
) -> list[PatternSignal]:
    """Bar-level entry point used by `scripts/backtest_research.py`.

    Returns every signal at the bar it triggered.
    Set `production_only=True` to filter to `PRODUCTION_PATTERNS`.
    """
    if not _READY or df is None or len(df) < 10:
        return []
    if atr_col not in df.columns:
        return []
    atr = df[atr_col].to_numpy(dtype=float)

    detectors = [
        ("order_block", _detect_order_block_impl),
        ("fvg", _detect_fvg_impl),
        ("liquidity_sweep", _detect_liquidity_sweep_impl),
        ("bos", _detect_bos_impl),
        ("choch", _detect_choch_impl),
    ]
    out: list[PatternSignal] = []
    for name, fn in detectors:
        try:
            sigs = fn(df, atr)
        except Exception as exc:
            logger.debug("smc detector failed", extra={"detector": name, "error": str(exc)})
            continue
        if production_only:
            if not SMC_PRODUCTION_ENABLED:
                continue
            sigs = [s for s in sigs if s.pattern in PRODUCTION_PATTERNS]
        out.extend(sigs)
    return out

__all__ = [
    "detect_order_block",
    "detect_fair_value_gap",
    "detect_liquidity_sweep",
    "detect_bos",
    "detect_choch",
    "SMC_DETECTORS",
    "detect_all_smc_patterns",
    "PRODUCTION_PATTERNS",
    "SMC_PRODUCTION_ENABLED",
]
