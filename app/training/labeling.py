"""
app/training/labeling.py — Triple-barrier event labeling.

Reference: Marcos Lopez de Prado, "Advances in Financial Machine Learning", Ch.3.

Definition. For each event (a pattern firing at bar t0 with entry/stop/target levels
and ATR), three barriers are placed:

  - **top** barrier: a price level above entry (long) or below entry (short).
    Either explicit `target_level` (geometric R:R target of the pattern) or
    `entry + atr_mult_top * ATR_at_t0`.
  - **bottom** barrier: a price level below entry (long) or above entry (short).
    Either explicit `stop_level` or `entry - atr_mult_bot * ATR_at_t0`.
  - **vertical** barrier: a time-based stop `t0 + horizon_bars`.

Forward-walk through bars (t0+1 .. t0+horizon), checking which barrier hits
first. The label is:

  - **+1** (success): top barrier hit first
  - **-1** (failure): bottom barrier hit first
  - **0**  (timeout): vertical barrier reached without either side hit

For binary classification (used by CatBoost and meta-classifier), we map:
  +1 → 1 (positive class), -1 OR 0 → 0 (negative class).

The barrier-exit time `t1` is returned so that downstream `purged_kfold` can
embargo overlapping samples.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

try:
    import numpy as np  # noqa: F401
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError("triple-barrier labeling requires numpy + pandas") from e

@dataclass
class TripleBarrierLabel:
    """Result of triple-barrier labeling for ONE event."""

    label: int
    binary: int
    exit_bar_idx: int
    exit_price: float
    holding_bars: int
    barrier_hit: str

def label_triple_barrier(
    df: pd.DataFrame,
    *,
    bar_idx: int,
    direction: str,
    entry: float,
    stop: float | None = None,
    target: float | None = None,
    atr_at_entry: float | None = None,
    horizon_bars: int = 24,
    atr_mult_top: float = 2.0,
    atr_mult_bot: float = 1.0,
) -> TripleBarrierLabel:
    """
    Label one event using the triple-barrier method.

    Parameters
    ----------
    df : DataFrame
        OHLCV with columns ['high', 'low', 'close'] (others ignored).
        Must be indexed 0..N-1 (call .reset_index(drop=True) before passing).
    bar_idx : int
        Index of the bar where the event fires (t0). Forward-walk starts at
        `bar_idx + 1`.
    direction : str
        "BUY" or "SELL". Long means top = profit, bottom = stop.
    entry : float
        Entry price at t0.
    stop : float | None
        Explicit stop barrier price. If None, fall back to ATR-based.
    target : float | None
        Explicit target barrier price. If None, fall back to ATR-based.
    atr_at_entry : float | None
        ATR value at t0 (used only when `stop` or `target` is None).
    horizon_bars : int
        Vertical barrier: timeout after this many bars.
    atr_mult_top / atr_mult_bot : float
        ATR multipliers for the synthetic barriers (used as fallback).

    Returns
    -------
    TripleBarrierLabel
        - .label   ∈ {+1, -1, 0}        (top / bottom / timeout)
        - .binary  ∈ {0, 1}              (1 iff +1, used by CatBoost binary)
        - .exit_bar_idx                  (which bar closed the trade)
        - .barrier_hit ∈ {"top","bottom","timeout","no_data"}
    """
    is_buy = direction.upper() == "BUY"
    n_bars = len(df)

    if target is not None and target > 0:
        top_level = float(target) if is_buy else float(target)
    else:
        atr = float(atr_at_entry) if atr_at_entry and atr_at_entry > 0 else 0.0
        top_level = entry + atr_mult_top * atr if is_buy else entry - atr_mult_top * atr

    if stop is not None and stop > 0:
        bot_level = float(stop)
    else:
        atr = float(atr_at_entry) if atr_at_entry and atr_at_entry > 0 else 0.0
        bot_level = entry - atr_mult_bot * atr if is_buy else entry + atr_mult_bot * atr

    start = bar_idx + 1
    end = min(n_bars, start + horizon_bars)

    if end <= start:
        return TripleBarrierLabel(
            label=0,
            binary=0,
            exit_bar_idx=-1,
            exit_price=0.0,
            holding_bars=0,
            barrier_hit="no_data",
        )

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    for i in range(start, end):
        h = float(highs[i])
        lo = float(lows[i])
        if is_buy:
            if lo <= bot_level:
                return TripleBarrierLabel(
                    label=-1,
                    binary=0,
                    exit_bar_idx=i,
                    exit_price=bot_level,
                    holding_bars=i - bar_idx,
                    barrier_hit="bottom",
                )
            if h >= top_level:
                return TripleBarrierLabel(
                    label=1,
                    binary=1,
                    exit_bar_idx=i,
                    exit_price=top_level,
                    holding_bars=i - bar_idx,
                    barrier_hit="top",
                )
        else:
            if h >= bot_level:
                return TripleBarrierLabel(
                    label=-1,
                    binary=0,
                    exit_bar_idx=i,
                    exit_price=bot_level,
                    holding_bars=i - bar_idx,
                    barrier_hit="bottom",
                )
            if lo <= top_level:
                return TripleBarrierLabel(
                    label=1,
                    binary=1,
                    exit_bar_idx=i,
                    exit_price=top_level,
                    holding_bars=i - bar_idx,
                    barrier_hit="top",
                )

    exit_idx = end - 1
    return TripleBarrierLabel(
        label=0,
        binary=0,
        exit_bar_idx=exit_idx,
        exit_price=float(closes[exit_idx]),
        holding_bars=exit_idx - bar_idx,
        barrier_hit="timeout",
    )

def label_events_batch(
    df: pd.DataFrame,
    events: Iterable[dict],
    *,
    horizon_bars: int = 24,
    atr_mult_top: float = 2.0,
    atr_mult_bot: float = 1.0,
) -> pd.DataFrame:
    """
    Label many events at once. Each `event` dict must contain:
        bar_idx, direction, entry, [stop], [target], [atr_at_entry]

    Returns a DataFrame with original keys plus:
        label, binary, exit_bar_idx, exit_price, holding_bars, barrier_hit
    """
    rows = []
    for ev in events:
        res = label_triple_barrier(
            df,
            bar_idx=int(ev["bar_idx"]),
            direction=str(ev["direction"]),
            entry=float(ev["entry"]),
            stop=ev.get("stop"),
            target=ev.get("target"),
            atr_at_entry=ev.get("atr_at_entry"),
            horizon_bars=horizon_bars,
            atr_mult_top=atr_mult_top,
            atr_mult_bot=atr_mult_bot,
        )
        row = dict(ev)
        row.update(
            {
                "label": res.label,
                "binary": res.binary,
                "exit_bar_idx": res.exit_bar_idx,
                "exit_price": res.exit_price,
                "holding_bars": res.holding_bars,
                "barrier_hit": res.barrier_hit,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)

__all__ = ["TripleBarrierLabel", "label_triple_barrier", "label_events_batch"]
