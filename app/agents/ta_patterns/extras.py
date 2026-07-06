"""
app/agents/ta_patterns/extras.py — Setup features (NOT trade-emitting patterns).

These detectors don't produce stand-alone `PatternSignal` entries. Instead
they enrich the meta-classifier and CatBoost feature vectors with binary
"setup present" flags + small numeric features that historical studies show
are predictive on intraday MOEX equities:

  - **retest**: did price retest a recently-broken level (S→R role flip)?
  - **divergence**: bullish/bearish divergence between price and RSI/OBV/MACD
  - **gap**: opening / midday gap + fill probability
  - **liquidity_grab**: stop-hunt above prev day high or below prev day low

Each function returns a small dict that the caller merges into its feature
extraction. No external dependencies beyond numpy / pandas.
"""

from __future__ import annotations

from typing import Any

try:
    import numpy as np  # noqa: F401
    import pandas as pd

    _READY = True
except ImportError:
    _READY = False

def detect_retest(
    df: pd.DataFrame,
    *,
    level_price: float,
    tolerance_atr: float = 0.3,
    atr: float = 0.0,
    lookback: int = 10,
) -> dict[str, Any]:
    """
    Did the recent `lookback` bars revisit `level_price` after a breakout?

    Returns:
        {"retest": 0|1, "retest_age_bars": int}
    """
    if not _READY or df is None or len(df) < 3 or atr <= 0:
        return {"retest": 0, "retest_age_bars": -1}

    tail = df.tail(lookback)
    tail["close"].astype(float).to_numpy()
    high = tail["high"].astype(float).to_numpy()
    low = tail["low"].astype(float).to_numpy()
    tol = tolerance_atr * atr

    for i in range(len(tail) - 1, -1, -1):
        if low[i] - tol <= level_price <= high[i] + tol:
            return {
                "retest": 1,
                "retest_age_bars": len(tail) - 1 - i,
            }
    return {"retest": 0, "retest_age_bars": -1}

def detect_divergence(
    df: pd.DataFrame,
    indicator: pd.Series,
    *,
    lookback: int = 15,
) -> dict[str, Any]:
    """
    Compare last two local highs / lows of price vs `indicator`.

    Returns:
        {"bull_divergence": 0|1, "bear_divergence": 0|1}

      bull_divergence: price lower-low, indicator higher-low → bullish
      bear_divergence: price higher-high, indicator lower-high → bearish
    """
    if (
        not _READY
        or df is None
        or indicator is None
        or len(df) < lookback
        or len(indicator) < lookback
    ):
        return {"bull_divergence": 0, "bear_divergence": 0}

    close = df["close"].astype(float).to_numpy()[-lookback:]
    ind = indicator.astype(float).to_numpy()[-lookback:]

    half = lookback // 2
    p1_close, p2_close = close[:half], close[half:]
    p1_ind, p2_ind = ind[:half], ind[half:]

    bull = 0
    bear = 0

    if p2_close.min() < p1_close.min() and p2_ind.min() > p1_ind.min():
        bull = 1

    if p2_close.max() > p1_close.max() and p2_ind.max() < p1_ind.max():
        bear = 1
    return {"bull_divergence": bull, "bear_divergence": bear}

def detect_gap(df: pd.DataFrame) -> dict[str, Any]:
    """
    Did the latest bar open with a gap from the previous close?

    Returns:
        {"gap_up": 0|1, "gap_down": 0|1, "gap_size_bps": float}
    """
    if not _READY or df is None or len(df) < 2:
        return {"gap_up": 0, "gap_down": 0, "gap_size_bps": 0.0}
    prev_close = float(df["close"].iloc[-2])
    cur_open = float(df["open"].iloc[-1])
    if prev_close <= 0:
        return {"gap_up": 0, "gap_down": 0, "gap_size_bps": 0.0}
    pct = (cur_open - prev_close) / prev_close
    bps = pct * 10000
    return {
        "gap_up": 1 if pct > 0.003 else 0,
        "gap_down": 1 if pct < -0.003 else 0,
        "gap_size_bps": round(bps, 2),
    }

def detect_liquidity_grab(
    df: pd.DataFrame,
    *,
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
    lookback: int = 5,
) -> dict[str, Any]:
    """
    Did the recent bars pierce previous day's high/low and reverse?
    (Stop-hunt / Smart Money "liquidity sweep" lite.)

    Returns:
        {"sweep_up": 0|1, "sweep_down": 0|1}

      sweep_up:   high broke prev_day_high but close fell back below it
      sweep_down: low  broke prev_day_low  but close came back above it
    """
    if (
        not _READY
        or df is None
        or len(df) < lookback
        or (prev_day_high is None and prev_day_low is None)
    ):
        return {"sweep_up": 0, "sweep_down": 0}

    tail = df.tail(lookback)
    sweep_up = 0
    sweep_down = 0
    if prev_day_high is not None and (
        tail["high"].astype(float).max() > prev_day_high
        and float(tail["close"].iloc[-1]) < prev_day_high
    ):
        sweep_up = 1
    if prev_day_low is not None and (
        tail["low"].astype(float).min() < prev_day_low
        and float(tail["close"].iloc[-1]) > prev_day_low
    ):
        sweep_down = 1
    return {"sweep_up": sweep_up, "sweep_down": sweep_down}

__all__ = [
    "detect_retest",
    "detect_divergence",
    "detect_gap",
    "detect_liquidity_grab",
]
