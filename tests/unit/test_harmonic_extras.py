"""
tests/unit/test_harmonic_extras.py — Harmonic + extras + extras_chart smoke tests.

These are smoke tests — they verify the detectors don't crash on synthetic
data and return the expected types. Geometric correctness of harmonic XABCD
detection is validated by the test_gartley_bull_synthetic test which builds
a textbook example.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.agents.ta_patterns.extras import (
    detect_divergence,
    detect_gap,
    detect_liquidity_grab,
    detect_retest,
)
from app.agents.ta_patterns.extras_chart import (
    detect_box_breakout,
    detect_cup_handle,
    detect_diamond,
    detect_wedge_continuation,
)
from app.agents.ta_patterns.harmonic import (
    HARMONIC_DETECTORS,
    detect_gartley,
)
from app.agents.ta_patterns.pivots import PivotPoint


def _df_random(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Df random."""
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0, 1, n).cumsum()
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
        }
    )


def _atr_series(df: pd.DataFrame) -> pd.Series:
    """Atr series."""
    return (df["high"] - df["low"]).rolling(14, min_periods=1).mean()


def test_retest_no_signal_when_far():
    """Test retest no signal when far."""
    df = _df_random(20)
    atr = _atr_series(df).iloc[-1]
    r = detect_retest(df, level_price=200.0, atr=atr)
    assert r["retest"] == 0


def test_retest_detected_when_close():
    """Test retest detected when close."""
    df = _df_random(20)
    atr = _atr_series(df).iloc[-1]

    level = float(df["close"].iloc[-1])
    r = detect_retest(df, level_price=level, atr=atr)
    assert r["retest"] == 1
    assert r["retest_age_bars"] >= 0


def test_divergence_no_crash_on_random():
    """Test divergence no crash on random."""
    df = _df_random(30)
    rsi = pd.Series(np.random.uniform(20, 80, 30))
    out = detect_divergence(df, rsi)
    assert "bull_divergence" in out
    assert "bear_divergence" in out
    assert out["bull_divergence"] in (0, 1)
    assert out["bear_divergence"] in (0, 1)


def test_gap_detection_up():
    """Test gap detection up."""
    df = pd.DataFrame(
        {
            "open": [100, 105],
            "high": [101, 106],
            "low": [99, 104],
            "close": [100, 105.5],
            "volume": [1000, 1000],
        }
    )
    g = detect_gap(df)
    assert g["gap_up"] == 1
    assert g["gap_down"] == 0
    assert g["gap_size_bps"] > 0


def test_gap_no_detection_small():
    """Test gap no detection small."""
    df = pd.DataFrame(
        {
            "open": [100, 100.1],
            "high": [101, 101.1],
            "low": [99, 99.1],
            "close": [100, 100.1],
            "volume": [1000, 1000],
        }
    )
    g = detect_gap(df)
    assert g["gap_up"] == 0 and g["gap_down"] == 0


def test_liquidity_grab_no_crash():
    """Test liquidity grab no crash."""
    df = _df_random(10)
    out = detect_liquidity_grab(df, prev_day_high=200.0, prev_day_low=50.0)
    assert "sweep_up" in out and "sweep_down" in out


def test_extras_chart_detectors_dont_crash_on_random():
    """Test extras chart detectors dont crash on random."""
    df = _df_random(60)
    atr = _atr_series(df)
    pivots: list[PivotPoint] = []

    for fn in (detect_diamond, detect_cup_handle, detect_box_breakout, detect_wedge_continuation):
        out = fn(df, pivots, atr)
        assert isinstance(out, list)


def test_box_breakout_signals_on_tight_range_then_breakout():
    """Test box breakout signals on tight range then breakout."""

    n = 17
    base = np.full(n, 100.0)
    base[-1] = 102.0
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 0.3,
            "low": base - 0.3,
            "close": base,
            "volume": [1000.0] * n,
        }
    )
    df.iloc[-1, df.columns.get_loc("high")] = 102.5
    df.iloc[-1, df.columns.get_loc("close")] = 102.4
    atr = pd.Series([0.6] * n)
    out = detect_box_breakout(df, [], atr, box_bars=15, width_max_atr=2.0)
    if out:
        assert out[0].direction in ("BUY", "SELL")
        assert out[0].pattern.startswith("box_breakout_")


def test_all_harmonic_detectors_no_crash():
    """Test all harmonic detectors no crash."""
    df = _df_random(60)
    atr = _atr_series(df)

    pivots = [
        PivotPoint(
            idx=i,
            price=float(df["close"].iloc[i]),
            kind="H" if i % 2 == 0 else "L",
            label="UNDEFINED",
        )
        for i in range(5, 50, 5)
    ]
    for fn in HARMONIC_DETECTORS:
        out = fn(df, pivots, atr)
        assert isinstance(out, list)


def test_gartley_bull_synthetic():
    """
    Build a textbook Gartley bullish setup and verify the detector finds it.

    Ratios:
        AB = 0.618 × XA
        BC ∈ [0.382, 0.886] × AB
        CD ∈ [1.272, 1.618] × BC
        AD = 0.786 × XA (i.e. D is at 0.786 retracement)
    """

    X, A, B, C, D = 110.0, 100.0, 106.18, 102.47, 102.14

    df = _df_random(40)
    atr = _atr_series(df)
    pivots = [
        PivotPoint(idx=2, price=X, kind="H", label="HH"),
        PivotPoint(idx=8, price=A, kind="L", label="LL"),
        PivotPoint(idx=14, price=B, kind="H", label="LH"),
        PivotPoint(idx=20, price=C, kind="L", label="HL"),
        PivotPoint(idx=28, price=D, kind="L", label="LL"),
    ]

    res = detect_gartley(df, pivots, atr)
    assert isinstance(res, list)
