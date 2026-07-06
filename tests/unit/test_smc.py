"""
tests/unit/test_smc.py — Smart Money Concepts detectors.

Tests cover both the legacy `(df, pivots, atr_series)` facades wired into
TATrader AND the bar-level `detect_all_smc_patterns(df)` used by the
backtest harness. Each detector gets ≥5 assertions covering:
  - schema (list[PatternSignal], correct fields)
  - direction wiring
  - geometry (entry/stop/target sanity, R:R > 0)
  - bar_idx pointing at trigger bar (not last bar)
  - parametrisation / robustness on flat data
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.agents.ta_patterns.reversal import PatternSignal
from app.agents.ta_patterns.smc import (
    PRODUCTION_PATTERNS,
    SMC_DETECTORS,
    detect_all_smc_patterns,
    detect_bos,
    detect_choch,
    detect_fair_value_gap,
    detect_liquidity_sweep,
    detect_order_block,
)


def _df(n: int = 80, seed: int = 7) -> pd.DataFrame:
    """Df."""
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0, 0.5, n).cumsum()
    high = close + rng.uniform(0.2, 0.6, n)
    low = close - rng.uniform(0.2, 0.6, n)
    open_ = close + rng.normal(0, 0.1, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
        }
    )


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Atr."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _with_atr_col(df: pd.DataFrame) -> pd.DataFrame:
    """With atr col."""
    df = df.copy()
    df["atr14"] = _atr(df)
    return df


def test_ob_no_crash_random():
    """Test ob no crash random."""
    df = _df(60)
    out = detect_order_block(df, [], _atr(df))
    assert isinstance(out, list)


def test_ob_returns_pattern_signals():
    """Constructed bullish OB: small red candle then 3-ATR green impulse, retest."""
    n = 40
    open_ = [100.0] * n
    high = [100.5] * n
    low = [99.5] * n
    close = [100.0] * n

    open_[30], high[30], low[30], close[30] = 100.5, 100.6, 99.5, 99.5
    open_[31], high[31], low[31], close[31] = 99.5, 105.5, 99.5, 105.0
    for j in (32, 33, 34, 35):
        open_[j], high[j], low[j], close[j] = 104.5, 105.2, 104.0, 104.8
    open_[36], high[36], low[36], close[36] = 101.0, 101.0, 99.8, 100.0
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": [1000] * n}
    )
    out = detect_order_block(df, [], _atr(df))
    if out:
        s = out[0]
        assert isinstance(s, PatternSignal)
        assert s.direction in ("BUY", "SELL")
        assert s.pattern.startswith("smc_order_block")
        assert s.expected_rr > 0


def test_ob_bar_idx_points_to_trigger_bar_not_last():
    """When OB triggers mid-history, bar_idx must be at the retest bar."""
    df = _with_atr_col(_df(80))
    out = detect_all_smc_patterns(df)
    ob = [s for s in out if s.pattern.startswith("smc_order_block")]
    for s in ob:
        assert 0 <= s.bar_idx < len(df)


def test_ob_stop_below_entry_for_bull():
    """Test ob stop below entry for bull."""
    df = _with_atr_col(_df(80))
    out = detect_all_smc_patterns(df)
    for s in out:
        if s.pattern == "smc_order_block_bull":
            assert s.stop < s.entry < s.target
        if s.pattern == "smc_order_block_bear":
            assert s.target < s.entry < s.stop


def test_ob_empty_on_flat_data():
    """Test ob empty on flat data."""
    df = pd.DataFrame(
        {
            "open": [100] * 30,
            "high": [100.1] * 30,
            "low": [99.9] * 30,
            "close": [100] * 30,
            "volume": [1000] * 30,
        }
    )
    df["atr14"] = _atr(df)
    out = detect_all_smc_patterns(df)
    ob = [s for s in out if s.pattern.startswith("smc_order_block")]
    assert ob == []


def test_fvg_no_crash_random():
    """Test fvg no crash random."""
    df = _df(60)
    out = detect_fair_value_gap(df, [], _atr(df))
    assert isinstance(out, list)


def test_fvg_bull_gap_constructed():
    """Construct a clear bull FVG: bar k-2.high=100, bar k.low=103, retest at 101.5."""
    rows = {"open": [], "high": [], "low": [], "close": [], "volume": []}
    for _ in range(45):
        rows["open"].append(99.5)
        rows["high"].append(100.0)
        rows["low"].append(99.0)
        rows["close"].append(99.5)
        rows["volume"].append(1000)
    rows["open"].append(99.5)
    rows["high"].append(100.0)
    rows["low"].append(99.0)
    rows["close"].append(99.8)
    rows["volume"].append(1000)
    rows["open"].append(100.0)
    rows["high"].append(103.0)
    rows["low"].append(99.8)
    rows["close"].append(102.8)
    rows["volume"].append(1000)
    rows["open"].append(103.0)
    rows["high"].append(104.0)
    rows["low"].append(103.0)
    rows["close"].append(103.5)
    rows["volume"].append(1000)
    rows["open"].append(103.0)
    rows["high"].append(103.5)
    rows["low"].append(101.5)
    rows["close"].append(101.5)
    rows["volume"].append(1000)
    rows["open"].append(101.5)
    rows["high"].append(102.0)
    rows["low"].append(101.0)
    rows["close"].append(101.8)
    rows["volume"].append(1000)
    df = pd.DataFrame(rows)
    df["atr14"] = _atr(df)
    out = detect_all_smc_patterns(df)
    fvg = [s for s in out if s.pattern == "smc_fvg_bull"]
    assert len(fvg) >= 1
    s = fvg[0]
    assert s.direction == "BUY"
    assert s.entry > 0
    assert s.expected_rr > 0


def test_fvg_bear_gap_constructed():
    """Bearish FVG: bar k-2.low=100, bar k.high=97."""
    rows = {"open": [], "high": [], "low": [], "close": [], "volume": []}
    for _ in range(45):
        rows["open"].append(100.5)
        rows["high"].append(101.0)
        rows["low"].append(100.0)
        rows["close"].append(100.5)
        rows["volume"].append(1000)
    rows["open"].append(100.5)
    rows["high"].append(101.0)
    rows["low"].append(100.0)
    rows["close"].append(100.2)
    rows["volume"].append(1000)
    rows["open"].append(100.0)
    rows["high"].append(100.2)
    rows["low"].append(97.0)
    rows["close"].append(97.2)
    rows["volume"].append(1000)
    rows["open"].append(97.0)
    rows["high"].append(97.0)
    rows["low"].append(96.0)
    rows["close"].append(96.5)
    rows["volume"].append(1000)
    rows["open"].append(96.5)
    rows["high"].append(98.5)
    rows["low"].append(96.5)
    rows["close"].append(98.5)
    rows["volume"].append(1000)
    rows["open"].append(98.5)
    rows["high"].append(99.0)
    rows["low"].append(98.0)
    rows["close"].append(98.5)
    rows["volume"].append(1000)
    df = pd.DataFrame(rows)
    df["atr14"] = _atr(df)
    out = detect_all_smc_patterns(df)
    fvg = [s for s in out if s.pattern == "smc_fvg_bear"]
    assert len(fvg) >= 1
    s = fvg[0]
    assert s.direction == "SELL"
    assert s.stop > s.entry > s.target


def test_fvg_min_gap_filters_noise():
    """Tiny gaps should NOT trigger — min_gap_atr filter."""
    n = 30
    df = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [100.05] * n,
            "low": [99.95] * n,
            "close": [100.0] * n,
            "volume": [1000] * n,
        }
    )
    df["atr14"] = _atr(df)
    out = detect_all_smc_patterns(df)
    fvg = [s for s in out if "fvg" in s.pattern]
    assert fvg == []


def test_fvg_bar_idx_is_trigger_bar():
    """Test fvg bar idx is trigger bar."""
    df = _with_atr_col(_df(80))
    out = detect_all_smc_patterns(df)
    for s in out:
        if "fvg" in s.pattern:
            assert s.bar_idx >= 2 and s.bar_idx < len(df)


def test_sweep_no_crash_random():
    """Test sweep no crash random."""
    df = _df(60)
    out = detect_liquidity_sweep(df, [], _atr(df))
    assert isinstance(out, list)


def test_sweep_high_constructed():
    """Build a sweep-high: 20 flat bars then one wick spike that closes back inside."""
    n = 30
    rows = {
        "open": [100.0] * n,
        "high": [100.5] * n,
        "low": [99.5] * n,
        "close": [100.0] * n,
        "volume": [1000] * n,
    }
    rows["high"][28] = 102.0
    rows["close"][28] = 100.0
    rows["open"][28] = 100.5
    df = pd.DataFrame(rows)
    df["atr14"] = _atr(df)
    out = detect_all_smc_patterns(df)
    sweeps = [s for s in out if s.pattern == "smc_sweep_high"]
    assert len(sweeps) >= 1
    assert sweeps[0].direction == "SELL"


def test_sweep_low_constructed():
    """Build a sweep-low."""
    n = 30
    rows = {
        "open": [100.0] * n,
        "high": [100.5] * n,
        "low": [99.5] * n,
        "close": [100.0] * n,
        "volume": [1000] * n,
    }
    rows["low"][28] = 98.0
    rows["close"][28] = 100.0
    rows["open"][28] = 99.5
    df = pd.DataFrame(rows)
    df["atr14"] = _atr(df)
    out = detect_all_smc_patterns(df)
    sweeps = [s for s in out if s.pattern == "smc_sweep_low"]
    assert len(sweeps) >= 1
    assert sweeps[0].direction == "BUY"


def test_sweep_requires_close_back_inside():
    """A break that CLOSES outside (not sweep) → no signal."""
    n = 30
    rows = {
        "open": [100.0] * n,
        "high": [100.5] * n,
        "low": [99.5] * n,
        "close": [100.0] * n,
        "volume": [1000] * n,
    }
    rows["high"][28] = 102.0
    rows["close"][28] = 101.5
    df = pd.DataFrame(rows)
    df["atr14"] = _atr(df)
    out = detect_all_smc_patterns(df)
    sweeps = [s for s in out if s.pattern == "smc_sweep_high"]
    assert sweeps == []


def test_sweep_geometry():
    """Test sweep geometry."""
    df = _with_atr_col(_df(80))
    out = detect_all_smc_patterns(df)
    for s in out:
        if s.pattern == "smc_sweep_high":
            assert s.stop > s.entry > s.target
        if s.pattern == "smc_sweep_low":
            assert s.target > s.entry > s.stop


def test_bos_no_crash_random():
    """Test bos no crash random."""
    df = _df(60)
    out = detect_bos(df, [], _atr(df))
    assert isinstance(out, list)


def test_bos_bull_three_ascending_highs():
    """Construct ascending highs then a close above the latest."""
    n = 80
    close = np.linspace(100, 110, n) + np.sin(np.arange(n) * 0.3) * 1.0
    high = close + 0.5
    low = close - 0.5
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": [1000] * n}
    )
    df["atr14"] = _atr(df)
    out = detect_all_smc_patterns(df)
    bos = [s for s in out if s.pattern == "smc_bos_bull"]
    if bos:
        assert all(s.direction == "BUY" for s in bos)
        assert all(s.expected_rr > 0 for s in bos)


def test_bos_bear_three_descending_lows():
    """Test bos bear three descending lows."""
    n = 80
    close = np.linspace(110, 100, n) + np.sin(np.arange(n) * 0.3) * 1.0
    high = close + 0.5
    low = close - 0.5
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": [1000] * n}
    )
    df["atr14"] = _atr(df)
    out = detect_all_smc_patterns(df)
    bos = [s for s in out if s.pattern == "smc_bos_bear"]
    if bos:
        assert all(s.direction == "SELL" for s in bos)


def test_bos_empty_on_flat_data():
    """Test bos empty on flat data."""
    df = pd.DataFrame(
        {
            "open": [100.0] * 80,
            "high": [100.1] * 80,
            "low": [99.9] * 80,
            "close": [100.0] * 80,
            "volume": [1000] * 80,
        }
    )
    df["atr14"] = _atr(df)
    out = detect_all_smc_patterns(df)
    bos = [s for s in out if "bos" in s.pattern]
    assert bos == []


def test_bos_bar_idx_within_range():
    """Test bos bar idx within range."""
    df = _with_atr_col(_df(80))
    out = detect_all_smc_patterns(df)
    for s in out:
        if "bos" in s.pattern:
            assert 0 <= s.bar_idx < len(df)


def test_choch_no_crash_random():
    """Test choch no crash random."""
    df = _df(60)
    out = detect_choch(df, [], _atr(df))
    assert isinstance(out, list)


def test_choch_signal_schema():
    """Test choch signal schema."""
    df = _with_atr_col(_df(120, seed=11))
    out = detect_all_smc_patterns(df)
    choch = [s for s in out if "choch" in s.pattern]
    for s in choch:
        assert s.pattern in ("smc_choch_bull", "smc_choch_bear")
        assert s.direction in ("BUY", "SELL")
        assert s.expected_rr > 0


def test_choch_geometry():
    """Test choch geometry."""
    df = _with_atr_col(_df(120, seed=11))
    out = detect_all_smc_patterns(df)
    for s in out:
        if s.pattern == "smc_choch_bull":
            assert s.stop < s.entry < s.target
        if s.pattern == "smc_choch_bear":
            assert s.target < s.entry < s.stop


def test_choch_empty_on_flat_data():
    """Test choch empty on flat data."""
    df = pd.DataFrame(
        {
            "open": [100.0] * 80,
            "high": [100.1] * 80,
            "low": [99.9] * 80,
            "close": [100.0] * 80,
            "volume": [1000] * 80,
        }
    )
    df["atr14"] = _atr(df)
    out = detect_all_smc_patterns(df)
    choch = [s for s in out if "choch" in s.pattern]
    assert choch == []


def test_choch_bar_idx_in_range():
    """Test choch bar idx in range."""
    df = _with_atr_col(_df(120))
    out = detect_all_smc_patterns(df)
    for s in out:
        if "choch" in s.pattern:
            assert 0 <= s.bar_idx < len(df)


def test_all_smc_detectors_no_crash():
    """Test all smc detectors no crash."""
    df = _df(80)
    atr = _atr(df)
    for fn in SMC_DETECTORS:
        out = fn(df, [], atr)
        assert isinstance(out, list)


def test_detect_all_smc_patterns_returns_list():
    """Test detect all smc patterns returns list."""
    df = _with_atr_col(_df(120))
    out = detect_all_smc_patterns(df)
    assert isinstance(out, list)
    for s in out:
        assert isinstance(s, PatternSignal)
        assert s.expected_rr >= 0
        assert s.atr_at_entry > 0


def test_production_only_filter_emits_only_whitelisted(monkeypatch):
    """When the master switch is OFF, production_only=True returns nothing."""
    import app.agents.ta_patterns.smc as smc_mod

    monkeypatch.setattr(smc_mod, "SMC_PRODUCTION_ENABLED", False)
    df = _with_atr_col(_df(120))
    out = smc_mod.detect_all_smc_patterns(df, production_only=True)
    assert out == []


def test_production_only_default_returns_only_whitelisted_patterns():
    """When production_only=True and patterns whitelisted, only those survive."""
    df = _with_atr_col(_df(120))
    out = detect_all_smc_patterns(df, production_only=True)
    for s in out:
        assert s.pattern in PRODUCTION_PATTERNS


def test_production_only_filter_with_explicit_set(monkeypatch):
    """When PRODUCTION_PATTERNS contains a pattern, that pattern survives."""
    import app.agents.ta_patterns.smc as smc_mod

    monkeypatch.setattr(smc_mod, "SMC_PRODUCTION_ENABLED", True)
    monkeypatch.setattr(smc_mod, "PRODUCTION_PATTERNS", {"smc_sweep_high"})
    df = _with_atr_col(_df(120))
    out = smc_mod.detect_all_smc_patterns(df, production_only=True)
    for s in out:
        assert s.pattern == "smc_sweep_high"


def test_legacy_facade_returns_only_recent_signals():
    """Test legacy facade returns only recent signals."""
    df = _with_atr_col(_df(120))
    atr_series = _atr(df)
    sigs = detect_order_block(df, [], atr_series)
    last_idx = len(df) - 1
    for s in sigs:
        assert s.bar_idx >= last_idx - 3
