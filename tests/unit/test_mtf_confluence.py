"""Tests for multi-timeframe (MTF) confluence module + aggregator wiring.

v0.0.38 — see app/agents/ta_patterns/mtf_confluence.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.config as cfg
from app.agents.ta_patterns.mtf_confluence import (
    compute_mtf_trend,
    mtf_confluence_score,
    resample_ohlcv,
)
from app.dispatcher.aggregator import SignalAggregator
from app.dispatcher.signal import (
    DecisionAction,
    Direction,
    SignalSource,
    UnifiedSignal,
)
from app.dispatcher.tier_classifier import apply_tier


def _trending_df(
    direction: int = 1,
    n: int = 120,
    base: float = 100.0,
    step: float = 0.30,
    noise: float = 0.02,
    freq: str = "10min",
) -> pd.DataFrame:
    """Build an OHLCV frame with a clean trend that triggers ADX > 20."""
    rng = np.random.default_rng(42)
    drift = direction * step
    close = base + np.cumsum(np.full(n, drift) + rng.normal(0, noise, n))
    open_ = np.concatenate([[base], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0.01, 0.05, n)
    low = np.minimum(open_, close) - rng.uniform(0.01, 0.05, n)
    volume = rng.integers(1000, 5000, n)
    idx = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def _flat_df(n: int = 120, freq: str = "10min") -> pd.DataFrame:
    """Flat df."""
    rng = np.random.default_rng(7)
    base = 100.0
    close = base + rng.normal(0, 0.05, n)
    open_ = base + rng.normal(0, 0.05, n)
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    volume = rng.integers(1000, 5000, n)
    idx = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def _make_sig(source, direction, mag=0.7, rr=2.0, ticker="SBER"):
    """Make sig."""
    return UnifiedSignal(
        source=source,
        detector="test",
        ticker=ticker,
        direction=direction,
        magnitude=mag,
        raw_confidence=mag,
        horizon_min=60,
        price=100.0,
        entry_level=100.0,
        stop_level=98.0,
        target_level=104.0,
        expected_rr=rr,
        atr=0.5,
    )


def test_score_3_of_3_agree_full_credit():
    """Test score 3 of 3 agree full credit."""
    trends = {"trend_10m": 1, "trend_60m": 1, "trend_daily": 1}
    assert mtf_confluence_score("BUY", trends) == 1.0


def test_score_2_of_3_agree():
    """Test score 2 of 3 agree."""
    trends = {"trend_10m": 1, "trend_60m": 1, "trend_daily": -1}
    assert mtf_confluence_score("BUY", trends) == 0.7


def test_score_1_of_3_agree():
    """Test score 1 of 3 agree."""
    trends = {"trend_10m": 1, "trend_60m": -1, "trend_daily": -1}
    assert mtf_confluence_score("BUY", trends) == 0.3


def test_score_counter_trend_negative():
    """Test score counter trend negative."""
    trends = {"trend_10m": -1, "trend_60m": -1, "trend_daily": -1}
    assert mtf_confluence_score("BUY", trends) == -0.5


def test_score_sell_against_uptrend():
    """Test score sell against uptrend."""
    trends = {"trend_10m": 1, "trend_60m": 1, "trend_daily": 1}
    assert mtf_confluence_score("SELL", trends) == -0.5


def test_score_all_flat_neutral():
    """All flat trends → no opinion, return 1.0 (multiplier untouched)."""
    trends = {"trend_10m": 0, "trend_60m": 0, "trend_daily": 0}
    assert mtf_confluence_score("BUY", trends) == 1.0


def test_score_two_flat_one_agree():
    """1 opinion, agrees → full credit; 0 opinions remaining."""
    trends = {"trend_10m": 1, "trend_60m": 0, "trend_daily": 0}
    assert mtf_confluence_score("BUY", trends) == 1.0


def test_score_two_flat_one_disagree():
    """Test score two flat one disagree."""
    trends = {"trend_10m": -1, "trend_60m": 0, "trend_daily": 0}
    assert mtf_confluence_score("BUY", trends) == -0.5


def test_score_neutral_direction_passes():
    """Test score neutral direction passes."""
    trends = {"trend_10m": 1, "trend_60m": 1, "trend_daily": 1}
    assert mtf_confluence_score("NEUTRAL", trends) == 1.0


def test_compute_mtf_trend_uptrend():
    """Test compute mtf trend uptrend."""
    df = _trending_df(direction=1)
    trends = compute_mtf_trend(df, df, df)
    assert trends["trend_10m"] == 1
    assert trends["trend_60m"] == 1
    assert trends["trend_daily"] == 1


def test_compute_mtf_trend_downtrend():
    """Test compute mtf trend downtrend."""
    df = _trending_df(direction=-1)
    trends = compute_mtf_trend(df, df, df)
    assert trends["trend_10m"] == -1
    assert trends["trend_60m"] == -1


def test_compute_mtf_trend_none_dfs():
    """Test compute mtf trend none dfs."""
    trends = compute_mtf_trend(None, None, None)
    assert trends == {"trend_10m": 0, "trend_60m": 0, "trend_daily": 0}


def test_compute_mtf_trend_short_df():
    """Short DF → safe default 0 (no opinion)."""
    df = _trending_df(direction=1, n=10)
    trends = compute_mtf_trend(df, df, df)
    assert trends["trend_10m"] == 0


def test_compute_mtf_trend_flat_market():
    """Test compute mtf trend flat market."""
    df = _flat_df()
    trends = compute_mtf_trend(df, df, df)
    assert trends == {"trend_10m": 0, "trend_60m": 0, "trend_daily": 0}


def test_resample_to_60min_reduces_rows():
    """Test resample to 60min reduces rows."""
    df = _trending_df(direction=1, n=180, freq="10min")
    out = resample_ohlcv(df, "60min")
    assert len(out) < len(df)
    assert set(["open", "high", "low", "close"]).issubset(out.columns)


def test_resample_handles_timestamp_column():
    """Test resample handles timestamp column."""
    df = _trending_df(direction=1, n=60)
    df = df.reset_index().rename(columns={"index": "timestamp"})
    out = resample_ohlcv(df, "60min")
    assert hasattr(out, "columns")


@pytest.mark.asyncio
async def test_aggregator_no_mtf_when_dfs_missing(monkeypatch):
    """With MTF on but no DFs supplied → no veto, no magnitude change."""
    monkeypatch.setattr(cfg, "MTF_CONFLUENCE_ENABLED", True)
    agg = SignalAggregator()
    sigs = [_make_sig(SignalSource.TA, Direction.BUY, mag=0.7)]
    d = await agg.aggregate("SBER", "c1", sigs)
    apply_tier(d)
    assert d.action == DecisionAction.EXECUTE


@pytest.mark.asyncio
async def test_aggregator_mtf_full_confluence_buy(monkeypatch):
    """Test aggregator mtf full confluence buy."""
    monkeypatch.setattr(cfg, "MTF_CONFLUENCE_ENABLED", True)
    agg = SignalAggregator()
    df_up = _trending_df(direction=1)
    sigs = [_make_sig(SignalSource.TA, Direction.BUY, mag=0.6)]
    d = await agg.aggregate(
        "SBER",
        "c1",
        sigs,
        df_10m=df_up,
        df_60m=df_up,
        df_daily=df_up,
    )
    apply_tier(d)
    assert d.action == DecisionAction.EXECUTE
    assert "mtf_score=1.00" in d.rationale


@pytest.mark.asyncio
async def test_aggregator_mtf_counter_trend_veto(monkeypatch):
    """BUY signal against a downtrend on all 3 timeframes → VETO."""
    monkeypatch.setattr(cfg, "MTF_CONFLUENCE_ENABLED", True)
    agg = SignalAggregator()
    df_down = _trending_df(direction=-1)
    sigs = [_make_sig(SignalSource.TA, Direction.BUY, mag=0.8)]
    d = await agg.aggregate(
        "SBER",
        "c1",
        sigs,
        df_10m=df_down,
        df_60m=df_down,
        df_daily=df_down,
    )
    apply_tier(d)
    assert d.action == DecisionAction.VETO
    assert "MTF" in d.rationale or "mtf" in d.rationale.lower()


@pytest.mark.asyncio
async def test_aggregator_mtf_disabled_no_effect(monkeypatch):
    """When MTF disabled, counter-trend signal still EXECUTEs."""
    monkeypatch.setattr(cfg, "MTF_CONFLUENCE_ENABLED", False)
    agg = SignalAggregator()
    df_down = _trending_df(direction=-1)
    sigs = [_make_sig(SignalSource.TA, Direction.BUY, mag=0.8)]
    d = await agg.aggregate(
        "SBER",
        "c1",
        sigs,
        df_10m=df_down,
        df_60m=df_down,
        df_daily=df_down,
    )
    apply_tier(d)
    assert d.action == DecisionAction.EXECUTE


@pytest.mark.asyncio
async def test_aggregator_mtf_partial_confluence_reduces_magnitude(monkeypatch):
    """1-of-3 agreement reduces magnitude by 0.3x but still EXECUTEs."""
    monkeypatch.setattr(cfg, "MTF_CONFLUENCE_ENABLED", True)
    agg = SignalAggregator()
    df_up = _trending_df(direction=1)
    df_down = _trending_df(direction=-1)
    sigs = [_make_sig(SignalSource.TA, Direction.BUY, mag=0.9)]
    d = await agg.aggregate(
        "SBER",
        "c1",
        sigs,
        df_10m=df_up,
        df_60m=df_down,
        df_daily=df_down,
    )
    apply_tier(d)
    assert d.action == DecisionAction.EXECUTE
    assert d.combined_magnitude <= 0.9 * 0.3 + 1e-6
