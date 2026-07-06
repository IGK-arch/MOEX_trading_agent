"""
tests/unit/test_anomaly_timeout.py — regression guard for the 14:36/14:52 UTC
production outage where AnomalyDetector hit 100% timeout under the 500ms cap.

The fix (Phase 28 / v0.5.0):
1. Concurrent obstats fetch + concurrent per-ticker analysis via asyncio.gather.
2. Per-ticker hard timeout (1.5s) so one stuck ticker can't starve the cycle.
3. Global POLL_TIMEOUT_SECS bumped from 0.5 → 5.0 (still well below the 30s
   dispatcher cycle).

This test verifies poll() completes in under 4 seconds on a 20-ticker
synthetic workload, even with a deliberately slow obstats fetch.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pandas as pd
import pytest

from app.agents.anomaly_detector import AnomalyDetectorAgent


def _make_candles(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Realistic-shaped 10m OHLCV with enough bars for all detectors."""
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + np.abs(rng.standard_normal(n)) * 0.3
    lows = np.minimum(opens, closes) - np.abs(rng.standard_normal(n)) * 0.3
    volumes = rng.integers(50_000, 150_000, n).astype(float)
    begin = pd.date_range("2026-01-01", periods=n, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "begin": begin,
        }
    )


class _FakeCandleStore:
    """In-memory candle store — pre-loaded with synthetic OHLCV per ticker."""

    def __init__(self, tickers: list[str]) -> None:
        """Init."""
        self._df_by_ticker = {t: _make_candles(seed=i) for i, t in enumerate(tickers)}

    def get(self, ticker: str, interval_min: int) -> pd.DataFrame | None:
        """Get."""
        return self._df_by_ticker.get(ticker)


class _SlowAlgoPack:
    """Stand-in AlgoPack client where get_obstats sleeps 100ms per call.

    With the old serial code (20 × 100ms = 2 sec) this would already
    blow the 500ms budget. With the gathered fan-out the obstats step
    finishes in roughly 100ms wall-clock (well under the per-ticker
    1.5s analysis budget).
    """

    def __init__(self) -> None:
        """Init."""
        self._started = True
        self._auth_failed = False

    async def startup(self) -> None:
        """Startup."""
        self._started = True

    async def get_obstats(self, ticker: str) -> pd.DataFrame:
        """Get obstats."""
        await asyncio.sleep(0.1)
        return pd.DataFrame(
            {
                "ts": [pd.Timestamp("2026-01-01", tz="UTC")],
                "secid": [ticker],
                "imbalance_vol_bbo": [0.1],
                "vwap_b": [100.0],
            }
        )


@pytest.mark.asyncio
async def test_anomaly_poll_completes_under_4_seconds() -> None:
    """poll() must complete under 4 seconds on a 20-ticker workload."""
    tickers = [
        "LKOH",
        "SBER",
        "ROSN",
        "GAZP",
        "VTBR",
        "YDEX",
        "PLZL",
        "T",
        "NVTK",
        "X5",
        "GMKN",
        "MGNT",
        "ALRS",
        "AFLT",
        "CHMF",
        "NLMK",
        "MOEX",
        "SNGSP",
        "MTSS",
        "PIKK",
    ]

    agent = AnomalyDetectorAgent(tickers=tickers, interval_min=10, min_bars_required=15)
    agent.candle_store = _FakeCandleStore(tickers)
    agent.algopack = _SlowAlgoPack()
    agent._started = True

    start = time.monotonic()
    signals = await agent.poll()
    elapsed = time.monotonic() - start

    assert elapsed < 4.0, (
        f"AnomalyDetector.poll() took {elapsed:.2f}s — must stay under 4s "
        f"to fit comfortably under POLL_TIMEOUT_SECS=5.0. Old serial code "
        f"would take ~2s just on the obstats fan-out."
    )
    assert isinstance(signals, list)
    assert agent._poll_count == 1


@pytest.mark.asyncio
async def test_anomaly_poll_under_500ms_with_cached_obstats() -> None:
    """Once obstats is cached (TTL 30s), subsequent polls should be very fast."""
    tickers = ["SBER", "GAZP", "LKOH", "VTBR", "ROSN"]
    agent = AnomalyDetectorAgent(tickers=tickers, interval_min=10, min_bars_required=15)
    agent.candle_store = _FakeCandleStore(tickers)
    agent.algopack = _SlowAlgoPack()
    agent._started = True

    await agent.poll()

    start = time.monotonic()
    await agent.poll()
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, (
        f"Warm-cache poll took {elapsed:.2f}s — expected <500ms once "
        f"obstats is cached (TTL 30s) and only pandas work remains."
    )
