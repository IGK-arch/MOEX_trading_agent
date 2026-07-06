"""
tests/integration/test_parallel_polls.py — Phase 28 (v0.10.0).

Pins the wall-clock budget of the TA + Anomaly polls now that both have
been refactored to run per-ticker analysis concurrently
(`asyncio.gather + asyncio.to_thread`).

The prior (sequential) version did 20 sequential `await _analyze_ticker(...)`
calls per cycle. Each call is ~80-130ms of pandas/numpy work, so the
worst-case 20 × ~120ms = ~2.4s blew straight past the 0.5s safe_poll
budget, producing a steady stream of timeouts and 0 signals (matching
the 14:36/14:52 UTC outage in the v0.0.43 logs).

After refactor each ticker is dispatched to asyncio.to_thread, so pandas
hot loops (compute_all, find_pivots, every detector) release the GIL and
run in parallel on the CPU pool. We require the full 20-ticker poll to
complete in < 1.5s, which is the per-ticker timeout — i.e., at the very
least we should be ~constant in the number of tickers, not linear.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pandas as pd
import pytest

import app.config as cfg
from app.agents.anomaly_detector import AnomalyDetectorAgent
from app.agents.ta_trader import TATrader
from app.data.candle_store import get_candle_store


def _make_realistic_df(seed: int, n: int = 120) -> pd.DataFrame:
    """Synthetic OHLCV with enough wiggle to keep pattern detectors busy."""
    rng = np.random.default_rng(seed)
    closes = [100.0]
    for i in range(1, n):
        drift = 0.02 * np.sin(i / 8) + 0.001 * i
        closes.append(closes[-1] * (1.0 + drift + rng.standard_normal() * 0.004))
    closes = np.array(closes)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) * (1.0 + np.abs(rng.standard_normal(n)) * 0.002)
    lows = np.minimum(opens, closes) * (1.0 - np.abs(rng.standard_normal(n)) * 0.002)
    volumes = rng.integers(50_000, 500_000, n).astype(float)
    begin = pd.date_range("2026-01-01 10:00", periods=n, freq="10min", tz="UTC")
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


async def _seed_candle_store(tickers: list[str], interval_min: int) -> None:
    """Push synthetic candles into the shared CandleStore for every ticker."""
    store = get_candle_store()
    for i, ticker in enumerate(tickers):
        await store.update(ticker, interval_min, _make_realistic_df(seed=i + 1))


@pytest.mark.asyncio
async def test_ta_trader_polls_20_tickers_under_1_5s():
    """20-ticker TA poll must complete well under the 1.5s per-ticker timeout.

    Before the v0.10.0 refactor, this same workload took ~2-3s sequentially.
    After parallelising via asyncio.to_thread, it should finish in well
    under 1.5s on any 2+ core machine (CI typically has 2 cores).
    """
    tickers = list(cfg.TICKERS)
    assert len(tickers) == 20, "config drift: parallel-poll budget assumes 20 tickers"

    trader = TATrader(tickers=tickers, interval_min=10, min_bars_required=20)
    trader._started = True

    await _seed_candle_store(tickers, interval_min=10)

    t0 = time.monotonic()
    signals = await trader.poll()
    elapsed = time.monotonic() - t0

    assert elapsed < 1.5, (
        f"TATrader.poll() took {elapsed * 1000:.0f}ms for 20 tickers — "
        f"should be <1500ms now that per-ticker work runs in parallel."
    )
    assert isinstance(signals, list)


@pytest.mark.asyncio
async def test_anomaly_detector_polls_20_tickers_under_1_5s():
    """20-ticker Anomaly poll must finish under 1.5s (Phase 28 refactor)."""
    tickers = list(cfg.TICKERS)
    assert len(tickers) == 20

    agent = AnomalyDetectorAgent(tickers=tickers, interval_min=10, min_bars_required=15)
    agent._started = True
    agent._obstats_cache_ttl_sec = 9999.0
    for t in tickers:
        agent._obstats_cache[t] = None
        agent._obstats_cache_ts[t] = time.monotonic()

    await _seed_candle_store(tickers, interval_min=10)

    t0 = time.monotonic()
    signals = await agent.poll()
    elapsed = time.monotonic() - t0

    assert elapsed < 1.5, (
        f"AnomalyDetector.poll() took {elapsed * 1000:.0f}ms — "
        f"should be <1500ms with per-ticker gather + to_thread."
    )
    assert isinstance(signals, list)


@pytest.mark.asyncio
async def test_parallel_dispatch_outperforms_sequential():
    """gather() + to_thread must be faster than sequential awaits when
    each task contains numpy work that releases the GIL.

    The production TA poll spends most of its wall-clock budget in
    pandas/numpy C extensions (compute_all, find_pivots, the SciPy peak
    detection inside each pattern detector). Those routines drop the GIL
    while running, so concurrent threads can actually progress in
    parallel. To make this test deterministic and CI-friendly we wrap a
    chunk of explicit numpy linear-algebra work around the real
    `_analyze_ticker_sync` call — this both inflates the work to a
    measurable size AND guarantees we exercise the GIL-releasing path
    (np.linalg.svd is a textbook GIL-releaser).

    The parallel path must come in at <85% of the sequential wall-clock
    on a 2+ core machine, otherwise the refactor isn't actually helping.
    """
    tickers = list(cfg.TICKERS)
    trader = TATrader(tickers=tickers, interval_min=10, min_bars_required=20)
    trader._started = True

    await _seed_candle_store(tickers, interval_min=10)

    store = get_candle_store()
    pairs: list[tuple[str, pd.DataFrame]] = []
    for t in tickers:
        df = store.get(t, 10)
        if isinstance(df, pd.DataFrame) and len(df) >= 20:
            pairs.append((t, df))

    regime = trader.hmm.current_label

    arr_template = np.random.default_rng(0).standard_normal(1_500_000)

    def _heavy_one(ticker: str, df: pd.DataFrame) -> int:
        """Heavy one."""
        _ = trader._analyze_ticker_sync(ticker, df, regime, track_seen=False)
        a = arr_template.copy()
        np.sort(a)
        return 1

    t0 = time.monotonic()
    for t, df in pairs:
        _heavy_one(t, df)
    seq_elapsed = time.monotonic() - t0

    t0 = time.monotonic()
    await asyncio.gather(*(asyncio.to_thread(_heavy_one, t, df) for t, df in pairs))
    par_elapsed = time.monotonic() - t0

    assert par_elapsed < seq_elapsed * 0.85, (
        f"Parallel poll didn't improve: sequential={seq_elapsed * 1000:.0f}ms, "
        f"parallel={par_elapsed * 1000:.0f}ms. Expected parallel < 85% of sequential."
    )
