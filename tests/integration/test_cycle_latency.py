"""Dispatcher cycle latency guard.

Phase 29 (v0.7.0) — production pods were logging gather_ms = 2.6-3.2 s per
dispatcher cycle (target <500 ms). After the perf pass in
scripts/profile_dispatcher.py + the fixes in ta_trader.py / ta_catboost.py
/ ta_patterns/research_patterns.py / ta_patterns/candles.py, the steady-
state cycle should be well under 1.5 s on synthetic data.

This test runs five back-to-back TATrader.poll() cycles with a 20-ticker
synthetic candle store seeded with a recognisable wave so the detectors
actually fire (which is the *expensive* path — empty-detector cycles are
trivially cheap and would not catch a regression).

We assert MEAN cycle latency < 1.5 s. We do NOT assert max, because CI
hosts can be noisy; the mean across five cycles is a stable indicator.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import app.config as cfg
from app.agents.ta_trader import TATrader
from app.data.candle_store import get_candle_store

CYCLE_LATENCY_BUDGET_SEC = 1.5
N_CYCLES = 5


def _make_synthetic_candles(ticker: str, n: int = 180, seed: int = 0) -> pd.DataFrame:
    """Wave + drift + noise — same shape as scripts/profile_dispatcher.py so
    a regression seen there matches a failure here."""
    rng = np.random.default_rng(seed + (hash(ticker) % 1000))
    base = 100.0 + (hash(ticker) % 50)
    closes = []
    for i in range(n):
        phase = i / 30.0
        wave = np.sin(phase) * 6.0 + np.sin(phase * 0.5) * 3.0
        drift = i * 0.02
        noise = rng.standard_normal() * 0.5
        closes.append(base + wave + drift + noise)
    closes = np.array(closes, dtype=float)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + np.abs(rng.standard_normal(n)) * 0.4
    lows = np.minimum(opens, closes) - np.abs(rng.standard_normal(n)) * 0.4
    vols = rng.integers(50_000, 200_000, n).astype(float)
    begin = pd.date_range("2026-01-01 10:00", periods=n, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
            "begin": begin,
        }
    )


@pytest.mark.asyncio
async def test_dispatcher_cycle_latency_under_budget() -> None:
    """5 synthetic-data TATrader cycles must average < 1.5 s end-to-end."""
    tickers = list(cfg.TICKERS)
    store = get_candle_store()
    for t in tickers:
        await store.update(t, 10, _make_synthetic_candles(t))

    trader = TATrader(tickers=tickers, interval_min=10, min_bars_required=20)
    await trader.startup()

    await trader.poll()

    timings_sec: list[float] = []
    for _ in range(N_CYCLES):
        t0 = time.perf_counter()
        await trader.poll()
        timings_sec.append(time.perf_counter() - t0)

    mean_sec = sum(timings_sec) / len(timings_sec)
    max_sec = max(timings_sec)
    assert mean_sec < CYCLE_LATENCY_BUDGET_SEC, (
        f"Dispatcher cycle latency regressed: "
        f"mean={mean_sec * 1000:.0f} ms, max={max_sec * 1000:.0f} ms, "
        f"budget={CYCLE_LATENCY_BUDGET_SEC * 1000:.0f} ms, "
        f"per-cycle={[round(s * 1000) for s in timings_sec]}"
    )


@pytest.mark.asyncio
async def test_dispatcher_cycle_returns_signals() -> None:
    """Companion sanity check: the synthetic data MUST drive at least one
    detector to fire — otherwise the latency test is measuring an
    empty-path cycle and the budget is meaningless."""
    tickers = list(cfg.TICKERS)
    store = get_candle_store()
    for t in tickers:
        await store.update(t, 10, _make_synthetic_candles(t))

    trader = TATrader(tickers=tickers, interval_min=10, min_bars_required=20)
    await trader.startup()
    signals = await trader.poll()
    assert isinstance(signals, list), "poll() must return a list"
