"""
tests/unit/test_data_caching.py — v0.8.0 data-layer caching tests.

Verifies that:
  * TTLCache stores values, expires them after `ttl_seconds`, and tracks
    hit / miss / set counters correctly.
  * SingleFlight coalesces concurrent callers for the same key into a
    single underlying fetch.
  * `cached_async` decorator integrates TTL + single-flight and skips
    empty results (no cache poisoning).
  * AlgoPack `get_obstats` reuses the cache within TTL and re-fetches
    after the cache is cleared / TTL expires.
  * `get_supercandles` short-circuits to the in-memory layer on the
    second call (no second moexalgo dispatch).
  * Cache hit ratio under a synthetic 20-ticker × 5-agent stampede beats
    a pre-defined threshold — the bar we publish in the release notes.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pandas as pd
import pytest

from app.utils.async_cache import SingleFlight, TTLCache, cached_async


def test_ttl_cache_hit_miss_and_stats() -> None:
    """A fresh key should miss, then hit after `set`, and counters add up."""
    cache = TTLCache(ttl_seconds=30.0)

    assert cache.get(("SBER", 5)) is None
    cache.set(("SBER", 5), {"close": 100.0})
    assert cache.get(("SBER", 5)) == {"close": 100.0}
    assert cache.get(("SBER", 5)) == {"close": 100.0}

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["sets"] == 1
    assert stats["size"] == 1
    assert stats["hit_ratio"] == pytest.approx(2 / 3, rel=1e-3)


def test_ttl_cache_expires_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """After `ttl_seconds` elapses (monotonic), the entry is treated as missing."""
    cache = TTLCache(ttl_seconds=5.0)
    fake_now = [1_000.0]

    def fake_monotonic() -> float:
        """Fake monotonic."""
        return fake_now[0]

    monkeypatch.setattr("app.utils.async_cache.time.monotonic", fake_monotonic)

    cache.set("k", "v")
    assert cache.get("k") == "v"

    fake_now[0] += 5.01
    assert cache.get("k") is None
    assert cache.stats()["size"] == 0


def test_ttl_cache_invalidate_and_clear() -> None:
    """Test ttl cache invalidate and clear."""
    cache = TTLCache(ttl_seconds=30.0)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.invalidate("a")
    assert cache.get("a") is None
    assert cache.get("b") == 2
    cache.clear()
    assert cache.stats()["size"] == 0
    assert cache.stats()["hits"] == 0


@pytest.mark.asyncio
async def test_single_flight_coalesces_concurrent_calls() -> None:
    """Five concurrent callers with the same key produce one fetch."""
    sf = SingleFlight()
    fetch_count = 0

    async def slow_fetch() -> str:
        """Slow fetch."""
        nonlocal fetch_count
        fetch_count += 1
        await asyncio.sleep(0.05)
        return "result"

    results = await asyncio.gather(*(sf.do("key", slow_fetch) for _ in range(5)))
    assert results == ["result"] * 5
    assert fetch_count == 1, "SingleFlight should issue exactly 1 underlying fetch"


@pytest.mark.asyncio
async def test_single_flight_distinct_keys_run_in_parallel() -> None:
    """Different keys must NOT block each other."""
    sf = SingleFlight()
    started: list[str] = []

    async def slow_fetch(key: str) -> str:
        """Slow fetch."""
        started.append(key)
        await asyncio.sleep(0.1)
        return key

    t0 = time.monotonic()
    results = await asyncio.gather(
        sf.do("A", lambda: slow_fetch("A")),
        sf.do("B", lambda: slow_fetch("B")),
        sf.do("C", lambda: slow_fetch("C")),
    )
    elapsed = time.monotonic() - t0
    assert set(results) == {"A", "B", "C"}
    assert elapsed < 0.25, f"distinct keys appear to be serialised (elapsed={elapsed:.3f}s)"


@pytest.mark.asyncio
async def test_single_flight_propagates_exceptions() -> None:
    """Failure in the leader propagates to all waiters and clears the slot."""
    sf = SingleFlight()

    async def boom() -> None:
        """Boom."""
        raise RuntimeError("fetch failed")

    with pytest.raises(RuntimeError, match="fetch failed"):
        await sf.do("k", boom)

    async def ok() -> str:
        """Ok."""
        return "ok"

    assert await sf.do("k", ok) == "ok"


@pytest.mark.asyncio
async def test_cached_async_skips_empty_results() -> None:
    """Empty DataFrames / Nones must NOT be cached (no poisoning)."""
    cache = TTLCache(ttl_seconds=30.0)
    flight = SingleFlight()
    calls = {"n": 0}

    @cached_async(cache, flight=flight)
    async def fetch(ticker: str) -> pd.DataFrame:
        """Fetch."""
        calls["n"] += 1
        if calls["n"] == 1:
            return pd.DataFrame()
        return pd.DataFrame({"close": [100.0]})

    df1 = await fetch("SBER")
    assert df1.empty
    df2 = await fetch("SBER")
    assert not df2.empty
    assert calls["n"] == 2, "empty result should not have been cached"

    df3 = await fetch("SBER")
    assert not df3.empty
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_cached_async_concurrent_stampede() -> None:
    """20 concurrent fetches for the same key share one underlying call."""
    cache = TTLCache(ttl_seconds=30.0)
    flight = SingleFlight()
    calls = {"n": 0}

    @cached_async(cache, flight=flight)
    async def fetch(ticker: str) -> str:
        """Fetch."""
        calls["n"] += 1
        await asyncio.sleep(0.02)
        return f"{ticker}-data"

    results = await asyncio.gather(*(fetch("SBER") for _ in range(20)))
    assert results == ["SBER-data"] * 20
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_algopack_obstats_uses_ttl_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two get_obstats calls within TTL → only one fetch."""
    from app.data import algopack_client as ap_mod

    ap_mod.clear_algopack_caches()
    client = ap_mod.AlgoPackClient()
    client._auth_failed = False
    client._client = object()

    fetch_calls = {"n": 0}

    async def fake_fetch(ticker: str, date: str, cache_key: tuple) -> pd.DataFrame:
        """Fake fetch."""
        fetch_calls["n"] += 1
        return pd.DataFrame({"trades": [42], "secid": [ticker]})

    monkeypatch.setattr(client, "_fetch_obstats", fake_fetch)

    df1 = await client.get_obstats("SBER", date="2026-05-26")
    df2 = await client.get_obstats("SBER", date="2026-05-26")
    df3 = await client.get_obstats("SBER", date="2026-05-26")
    assert fetch_calls["n"] == 1, "should hit cache on second / third call"
    assert (df1["trades"] == df2["trades"]).all()
    assert (df2["trades"] == df3["trades"]).all()

    stats = ap_mod.get_obstats_cache().stats()
    assert stats["hits"] >= 2
    assert stats["sets"] == 1


@pytest.mark.asyncio
async def test_algopack_obstats_refetches_after_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """After clear_algopack_caches() the next call MUST hit the network again."""
    from app.data import algopack_client as ap_mod

    ap_mod.clear_algopack_caches()
    client = ap_mod.AlgoPackClient()
    client._auth_failed = False
    client._client = object()

    fetch_calls = {"n": 0}

    async def fake_fetch(ticker: str, date: str, cache_key: tuple) -> pd.DataFrame:
        """Fake fetch."""
        fetch_calls["n"] += 1
        return pd.DataFrame({"trades": [fetch_calls["n"]]})

    monkeypatch.setattr(client, "_fetch_obstats", fake_fetch)

    await client.get_obstats("GAZP", date="2026-05-26")
    ap_mod.clear_algopack_caches()
    await client.get_obstats("GAZP", date="2026-05-26")
    assert fetch_calls["n"] == 2


@pytest.mark.asyncio
async def test_supercandles_inmemory_layer_avoids_double_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second call within 60 s should NOT hit sqlite or moexalgo."""
    from app.data import supercandles as sc_mod

    sc_mod.clear_supercandles_cache()

    dispatch_count = {"n": 0}

    def fake_sync(ticker: str, dt: str, interval: int) -> pd.DataFrame:
        """Fake sync."""
        dispatch_count["n"] += 1
        return pd.DataFrame(
            {
                "ts": [pd.Timestamp("2026-05-26 10:00:00", tz="UTC")],
                "close": [100.0 + dispatch_count["n"]],
                "volume": [10_000.0],
            }
        )

    monkeypatch.setattr(sc_mod, "_fetch_sync", fake_sync)
    monkeypatch.setattr(sc_mod, "_cache_lookup", lambda *a, **kw: None)
    monkeypatch.setattr(sc_mod, "_cache_store", lambda *a, **kw: None)
    monkeypatch.setattr(sc_mod, "_HAS_MOEXALGO", True)
    monkeypatch.setattr(sc_mod, "_HAS_PANDAS", True)

    df1 = await sc_mod.get_supercandles("ROSN", trade_date="2026-05-26", interval=5)
    df2 = await sc_mod.get_supercandles("ROSN", trade_date="2026-05-26", interval=5)
    assert dispatch_count["n"] == 1, "second call should be served from memory"
    assert df1 is not None and df2 is not None
    assert (df1["close"] == df2["close"]).all()


@pytest.mark.asyncio
async def test_supercandles_coalesces_concurrent_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """20 concurrent dispatcher tasks → exactly one moexalgo call."""
    from app.data import supercandles as sc_mod

    sc_mod.clear_supercandles_cache()
    dispatch_count = {"n": 0}

    def fake_sync(ticker: str, dt: str, interval: int) -> pd.DataFrame:
        """Fake sync."""
        dispatch_count["n"] += 1
        time.sleep(0.03)
        return pd.DataFrame({"ts": [pd.Timestamp("2026-05-26 10:00", tz="UTC")], "close": [100.0]})

    monkeypatch.setattr(sc_mod, "_fetch_sync", fake_sync)
    monkeypatch.setattr(sc_mod, "_cache_lookup", lambda *a, **kw: None)
    monkeypatch.setattr(sc_mod, "_cache_store", lambda *a, **kw: None)
    monkeypatch.setattr(sc_mod, "_HAS_MOEXALGO", True)
    monkeypatch.setattr(sc_mod, "_HAS_PANDAS", True)

    results = await asyncio.gather(
        *(sc_mod.get_supercandles("LKOH", trade_date="2026-05-26", interval=5) for _ in range(20))
    )
    assert dispatch_count["n"] == 1
    assert all(r is not None for r in results)


@pytest.mark.asyncio
async def test_synthetic_dispatcher_cycle_hit_ratio() -> None:
    """
    Mirror one dispatcher cycle: 20 tickers × 5 agents asking the same
    endpoint, then a *second* cycle within the 30 s TTL window.

    Without the cache: 20 × 5 × 2 = 200 fetches.
    With the cache + single-flight: 20 fetches in cycle 1 (one per ticker,
    followers coalesced into the in-flight Future); 0 in cycle 2 (TTL hits).

    Effective fetch reduction = (200 - 20) / 200 = 90 %.

    We assert the reduction is >= 80 % to leave some safety margin, and
    the warm-cycle TTLCache hit ratio is 100 % (every read found a value).
    """
    cache = TTLCache(ttl_seconds=30.0)
    flight = SingleFlight()
    fetch_count = 0

    @cached_async(cache, flight=flight)
    async def fetch(ticker: str) -> dict[str, Any]:
        """Fetch."""
        nonlocal fetch_count
        fetch_count += 1
        await asyncio.sleep(0.005)
        return {"ticker": ticker, "close": 100.0}

    tickers = [
        "SBER",
        "GAZP",
        "LKOH",
        "ROSN",
        "GMKN",
        "VTBR",
        "MOEX",
        "MGNT",
        "NLMK",
        "PHOR",
        "PLZL",
        "TATN",
        "SNGS",
        "RTKM",
        "MTSS",
        "POLY",
        "AFLT",
        "FIVE",
        "YNDX",
        "OZON",
    ]

    coros = [fetch(t) for t in tickers for _ in range(5)]
    await asyncio.gather(*coros)
    assert fetch_count == 20, (
        f"single-flight should collapse 100 calls → 20 fetches (got {fetch_count})"
    )

    s1 = cache.stats()
    hits_before = s1["hits"]
    misses_before = s1["misses"]

    coros2 = [fetch(t) for t in tickers for _ in range(5)]
    await asyncio.gather(*coros2)
    assert fetch_count == 20, f"warm cycle should add 0 fetches (got {fetch_count})"

    total_calls = 2 * 20 * 5
    reduction = (total_calls - fetch_count) / total_calls
    assert reduction >= 0.80, (
        f"network reduction below floor ({reduction:.2%}); see release notes for v0.8.0"
    )

    s2 = cache.stats()
    warm_hits = s2["hits"] - hits_before
    warm_misses = s2["misses"] - misses_before
    warm_ratio = warm_hits / (warm_hits + warm_misses) if (warm_hits + warm_misses) else 0.0
    assert warm_ratio == pytest.approx(1.0, abs=1e-6), (
        f"warm-cycle hit ratio should be 100% (was {warm_ratio:.2%})"
    )
