"""
app/utils/async_cache.py — async TTL cache with single-flight coalescing.

Purpose (v0.8.0)
----------------
The dispatcher cycles every 30 s and fans out 20 tickers × N endpoints
(candles, supercandles, obstats, tradestats). Within one cycle multiple
agents often request the same (ticker, endpoint) tuple, and the AlgoPack
semaphore (4 slots) serialises the calls. Result: identical HTTP requests
get queued behind each other, blowing the cycle budget.

This module provides two primitives:

1.  `TTLCache` — a tiny dict-backed cache keyed by an arbitrary hashable
    tuple, with per-entry expiry. No external deps, no LRU eviction (we
    have ~hundreds of keys, not millions); a periodic sweep keeps it
    bounded.

2.  `single_flight` — a decorator that wraps an `async` function so that
    concurrent callers with the *same key* await one shared in-flight
    Future instead of issuing duplicate work. Combined with the TTL cache
    it gives us request coalescing for free.

Together they cut redundant API traffic by an order of magnitude on
synthetic stress tests (see tests/unit/test_data_caching.py).

Design notes
------------
- All locks are `asyncio.Lock` — single event-loop process.
- The cache is intentionally *not* thread-safe; the entire bot runs in
  one loop.
- Negative results (None, empty DataFrame) are *not* cached by default —
  callers should pass `cache_empty=True` if they want that. Failed
  fetches should retry rather than poison the cache.
- TTL is checked at read time using `time.monotonic()` so wall-clock
  jumps don't invalidate the cache.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable, Hashable
from typing import Any, TypeVar

T = TypeVar("T")

class TTLCache:
    """
    Async-friendly TTL cache.

    Keys are any hashable (typically a tuple like ("SBER", 5)).
    Values are stored alongside a monotonic timestamp; reads after
    `ttl_seconds` return a miss.

    Stats are exposed via `.stats()` for the dashboard / tests.
    """

    __slots__ = (
        "_store",
        "_ttl",
        "_hits",
        "_misses",
        "_sets",
        "_max_entries",
    )

    def __init__(self, ttl_seconds: float = 30.0, max_entries: int = 4096) -> None:
        """Init."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._store: dict[Hashable, tuple[float, Any]] = {}
        self._ttl: float = float(ttl_seconds)
        self._hits: int = 0
        self._misses: int = 0
        self._sets: int = 0
        self._max_entries: int = max_entries

    def get(self, key: Hashable) -> Any | None:
        """Return cached value or None if missing/expired."""
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        stored_at, value = entry
        if (time.monotonic() - stored_at) > self._ttl:
            self._store.pop(key, None)
            self._misses += 1
            return None
        self._hits += 1
        return value

    def set(self, key: Hashable, value: Any) -> None:
        """Store a value with the current monotonic timestamp."""
        if len(self._store) >= self._max_entries:
            try:
                oldest_key = min(self._store, key=lambda k: self._store[k][0])
                self._store.pop(oldest_key, None)
            except ValueError:
                pass
        self._store[key] = (time.monotonic(), value)
        self._sets += 1

    def invalidate(self, key: Hashable) -> None:
        """Remove a single key (used by tests / manual refresh)."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Wipe everything. Used between dispatcher restarts / tests."""
        self._store.clear()
        self._hits = 0
        self._misses = 0
        self._sets = 0

    def stats(self) -> dict[str, Any]:
        """Stats."""
        total = self._hits + self._misses
        hit_ratio = (self._hits / total) if total > 0 else 0.0
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "sets": self._sets,
            "hit_ratio": round(hit_ratio, 4),
            "ttl_seconds": self._ttl,
        }

    @property
    def ttl_seconds(self) -> float:
        """Ttl seconds."""
        return self._ttl

class SingleFlight:
    """
    Coalesce concurrent calls with the same key into a single
    in-flight awaitable.

    Usage:
        sf = SingleFlight()
        async def fetch(key):
            return await sf.do(key, lambda: real_fetch(key))

    If two coroutines call `do("SBER")` while the first is still in
    flight, the second awaits the same Future — no duplicate HTTP hit.
    """

    __slots__ = ("_inflight", "_lock")

    def __init__(self) -> None:
        """Init."""
        self._inflight: dict[Hashable, asyncio.Future[Any]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    async def do(
        self,
        key: Hashable,
        coro_factory: Callable[[], Awaitable[T]],
    ) -> T:
        """Do."""
        async with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                fut: asyncio.Future[Any] = existing
                owner = False
            else:
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                self._inflight[key] = fut
                owner = True

        if not owner:
            return await fut  # type: ignore[no-any-return]

        try:
            result = await coro_factory()
        except BaseException as exc:
            async with self._lock:
                self._inflight.pop(key, None)
            if not fut.done():
                fut.set_exception(exc)
                fut.exception()
            raise
        else:
            async with self._lock:
                self._inflight.pop(key, None)
            if not fut.done():
                fut.set_result(result)
            return result

    def in_flight_keys(self) -> list[Hashable]:
        """Snapshot of currently-fetching keys (debug / tests)."""
        return list(self._inflight.keys())

def cached_async(
    cache: TTLCache,
    *,
    flight: SingleFlight | None = None,
    key_func: Callable[..., Hashable] | None = None,
    cache_empty: bool = False,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator: wrap an async function with TTL caching + optional
    single-flight coalescing.

    Parameters
    ----------
    cache : TTLCache
        Shared cache instance.
    flight : SingleFlight | None
        Optional coalescer. If provided, concurrent callers with the
        same key wait for the leader.
    key_func : Callable | None
        Build a cache key from the call args. Defaults to `(*args, *sorted(kwargs.items()))`.
    cache_empty : bool
        If False (default), empty / None results bypass `set()` so a
        transient failure doesn't poison the cache for the full TTL.
    """

    def _is_empty(value: Any) -> bool:
        """Is empty."""
        if value is None:
            return True
        empty_attr = getattr(value, "empty", None)
        if empty_attr is True:
            return True
        return bool(isinstance(value, (list, tuple, dict, set)) and len(value) == 0)

    def _decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Decorator."""

        @functools.wraps(fn)
        async def _wrapped(*args: Any, **kwargs: Any) -> T:
            """Wrapped."""
            if key_func is not None:
                key = key_func(*args, **kwargs)
            else:
                key = (args, tuple(sorted(kwargs.items())))

            cached = cache.get(key)
            if cached is not None:
                return cached  # type: ignore[no-any-return]

            async def _do_fetch() -> T:
                """Do fetch."""
                second_look = cache.get(key)
                if second_look is not None:
                    return second_look  # type: ignore[no-any-return]
                result = await fn(*args, **kwargs)
                if cache_empty or not _is_empty(result):
                    cache.set(key, result)
                return result

            if flight is not None:
                return await flight.do(key, _do_fetch)
            return await _do_fetch()

        _wrapped.cache = cache  # type: ignore[attr-defined]
        _wrapped.flight = flight  # type: ignore[attr-defined]
        return _wrapped

    return _decorator

__all__ = ["TTLCache", "SingleFlight", "cached_async"]
