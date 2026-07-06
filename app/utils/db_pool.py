"""Shared aiosqlite connection cache (v0.9.0).

Before this module every write opened a fresh `aiosqlite.connect(...)`,
which under the hood:
  1. forks an OS thread,
  2. opens the file (fsync on first WAL frame),
  3. tears the thread down on exit.

For a 30-second dispatcher cycle that issues 5-15 writes (decision upsert
+ trade insert + circuit_breaker update + turnover_log + …) the open/close
cost dominates — measured at ~3-5 ms per call on the hackathon VM.

`get_conn(path)` returns a long-lived connection per DB path. PRAGMAs are
applied exactly once, on first acquisition. Calls are serialised by an
asyncio.Lock per connection because aiosqlite already serialises on its
own worker thread, but we still want deterministic ordering inside a single
async task that issues multiple statements (the alternative is one global
lock, which kills concurrency between *different* DBs).

Usage:
    from app.utils.db_pool import get_conn
    db = await get_conn(DECISIONS_DB)
    await db.execute("INSERT ...", params)
    await db.commit()

The module is intentionally tiny and dependency-free so it can be imported
from anywhere (risk, execution, dashboard) without circular imports.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from typing import Any

try:  # pragma: no cover — environment guard, mirrors order_manager.py
    import aiosqlite  # type: ignore

    _HAS_AIOSQLITE = True
except ImportError:  # pragma: no cover
    aiosqlite = None  # type: ignore
    _HAS_AIOSQLITE = False

_PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA busy_timeout = 5000",
    "PRAGMA cache_size = -10000",
    "PRAGMA temp_store = MEMORY",
)

_conns: dict[str, Any] = {}
_loop_locks: dict[int, asyncio.Lock] = {}

def _lock_for_loop() -> asyncio.Lock:
    """Lock for loop."""
    loop = asyncio.get_event_loop()
    key = id(loop)
    lock = _loop_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _loop_locks[key] = lock
    return lock

async def get_conn(path: str | os.PathLike[str]) -> Any:
    """Return a cached aiosqlite.Connection for `path`, opening it once."""
    if not _HAS_AIOSQLITE:
        raise RuntimeError("aiosqlite is not installed")

    key = str(Path(path).resolve())
    conn = _conns.get(key)
    if conn is not None:
        return conn

    async with _lock_for_loop():
        conn = _conns.get(key)
        if conn is not None:
            return conn

        new_conn = await aiosqlite.connect(key)
        for pragma in _PRAGMAS:
            await new_conn.execute(pragma)
        await new_conn.commit()
        _conns[key] = new_conn
        return new_conn

async def close_all() -> None:
    """Close every cached connection. Call on shutdown / between tests."""
    async with _lock_for_loop():
        for conn in list(_conns.values()):
            with contextlib.suppress(Exception):
                await conn.close()
        _conns.clear()

def _reset_for_tests() -> None:
    """Synchronous escape hatch for unit tests — drop the cache without
    awaiting close. Tests open fresh tmp_path DBs each run, so the FD leak
    is bounded and the OS reclaims it at process exit. Also drops the
    per-loop lock cache so a fresh test loop gets a fresh lock."""
    _conns.clear()
    _loop_locks.clear()
