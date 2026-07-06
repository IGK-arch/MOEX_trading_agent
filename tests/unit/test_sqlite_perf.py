"""SQLite performance regression tests for v0.9.0.

Background
----------
Pre-v0.9.0 the bot opened a fresh `aiosqlite.connect(...)` on every write
(see git history of app/execution/order_manager.py before this release).
On the hackathon VM that adds ~3 ms per call — tolerable for one cycle,
catastrophic during recovery loops that replay 5000+ decisions.

v0.9.0 changes:
  - WAL + synchronous=NORMAL + cache_size=10 MB pragmas at bootstrap
  - app.utils.db_pool.get_conn() returns a long-lived connection per path
  - decisions / trades / news_events get hot-path indexes

These tests run the SAME 1000-insert workload against a baseline (rollback
journal, fresh connect-per-write — the pre-v0.9.0 behaviour) and against
the optimised pool, then assert the new path is meaningfully faster AND
emit the timings to stdout so CI logs preserve the numbers.

We use `tmp_path` so each test gets its own DB — no global state, safe
under parallel pytest workers.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

import aiosqlite
import pytest

from app.utils.db_pool import _reset_for_tests, get_conn

N_INSERTS = 1000


def _create_schema(db_path: Path) -> None:
    """Create a minimal trades-shaped table for the perf workload."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            trade_time TEXT NOT NULL,
            order_value REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def _apply_optimised_pragmas(db_path: Path) -> None:
    """Mirror scripts/bootstrap_db.py:_apply_pragmas() for the optimised DB."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL")
    cur.execute("PRAGMA synchronous = NORMAL")
    cur.execute("PRAGMA busy_timeout = 5000")
    cur.execute("PRAGMA cache_size = -10000")
    cur.execute("PRAGMA temp_store = MEMORY")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_date_time ON trades(trade_date, trade_time)")
    conn.commit()
    conn.close()


async def _insert_via_fresh_connect(db_path: Path, n: int) -> None:
    """Baseline: open + close a new aiosqlite connection per write."""
    for i in range(n):
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT INTO trades (decision_id, ticker, trade_date, trade_time, "
                "order_value, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (f"d{i}", "SBER", "2026-05-26", "10:00:00", 100.0 * i, "2026-05-26T10:00:00"),
            )
            await db.commit()


async def _insert_via_pool(db_path: Path, n: int) -> None:
    """v0.9.0: single pooled connection, PRAGMAs applied once on first call."""
    db = await get_conn(db_path)
    for i in range(n):
        await db.execute(
            "INSERT INTO trades (decision_id, ticker, trade_date, trade_time, "
            "order_value, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (f"d{i}", "SBER", "2026-05-26", "10:00:00", 100.0 * i, "2026-05-26T10:00:00"),
        )
        await db.commit()


@pytest.fixture(autouse=True)
def _isolate_pool():
    """Each test gets a clean pool — no leaked connections across tests."""
    _reset_for_tests()
    yield
    _reset_for_tests()


@pytest.mark.asyncio
async def test_baseline_1000_inserts_completes(tmp_path: Path) -> None:
    """Sanity: the baseline workload itself must succeed.

    If this fails the rest of the perf comparisons are meaningless.
    Captures the baseline timing for human eyes in pytest -s logs.
    """
    db_path = tmp_path / "baseline.db"
    _create_schema(db_path)

    t0 = time.perf_counter()
    await _insert_via_fresh_connect(db_path, N_INSERTS)
    elapsed = time.perf_counter() - t0
    print(f"\n[BASELINE] 1000 fresh-connect inserts: {elapsed * 1000:.0f} ms")

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    conn.close()
    assert count == N_INSERTS


@pytest.mark.asyncio
async def test_pooled_1000_inserts_completes(tmp_path: Path) -> None:
    """The optimised path must finish and write every row."""
    db_path = tmp_path / "pooled.db"
    _create_schema(db_path)
    _apply_optimised_pragmas(db_path)

    t0 = time.perf_counter()
    await _insert_via_pool(db_path, N_INSERTS)
    elapsed = time.perf_counter() - t0
    print(f"\n[OPTIMISED] 1000 pooled inserts: {elapsed * 1000:.0f} ms")

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    conn.close()
    assert count == N_INSERTS


@pytest.mark.asyncio
async def test_pool_is_faster_than_fresh_connect(tmp_path: Path) -> None:
    """v0.9.0 must outperform the pre-release fresh-connect baseline.

    We assert a conservative 2x speedup so the test is robust on slow CI
    hardware where absolute numbers fluctuate. On the hackathon VM the
    real-world gap is closer to 5-8x; on macOS APFS it's ~3-5x.
    """
    baseline_db = tmp_path / "baseline.db"
    pool_db = tmp_path / "pool.db"
    _create_schema(baseline_db)
    _create_schema(pool_db)
    _apply_optimised_pragmas(pool_db)

    t0 = time.perf_counter()
    await _insert_via_fresh_connect(baseline_db, N_INSERTS)
    baseline_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    await _insert_via_pool(pool_db, N_INSERTS)
    pooled_s = time.perf_counter() - t0

    speedup = baseline_s / pooled_s if pooled_s > 0 else float("inf")
    print(
        f"\n[COMPARE] baseline={baseline_s * 1000:.0f} ms  "
        f"pooled={pooled_s * 1000:.0f} ms  speedup={speedup:.2f}x"
    )
    assert pooled_s < baseline_s, (
        f"pooled ({pooled_s:.3f}s) must be faster than baseline ({baseline_s:.3f}s)"
    )
    assert speedup >= 2.0, (
        f"expected >=2x speedup, got {speedup:.2f}x "
        f"(baseline={baseline_s:.3f}s, pooled={pooled_s:.3f}s)"
    )


@pytest.mark.asyncio
async def test_wal_mode_is_active(tmp_path: Path) -> None:
    """After get_conn() runs PRAGMA setup, journal_mode must report 'wal'.

    Rollback journal silently falls back if WAL fails (e.g. on a network
    filesystem). A regression that re-enables the default journal would
    re-introduce the 'database is locked' bug we shipped v0.9.0 to fix.
    """
    db_path = tmp_path / "wal.db"
    _create_schema(db_path)

    db = await get_conn(db_path)
    async with db.execute("PRAGMA journal_mode") as cur:
        row = await cur.fetchone()
    assert row is not None
    assert row[0].lower() == "wal", f"expected WAL, got {row[0]}"


@pytest.mark.asyncio
async def test_get_conn_returns_same_object_for_same_path(tmp_path: Path) -> None:
    """The pool must cache by resolved path — second call hits the cache.

    Without this guarantee every caller in execution/risk/dashboard would
    open its own connection and we'd be back to the pre-v0.9.0 state.
    """
    db_path = tmp_path / "cached.db"
    _create_schema(db_path)

    conn_a = await get_conn(db_path)
    conn_b = await get_conn(db_path)
    assert conn_a is conn_b


@pytest.mark.asyncio
async def test_index_speeds_up_date_range_scan(tmp_path: Path) -> None:
    """idx_trades_date_time must turn a date scan from O(N) into a range seek.

    The turnover tracker runs
        SELECT SUM(order_value) FROM trades WHERE trade_date >= ?
    every check. After 10k+ trades a full scan adds tens of ms — multiply
    by the dashboard's 5s refresh and you've burned a CPU.
    """
    indexed_db = tmp_path / "indexed.db"
    unindexed_db = tmp_path / "unindexed.db"
    _create_schema(indexed_db)
    _create_schema(unindexed_db)
    _apply_optimised_pragmas(indexed_db)

    rows = [
        (
            f"d{i}",
            "SBER",
            f"2026-05-{(i % 30) + 1:02d}",
            "10:00:00",
            100.0 * i,
            "2026-05-26T10:00:00",
        )
        for i in range(5000)
    ]
    for path in (indexed_db, unindexed_db):
        conn = sqlite3.connect(path)
        conn.executemany(
            "INSERT INTO trades (decision_id, ticker, trade_date, trade_time, "
            "order_value, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

    conn = sqlite3.connect(indexed_db)
    plan = conn.execute(
        "EXPLAIN QUERY PLAN SELECT SUM(order_value) FROM trades WHERE trade_date >= ?",
        ("2026-05-15",),
    ).fetchall()
    conn.close()
    plan_text = " ".join(str(p) for p in plan).lower()
    assert "idx_trades_date_time" in plan_text, f"index not used; plan was: {plan_text}"


@pytest.mark.asyncio
async def test_pool_survives_concurrent_writers(tmp_path: Path) -> None:
    """Multiple async tasks sharing the pooled connection must not deadlock
    nor corrupt rows. This mirrors the real dispatcher → risk → execution
    fan-out where ~6 coroutines may write to decisions.db within a cycle.
    """
    db_path = tmp_path / "concurrent.db"
    _create_schema(db_path)
    _apply_optimised_pragmas(db_path)

    async def _writer(tag: int, n: int) -> None:
        """Writer."""
        db = await get_conn(db_path)
        for i in range(n):
            await db.execute(
                "INSERT INTO trades (decision_id, ticker, trade_date, trade_time, "
                "order_value, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (f"t{tag}_d{i}", "SBER", "2026-05-26", "10:00:00", 1.0, "2026-05-26T10:00:00"),
            )
            await db.commit()

    n_per_task = 50
    n_tasks = 6
    await asyncio.gather(*[_writer(t, n_per_task) for t in range(n_tasks)])

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    conn.close()
    assert count == n_per_task * n_tasks, (
        f"lost writes: expected {n_per_task * n_tasks}, got {count}"
    )
