"""Broker reconciliation tests (Phase 30 — orphan-order safety net).

These tests pin down the contract for `app.execution.broker_reconciler`:
the reconciler must converge local DB state with ArenaGo on every cycle,
and the OrderManager dedup guard must drop same-direction duplicates
within a 5-min window so two agents can't stack exposure.

Test design notes
-----------------
- Each test uses tmp_path and clears the db_pool cache + reconciler
  singleton via the `clean_state` fixture.  Tests inject fake clients
  (fake ArenaGo, fake PositionBook) directly into BrokerReconciler so
  no networking and no global singletons are touched.
- We bootstrap minimal `decisions` + `trades` schemas with the same
  PRAGMAs as production so PRAGMA-sensitive code paths (the pool's
  one-time setup) behave identically.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest

from app.execution import broker_reconciler as br
from app.execution.broker_reconciler import (
    BrokerPosition,
    BrokerReconciler,
    ReconcileReport,
    is_duplicate_today,
)
from app.utils.db_pool import _reset_for_tests


def _create_schemas(decisions_db: Path, trades_db: Path) -> None:
    """Mirror scripts/bootstrap_db.py — minimal columns the reconciler touches."""
    conn = sqlite3.connect(decisions_db)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL")
    cur.execute("PRAGMA synchronous = NORMAL")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            decision_id     TEXT PRIMARY KEY,
            cycle_id        TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            action          TEXT NOT NULL,
            tier            TEXT NOT NULL DEFAULT 'NONE',
            direction       TEXT NOT NULL DEFAULT 'NEUTRAL',
            combined_magnitude REAL DEFAULT 0.0,
            risk_check      TEXT NOT NULL DEFAULT 'PASSED',
            stop_loss       REAL,
            take_profit     REAL,
            expected_holding_min INTEGER DEFAULT 0,
            rationale       TEXT DEFAULT '',
            signals_json    TEXT DEFAULT '[]',
            trade_request_json TEXT,
            git_commit      TEXT DEFAULT '',
            executed_bool   INTEGER DEFAULT 0,
            arena_response_json TEXT,
            pnl_rub         REAL,
            reflection_status TEXT DEFAULT 'PENDING',
            created_at      TEXT NOT NULL,
            executed_at     TEXT
        )
    """)
    conn.commit()
    conn.close()

    conn = sqlite3.connect(trades_db)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL")
    cur.execute("PRAGMA synchronous = NORMAL")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id     TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            direction       TEXT NOT NULL,
            quantity        INTEGER NOT NULL,
            price           REAL NOT NULL,
            order_value     REAL NOT NULL,
            remaining_cash  REAL NOT NULL,
            trade_date      TEXT NOT NULL,
            trade_time      TEXT NOT NULL,
            bot             TEXT NOT NULL,
            source_model    TEXT DEFAULT '',
            arena_raw_json  TEXT DEFAULT '',
            created_at      TEXT NOT NULL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_date_time ON trades(trade_date, trade_time)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker_date ON trades(ticker, trade_date)")
    conn.commit()
    conn.close()


@dataclass
class _FakePosition:
    """Mirror app.risk.position_book.Position with the only fields the
    reconciler reads. Keeping this in-test avoids importing the real
    Position dataclass which pulls in the full PositionBook singleton."""

    ticker: str
    quantity: int
    avg_price: float
    bot: str = "test_bot"


class FakeBook:
    """Stand-in for PositionBook — just exposes positions + cash."""

    def __init__(self, positions: dict[str, _FakePosition], cash: float = 1_000_000.0) -> None:
        """Init."""
        self.positions = positions
        self.cash_balance = cash


class FakeArena:
    """Stand-in for ArenaGoClient — returns whatever the test seeded."""

    def __init__(
        self,
        positions: list[dict[str, Any]] | None = None,
        trades: list[dict[str, Any]] | None = None,
        cash: float = 1_000_000.0,
    ) -> None:
        """Init."""
        self._positions = positions or []
        self._trades = trades or []
        self._cash = cash
        self._bot_name = "test_bot"

    async def get_positions(self) -> list[dict[str, Any]]:
        """Get positions."""
        return self._positions

    async def get_trades(self) -> list[dict[str, Any]]:
        """Get trades."""
        return self._trades

    async def get_cash_balance(self) -> float:
        """Get cash balance."""
        return self._cash


@pytest.fixture(autouse=True)
def clean_state():
    """Per-test isolation: reset both the connection pool and the
    reconciler singleton so no global state leaks between tests."""
    _reset_for_tests()
    br._reset_for_tests()
    yield
    _reset_for_tests()
    br._reset_for_tests()


@pytest.fixture
def tmp_dbs(tmp_path: Path, monkeypatch):
    """Set up tmp decisions.db + trades.db and point the module-level
    DECISIONS_DB / TRADES_DB constants at them. The reconciler reads
    these at function-call time (not at import time, via get_conn) so
    monkeypatching the module attributes is enough."""
    dec = tmp_path / "decisions.db"
    tr = tmp_path / "trades.db"
    _create_schemas(dec, tr)
    monkeypatch.setattr(br, "DECISIONS_DB", dec, raising=False)
    monkeypatch.setattr(br, "TRADES_DB", tr, raising=False)
    return dec, tr


@pytest.mark.asyncio
async def test_broker_position_normalisation_from_arenago_raw():
    """`BrokerPosition.from_arenago` must uppercase the ticker, coerce
    quantity to int, and pull the bot from the raw payload when present
    or fall back to the default. Without this normalisation downstream
    dict lookups (keyed by upper ticker) would silently miss positions."""
    raw = {"secid": "sber", "position": "5", "average_price": "250.5", "bot": "alpha"}
    bp = BrokerPosition.from_arenago(raw, default_bot="fallback")
    assert bp.ticker == "SBER"
    assert bp.quantity == 5
    assert bp.avg_price == pytest.approx(250.5)
    assert bp.bot == "alpha"

    raw2 = {"secid": "GAZP", "position": -3, "average_price": 180.0}
    bp2 = BrokerPosition.from_arenago(raw2, default_bot="fallback")
    assert bp2.bot == "fallback"
    assert bp2.quantity == -3


@pytest.mark.asyncio
async def test_no_divergence_when_state_matches(tmp_dbs):
    """When broker and local agree, the reconciler reports zero changes
    and `has_mismatch=False` so periodic logging stays quiet."""
    arena = FakeArena(
        positions=[{"secid": "SBER", "position": 10, "average_price": 250.0}],
        cash=999_000.0,
    )
    book = FakeBook(
        positions={"SBER": _FakePosition("SBER", 10, 250.0)},
        cash=999_000.0,
    )
    rec = BrokerReconciler(arenago=arena, position_book=book)
    report = await rec.reconcile_once()

    assert report.fetched_positions == 1
    assert report.synthetic_added == []
    assert report.marked_closed == []
    assert report.cash_divergence_rub <= br.CASH_DIVERGENCE_TOL_RUB
    assert report.has_mismatch is False


@pytest.mark.asyncio
async def test_synthetic_entry_created_for_unknown_broker_position(tmp_dbs):
    """Broker holds a position the bot has no record of → reconciler
    must insert a synthetic row in BOTH decisions.db (so SL monitor can
    arm a stop) and trades.db (so PositionBook FIFO sees the position)."""
    dec_db, tr_db = tmp_dbs
    arena = FakeArena(
        positions=[{"secid": "LKOH", "position": 5, "average_price": 6000.0}],
    )
    book = FakeBook(positions={}, cash=1_000_000.0)
    rec = BrokerReconciler(arenago=arena, position_book=book)

    report = await rec.reconcile_once()

    assert "LKOH" in report.synthetic_added
    assert report.has_mismatch is True

    conn = sqlite3.connect(dec_db)
    rows = conn.execute(
        "SELECT ticker, action, direction, stop_loss, take_profit, "
        "executed_bool, rationale FROM decisions WHERE ticker = 'LKOH'"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    ticker, action, direction, sl, tp, executed_bool, rationale = rows[0]
    assert action == "EXECUTE"
    assert direction == "BUY"
    assert executed_bool == 1
    assert sl is not None and tp is not None
    assert sl < 6000.0 < tp
    assert "synthetic" in rationale.lower()

    conn = sqlite3.connect(tr_db)
    trade_rows = conn.execute(
        "SELECT ticker, direction, quantity, price FROM trades WHERE ticker = 'LKOH'"
    ).fetchall()
    conn.close()
    assert len(trade_rows) == 1
    assert trade_rows[0] == ("LKOH", "BUY", 5, 6000.0)


@pytest.mark.asyncio
async def test_synthetic_entry_sl_tp_armed_with_standard_atr_rules(tmp_dbs):
    """The SL/TP on the synthetic row must follow the ATR-percentage
    heuristic so the stop_loss_monitor has SOMETHING to fire on. This
    pins down the math: for a BUY at price P with default ATR pct,
    SL = P - 1.75 * (P * 0.015), TP = P + 3.5 * (P * 0.015)."""
    dec_db, _ = tmp_dbs
    entry = 100.0
    arena = FakeArena(
        positions=[{"secid": "MGNT", "position": 10, "average_price": entry}],
    )
    rec = BrokerReconciler(arenago=arena, position_book=FakeBook({}))
    await rec.reconcile_once()

    conn = sqlite3.connect(dec_db)
    sl, tp = conn.execute(
        "SELECT stop_loss, take_profit FROM decisions WHERE ticker='MGNT'"
    ).fetchone()
    conn.close()

    atr_proxy = entry * 0.015
    expected_sl = entry - 1.75 * atr_proxy
    expected_tp = entry + 2.0 * 1.75 * atr_proxy
    assert sl == pytest.approx(expected_sl, rel=1e-6)
    assert tp == pytest.approx(expected_tp, rel=1e-6)


@pytest.mark.asyncio
async def test_local_only_position_marked_closed(tmp_dbs):
    """Bot thinks it holds SBER but broker shows nothing → reconciler
    inserts a flattening SELL row in trades.db so the next FIFO net
    nets to zero, AND emits the closure in the report."""
    _, tr_db = tmp_dbs
    arena = FakeArena(positions=[])
    book = FakeBook(
        positions={"SBER": _FakePosition("SBER", 7, 280.0, "test_bot")},
        cash=1_000_000.0,
    )
    rec = BrokerReconciler(arenago=arena, position_book=book)

    report = await rec.reconcile_once()
    assert report.marked_closed == ["SBER"]
    assert report.has_mismatch is True

    conn = sqlite3.connect(tr_db)
    rows = conn.execute(
        "SELECT direction, quantity, price FROM trades WHERE ticker='SBER'"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0] == ("SELL", 7, 280.0)


@pytest.mark.asyncio
async def test_marked_closed_idempotent_within_run(tmp_dbs):
    """Calling reconcile_once twice for the same stale local position
    must NOT write two flattening rows — the in-process closed-marker
    set short-circuits the second attempt. Without this guard the
    every-5-min loop would pile up duplicate close markers."""
    _, tr_db = tmp_dbs
    arena = FakeArena(positions=[])
    book = FakeBook(
        positions={"GAZP": _FakePosition("GAZP", 4, 180.0)},
        cash=1_000_000.0,
    )
    rec = BrokerReconciler(arenago=arena, position_book=book)

    await rec.reconcile_once()
    await rec.reconcile_once()

    conn = sqlite3.connect(tr_db)
    n = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE ticker='GAZP' AND source_model='reconciler'"
    ).fetchone()[0]
    conn.close()
    assert n == 1, f"expected exactly one close marker, got {n}"


@pytest.mark.asyncio
async def test_cash_divergence_above_tolerance_marks_mismatch(tmp_dbs):
    """A 1+ RUB gap between broker and local cash must flip has_mismatch=True
    even when positions are perfectly aligned. The threshold is 0.01 RUB."""
    arena = FakeArena(positions=[], cash=900_000.0)
    book = FakeBook(positions={}, cash=905_000.0)
    rec = BrokerReconciler(arenago=arena, position_book=book)

    report = await rec.reconcile_once()
    assert report.cash_divergence_rub == pytest.approx(5_000.0)
    assert report.has_mismatch is True
    assert "Δcash" in report.diff_summary


@pytest.mark.asyncio
async def test_cash_divergence_within_tolerance_is_quiet(tmp_dbs):
    """Sub-kopeck float noise (< 0.01 RUB) must NOT flip has_mismatch
    so the periodic log line stays at INFO under normal operation."""
    arena = FakeArena(positions=[], cash=1_000_000.0)
    book = FakeBook(positions={}, cash=1_000_000.0009)
    rec = BrokerReconciler(arenago=arena, position_book=book)

    report = await rec.reconcile_once()
    assert report.cash_divergence_rub < br.CASH_DIVERGENCE_TOL_RUB
    assert report.has_mismatch is False


@pytest.mark.asyncio
async def test_short_position_synthetic_is_sell_with_inverted_sl_tp(tmp_dbs):
    """Negative-quantity broker position → synthetic decision direction=SELL
    and SL is above entry, TP below. This guards against a sign-flip bug
    that would otherwise make the SL fire instantly on price > entry."""
    dec_db, _ = tmp_dbs
    arena = FakeArena(
        positions=[{"secid": "NLMK", "position": -8, "average_price": 200.0}],
    )
    rec = BrokerReconciler(arenago=arena, position_book=FakeBook({}))
    await rec.reconcile_once()

    conn = sqlite3.connect(dec_db)
    direction, sl, tp = conn.execute(
        "SELECT direction, stop_loss, take_profit FROM decisions WHERE ticker='NLMK'"
    ).fetchone()
    conn.close()
    assert direction == "SELL"
    assert sl > 200.0 > tp


@pytest.mark.asyncio
async def test_reconciler_ignores_zero_quantity_broker_rows(tmp_dbs):
    """Some sandbox responses echo zero-qty rows for tickers the bot
    once held. The reconciler must SKIP those so we don't create a
    synthetic for nothing."""
    arena = FakeArena(
        positions=[
            {"secid": "ALRS", "position": 0, "average_price": 50.0},
            {"secid": "MOEX", "position": 3, "average_price": 220.0},
        ],
    )
    rec = BrokerReconciler(arenago=arena, position_book=FakeBook({}))
    report = await rec.reconcile_once()

    assert "ALRS" not in report.synthetic_added
    assert "MOEX" in report.synthetic_added


@pytest.mark.asyncio
async def test_is_duplicate_today_detects_recent_same_direction(tmp_dbs):
    """OrderManager calls is_duplicate_today before submit. A BUY for
    SBER that hit trades.db 1 second ago must trigger True; a SELL or
    a different ticker must NOT trigger."""
    _, tr_db = tmp_dbs
    from datetime import datetime

    now = datetime.now(tz=UTC)
    conn = sqlite3.connect(tr_db)
    conn.execute(
        """INSERT INTO trades
           (decision_id, ticker, direction, quantity, price, order_value,
            remaining_cash, trade_date, trade_time, bot, source_model,
            arena_raw_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "d1",
            "SBER",
            "BUY",
            1,
            250.0,
            250.0,
            999_750.0,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            "test_bot",
            "ta",
            "{}",
            now.isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    assert await is_duplicate_today("SBER", "BUY", window_sec=300, db_path=tr_db) is True
    assert await is_duplicate_today("SBER", "SELL", window_sec=300, db_path=tr_db) is False
    assert await is_duplicate_today("GAZP", "BUY", window_sec=300, db_path=tr_db) is False


@pytest.mark.asyncio
async def test_is_duplicate_today_window_expires(tmp_dbs):
    """A trade older than window_sec must NOT count as duplicate. We
    seed a row with created_at 10 minutes ago and probe with a 5-min
    window — must return False."""
    _, tr_db = tmp_dbs
    from datetime import datetime, timedelta

    now = datetime.now(tz=UTC)
    old = now - timedelta(minutes=10)
    conn = sqlite3.connect(tr_db)
    conn.execute(
        """INSERT INTO trades
           (decision_id, ticker, direction, quantity, price, order_value,
            remaining_cash, trade_date, trade_time, bot, source_model,
            arena_raw_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "d_old",
            "VTBR",
            "BUY",
            100,
            1.0,
            100.0,
            999_900.0,
            old.strftime("%Y-%m-%d"),
            old.strftime("%H:%M:%S"),
            "test_bot",
            "ta",
            "{}",
            old.isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    assert await is_duplicate_today("VTBR", "BUY", window_sec=300, db_path=tr_db) is False
    assert await is_duplicate_today("VTBR", "BUY", window_sec=1800, db_path=tr_db) is True


@pytest.mark.asyncio
async def test_report_log_warning_silent_when_aligned(caplog, tmp_dbs):
    """When state is aligned, log_warning_if_needed must NOT emit
    a warning line. Otherwise the every-5-min loop would spam logs."""
    import logging

    caplog.set_level(logging.WARNING)
    arena = FakeArena(positions=[], cash=1_000.0)
    book = FakeBook(positions={}, cash=1_000.0)
    rec = BrokerReconciler(arenago=arena, position_book=book)

    report = await rec.reconcile_once()
    report.log_warning_if_needed()

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "Broker reconciliation" in r.getMessage()
    ]
    assert warnings == []


@pytest.mark.asyncio
async def test_report_log_warning_fires_on_mismatch(caplog, tmp_dbs):
    """When there IS a mismatch the report MUST log a WARNING so a
    human operator can be alerted via the standard log pipeline."""
    import logging

    caplog.set_level(logging.WARNING)
    arena = FakeArena(
        positions=[{"secid": "T", "position": 2, "average_price": 3000.0}],
    )
    book = FakeBook(positions={}, cash=1_000_000.0)
    rec = BrokerReconciler(arenago=arena, position_book=book)

    report = await rec.reconcile_once()
    report.log_warning_if_needed()

    matches = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "Broker reconciliation" in r.getMessage()
    ]
    assert len(matches) >= 1


@pytest.mark.asyncio
async def test_periodic_loop_runs_then_stops(tmp_dbs):
    """The background loop must execute at least one cycle and shut down
    cleanly. This protects against a typo in the cancel handling that
    would leave the asyncio.Task pending across the whole process."""
    arena = FakeArena(positions=[], cash=1_000.0)
    book = FakeBook(positions={}, cash=1_000.0)
    rec = BrokerReconciler(arenago=arena, position_book=book, interval_sec=0.05)

    await rec.start_periodic()
    import asyncio

    await asyncio.sleep(0.15)
    await rec.stop()
    assert rec._task is None or rec._task.done()


@pytest.mark.asyncio
async def test_broker_get_positions_failure_is_graceful(tmp_dbs):
    """A network blowup on broker.get_positions must NOT prevent the
    reconciler from reporting — it falls back to "no broker data" which
    means "leave local state alone"."""

    class FlakyArena(FakeArena):
        """Flaky Arena."""

        async def get_positions(self):
            """Get positions."""
            raise RuntimeError("simulated network error")

    arena = FlakyArena(positions=[], cash=500_000.0)
    book = FakeBook(
        positions={"CHMF": _FakePosition("CHMF", 1, 1200.0)},
        cash=500_000.0,
    )
    rec = BrokerReconciler(arenago=arena, position_book=book)

    report = await rec.reconcile_once()
    assert isinstance(report, ReconcileReport)
    assert report.fetched_positions == 0
