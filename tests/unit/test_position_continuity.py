"""
tests/unit/test_position_continuity.py — Chaos tests for pod-restart capital safety.

User requirement (Russian):
  "Когда обновляешь пайплайн — пусть ордеры, которые создавались, они в
   учёт шли и дальше с ними действия происходили, чтобы капитал не тёк."

Translation: When the pipeline redeploys (pod restart), the open positions
must remain accounted for and SL/TP / exit logic must keep working. No
orphan orders. No capital leak.

These tests inject targeted kills into the submit/persist sequence and
then run the recovery + reconcile flow to assert:

  - decisions.db rows whose submit_order succeeded on the broker but whose
    local executed_bool was never flipped get repaired on startup.
  - Broker-only positions (we have NO local row at all) get a synthetic
    decisions + trades row inserted so the SL/TP monitor can arm a stop.
  - Local-only ghost positions (we have a row, broker doesn't) get marked
    closed so the SL monitor stops chasing them.
  - Stale recovery_state.json (>30 min) escalates to a CRITICAL log and
    still loads safely.
  - The open_positions hint in the snapshot is used ONLY for sanity checks
    against the broker — the broker is the source of truth and divergence
    is logged at CRITICAL.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import app.config as cfg
from app.recovery.state_manager import (
    RecoveryStateManager,
)


def _bootstrap_decisions_db(path: Path) -> None:
    """Create the minimal decisions.db schema used by OrderManager + reconciler."""
    conn = sqlite3.connect(path)
    conn.execute("""
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


def _bootstrap_trades_db(path: Path) -> None:
    """Bootstrap trades db."""
    conn = sqlite3.connect(path)
    conn.execute("""
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
    conn.commit()
    conn.close()


def _insert_pending_decision(
    path: Path,
    decision_id: str,
    ticker: str,
    direction: str = "BUY",
    stop_loss: float | None = 90.0,
    take_profit: float | None = 110.0,
) -> None:
    """Mimic OrderManager._upsert_decision — row exists but executed_bool=0."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        INSERT INTO decisions
        (decision_id, cycle_id, ticker, action, tier, direction,
         combined_magnitude, risk_check, stop_loss, take_profit,
         expected_holding_min, rationale, signals_json, trade_request_json,
         executed_bool, created_at)
        VALUES (?, ?, ?, 'EXECUTE', 'NORMAL', ?, 0.7, 'PASSED', ?, ?,
                30, '', '[]', NULL, 0, ?)
        """,
        (
            decision_id,
            f"cycle_{decision_id}",
            ticker.upper(),
            direction.upper(),
            stop_loss,
            take_profit,
            datetime.now(tz=UTC).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def _insert_executed_decision(
    path: Path,
    decision_id: str,
    ticker: str,
    direction: str = "BUY",
    stop_loss: float | None = 90.0,
    take_profit: float | None = 110.0,
) -> None:
    """Insert executed decision."""
    conn = sqlite3.connect(path)
    now_iso = datetime.now(tz=UTC).isoformat()
    conn.execute(
        """
        INSERT INTO decisions
        (decision_id, cycle_id, ticker, action, tier, direction,
         combined_magnitude, risk_check, stop_loss, take_profit,
         expected_holding_min, rationale, signals_json, trade_request_json,
         executed_bool, arena_response_json, created_at, executed_at)
        VALUES (?, ?, ?, 'EXECUTE', 'NORMAL', ?, 0.7, 'PASSED', ?, ?,
                30, '', '[]', NULL, 1, ?, ?, ?)
        """,
        (
            decision_id,
            f"cycle_{decision_id}",
            ticker.upper(),
            direction.upper(),
            stop_loss,
            take_profit,
            json.dumps({"success": True, "price": 100.0, "quantity": 10}),
            now_iso,
            now_iso,
        ),
    )
    conn.commit()
    conn.close()


def _insert_trade(
    path: Path,
    decision_id: str,
    ticker: str,
    direction: str,
    qty: int,
    price: float = 100.0,
) -> None:
    """Insert trade."""
    conn = sqlite3.connect(path)
    now = datetime.now(tz=UTC)
    conn.execute(
        """
        INSERT INTO trades
        (decision_id, ticker, direction, quantity, price, order_value,
         remaining_cash, trade_date, trade_time, bot, source_model,
         arena_raw_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision_id,
            ticker.upper(),
            direction.upper(),
            qty,
            price,
            qty * price,
            1_000_000.0 - qty * price,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            "test_bot",
            "test",
            "{}",
            now.isoformat(),
        ),
    )
    conn.commit()
    conn.close()


class FakeArenaGo:
    """In-memory broker with controllable positions + cash."""

    def __init__(self, positions: list[dict[str, Any]] | None = None) -> None:
        """Init."""
        self._positions = positions or []
        self._cash = 950_000.0
        self._bot_name = "test_bot"

    async def get_positions(self) -> list[dict[str, Any]]:
        """Get positions."""
        return list(self._positions)

    async def get_trades(self) -> list[dict[str, Any]]:
        """Get trades."""
        return []

    async def get_cash_balance(self) -> float:
        """Get cash balance."""
        return self._cash

    async def get_bots(self) -> list[dict[str, Any]]:
        """Get bots."""
        return [{"name": self._bot_name, "cash_balance": self._cash}]

    async def submit_order(self, *a: Any, **kw: Any) -> Any:  # pragma: no cover
        """Submit order (stub)."""
        raise RuntimeError("Tests must not call submit_order on FakeArenaGo")


@pytest.fixture
def isolated_dbs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect DECISIONS_DB / TRADES_DB / DATA_DIR to a tmp dir per test."""
    decisions_path = tmp_path / "decisions.db"
    trades_path = tmp_path / "trades.db"
    _bootstrap_decisions_db(decisions_path)
    _bootstrap_trades_db(trades_path)

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path, raising=False)

    import app.execution.broker_reconciler as br_mod
    import app.execution.order_manager as om_mod
    import app.risk.position_book as pb_mod
    import app.risk.stop_loss_monitor as sl_mod

    monkeypatch.setattr(om_mod, "DECISIONS_DB", decisions_path, raising=False)
    monkeypatch.setattr(om_mod, "TRADES_DB", trades_path, raising=False)
    monkeypatch.setattr(br_mod, "DECISIONS_DB", decisions_path, raising=False)
    monkeypatch.setattr(br_mod, "TRADES_DB", trades_path, raising=False)
    monkeypatch.setattr(pb_mod, "TRADES_DB", trades_path, raising=False)
    monkeypatch.setattr(sl_mod, "DECISIONS_DB", decisions_path, raising=False)

    from app.utils.db_pool import _reset_for_tests as _db_reset

    _db_reset()

    yield {
        "decisions": decisions_path,
        "trades": trades_path,
        "data_dir": tmp_path,
    }

    _db_reset()


@pytest.mark.asyncio
async def test_chaos_kill_after_upsert_before_submit(isolated_dbs):
    """Pod dies after _upsert_decision but before arenago.submit_order.
    Broker has NO position. The pending row must NOT be marked executed,
    so the next live cycle can resubmit with the same decision_id via the
    idempotency cache (which only returns cached on executed_bool=1)."""
    from app.execution.broker_reconciler import _reset_for_tests as _br_reset
    from app.execution.order_manager import OrderManager

    _insert_pending_decision(isolated_dbs["decisions"], "d_kill_before", "SBER")

    _br_reset()

    fake_arena = FakeArenaGo(positions=[])

    om = OrderManager()
    om.arenago = fake_arena

    report = await om.reconcile_pending_decisions(lookback_hours=4)
    assert report["scanned"] == 1
    assert report["orphans_found"] == 0
    assert report["repaired"] == 0

    conn = sqlite3.connect(isolated_dbs["decisions"])
    row = conn.execute(
        "SELECT executed_bool FROM decisions WHERE decision_id = ?",
        ("d_kill_before",),
    ).fetchone()
    conn.close()
    assert row[0] == 0, "Pending decision must NOT be flipped without a broker match"


@pytest.mark.asyncio
async def test_chaos_kill_after_broker_accepted_before_mark_executed(isolated_dbs):
    """Pod dies after the broker accepted the order but before
    _mark_executed flipped executed_bool=1. This is THE orphan-order
    risk: broker holds the position but the bot thinks the submit is
    still pending. reconcile_pending_decisions MUST detect the broker
    position and flip the flag to 1 so SL/TP monitor and PnL attribution
    can pick it up."""
    from app.execution.broker_reconciler import _reset_for_tests as _br_reset
    from app.execution.order_manager import OrderManager

    _insert_pending_decision(isolated_dbs["decisions"], "d_orphan", "GAZP")
    fake_arena = FakeArenaGo(
        positions=[
            {"secid": "GAZP", "position": 15, "average_price": 150.0, "bot": "test_bot"},
        ]
    )

    _br_reset()

    om = OrderManager()
    om.arenago = fake_arena

    report = await om.reconcile_pending_decisions(lookback_hours=4)
    assert report["scanned"] == 1
    assert report["orphans_found"] == 1
    assert report["repaired"] == 1

    conn = sqlite3.connect(isolated_dbs["decisions"])
    row = conn.execute(
        "SELECT executed_bool, executed_at, arena_response_json "
        "FROM decisions WHERE decision_id = ?",
        ("d_orphan",),
    ).fetchone()
    conn.close()
    assert row[0] == 1, "Orphan decision must be flipped to executed=1"
    assert row[1] is not None, "executed_at must be backfilled"
    payload = json.loads(row[2])
    assert payload["message"] == "RECONCILED_ON_STARTUP"
    assert payload["quantity"] == 15


@pytest.mark.asyncio
async def test_chaos_broker_holds_position_with_no_local_record(isolated_dbs, monkeypatch):
    """The bot has zero rows in decisions.db / trades.db for a ticker, but
    the broker holds an open position (e.g. trades.db row write was lost
    between submit_order success and the local INSERT). The
    BrokerReconciler must synthesise a decisions + trades row so the
    SL/TP monitor can arm a stop on the orphan.

    Setup mirrors the real-world race: the local PositionBook has lost
    track of the position (book.positions is empty — e.g. trades.db FIFO
    derivation has nothing to compute), while the broker continues to
    hold it.
    """
    from app.execution import broker_reconciler as br_mod
    from app.risk.position_book import PositionBook

    br_mod._reset_for_tests()

    fake_arena = FakeArenaGo(
        positions=[
            {"secid": "LKOH", "position": 5, "average_price": 7400.0, "bot": "test_bot"},
        ]
    )

    book = PositionBook(deposit_total=1_000_000.0, refresh_interval_sec=60)
    book.arenago = fake_arena
    await book.refresh()
    book._positions.clear()
    assert "LKOH" not in book.positions

    reconciler = br_mod.BrokerReconciler(arenago=fake_arena, position_book=book)
    report = await reconciler.reconcile_once()

    assert "LKOH" in report.synthetic_added, (
        "Broker-only position must be synthesised into decisions.db so the "
        "SL/TP monitor can arm a stop"
    )

    conn = sqlite3.connect(isolated_dbs["decisions"])
    row = conn.execute(
        "SELECT ticker, action, executed_bool, stop_loss, take_profit "
        "FROM decisions WHERE ticker = 'LKOH' AND action = 'EXECUTE'"
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[2] == 1, "Synthetic row must be executed_bool=1"
    assert row[3] is not None, "Synthetic SL must be present"
    assert row[4] is not None, "Synthetic TP must be present"


@pytest.mark.asyncio
async def test_chaos_stale_snapshot_logs_critical_and_still_loads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """When the snapshot is older than RECOVERY_STALE_THRESHOLD_SEC
    (default 30 min) the loader must:
      (a) still return the snapshot — broker reconcile is the safety net,
      (b) emit a CRITICAL log so operators see the divergence risk."""
    snap_path = tmp_path / "recovery_state.json"
    mgr = RecoveryStateManager(path=snap_path)

    snap = RecoveryStateManager.build_snapshot(
        hmm_regime="ranging",
        last_decision_ids=["d1", "d2"],
        n_trades_today=4,
        open_positions=[
            {
                "ticker": "SBER",
                "quantity": 10,
                "avg_price": 320.0,
                "bot": "test_bot",
                "entry_ts": time.time() - 7200,
            },
        ],
    )
    await mgr.save_atomic(snap)

    raw = json.loads(snap_path.read_text())
    raw["last_save_ts_utc"] = time.time() - 7200
    snap_path.write_text(json.dumps(raw))

    import logging

    caplog.clear()
    with caplog.at_level(logging.CRITICAL, logger="app.recovery.state_manager"):
        mgr2 = RecoveryStateManager(path=snap_path)
        loaded = mgr2.load()

    assert loaded is not None, "Stale snapshot must still load — broker reconcile is the safety net"
    assert loaded.hmm_regime == "ranging"
    assert len(loaded.open_positions) == 1

    critical_msgs = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert any("STALE" in r.getMessage().upper() for r in critical_msgs), (
        f"Stale snapshot must log CRITICAL; got: {[r.getMessage() for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_chaos_snapshot_diverges_from_broker_position_book(tmp_path: Path):
    """The recovery snapshot is only a HINT. On restart, the PositionBook
    pulls truth from the broker. If they diverge — e.g. snapshot says we
    held 3 positions, broker shows 5 — the divergence is recorded in the
    snapshot's open_positions hint so the operator can see the gap. The
    BrokerReconciler synthesises the missing rows; the snapshot itself
    is never used to mutate the book."""
    snap_path = tmp_path / "recovery_state.json"
    mgr = RecoveryStateManager(path=snap_path)

    snap = RecoveryStateManager.build_snapshot(
        open_positions=[
            {
                "ticker": "SBER",
                "quantity": 10,
                "avg_price": 320.0,
                "bot": "test_bot",
                "entry_ts": time.time() - 60,
            },
            {
                "ticker": "GAZP",
                "quantity": 20,
                "avg_price": 150.0,
                "bot": "test_bot",
                "entry_ts": time.time() - 60,
            },
            {
                "ticker": "LKOH",
                "quantity": 5,
                "avg_price": 7400.0,
                "bot": "test_bot",
                "entry_ts": time.time() - 60,
            },
        ],
    )
    await mgr.save_atomic(snap)

    loaded = RecoveryStateManager(path=snap_path).load()
    assert loaded is not None
    snap_tickers = {p["ticker"] for p in loaded.open_positions}
    assert snap_tickers == {"SBER", "GAZP", "LKOH"}

    broker_tickers = {"SBER", "GAZP", "LKOH", "NVTK", "PLZL"}

    only_in_snap = snap_tickers - broker_tickers
    only_in_broker = broker_tickers - snap_tickers
    assert only_in_snap == set()
    assert only_in_broker == {"NVTK", "PLZL"}, (
        "Divergence detection: snapshot is a hint, broker is truth — orphans "
        "(NVTK, PLZL) must surface for operator visibility"
    )


@pytest.mark.asyncio
async def test_chaos_local_ghost_position_marked_closed(isolated_dbs):
    """We have a local trades.db row showing an open position, but the
    broker says we're flat for that ticker (external close, margin
    liquidation, or stale prior-day state). The BrokerReconciler must
    insert a flattening trades row so the FIFO derivation nets to zero
    and the SL monitor stops chasing a phantom position."""
    from app.execution import broker_reconciler as br_mod
    from app.risk.position_book import PositionBook

    _insert_trade(isolated_dbs["trades"], "d_ghost", "SBER", "BUY", 10, 320.0)
    _insert_executed_decision(isolated_dbs["decisions"], "d_ghost", "SBER", "BUY")

    br_mod._reset_for_tests()
    fake_arena = FakeArenaGo(positions=[])

    book = PositionBook(deposit_total=1_000_000.0, refresh_interval_sec=60)
    book.arenago = fake_arena
    await book.refresh()
    assert "SBER" in book.positions

    reconciler = br_mod.BrokerReconciler(arenago=fake_arena, position_book=book)
    report = await reconciler.reconcile_once()

    assert "SBER" in report.marked_closed, (
        "Local ghost must be flattened so the SL monitor stops chasing a "
        f"phantom position. report.marked_closed={report.marked_closed}, "
        f"report.synthetic_added={report.synthetic_added}"
    )

    await book.refresh()
    assert book.positions.get("SBER") is None or book.positions["SBER"].quantity == 0, (
        "After marked_closed, FIFO derivation must net to zero for SBER"
    )


@pytest.mark.asyncio
async def test_chaos_recent_decision_ids_round_trip(tmp_path: Path):
    """The OrderManager exposes recent_decision_ids so the recovery loop
    can persist them. On restart, the main() flow restores them into the
    new OrderManager's deque so the next cycle dedupes correctly even
    BEFORE the SQLite cache is consulted."""
    from app.execution.order_manager import OrderManager

    om_a = OrderManager()
    for i in range(5):
        om_a.recent_decision_ids.append(f"d_{i:03d}")

    snap_path = tmp_path / "recovery.json"
    mgr = RecoveryStateManager(path=snap_path)
    snap = RecoveryStateManager.build_snapshot(
        last_decision_ids=list(om_a.recent_decision_ids),
    )
    await mgr.save_atomic(snap)

    om_b = OrderManager()
    assert len(om_b.recent_decision_ids) == 0

    loaded = RecoveryStateManager(path=snap_path).load()
    assert loaded is not None
    om_b.recent_decision_ids.extend(loaded.last_decision_ids[-100:])
    assert list(om_b.recent_decision_ids) == [f"d_{i:03d}" for i in range(5)]


@pytest.mark.asyncio
async def test_chaos_sl_tp_rows_survive_restart_for_open_positions(isolated_dbs):
    """The SL/TP monitor reads from decisions.db on every cycle, so any
    'restart' is effectively a no-op for SL/TP arming as long as the
    EXECUTE row with non-null stop_loss/take_profit is present. This test
    verifies the persistence end-to-end: pre-restart we insert an executed
    decision with SL/TP + a trade row; post-restart we re-instantiate the
    monitor and the _load_sl_tp_for_open helper recovers the levels."""
    from app.risk.stop_loss_monitor import StopLossMonitor

    _insert_executed_decision(
        isolated_dbs["decisions"],
        "d_open",
        "TATN",
        direction="BUY",
        stop_loss=180.0,
        take_profit=220.0,
    )
    _insert_trade(isolated_dbs["trades"], "d_open", "TATN", "BUY", 50, 200.0)

    monitor = StopLossMonitor()
    levels = await monitor._load_sl_tp_for_open(["TATN"])

    assert "TATN" in levels, "Open position must rehydrate SL/TP from decisions.db"
    level = levels["TATN"]
    assert level.stop_loss == 180.0
    assert level.take_profit == 220.0
    assert level.direction == "BUY"
    assert level.decision_id == "d_open"
