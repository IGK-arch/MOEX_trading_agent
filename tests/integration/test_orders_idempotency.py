"""
tests/integration/test_orders_idempotency.py — Submitting the same
decision_id twice must hit ArenaGo exactly once.

Phase 19 (v0.0.16) made the cycle_id deterministic so a process restart
inside the same time-bucket produces the same decision_id. This test
exercises that contract end-to-end with the fake ArenaGo client.
"""

from __future__ import annotations

import pytest

from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    DecisionTier,
    Direction,
    RiskCheckResult,
    SignalSource,
    TradeRequest,
    UnifiedSignal,
)
from app.execution import order_manager as om_mod
from app.execution.arenago_client import SubmitResult


class IdempotentFakeArenaGo:
    """Fake ArenaGo client that:
    * Tracks every submit_order call in `calls`.
    * Returns the cached result on duplicate decision_id (matches
      the contract enforced by ArenaGoClient._is_already_executed).
    """

    def __init__(self) -> None:
        """Init."""
        self._started = True
        self.calls: list[dict] = []
        self._cache: dict[str, SubmitResult] = {}

    async def startup(self) -> None:
        """Startup."""
        ...

    async def shutdown(self) -> None:
        """Shutdown."""
        ...

    async def submit_order(
        self,
        direction: str,
        ticker: str,
        quantity: int,
        decision_id: str,
        bot: str | None = None,
    ) -> SubmitResult:
        """Submit order."""
        if decision_id in self._cache:
            return self._cache[decision_id]

        self.calls.append(
            {
                "direction": direction,
                "ticker": ticker,
                "quantity": quantity,
                "decision_id": decision_id,
                "bot": bot,
            }
        )
        result = SubmitResult(
            success=True,
            message="OK",
            order_value=10_000.0,
            price=100.0,
            quantity=quantity,
            remaining_cash=990_000.0,
            decision_id=decision_id,
        )
        self._cache[decision_id] = result
        return result

    async def get_positions(self):
        """Get positions."""
        return []

    async def get_cash_balance(self):
        """Get cash balance."""
        return 990_000.0

    async def get_bots(self):
        """Get bots."""
        return [{"name": "test_bot", "cash_balance": 990_000.0}]

    async def get_trades(self):
        """Get trades."""
        return []


def _build_execute_decision(decision_id: str = "stable_test_id") -> Decision:
    """Build execute decision."""
    sig = UnifiedSignal(
        source=SignalSource.TA,
        detector="idempotency_test",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=0.80,
        raw_confidence=0.80,
        horizon_min=60,
        price=100.0,
        entry_level=100.0,
        stop_level=98.0,
        target_level=105.0,
        expected_rr=2.5,
        atr=1.0,
    )
    tr = TradeRequest(
        decision_id=decision_id,
        ticker="SBER",
        direction=Direction.BUY,
        quantity=100,
        bot="test_bot",
        price_at_signal=100.0,
    )
    return Decision(
        decision_id=decision_id,
        cycle_id="bucket_X",
        ticker="SBER",
        action=DecisionAction.EXECUTE,
        tier=DecisionTier.TIER1,
        direction=Direction.BUY,
        combined_magnitude=0.80,
        signals=[sig],
        risk_check=RiskCheckResult.PASSED,
        trade_request=tr,
        expected_holding_min=60,
        stop_loss=98.0,
        take_profit=105.0,
        expected_rr=2.5,
        rationale="idempotency_test",
    )


@pytest.mark.asyncio
async def test_same_decision_id_submitted_twice_calls_arenago_once(monkeypatch, tmp_path):
    """Submit the same decision_id twice → ArenaGo sees exactly one call."""
    fake = IdempotentFakeArenaGo()

    monkeypatch.setattr(om_mod, "_order_manager", None)
    monkeypatch.setattr("app.execution.order_manager.get_arenago_client", lambda: fake)
    monkeypatch.setattr(om_mod, "DECISIONS_DB", tmp_path / "decisions.db")
    monkeypatch.setattr(om_mod, "TRADES_DB", tmp_path / "trades.db")

    om = om_mod.OrderManager()
    decision = _build_execute_decision("dup_test_001")

    r1 = await om.submit(decision)
    assert r1 is not None and r1.success

    r2 = await om.submit(decision)
    assert r2 is not None and r2.success

    assert len(fake.calls) == 1, (
        f"ArenaGo should see EXACTLY one call for duplicate decision_id, got {len(fake.calls)}"
    )
    assert r1.decision_id == r2.decision_id == "dup_test_001"


@pytest.mark.asyncio
async def test_different_decision_ids_both_reach_arenago(monkeypatch, tmp_path):
    """Sanity check: two genuinely different decisions both make it through.

    Phase 30 caveat — OrderManager.submit now ALSO drops same-direction
    same-ticker duplicates inside a 5-min window (broker_reconciler.is_duplicate_today).
    To keep this test exercising the decision_id path and not the
    duplicate-window path, the two decisions target DIFFERENT tickers.
    """
    fake = IdempotentFakeArenaGo()

    monkeypatch.setattr(om_mod, "_order_manager", None)
    monkeypatch.setattr("app.execution.order_manager.get_arenago_client", lambda: fake)
    monkeypatch.setattr(om_mod, "DECISIONS_DB", tmp_path / "decisions.db")
    monkeypatch.setattr(om_mod, "TRADES_DB", tmp_path / "trades.db")
    import sqlite3

    for db in (tmp_path / "decisions.db", tmp_path / "trades.db"):
        c = sqlite3.connect(db)
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY, cycle_id TEXT, ticker TEXT,
                action TEXT, tier TEXT, direction TEXT,
                combined_magnitude REAL, risk_check TEXT,
                stop_loss REAL, take_profit REAL, expected_holding_min INTEGER,
                rationale TEXT, signals_json TEXT, trade_request_json TEXT,
                created_at TEXT, executed_at TEXT, executed_bool INTEGER,
                arena_response_json TEXT, reflection_status TEXT,
                pnl_rub REAL, git_commit TEXT
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT,
                ticker TEXT, direction TEXT, quantity INTEGER, price REAL,
                order_value REAL, remaining_cash REAL, trade_date TEXT,
                trade_time TEXT, bot TEXT, source_model TEXT,
                arena_raw_json TEXT, created_at TEXT
            );
            """
        )
        c.commit()
        c.close()
    from app.execution import broker_reconciler as br_mod

    monkeypatch.setattr(br_mod, "TRADES_DB", tmp_path / "trades.db")
    from app.utils.db_pool import _reset_for_tests

    _reset_for_tests()

    om = om_mod.OrderManager()
    d1 = _build_execute_decision("alpha_001")
    d2_sig = UnifiedSignal(
        source=SignalSource.TA,
        detector="idempotency_test",
        ticker="GAZP",
        direction=Direction.BUY,
        magnitude=0.80,
        raw_confidence=0.80,
        horizon_min=60,
        price=100.0,
        entry_level=100.0,
        stop_level=98.0,
        target_level=105.0,
        expected_rr=2.5,
        atr=1.0,
    )
    d2_tr = TradeRequest(
        decision_id="alpha_002",
        ticker="GAZP",
        direction=Direction.BUY,
        quantity=100,
        bot="test_bot",
        price_at_signal=100.0,
    )
    d2 = Decision(
        decision_id="alpha_002",
        cycle_id="bucket_X",
        ticker="GAZP",
        action=DecisionAction.EXECUTE,
        tier=DecisionTier.TIER1,
        direction=Direction.BUY,
        combined_magnitude=0.80,
        signals=[d2_sig],
        risk_check=RiskCheckResult.PASSED,
        trade_request=d2_tr,
        expected_holding_min=60,
        stop_loss=98.0,
        take_profit=105.0,
        expected_rr=2.5,
        rationale="idempotency_test",
    )

    await om.submit(d1)
    await om.submit(d2)

    assert len(fake.calls) == 2
    ids = {c["decision_id"] for c in fake.calls}
    assert ids == {"alpha_001", "alpha_002"}


@pytest.mark.asyncio
async def test_decision_id_is_deterministic_for_same_inputs():
    """SHA1-based decision_id MUST be identical for same (cycle_id, ticker,
    signal_ids). This is what makes restart-idempotency possible."""
    sig_ids = ["sigA", "sigB", "sigC"]
    id1 = Decision.make_id("cycle_42", "SBER", sig_ids)
    id2 = Decision.make_id("cycle_42", "SBER", list(reversed(sig_ids)))
    assert id1 == id2, "decision_id must be order-independent (signals are sorted)"

    id3 = Decision.make_id("cycle_42", "SBER", sig_ids + ["sigD"])
    assert id3 != id1, "Adding a new signal must change decision_id"
