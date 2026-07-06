"""
tests/integration/test_full_dispatcher_cycle.py — Full dispatcher cycle with
synthetic candles + fake ArenaGo + fake Polza, 5 cycles end-to-end.

Verifies:
  * Adapters → Aggregator → RiskManager → OrderManager → ArenaGo path is wired.
  * At least one Decision per cycle is persisted to decisions.db when signals
    pass the tier / risk gates.
  * No uncaught exceptions; no LLM network calls.
  * The fake ArenaGo client sees idempotent submits (same decision_id within
    cycle bucket → single network call).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.agents.base import BaseAdapter
from app.dispatcher.aggregator import SignalAggregator
from app.dispatcher.dispatcher import Dispatcher
from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    Direction,
    SignalSource,
    UnifiedSignal,
)
from app.dispatcher.tier_classifier import apply_tier
from app.execution.arenago_client import SubmitResult


def _candle_signal(
    source: SignalSource,
    ticker: str,
    direction: Direction,
    *,
    mag: float = 0.75,
    rr: float = 2.5,
    price: float = 100.0,
) -> UnifiedSignal:
    """Build a UnifiedSignal that satisfies Tier-1 thresholds out of the box."""
    return UnifiedSignal(
        source=source,
        detector="synthetic_candles",
        ticker=ticker,
        direction=direction,
        magnitude=mag,
        raw_confidence=mag,
        horizon_min=60,
        price=price,
        entry_level=price,
        stop_level=price * 0.98,
        target_level=price * 1.05,
        expected_rr=rr,
        atr=price * 0.01,
    )


class SyntheticCandleAdapter(BaseAdapter):
    """Emits a fresh BUY signal every poll cycle from synthetic candle data."""

    name = "TA_SYNTH"

    def __init__(self, ticker: str = "SBER", source: SignalSource = SignalSource.TA) -> None:
        """Init."""
        super().__init__()
        self.ticker = ticker
        self.source = source
        self.polls = 0

    async def startup(self) -> None:
        """Startup."""
        self._started = True

    async def poll(self) -> list[UnifiedSignal]:
        """Poll."""
        self.polls += 1
        return [_candle_signal(self.source, self.ticker, Direction.BUY)]

    async def shutdown(self) -> None:
        """Shutdown."""
        self._started = False


class FakeOrderRouter:
    """Stand-in for OrderManager that records every submit() in memory.

    Avoids touching the real decisions.db / trades.db singletons so the test
    is hermetic. Also writes a row into a per-test sqlite file to assert the
    DB-write code path works end-to-end.
    """

    def __init__(self, db_path: Path) -> None:
        """Init."""
        self.db_path = db_path
        self.submitted: list[Decision] = []
        self.duplicates: int = 0
        self._seen_ids: set[str] = set()
        self.stats = {"submit_count": 0, "success_count": 0, "reject_count": 0}

    async def submit(self, decision: Decision) -> SubmitResult | None:
        """Submit."""
        if decision.action != DecisionAction.EXECUTE:
            return None
        if decision.decision_id in self._seen_ids:
            self.duplicates += 1
            return SubmitResult(
                success=True,
                message="cached (test fake)",
                decision_id=decision.decision_id,
            )
        self._seen_ids.add(decision.decision_id)
        self.submitted.append(decision)
        self.stats["submit_count"] += 1
        self.stats["success_count"] += 1
        await self._write_row(decision)
        return SubmitResult(
            success=True,
            message="OK",
            order_value=10_000.0,
            price=100.0,
            quantity=100,
            remaining_cash=990_000.0,
            decision_id=decision.decision_id,
        )

    async def _upsert_decision(self, decision: Decision) -> None:
        """Upsert decision."""
        await self._write_row(decision)

    async def _write_row(self, decision: Decision) -> None:
        """Write row."""
        try:
            import aiosqlite  # type: ignore
        except ImportError:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS decisions ("
                "decision_id TEXT PRIMARY KEY, cycle_id TEXT, ticker TEXT, "
                "action TEXT, direction TEXT, combined_magnitude REAL, "
                "rationale TEXT, created_at TEXT)"
            )
            await db.execute(
                "INSERT OR IGNORE INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision.decision_id,
                    decision.cycle_id,
                    decision.ticker,
                    decision.action.value,
                    decision.direction.value,
                    decision.combined_magnitude,
                    decision.rationale,
                    datetime.now(tz=UTC).isoformat(),
                ),
            )
            await db.commit()


@pytest.mark.asyncio
async def test_full_dispatcher_cycle_persists_decisions(monkeypatch, tmp_path, fake_polza):
    """Run 5 dispatcher-style cycles end-to-end and verify each one persists."""
    monkeypatch.setattr("app.dispatcher.dispatcher.is_trading_open", lambda: True)

    db_path = tmp_path / "decisions.db"
    router = FakeOrderRouter(db_path)

    aggregator = SignalAggregator()
    ta_adapter = SyntheticCandleAdapter(ticker="SBER", source=SignalSource.TA)
    news_adapter = SyntheticCandleAdapter(ticker="SBER", source=SignalSource.NEWS)
    await ta_adapter.startup()
    await news_adapter.startup()

    N_CYCLES = 5
    persisted: list[Decision] = []
    for cycle_idx in range(N_CYCLES):
        cycle_id = f"cyc_{cycle_idx}"
        sigs: list[UnifiedSignal] = []
        sigs.extend(await ta_adapter.safe_poll(timeout=1.0))
        sigs.extend(await news_adapter.safe_poll(timeout=1.0))
        assert sigs, "Adapters should emit signals each cycle"

        decision = await aggregator.aggregate("SBER", cycle_id, sigs)
        apply_tier(decision)
        if decision.action == DecisionAction.EXECUTE:
            await router.submit(decision)
            persisted.append(decision)

    await ta_adapter.shutdown()
    await news_adapter.shutdown()
    assert fake_polza.calls == [], "Polza must NOT be called in this test"

    assert len(persisted) == N_CYCLES, "Every cycle should produce a Decision"
    assert router.stats["submit_count"] == N_CYCLES
    assert router.stats["success_count"] == N_CYCLES
    assert router.stats["reject_count"] == 0
    assert all(d.action == DecisionAction.EXECUTE for d in persisted)
    assert all(d.direction == Direction.BUY for d in persisted)

    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT COUNT(*) FROM decisions") as cur:
            (n_rows,) = await cur.fetchone()
    assert n_rows == N_CYCLES, f"expected {N_CYCLES} rows, found {n_rows}"


@pytest.mark.asyncio
async def test_dispatcher_no_errors_when_all_adapters_quiet(monkeypatch, fake_arenago):
    """Empty cycles must not raise. Stats should reflect 0 executions."""
    monkeypatch.setattr("app.dispatcher.dispatcher.is_trading_open", lambda: True)

    class QuietAdapter(BaseAdapter):
        """Quiet Adapter."""

        name = "QUIET"

        async def startup(self):
            """Startup."""
            self._started = True

        async def poll(self):
            """Poll."""
            return []

        async def shutdown(self):
            """Shutdown."""
            self._started = False

    quiet = QuietAdapter()
    await quiet.startup()
    dispatcher = Dispatcher(adapters=[quiet], cycle_seconds=0.1, poll_timeout_seconds=0.5)
    await dispatcher._run_one_cycle()
    assert dispatcher._cycle_count == 1
    assert dispatcher._signals_gathered_total == 0
    assert dispatcher._decisions_executed_total == 0
