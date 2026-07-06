"""
tests/integration/test_failure_modes.py — Chaos-engineering coverage for
catastrophic external failures.

Each test simulates a hostile environment (corrupt DB, NaN candles, every
upstream timing out, etc.) and asserts the bot stays alive AND that the
patches landed during the 2026-05-26 sweep are wired up. The contract is
"degrade gracefully, log, continue". A crash here means the autonomous
trader would have died mid-evaluation on the hackathon VM.

Each test is intentionally narrow: it stresses ONE failure mode end to
end, with no real network calls, so the suite stays under 1 s wall-clock.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from unittest.mock import AsyncMock

import pytest

import app.config as cfg
from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    Direction,
    RiskCheckResult,
    SignalSource,
    UnifiedSignal,
)
from app.recovery import RecoveryStateManager


def test_bootstrap_db_rebuilds_corrupt_decisions_db(tmp_path, monkeypatch):
    """A corrupt decisions.db at boot must NOT crash the bootstrap call.
    Instead the file is unlinked and recreated from scratch."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    import importlib

    from scripts import bootstrap_db as bdb

    importlib.reload(bdb)

    decisions_path = tmp_path / "decisions.db"
    decisions_path.write_bytes(b"XXX_NOT_A_REAL_SQLITE_FILE_XXX" * 200)

    bdb.create_decisions_db()

    assert decisions_path.exists()
    with sqlite3.connect(str(decisions_path)) as cn:
        names = {
            row[0]
            for row in cn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    assert "decisions" in names
    assert "budget_log" in names


def test_bootstrap_db_preserves_intact_db(tmp_path, monkeypatch):
    """A valid pre-existing DB must NOT be wiped — verify the heal-only
    path doesn't accidentally nuke healthy data on every boot."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    import importlib

    from scripts import bootstrap_db as bdb

    importlib.reload(bdb)

    bdb.create_decisions_db()
    with sqlite3.connect(str(tmp_path / "decisions.db")) as cn:
        cn.execute(
            "INSERT INTO decisions (decision_id, cycle_id, ticker, action, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            ("survives_reboot", "c0", "SBER", "EXECUTE", "2026-05-26T00:00:00"),
        )
        cn.commit()

    bdb.create_decisions_db()
    with sqlite3.connect(str(tmp_path / "decisions.db")) as cn:
        row = cn.execute(
            "SELECT decision_id FROM decisions WHERE decision_id = ?",
            ("survives_reboot",),
        ).fetchone()
    assert row is not None
    assert row[0] == "survives_reboot"


def _make_nan_signal(ticker: str = "SBER") -> UnifiedSignal:
    """Signal carrying NaN where price/ATR should be — simulates corrupt
    ISS data feeding a buggy detector."""
    sig = UnifiedSignal(
        source=SignalSource.TA,
        detector="chaos_test",
        ticker=ticker,
        direction=Direction.BUY,
        magnitude=0.6,
        raw_confidence=0.6,
        horizon_min=60,
        price=float("nan"),
        entry_level=100.0,
        stop_level=98.0,
        target_level=104.0,
        expected_rr=2.0,
        atr=0.5,
    )
    return sig


@pytest.mark.asyncio
async def test_risk_manager_rejects_nan_price():
    """A signal with NaN price must be rejected with a 'sanity:' rationale.
    Without the chaos fix, NaN slipped past `qty <= 0` (NaN < 0 is False)
    and we'd POST quantity=0 or NaN to ArenaGo."""
    from app.risk.risk_manager import RiskManager

    rm = RiskManager(deposit_total=1_000_000.0, bot_name="chaos_bot")
    decision = Decision(
        decision_id="d_nan_price",
        cycle_id="c0",
        ticker="SBER",
        action=DecisionAction.EXECUTE,
        direction=Direction.BUY,
        combined_magnitude=0.6,
        expected_rr=2.0,
        signals=[_make_nan_signal()],
    )
    result = await rm.evaluate(decision)
    assert result.result == RiskCheckResult.REJECTED_HARD_CAP
    assert result.reason.startswith("sanity:")
    assert "nan" in result.reason.lower()


@pytest.mark.asyncio
async def test_risk_manager_rejects_inf_combined_magnitude():
    """+inf combined_magnitude (could come from divide-by-zero in
    aggregator weighting) must trigger the sanity reject."""
    from app.risk.risk_manager import RiskManager

    rm = RiskManager(deposit_total=1_000_000.0, bot_name="chaos_bot")
    good_sig = UnifiedSignal(
        source=SignalSource.TA,
        detector="ok",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=0.6,
        raw_confidence=0.6,
        horizon_min=60,
        price=100.0,
        entry_level=100.0,
        stop_level=98.0,
        target_level=104.0,
        expected_rr=2.0,
        atr=0.5,
    )
    decision = Decision(
        decision_id="d_inf_mag",
        cycle_id="c0",
        ticker="SBER",
        action=DecisionAction.EXECUTE,
        direction=Direction.BUY,
        combined_magnitude=float("inf"),
        expected_rr=2.0,
        signals=[good_sig],
    )
    result = await rm.evaluate(decision)
    assert result.result == RiskCheckResult.REJECTED_HARD_CAP
    assert "combined_magnitude" in result.reason


@pytest.mark.asyncio
async def test_recovery_corrupt_json_logs_and_starts_fresh(tmp_path, caplog):
    """Garbage in recovery_state.json must result in load() returning None
    (logged as ERROR), NOT raising — otherwise the process crash-loops."""
    snap_path = tmp_path / "recovery_state.json"
    snap_path.write_text("}{ this is not json at all", encoding="utf-8")

    mgr = RecoveryStateManager(path=snap_path)
    with caplog.at_level("ERROR"):
        loaded = mgr.load()

    assert loaded is None
    await mgr.save_atomic(RecoveryStateManager.build_snapshot(hmm_regime="trending"))
    fresh = mgr.load()
    assert fresh is not None
    assert fresh.hmm_regime == "trending"


@pytest.mark.asyncio
async def test_recovery_cleans_old_orphan_tmp_files(tmp_path):
    """SIGKILL during a save leaves a `.recovery_state_*.tmp` orphan. The
    next save must reap orphans older than 5 minutes so /data doesn't fill
    up over a long-running container."""
    snap_path = tmp_path / "recovery_state.json"

    fresh_orphan = tmp_path / ".recovery_state_fresh.tmp"
    fresh_orphan.write_text("recent")

    old_orphan = tmp_path / ".recovery_state_old.tmp"
    old_orphan.write_text("ancient")
    ten_min_ago = time.time() - 600.0
    os.utime(old_orphan, (ten_min_ago, ten_min_ago))

    mgr = RecoveryStateManager(path=snap_path)
    await mgr.save_atomic(RecoveryStateManager.build_snapshot(hmm_regime="ranging"))

    assert fresh_orphan.exists(), "fresh orphan must NOT be reaped"
    assert not old_orphan.exists(), "old orphan must be cleaned up"
    assert snap_path.exists()


class _SilentAdapter:
    """Adapter that always returns no signals (simulates polite failure
    after an upstream timeout)."""

    def __init__(self, name: str = "silent") -> None:
        """Init."""
        self.name = name
        self._started = True
        self._call_count = 0
        self._error_count = 0

    async def startup(self) -> None:
        """Startup."""
        pass

    async def shutdown(self) -> None:
        """Shutdown."""
        pass

    async def safe_poll(self, timeout: float = 5.0):
        """Safe poll."""
        self._call_count += 1
        return []

    async def poll(self):
        """Poll."""
        return []

    @property
    def stats(self):
        """Stats."""
        return {"adapter": self.name, "calls": self._call_count, "errors": 0, "error_rate": 0.0}


@pytest.mark.asyncio
async def test_dispatcher_critical_log_after_n_empty_cycles(caplog, monkeypatch):
    """When every adapter returns [] for `_empty_polls_threshold` consecutive
    cycles AND the market is open, the dispatcher emits CRITICAL so the
    operator sees that the bot has gone deaf (e.g. ISS + AlgoPack + Polza
    all down simultaneously)."""
    from app.dispatcher.aggregator import SignalAggregator
    from app.dispatcher.dispatcher import Dispatcher

    monkeypatch.setattr("app.dispatcher.dispatcher.is_trading_open", lambda: True)

    d = Dispatcher(
        adapters=[_SilentAdapter("a1"), _SilentAdapter("a2")],
        aggregator=SignalAggregator(),
        cycle_seconds=1.0,
        poll_timeout_seconds=0.5,
    )

    with caplog.at_level("CRITICAL"):
        for _ in range(3):
            await d._run_one_cycle()

    assert d._consecutive_empty_polls == 3
    critical_msgs = [r.message for r in caplog.records if r.levelname == "CRITICAL"]
    assert any("Dispatcher deaf" in m for m in critical_msgs)


@pytest.mark.asyncio
async def test_dispatcher_resets_counter_on_recovery(monkeypatch):
    """After the upstreams come back, a single non-empty cycle must reset
    the counter so the next outage triggers a fresh CRITICAL alert."""
    from app.dispatcher.aggregator import SignalAggregator
    from app.dispatcher.dispatcher import Dispatcher

    monkeypatch.setattr("app.dispatcher.dispatcher.is_trading_open", lambda: True)

    sig = UnifiedSignal(
        source=SignalSource.TA,
        detector="ok",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=0.6,
        raw_confidence=0.6,
        horizon_min=60,
        price=100.0,
        entry_level=100.0,
        stop_level=98.0,
        target_level=104.0,
        expected_rr=2.0,
        atr=0.5,
    )

    class _OnceAdapter(_SilentAdapter):
        """Once Adapter."""

        async def safe_poll(self, timeout: float = 5.0):
            """Safe poll."""
            return [sig]

    d = Dispatcher(
        adapters=[_SilentAdapter("a")],
        aggregator=SignalAggregator(),
        cycle_seconds=1.0,
        poll_timeout_seconds=0.5,
    )

    monkeypatch.setattr(d.aggregator, "aggregate_batch", AsyncMock(return_value=[]))
    monkeypatch.setattr(d, "_fetch_supercandles_batch", AsyncMock(return_value={}))

    await d._run_one_cycle()
    await d._run_one_cycle()
    assert d._consecutive_empty_polls == 2

    d.adapters = [_OnceAdapter("oa")]
    await d._run_one_cycle()
    assert d._consecutive_empty_polls == 0


@pytest.mark.asyncio
async def test_polza_chat_returns_neutral_on_timeout(monkeypatch):
    """A deadlocked Polza connection that never returns must NOT stall the
    dispatcher for the SDK's 60 s default — the chat() wrapper now caps each
    call at cfg.POLZA_REQUEST_TIMEOUT_SEC and returns a NEUTRAL stub."""
    from app.llm.polza_client import PolzaClient

    pc = PolzaClient()

    class _NeverReturns:
        """Never Returns."""

        class chat:  # noqa: N801 — mimic openai.AsyncClient interface
            """Chat."""

            class completions:  # noqa: N801
                """Completions."""

                @staticmethod
                async def create(**kwargs):
                    """Create."""
                    await asyncio.sleep(60.0)

    pc._client = _NeverReturns()
    pc._auth_disabled_until = 0.0
    monkeypatch.setattr(cfg, "POLZA_REQUEST_TIMEOUT_SEC", 0.2)
    pc._check_budget_and_select = AsyncMock(side_effect=lambda m: m)
    pc._get_cached = AsyncMock(return_value=None)

    started = time.monotonic()
    result = await pc.chat(
        messages=[{"role": "user", "content": "hello"}],
        model=cfg.POLZA_MODEL_REACTIVE,
        purpose="chaos_timeout_test",
        use_cache=False,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, f"chat() didn't honor timeout: {elapsed:.1f}s"
    assert result.get("timeout") is True
    parsed = json.loads(result["content"])
    assert parsed["direction"] == "NEUTRAL"
    assert parsed["magnitude"] == 0.0


@pytest.mark.asyncio
async def test_recovery_stale_snapshot_still_loads(tmp_path, caplog):
    """A 2-day-old snapshot should still load (with a WARN), not be rejected.
    Operators rely on stale state for the n_trades_today + decision-id dedup
    after long downtime windows."""
    snap_path = tmp_path / "recovery_state.json"

    two_days_ago = time.time() - 2 * 86400
    payload = {
        "schema_version": 1,
        "last_save_ts_utc": two_days_ago,
        "circuit_state": {"daily_pnl_rub": -1000},
        "hmm_regime": "ranging",
        "last_decision_ids": ["dX1", "dX2"],
        "meta_score_history": [0.55],
        "daily_turnover_rub": 99000,
        "n_trades_today": 4,
        "extras": {},
    }
    snap_path.write_text(json.dumps(payload), encoding="utf-8")

    mgr = RecoveryStateManager(path=snap_path)
    with caplog.at_level("WARNING"):
        loaded = mgr.load()

    assert loaded is not None
    assert loaded.hmm_regime == "ranging"
    assert loaded.last_decision_ids == ["dX1", "dX2"]
    stale_msgs = [r.message for r in caplog.records if "stale" in r.message.lower()]
    assert stale_msgs


class _ExplodingAdapter:
    """Adapter whose poll() raises — should NOT poison the dispatcher cycle."""

    def __init__(self, name: str = "boom") -> None:
        """Init."""
        self.name = name
        self._started = True
        self._call_count = 0
        self._error_count = 0

    async def startup(self) -> None:
        """Startup."""
        pass

    async def shutdown(self) -> None:
        """Shutdown."""
        pass

    async def safe_poll(self, timeout: float = 5.0):
        """Safe poll."""
        self._call_count += 1
        try:
            raise RuntimeError("simulated detector blow-up")
        except Exception:
            self._error_count += 1
            return []

    @property
    def stats(self):
        """Stats."""
        return {
            "adapter": self.name,
            "calls": self._call_count,
            "errors": self._error_count,
            "error_rate": self._error_count / max(1, self._call_count),
        }


@pytest.mark.asyncio
async def test_dispatcher_isolates_per_adapter_exceptions(monkeypatch):
    """If a single adapter raises but others return signals, the dispatcher
    must still process the good signals (no aggregator skip)."""
    from app.dispatcher.aggregator import SignalAggregator
    from app.dispatcher.dispatcher import Dispatcher

    monkeypatch.setattr("app.dispatcher.dispatcher.is_trading_open", lambda: True)

    good_sig = UnifiedSignal(
        source=SignalSource.TA,
        detector="ok",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=0.6,
        raw_confidence=0.6,
        horizon_min=60,
        price=100.0,
        entry_level=100.0,
        stop_level=98.0,
        target_level=104.0,
        expected_rr=2.0,
        atr=0.5,
    )

    class _HealthyAdapter(_SilentAdapter):
        """Healthy Adapter."""

        async def safe_poll(self, timeout: float = 5.0):
            """Safe poll."""
            return [good_sig]

    d = Dispatcher(
        adapters=[_ExplodingAdapter("boom"), _HealthyAdapter("ok")],
        aggregator=SignalAggregator(),
        cycle_seconds=1.0,
        poll_timeout_seconds=0.5,
    )
    monkeypatch.setattr(d.aggregator, "aggregate_batch", AsyncMock(return_value=[]))
    monkeypatch.setattr(d, "_fetch_supercandles_batch", AsyncMock(return_value={}))

    await d._run_one_cycle()
    assert d._consecutive_empty_polls == 0
