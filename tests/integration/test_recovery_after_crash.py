"""
tests/integration/test_recovery_after_crash.py — Crash + restart resume cycle.

Simulates the SIGKILL-then-restart scenario from the hackathon spec:
  1. Process A fills RecoveryStateManager with circuit state, decisions,
     turnover, regime cache.
  2. Process A is "killed" (we drop the manager reference).
  3. Process B boots fresh, calls load(), and must see all the state from A.
"""

from __future__ import annotations

import json

import pytest

from app.recovery import (
    SCHEMA_VERSION,
    RecoveryStateManager,
)


@pytest.mark.asyncio
async def test_recovery_resumes_from_snapshot(tmp_path):
    """Fill state, kill, restart — verify resume from recovery_state.json."""
    snap_path = tmp_path / "recovery_state.json"

    mgr_a = RecoveryStateManager(path=snap_path)
    decisions = [f"dec_{i:03d}" for i in range(50)]
    meta_scores = [0.55, 0.62, 0.71, 0.49, 0.68]
    snap_a = RecoveryStateManager.build_snapshot(
        circuit_state_dict={
            "daily_pnl_rub": -4_321.0,
            "n_trades_today": 17,
            "losing_streak": 2,
            "winning_streak": 0,
            "blocked_until_iso": None,
            "block_reason": "",
        },
        hmm_regime="trending",
        last_decision_ids=decisions,
        meta_score_history=meta_scores,
        daily_turnover_rub=275_000.0,
        n_trades_today=17,
        extras={"version": "v0.12.0", "phase": "test"},
    )
    await mgr_a.save_atomic(snap_a)
    assert snap_path.exists()
    raw = snap_path.read_text(encoding="utf-8")
    json.loads(raw)

    del mgr_a

    mgr_b = RecoveryStateManager(path=snap_path)
    loaded = mgr_b.load()
    assert loaded is not None, "RecoveryStateManager.load() must hydrate from disk"
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.hmm_regime == "trending"
    assert loaded.last_decision_ids == decisions
    assert loaded.meta_score_history == meta_scores
    assert loaded.daily_turnover_rub == 275_000.0
    assert loaded.n_trades_today == 17
    assert loaded.circuit_state["losing_streak"] == 2
    assert loaded.extras["version"] == "v0.12.0"


@pytest.mark.asyncio
async def test_recovery_survives_partial_write(tmp_path):
    """If a tmp file is left behind by a kill mid-write, the real file
    remains untouched and load() still returns last good snapshot."""
    snap_path = tmp_path / "recovery_state.json"
    mgr = RecoveryStateManager(path=snap_path)

    good_snap = RecoveryStateManager.build_snapshot(
        hmm_regime="ranging",
        last_decision_ids=["d1", "d2"],
        n_trades_today=2,
    )
    await mgr.save_atomic(good_snap)

    orphan = snap_path.parent / ".recovery_state_orphan.tmp"
    orphan.write_text('{ "schema_version": 999, "broken":', encoding="utf-8")

    mgr_restart = RecoveryStateManager(path=snap_path)
    loaded = mgr_restart.load()
    assert loaded is not None
    assert loaded.hmm_regime == "ranging"
    assert loaded.last_decision_ids == ["d1", "d2"]
    assert orphan.exists()


@pytest.mark.asyncio
async def test_recovery_dedupes_after_restart_via_last_decision_ids(tmp_path):
    """The `last_decision_ids` list is what protects us against re-submitting
    decisions that ArenaGo already executed but whose response wasn't yet
    flushed to decisions.db when the crash hit. Verify the list survives."""
    snap_path = tmp_path / "recovery_state.json"
    mgr = RecoveryStateManager(path=snap_path)

    all_ids = [f"d{i:05d}" for i in range(120)]
    snap = RecoveryStateManager.build_snapshot(last_decision_ids=all_ids)
    await mgr.save_atomic(snap)

    loaded = RecoveryStateManager(path=snap_path).load()
    assert loaded is not None
    assert len(loaded.last_decision_ids) == 100
    assert loaded.last_decision_ids[-1] == "d00119"
    assert loaded.last_decision_ids[0] == "d00020"


@pytest.mark.asyncio
async def test_recovery_missing_file_returns_none(tmp_path):
    """Cold first boot — no snapshot yet → load() returns None, not raises."""
    mgr = RecoveryStateManager(path=tmp_path / "fresh.json")
    assert mgr.load() is None
