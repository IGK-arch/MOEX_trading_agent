"""
tests/unit/test_recovery_state.py — Atomic save/load + corruption handling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.recovery import SCHEMA_VERSION, RecoveryStateManager


@pytest.mark.asyncio
async def test_save_then_load_roundtrip(tmp_path: Path):
    """Test save then load roundtrip."""
    path = tmp_path / "recovery.json"
    mgr = RecoveryStateManager(path=path)

    snap = RecoveryStateManager.build_snapshot(
        circuit_state_dict={"daily_pnl_rub": 1234.5, "n_trades_today": 7},
        hmm_regime="trending",
        last_decision_ids=["d1", "d2", "d3"],
        meta_score_history=[0.6, 0.7, 0.55],
        daily_turnover_rub=500_000.0,
        n_trades_today=7,
        extras={"version": "test"},
    )
    await mgr.save_atomic(snap)
    assert path.exists()

    loaded = mgr.load()
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.hmm_regime == "trending"
    assert loaded.last_decision_ids == ["d1", "d2", "d3"]
    assert loaded.meta_score_history == [0.6, 0.7, 0.55]
    assert loaded.daily_turnover_rub == 500_000.0
    assert loaded.n_trades_today == 7
    assert loaded.circuit_state["daily_pnl_rub"] == 1234.5


@pytest.mark.asyncio
async def test_load_returns_none_on_missing_file(tmp_path: Path):
    """Test load returns none on missing file."""
    mgr = RecoveryStateManager(path=tmp_path / "does_not_exist.json")
    assert mgr.load() is None


@pytest.mark.asyncio
async def test_load_returns_none_on_corrupt_file(tmp_path: Path):
    """Test load returns none on corrupt file."""
    path = tmp_path / "recovery.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    mgr = RecoveryStateManager(path=path)
    assert mgr.load() is None


@pytest.mark.asyncio
async def test_save_truncates_lists(tmp_path: Path):
    """build_snapshot must clip last_decision_ids to 100 and meta to 200."""
    snap = RecoveryStateManager.build_snapshot(
        last_decision_ids=[f"d{i}" for i in range(200)],
        meta_score_history=[0.5] * 500,
    )
    assert len(snap.last_decision_ids) == 100

    assert snap.last_decision_ids[-1] == "d199"
    assert len(snap.meta_score_history) == 200


@pytest.mark.asyncio
async def test_save_overwrites_atomically(tmp_path: Path):
    """Two consecutive saves must result in the second one being readable."""
    path = tmp_path / "recovery.json"
    mgr = RecoveryStateManager(path=path)
    s1 = RecoveryStateManager.build_snapshot(hmm_regime="trending", n_trades_today=1)
    s2 = RecoveryStateManager.build_snapshot(hmm_regime="crisis", n_trades_today=2)
    await mgr.save_atomic(s1)
    await mgr.save_atomic(s2)
    loaded = mgr.load()
    assert loaded is not None
    assert loaded.hmm_regime == "crisis"
    assert loaded.n_trades_today == 2


@pytest.mark.asyncio
async def test_json_is_parseable(tmp_path: Path):
    """The file content must be valid JSON (no half-writes after rename)."""
    path = tmp_path / "recovery.json"
    mgr = RecoveryStateManager(path=path)
    snap = RecoveryStateManager.build_snapshot(hmm_regime="trending")
    await mgr.save_atomic(snap)
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["hmm_regime"] == "trending"
    assert parsed["schema_version"] == SCHEMA_VERSION
