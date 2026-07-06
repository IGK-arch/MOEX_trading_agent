"""Unit tests for app.training.evening_pipeline (Phase 30 / v0.19.6).

Smoke + step-level tests for the daily evening retrain orchestrator.
We avoid spinning up real subprocesses for the meta/HMM scripts —
the test harness patches them out.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from app.training import evening_pipeline as ep_mod
from app.training.evening_pipeline import (
    DEFAULT_LOOKBACK_DAYS_ROLLING,
    EveningPipeline,
    get_evening_pipeline,
)


def test_singleton_returns_same_instance() -> None:
    """Test singleton returns same instance."""
    a = get_evening_pipeline()
    b = get_evening_pipeline()
    assert a is b
    assert isinstance(a, EveningPipeline)


def test_sql_count_missing_db_returns_zero(tmp_path) -> None:
    """Missing DB file → 0 (no crash)."""
    pipeline = EveningPipeline()
    assert (
        pipeline._sql_count(
            tmp_path / "missing.db",
            "SELECT COUNT(*) FROM foo",
            (),
        )
        == 0
    )


def test_sql_query_missing_db_returns_empty_list(tmp_path) -> None:
    """Test sql query missing db returns empty list."""
    pipeline = EveningPipeline()
    assert (
        pipeline._sql_query(
            tmp_path / "missing.db",
            "SELECT * FROM foo",
            (),
        )
        == []
    )


@pytest.mark.asyncio
async def test_snapshot_today_returns_zero_counts_when_db_empty(tmp_path, monkeypatch) -> None:
    """Snapshot step should work even when DBs don't exist yet."""
    monkeypatch.setattr("app.config.DATA_DIR", tmp_path)
    pipeline = EveningPipeline()
    res = await pipeline._snapshot_today()
    assert res["ok"] is True
    assert res["decisions_total"] == 0
    assert res["trades_count"] == 0


@pytest.mark.asyncio
async def test_update_direction_bias_writes_overlay(tmp_path, monkeypatch) -> None:
    """Round-trip wins/losses tally + JSON dump verified."""
    monkeypatch.setattr("app.config.DATA_DIR", tmp_path)
    overlay_path = tmp_path / "direction_bias_observed.json"
    monkeypatch.setattr(ep_mod, "DIRECTION_BIAS_OBS_PATH", overlay_path)

    trades_db = tmp_path / "trades.db"
    cn = sqlite3.connect(str(trades_db))
    cn.execute(
        "CREATE TABLE trades (ticker TEXT, direction TEXT, quantity INTEGER, "
        "price REAL, trade_date TEXT, created_at TEXT)"
    )
    today = "2026-05-26"
    cn.execute(
        "INSERT INTO trades VALUES (?,?,?,?,?,?)",
        ("SBER", "BUY", 10, 300.0, today, "2026-05-26T10:00:00+00:00"),
    )
    cn.execute(
        "INSERT INTO trades VALUES (?,?,?,?,?,?)",
        ("SBER", "SELL", 10, 310.0, today, "2026-05-26T11:00:00+00:00"),
    )
    cn.commit()
    cn.close()

    pipeline = EveningPipeline()
    res = await pipeline._update_direction_bias()
    assert res["ok"] is True
    assert overlay_path.exists()
    payload = json.loads(overlay_path.read_text())
    assert "per_ticker" in payload
    assert payload["lookback_days"] == DEFAULT_LOOKBACK_DAYS_ROLLING


@pytest.mark.asyncio
async def test_update_noisy_patterns_writes_blacklist(tmp_path, monkeypatch) -> None:
    """Pattern with 8 trades and 0 wins should appear in noisy list."""
    monkeypatch.setattr("app.config.DATA_DIR", tmp_path)
    blacklist_path = tmp_path / "noisy_patterns.json"
    monkeypatch.setattr(ep_mod, "NOISY_PATTERNS_PATH", blacklist_path)

    decisions_db = tmp_path / "decisions.db"
    cn = sqlite3.connect(str(decisions_db))
    cn.execute(
        "CREATE TABLE decisions (signals_json TEXT, executed_bool INTEGER, "
        "pnl_rub REAL, ticker TEXT, action TEXT, created_at TEXT)"
    )
    now_iso = "2026-05-26T20:00:00+00:00"
    bad_pattern = json.dumps([{"pattern": "rounding_bottom"}])
    for _ in range(8):
        cn.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?)",
            (bad_pattern, 1, -100.0, "T", "EXECUTE", now_iso),
        )
    cn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?)",
        (json.dumps([{"pattern": "bull_pennant"}]), 1, +120.0, "T", "EXECUTE", now_iso),
    )
    cn.commit()
    cn.close()

    pipeline = EveningPipeline()
    res = await pipeline._update_noisy_patterns()
    assert res["ok"] is True
    assert res["patterns_blacklisted"] >= 1
    blob = json.loads(blacklist_path.read_text())
    listed = {p["pattern"] for p in blob["patterns"]}
    assert "rounding_bottom" in listed
