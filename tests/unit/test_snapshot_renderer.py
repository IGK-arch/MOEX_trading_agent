"""
tests/unit/test_snapshot_renderer.py — Static HTML snapshot + metrics writer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.dashboard.metrics_writer import build_metrics_snapshot, write_once
from app.dashboard.snapshot_renderer import (
    get_dashboard_mode,
    render_snapshot_html,
)


def test_build_metrics_snapshot_has_required_fields():
    """Test build metrics snapshot has required fields."""
    snap = build_metrics_snapshot()

    for key in (
        "ts_utc",
        "equity_rub",
        "daily_pnl_pct",
        "max_dd_pct",
        "current_dd_pct",
        "n_trades_today",
        "n_open_positions",
        "hmm_regime",
        "run_mode",
        "live_sizing",
        "meta_min_proba",
    ):
        assert key in snap, f"snapshot missing key: {key}"


def test_render_snapshot_html_is_valid_string():
    """Test render snapshot html is valid string."""
    body = render_snapshot_html(refresh_sec=60)
    assert isinstance(body, str)
    assert body.startswith("<!DOCTYPE html>")

    assert "404: Loss Not Found" in body

    assert "Капитал" in body
    assert "Последние решения" in body
    assert 'http-equiv="refresh"' in body
    assert 'content="60"' in body


@pytest.mark.asyncio
async def test_write_once_creates_both_files(tmp_path: Path):
    """Test write once creates both files."""
    jsonl = tmp_path / "metrics_live.jsonl"
    summary = tmp_path / "metrics_summary.json"
    await write_once(jsonl, summary)
    assert jsonl.exists()
    assert summary.exists()

    line = jsonl.read_text(encoding="utf-8").strip()
    obj = json.loads(line)
    assert "ts_utc" in obj
    summary_obj = json.loads(summary.read_text(encoding="utf-8"))
    assert "ts_utc" in summary_obj


@pytest.mark.asyncio
async def test_write_once_appends(tmp_path: Path):
    """Test write once appends."""
    jsonl = tmp_path / "metrics_live.jsonl"
    summary = tmp_path / "metrics_summary.json"
    await write_once(jsonl, summary)
    await write_once(jsonl, summary)
    lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_get_dashboard_mode_defaults_to_snapshot(tmp_path: Path, monkeypatch):
    """Test get dashboard mode defaults to snapshot."""
    import app.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    assert get_dashboard_mode() == "snapshot"


def test_get_dashboard_mode_reads_file(tmp_path: Path, monkeypatch):
    """Test get dashboard mode reads file."""
    import app.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    (tmp_path / "dashboard_mode.txt").write_text("external\n", encoding="utf-8")
    assert get_dashboard_mode() == "external"


def test_get_dashboard_mode_ignores_invalid(tmp_path: Path, monkeypatch):
    """Test get dashboard mode ignores invalid."""
    import app.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    (tmp_path / "dashboard_mode.txt").write_text("garbage\n", encoding="utf-8")
    assert get_dashboard_mode() == "snapshot"
