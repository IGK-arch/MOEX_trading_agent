"""Tests for guarded reflection-driven runtime parameter control."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

import app.config as cfg
from app.memory.reflexive_overrides import (
    apply_reflexive_adjustments,
    apply_saved_reflexive_overrides,
)


@pytest.fixture
def data_dir(monkeypatch):
    path = Path.cwd() / f".test_reflexive_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "DATA_DIR", path)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _reset_meta(monkeypatch, data_dir):
    monkeypatch.setattr(cfg, "REFLEXIVE_CONTROL_ENABLED", True)
    monkeypatch.setattr(cfg, "REFLEXIVE_MIN_TRADES", 3)
    monkeypatch.setattr(cfg, "REFLEXIVE_MIN_CONFIDENCE", 0.65)
    monkeypatch.setattr(cfg, "META_MIN_PROBA", 0.55)
    monkeypatch.setattr(cfg, "META_MIN_PROBA_FLOOR", 0.45)
    monkeypatch.setattr(cfg, "META_MIN_PROBA_CEILING", 0.70)
    yield


def test_profitable_day_can_loosen_meta_slightly(data_dir):
    payload = {
        "parameter_adjustments": [
            {
                "parameter": "META_MIN_PROBA",
                "direction": "decrease",
                "delta": 0.05,
                "confidence": 0.80,
                "reason": "High win-rate and several missed high-confidence setups.",
            }
        ]
    }
    summary = apply_reflexive_adjustments(
        payload,
        date_str="2026-05-27",
        day_stats={"n_trades": 12, "total_pnl_rub": 5000, "win_rate": 0.75},
    )
    assert summary["applied"]
    assert cfg.META_MIN_PROBA == pytest.approx(0.53)

    saved = json.loads((data_dir / "runtime_overrides.json").read_text(encoding="utf-8"))
    assert saved["reflexive_control"]["applied"][0]["parameter"] == "META_MIN_PROBA"


def test_weak_day_blocks_loosening():
    payload = {
        "parameter_adjustments": [
            {
                "parameter": "META_MIN_PROBA",
                "direction": "decrease",
                "delta": 0.02,
                "confidence": 0.90,
                "reason": "LLM wants more trades despite a weak day.",
            }
        ]
    }
    summary = apply_reflexive_adjustments(
        payload,
        date_str="2026-05-27",
        day_stats={"n_trades": 10, "total_pnl_rub": -1000, "win_rate": 0.30},
    )
    assert not summary["applied"]
    assert "blocked loosening" in summary["skipped"][0]["reason"]
    assert cfg.META_MIN_PROBA == pytest.approx(0.55)


def test_weak_day_allows_tightening():
    payload = {
        "parameter_adjustments": [
            {
                "parameter": "META_MIN_PROBA",
                "direction": "increase",
                "delta": 0.04,
                "confidence": 0.90,
                "reason": "Losses came from low-quality marginal meta scores.",
            }
        ]
    }
    summary = apply_reflexive_adjustments(
        payload,
        date_str="2026-05-27",
        day_stats={"n_trades": 10, "total_pnl_rub": -1000, "win_rate": 0.30},
    )
    assert summary["applied"]
    assert cfg.META_MIN_PROBA == pytest.approx(0.57)


def test_low_sample_day_skips_all_changes():
    payload = {
        "parameter_adjustments": [
            {
                "parameter": "META_MIN_PROBA",
                "direction": "increase",
                "delta": 0.01,
                "confidence": 0.90,
                "reason": "Not enough data should block this.",
            }
        ]
    }
    summary = apply_reflexive_adjustments(
        payload,
        date_str="2026-05-27",
        day_stats={"n_trades": 1, "total_pnl_rub": 100, "win_rate": 1.0},
    )
    assert not summary["applied"]
    assert "not enough trades" in summary["skipped"][0]["reason"]


def test_saved_reflexive_override_is_restored_after_restart(data_dir):
    payload = {
        "parameter_adjustments": [
            {
                "parameter": "META_MIN_PROBA",
                "direction": "increase",
                "delta": 0.01,
                "confidence": 0.90,
                "reason": "Marginal low-quality trades should tighten meta gate.",
            }
        ]
    }
    apply_reflexive_adjustments(
        payload,
        date_str="2026-05-27",
        day_stats={"n_trades": 8, "total_pnl_rub": -500, "win_rate": 0.25},
    )

    cfg.META_MIN_PROBA = 0.55
    restored = apply_saved_reflexive_overrides()

    assert restored["applied"][0]["parameter"] == "META_MIN_PROBA"
    assert cfg.META_MIN_PROBA == pytest.approx(0.56)
