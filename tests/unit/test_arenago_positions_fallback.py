"""Unit tests for ArenaGoClient get_positions_safe fallback chain (v0.19.6).

When /api/positions returns 404 on all retries, the client should
fall back to:
  1. /api/trades reconstruction (cumulative qty per ticker)
  2. recovery_state.json cached positions

We never let the bot believe positions vanished due to broker 404.
"""

from __future__ import annotations

import json

import pytest

from app.execution.arenago_client import ArenaGoClient


def test_reconstruct_positions_from_trades_basic() -> None:
    """Cumulative qty + VWAP correctly computed from BUYs and SELLs."""
    trades = [
        {"secid": "SBER", "direction": "BUY", "quantity": 100, "price": 300.0},
        {"secid": "SBER", "direction": "BUY", "quantity": 50, "price": 310.0},
        {"secid": "SBER", "direction": "SELL", "quantity": 30, "price": 320.0},
        {"secid": "GAZP", "direction": "SELL", "quantity": 70, "price": 116.0},
    ]
    result = ArenaGoClient._reconstruct_positions_from_trades(trades)
    by_ticker = {r["secid"]: r for r in result}
    assert by_ticker["SBER"]["position"] == 120
    assert by_ticker["SBER"]["average_price"] == pytest.approx(303.333, rel=1e-3)
    assert by_ticker["GAZP"]["position"] == -70
    assert by_ticker["GAZP"]["average_price"] == 0.0


def test_reconstruct_handles_alt_field_names() -> None:
    """Tolerate ticker/side/qty/price field aliases."""
    trades = [
        {"ticker": "LKOH", "side": "B", "qty": 10, "price": 4000.0},
        {"ticker": "LKOH", "side": "S", "qty": 4, "price": 4100.0},
    ]
    result = ArenaGoClient._reconstruct_positions_from_trades(trades)
    assert len(result) == 1
    assert result[0]["secid"] == "LKOH"
    assert result[0]["position"] == 6


def test_reconstruct_empty_input_yields_empty_list() -> None:
    """Empty/None trades → empty list, no exception."""
    assert ArenaGoClient._reconstruct_positions_from_trades([]) == []
    assert ArenaGoClient._reconstruct_positions_from_trades(None) == []


def test_reconstruct_zero_net_position_dropped() -> None:
    """Positions that net to zero are dropped (no open exposure)."""
    trades = [
        {"secid": "PIKK", "direction": "BUY", "quantity": 5, "price": 540.0},
        {"secid": "PIKK", "direction": "SELL", "quantity": 5, "price": 545.0},
    ]
    result = ArenaGoClient._reconstruct_positions_from_trades(trades)
    assert result == []


def test_load_recovery_positions_missing_file(tmp_path, monkeypatch) -> None:
    """Missing recovery_state.json → empty list + inf age."""
    monkeypatch.setattr("app.config.RECOVERY_STATE_PATH", tmp_path / "missing.json")
    client = ArenaGoClient()
    positions, age = client._load_recovery_positions()
    assert positions == []
    assert age == float("inf")


def test_load_recovery_positions_valid_file(tmp_path, monkeypatch) -> None:
    """Recovery state with open_positions returns parsed list."""
    import time as _time

    state_path = tmp_path / "recovery_state.json"
    state = {
        "schema_version": 1,
        "last_save_ts_utc": _time.time() - 100.0,
        "open_positions": [
            {"ticker": "SBER", "quantity": 36, "avg_price": 320.05, "bot": "test-bot"},
            {"ticker": "GAZP", "quantity": -70, "avg_price": 116.20, "bot": "test-bot"},
            {"ticker": "EMPTY", "quantity": 0, "avg_price": 0.0},
        ],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr("app.config.RECOVERY_STATE_PATH", state_path)

    client = ArenaGoClient()
    positions, age = client._load_recovery_positions()
    assert len(positions) == 2
    tickers = {p["secid"] for p in positions}
    assert tickers == {"SBER", "GAZP"}
    assert age < 200.0


def test_load_recovery_positions_stale_file(tmp_path, monkeypatch) -> None:
    """Stale cache (>1h old) still returns positions but with large age."""
    import time as _time

    state_path = tmp_path / "recovery_state.json"
    state = {
        "schema_version": 1,
        "last_save_ts_utc": _time.time() - 7200.0,
        "open_positions": [
            {"ticker": "VTBR", "quantity": 185, "avg_price": 86.84},
        ],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr("app.config.RECOVERY_STATE_PATH", state_path)

    client = ArenaGoClient()
    positions, age = client._load_recovery_positions()
    assert len(positions) == 1
    assert age > 3600.0
