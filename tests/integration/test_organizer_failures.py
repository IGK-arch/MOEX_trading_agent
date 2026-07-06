"""
tests/integration/test_organizer_failures.py — Stage-2 chaos coverage
for ORGANIZER-SIDE outages.

Background
----------
The MOEX-AI-Hackathon-2026 organizers messaged (2026-05-26 18:53):

    "Да, но возможны непредвиденные сбои, которые мы оперативно будем
    исправлять. С вашей стороны рекомендуем также поработать над
    отказоустойчивостью"

Translation: ArenaGo / ISS / AlgoPack / Polza WILL flicker during Stage 2.
The bot must NOT crash, leak orphans, or melt CPU on infinite retry loops.

What these tests prove
----------------------
Each test forces ONE concrete organizer-side failure mode against the
real ArenaGoClient (or BrokerReconciler) wired against a stub httpx
transport, and asserts the v0.18.0 hardening kicks in:

  1.  502/503/504 → bounded retry, then 60 s circuit-break
  2.  Empty body  → soft-fail SubmitResult, no crash
  3.  Non-JSON   → soft-fail SubmitResult, no crash
  4.  Non-dict JSON → soft-fail SubmitResult, no crash
  5.  Missing fields → coerced via request body, no KeyError
  6.  Connection reset → resubmit ONCE (idempotency via decision_id)
  7.  401 → reauth via fresh SANDBOX_API_KEY, replay request
  8.  401 with no env key → fail fast, no infinite reauth loop
  9.  Daily-limit slowdown (>=950) → WARNING only, still submits
 10.  Daily-limit entry halt (999) → reject entries, accept exits
 11.  Daily-limit hard (1000) → reject ALL incl. exits
 12.  Cash drop without trades → CRITICAL log fires
 13.  Reconciler with transient get_positions failure → preserves local
 14.  Reconciler with confirmed-empty broker → marks closures

Total wall-clock: < 5 s. No real network. Single chaos suite hardened.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.config as cfg


class _FakeResponse:
    """Minimal stand-in for httpx.Response used in submit_order tests."""

    def __init__(
        self,
        status_code: int = 200,
        json_body: Any = None,
        raw_text: str | None = None,
    ) -> None:
        """Init."""
        self.status_code = status_code
        self._json_body = json_body
        if raw_text is not None:
            self.text = raw_text
        elif json_body is not None:
            self.text = json.dumps(json_body)
        else:
            self.text = ""

    def json(self) -> Any:
        """Json."""
        if self._json_body is None:
            raise json.JSONDecodeError("Expecting value", "", 0)
        return self._json_body

    def raise_for_status(self) -> None:
        """Raise for status."""
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=None,  # type: ignore[arg-type]
                response=self,  # type: ignore[arg-type]
            )


class _ScriptedHttpx:
    """Async httpx-like client whose post() returns a scripted sequence."""

    def __init__(self, sequence: list[Any]) -> None:
        """Init."""
        self._sequence = list(sequence)
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, json: Any = None, **kw: Any) -> Any:
        """Post."""
        self.calls.append({"url": url, "json": json})
        if not self._sequence:
            raise RuntimeError("scripted sequence exhausted")
        nxt = self._sequence.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    async def get(self, url: str, **kw: Any) -> Any:
        """Get."""
        return _FakeResponse(status_code=404)

    async def aclose(self) -> None:
        """Aclose."""
        return None


async def _make_client(monkeypatch, sequence: list[Any], api_key: str = "test_key"):
    """Build a started ArenaGoClient whose httpx is the scripted fake."""
    from app.execution import arenago_client as ac

    monkeypatch.setenv("SANDBOX_API_KEY", api_key)
    monkeypatch.setenv("ARENAGO_BOT_NAME", "test_bot")

    monkeypatch.setattr(ac, "ARENAGO_5XX_BACKOFF_BASE", 0.001)
    monkeypatch.setattr(ac, "ARENAGO_5XX_BACKOFF_MAX", 0.005)
    monkeypatch.setattr(ac, "ARENAGO_CIRCUIT_BREAK_SEC", 0.5)

    client = ac.ArenaGoClient()
    client._api_key = api_key
    client._bot_name = "test_bot"
    client._client = _ScriptedHttpx(sequence)
    client._started = True
    client._is_already_executed = AsyncMock(return_value=None)
    client._mark_executed = AsyncMock(return_value=None)
    return client


@pytest.mark.asyncio
async def test_5xx_storm_retries_then_circuit_breaks(monkeypatch):
    """502 every time → after ARENAGO_5XX_RETRY_ATTEMPTS the breaker trips
    and the next call returns a CIRCUIT_BREAKER_OPEN soft-fail."""
    sequence = [_FakeResponse(status_code=502) for _ in range(10)]
    client = await _make_client(monkeypatch, sequence)

    r1 = await client.submit_order(
        direction="BUY", ticker="SBER", quantity=10, decision_id="d_502_a"
    )
    assert r1.success is False
    assert "RETRIES_EXHAUSTED" in r1.message
    from app.execution import arenago_client as ac

    assert len(client._client.calls) == ac.ARENAGO_5XX_RETRY_ATTEMPTS + 1
    assert client._circuit_is_open(), "breaker must be open after 5xx storm"

    n_calls_before = len(client._client.calls)
    r2 = await client.submit_order(
        direction="BUY", ticker="GAZP", quantity=10, decision_id="d_502_b"
    )
    assert r2.success is False
    assert "CIRCUIT_BREAKER_OPEN" in r2.message
    assert len(client._client.calls) == n_calls_before, "network must NOT be touched"


@pytest.mark.asyncio
async def test_empty_response_body_soft_fails(monkeypatch):
    """200 OK but completely empty body must NOT crash on json()."""
    sequence = [_FakeResponse(status_code=200, raw_text="")]
    client = await _make_client(monkeypatch, sequence)

    r = await client.submit_order(
        direction="BUY", ticker="SBER", quantity=10, decision_id="d_empty"
    )
    assert r.success is False
    assert "EMPTY_RESPONSE_BODY" in r.message


@pytest.mark.asyncio
async def test_non_json_response_soft_fails(monkeypatch):
    """200 OK with HTML body must produce soft-fail, not raise."""
    sequence = [_FakeResponse(status_code=200, raw_text="<html>oops</html>")]
    client = await _make_client(monkeypatch, sequence)

    r = await client.submit_order(direction="BUY", ticker="SBER", quantity=10, decision_id="d_html")
    assert r.success is False
    assert "INVALID_JSON" in r.message


@pytest.mark.asyncio
async def test_non_dict_json_soft_fails(monkeypatch):
    """200 OK with a JSON string instead of dict → soft-fail, no crash."""
    sequence = [_FakeResponse(status_code=200, json_body=["unexpected", "list"])]
    client = await _make_client(monkeypatch, sequence)

    r = await client.submit_order(direction="BUY", ticker="SBER", quantity=10, decision_id="d_list")
    assert r.success is False
    assert "MALFORMED_JSON_NOT_DICT" in r.message


@pytest.mark.asyncio
async def test_missing_fields_use_request_defaults(monkeypatch):
    """Response missing price/quantity/remaining_cash → safe defaults so the
    rest of the pipeline doesn't crash on .price access."""
    sequence = [_FakeResponse(status_code=200, json_body={"success": True})]
    client = await _make_client(monkeypatch, sequence)

    r = await client.submit_order(
        direction="BUY", ticker="SBER", quantity=42, decision_id="d_partial"
    )
    assert r.success is True
    assert r.price == 0.0
    assert r.quantity == 42
    assert r.remaining_cash == 0.0


@pytest.mark.asyncio
async def test_connection_reset_replays_once(monkeypatch):
    """First attempt raises ConnectionError; replay succeeds. Exactly 2 calls."""
    try:
        import httpx
    except ImportError:
        pytest.skip("httpx not installed")

    sequence = [
        httpx.ConnectError("connection reset by peer"),
        _FakeResponse(
            status_code=200,
            json_body={
                "success": True,
                "price": 100.0,
                "quantity": 5,
                "order_value": 500.0,
                "remaining_cash": 990_000.0,
            },
        ),
    ]
    client = await _make_client(monkeypatch, sequence)
    r = await client.submit_order(
        direction="BUY", ticker="SBER", quantity=5, decision_id="d_reset_replay"
    )
    assert r.success is True
    assert len(client._client.calls) == 2, "must replay exactly once"


@pytest.mark.asyncio
async def test_connection_reset_twice_in_a_row_fails(monkeypatch):
    """Two consecutive connection errors → no second replay, soft-fail."""
    try:
        import httpx
    except ImportError:
        pytest.skip("httpx not installed")

    sequence = [
        httpx.ConnectError("first reset"),
        httpx.ConnectError("second reset"),
    ]
    client = await _make_client(monkeypatch, sequence)
    r = await client.submit_order(
        direction="BUY", ticker="SBER", quantity=5, decision_id="d_reset_twice"
    )
    assert r.success is False
    assert "NETWORK_ERROR" in r.message
    assert len(client._client.calls) == 2


@pytest.mark.asyncio
async def test_401_triggers_reauth_and_replay(monkeypatch):
    """First call gets 401; we read a fresh SANDBOX_API_KEY and replay
    successfully. Counter increments AT MOST once per submit (we don't
    want a reauth loop)."""
    sequence = [
        _FakeResponse(status_code=401),
        _FakeResponse(
            status_code=200,
            json_body={
                "success": True,
                "price": 100.0,
                "quantity": 5,
                "order_value": 500.0,
                "remaining_cash": 990_000.0,
            },
        ),
    ]
    client = await _make_client(monkeypatch, sequence, api_key="initial_key")
    fake_httpx = client._client

    async def _no_rebuild(_self):  # type: ignore[no-redef]
        """No rebuild stub."""
        _self._reauth_attempts += 1
        _self._api_key = "fresh_key"
        return True

    import types

    client._reauth_on_401 = types.MethodType(_no_rebuild, client)
    client._client = fake_httpx

    r = await client.submit_order(
        direction="BUY", ticker="SBER", quantity=5, decision_id="d_401_then_ok"
    )
    assert r.success is True
    assert client._reauth_attempts == 1


@pytest.mark.asyncio
async def test_401_no_env_key_fails_fast(monkeypatch):
    """401 + empty SANDBOX_API_KEY → no infinite reauth loop, immediate fail."""
    sequence = [_FakeResponse(status_code=401)]
    client = await _make_client(monkeypatch, sequence, api_key="will_be_blanked")

    monkeypatch.delenv("SANDBOX_API_KEY", raising=False)

    r = await client.submit_order(
        direction="BUY", ticker="SBER", quantity=5, decision_id="d_401_dead"
    )
    assert r.success is False
    assert "401" in r.message or "REAUTH" in r.message


@pytest.mark.asyncio
async def test_slowdown_threshold_logs_warning_still_submits(monkeypatch, caplog):
    """At >= ARENAGO_DAILY_TRADE_SLOWDOWN (950) the submit goes through but
    a WARNING is emitted so the operator knows we should narrow signals."""
    sequence = [
        _FakeResponse(
            status_code=200,
            json_body={
                "success": True,
                "price": 100.0,
                "quantity": 1,
                "order_value": 100.0,
                "remaining_cash": 999_999.0,
            },
        )
    ]
    client = await _make_client(monkeypatch, sequence)
    client._daily_trade_count = cfg.ARENAGO_DAILY_TRADE_SLOWDOWN

    with caplog.at_level("WARNING"):
        r = await client.submit_order(
            direction="BUY", ticker="SBER", quantity=1, decision_id="d_slow"
        )
    assert r.success is True
    msgs = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("slowdown" in m.lower() for m in msgs)


@pytest.mark.asyncio
async def test_entry_halt_rejects_entries_but_accepts_exits(monkeypatch):
    """At >= ARENAGO_DAILY_TRADE_ENTRY_HALT (999) new ENTRIES are rejected
    locally, but EXIT submits (is_exit=True) pass through so we can keep
    closing live positions."""
    sequence = [
        _FakeResponse(
            status_code=200,
            json_body={
                "success": True,
                "price": 100.0,
                "quantity": 1,
                "order_value": 100.0,
                "remaining_cash": 999_999.0,
            },
        )
    ]
    client = await _make_client(monkeypatch, sequence)
    client._daily_trade_count = cfg.ARENAGO_DAILY_TRADE_ENTRY_HALT

    r_entry = await client.submit_order(
        direction="BUY",
        ticker="SBER",
        quantity=1,
        decision_id="d_entry_at_halt",
        is_exit=False,
    )
    assert r_entry.success is False
    assert "ENTRY_HALT" in r_entry.message
    assert len(client._client.calls) == 0

    r_exit = await client.submit_order(
        direction="SELL",
        ticker="SBER",
        quantity=1,
        decision_id="d_exit_at_halt",
        is_exit=True,
    )
    assert r_exit.success is True
    assert len(client._client.calls) == 1


@pytest.mark.asyncio
async def test_hard_limit_rejects_all_including_exits(monkeypatch):
    """At ARENAGO_DAILY_TRADE_LIMIT (1000) we reject EVERYTHING, even exits,
    because the broker enforces the same wall on its side anyway."""
    client = await _make_client(monkeypatch, [])
    client._daily_trade_count = cfg.ARENAGO_DAILY_TRADE_LIMIT

    r = await client.submit_order(
        direction="SELL",
        ticker="SBER",
        quantity=1,
        decision_id="d_hard",
        is_exit=True,
    )
    assert r.success is False
    assert "LOCAL_HARD_LIMIT" in r.message


@pytest.mark.asyncio
async def test_cash_drift_logs_critical(monkeypatch, caplog):
    """When two successive submit_order responses show a cash drop >50k
    without a matching trade size, a CRITICAL fires for the operator."""
    sequence = [
        _FakeResponse(
            status_code=200,
            json_body={
                "success": True,
                "price": 100.0,
                "quantity": 10,
                "order_value": 1_000.0,
                "remaining_cash": 999_000.0,
            },
        ),
        _FakeResponse(
            status_code=200,
            json_body={
                "success": True,
                "price": 100.0,
                "quantity": 10,
                "order_value": 1_000.0,
                "remaining_cash": 800_000.0,
            },
        ),
    ]
    client = await _make_client(monkeypatch, sequence)

    await client.submit_order(direction="BUY", ticker="SBER", quantity=10, decision_id="d_cash_a")
    with caplog.at_level("CRITICAL"):
        await client.submit_order(
            direction="BUY", ticker="GAZP", quantity=10, decision_id="d_cash_b"
        )
    crits = [r.message for r in caplog.records if r.levelname == "CRITICAL"]
    assert any("cash drop" in m.lower() for m in crits)


@pytest.mark.asyncio
async def test_reconciler_preserves_local_on_transient_failure(monkeypatch):
    """If get_positions_safe returns confirmed=False (transient), the
    reconciler must NOT mark local positions closed. This is the most
    dangerous regression: a 30-second ArenaGo blip would flatten the
    entire book on paper otherwise."""
    from app.execution.broker_reconciler import BrokerReconciler

    fake_book = MagicMock()
    fake_pos = MagicMock(quantity=10, avg_price=100.0)
    fake_book.positions = {"SBER": fake_pos}
    fake_book.cash_balance = 990_000.0

    fake_arenago = MagicMock()
    fake_arenago._bot_name = "test_bot"
    fake_arenago.get_positions_safe = AsyncMock(return_value=([], False))
    fake_arenago.get_trades = AsyncMock(return_value=[])
    fake_arenago.get_cash_balance = AsyncMock(return_value=990_000.0)

    rec = BrokerReconciler(arenago=fake_arenago, position_book=fake_book)
    rec._mark_position_closed = AsyncMock()
    rec._create_synthetic_local = AsyncMock()

    report = await rec.reconcile_once()

    assert report.marked_closed == [], "transient failure MUST NOT mark local position closed"
    rec._mark_position_closed.assert_not_called()


@pytest.mark.asyncio
async def test_reconciler_marks_closed_when_broker_confirms_empty(monkeypatch):
    """confirmed=True + empty list → local position WAS closed by broker,
    so the reconciler synthesises the flattening trade. This is the
    happy-path baseline that the transient guard above must not break."""
    from app.execution.broker_reconciler import BrokerReconciler

    fake_book = MagicMock()
    fake_pos = MagicMock(quantity=10, avg_price=100.0)
    fake_book.positions = {"SBER": fake_pos}
    fake_book.cash_balance = 990_000.0

    fake_arenago = MagicMock()
    fake_arenago._bot_name = "test_bot"
    fake_arenago.get_positions_safe = AsyncMock(return_value=([], True))
    fake_arenago.get_trades = AsyncMock(return_value=[])
    fake_arenago.get_cash_balance = AsyncMock(return_value=990_000.0)

    rec = BrokerReconciler(arenago=fake_arenago, position_book=fake_book)
    rec._mark_position_closed = AsyncMock()
    rec._create_synthetic_local = AsyncMock()

    report = await rec.reconcile_once()

    assert "SBER" in report.marked_closed
    rec._mark_position_closed.assert_called_once()


@pytest.mark.asyncio
async def test_circuit_breaker_auto_closes(monkeypatch):
    """After ARENAGO_CIRCUIT_BREAK_SEC the breaker re-opens for traffic."""
    sequence = [_FakeResponse(status_code=502) for _ in range(10)]
    client = await _make_client(monkeypatch, sequence)

    await client.submit_order(direction="BUY", ticker="SBER", quantity=1, decision_id="d_trip")
    assert client._circuit_is_open()

    await asyncio.sleep(0.6)
    assert not client._circuit_is_open()


def test_recovery_save_interval_is_5_seconds_default():
    """The hardened default must be 5 s so SIGKILL never loses more than
    5 s of decision-id dedup + n_trades_today state."""
    assert cfg.RECOVERY_SAVE_INTERVAL_SEC <= 5.0, (
        f"organizer requested ≤5 s; got {cfg.RECOVERY_SAVE_INTERVAL_SEC}"
    )
