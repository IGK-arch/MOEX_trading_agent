"""
tests/integration/test_sanctions_e2e.py — Sanctions → priority queue → news_llm signal.

Verifies the fast path from a synthetic OFAC SDN event to a UnifiedSignal
in NewsLLM's signal buffer, with the dispatcher trigger fired.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from app.agents.news_llm import NewsLLM
from app.dispatcher.signal import Direction, SignalSource
from app.news.ingestion_bus import IngestionBus, NormalizedNewsEvent


class _StubPolza:
    """Stub polza client — always returns a strong sanctions SELL signal."""

    _started = True

    async def startup(self):
        """Startup."""
        return None

    async def chat_json(self, *_args, **_kwargs):
        """Chat json."""
        return {
            "direction": "SELL",
            "magnitude": 0.85,
            "affected_tickers": ["SBER", "GAZP"],
            "horizon_min": 180,
            "reason": "OFAC SDN block on Russian state banks",
            "classification": "SDN_BLOCK",
        }


def _sanctions_event() -> NormalizedNewsEvent:
    """Sanctions event."""
    return NormalizedNewsEvent(
        source="ofac_sdn",
        source_tier="S",
        ts_utc=datetime.now(tz=UTC),
        headline="OFAC adds SBER to SDN list",
        body="Treasury announced full block on Sberbank.",
        url="https://example.com",
        tickers=["SBER"],
        language="en",
        raw_payload={"jurisdiction": "US"},
    )


@pytest.mark.asyncio
async def test_priority_event_yields_signal_and_triggers_dispatcher():
    """Test priority event yields signal and triggers dispatcher."""
    bus = IngestionBus()
    nl = NewsLLM(bus=bus)
    nl.polza = _StubPolza()
    await nl.startup()

    nl.dedup.is_duplicate = lambda *_a, **_k: False  # type: ignore

    trigger = asyncio.Event()
    nl.set_dispatcher_trigger(trigger)

    await bus.publish_priority(_sanctions_event())

    deadline = asyncio.get_event_loop().time() + 2.0
    while asyncio.get_event_loop().time() < deadline:
        if nl._signal_buffer:
            break
        await asyncio.sleep(0.02)

    try:
        assert len(nl._signal_buffer) > 0, "No signal produced from priority event"
        assert trigger.is_set(), "Dispatcher trigger was not set"
        sig = nl._signal_buffer[0]
        assert sig.source == SignalSource.NEWS
        assert sig.direction == Direction.SELL

        assert sig.magnitude >= 0.8

        assert sig.metadata.get("is_sanctions") is True
        assert sig.metadata.get("priority") is True
    finally:
        await nl.shutdown()
