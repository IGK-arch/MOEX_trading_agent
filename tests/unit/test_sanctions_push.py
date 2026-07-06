"""
tests/unit/test_sanctions_push.py — Priority queue + DK-CoT prompt rendering.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.news.ingestion_bus import IngestionBus, NormalizedNewsEvent


def _make_event(
    source: str = "ofac_sdn", tier: str = "S", headline: str = "Sanctions on SBER"
) -> NormalizedNewsEvent:
    """Make event."""
    return NormalizedNewsEvent(
        source=source,
        source_tier=tier,
        ts_utc=datetime.now(tz=UTC),
        headline=headline,
        body="Body text...",
        url="https://example.com",
        tickers=[],
        language="en",
        raw_payload={"jurisdiction": "US"},
    )


@pytest.mark.asyncio
async def test_priority_publish_uses_separate_queue():
    """Test priority publish uses separate queue."""
    bus = IngestionBus()
    e = _make_event()
    ok = await bus.publish_priority(e)
    assert ok is True

    regular = await bus.consume(timeout=0.05)
    assert regular is None

    got = await bus.consume_priority(timeout=0.5)
    assert got is not None
    assert got.event_id == e.event_id


@pytest.mark.asyncio
async def test_priority_consume_returns_none_on_empty():
    """Test priority consume returns none on empty."""
    bus = IngestionBus()
    got = await bus.consume_priority(timeout=0.05)
    assert got is None


@pytest.mark.asyncio
async def test_priority_falls_back_to_regular_on_overflow():
    """Filling priority queue past PRIORITY_QUEUE_SIZE → fall back to regular."""
    bus = IngestionBus()

    for i in range(bus.PRIORITY_QUEUE_SIZE):
        await bus.publish_priority(_make_event(headline=f"S{i}"))
    assert bus._priority_queue.qsize() == bus.PRIORITY_QUEUE_SIZE

    overflow = _make_event(headline="overflow")
    ok = await bus.publish_priority(overflow)
    assert ok is True

    assert bus._queue.qsize() == 1


@pytest.mark.asyncio
async def test_stats_track_priority_separately():
    """Test stats track priority separately."""
    bus = IngestionBus()
    await bus.publish_priority(_make_event())
    await bus.publish(_make_event(source="other", tier="A"))
    stats = bus.stats()
    assert stats["priority_queue_size"] == 1
    assert stats["queue_size"] == 1
    assert "priority" in stats["per_source"]["ofac_sdn"]


def test_v2_dkcot_prompts_exist_and_have_required_placeholders():
    """The DK-CoT prompt files must exist and contain the documented templates."""
    base = Path(__file__).resolve().parent.parent.parent / "app" / "news" / "prompts"
    sanctions = base / "sanctions_v2_dkcot.txt"
    reactive = base / "reactive_v2_dkcot.txt"
    assert sanctions.exists(), "sanctions_v2_dkcot.txt should exist"
    assert reactive.exists(), "reactive_v2_dkcot.txt should exist"
    s_text = sanctions.read_text(encoding="utf-8")
    r_text = reactive.read_text(encoding="utf-8")

    assert "Шаг 1" in s_text or "Step 1" in s_text
    assert "Шаг 5" in s_text or "JSON" in s_text

    for var in ("{{headline}}", "{{body}}", "{{jurisdiction}}", "{{ts}}"):
        assert var in s_text, f"sanctions prompt missing {var}"

    for var in ("{{headline}}", "{{body}}", "{{tickers}}"):
        assert var in r_text, f"reactive prompt missing {var}"
