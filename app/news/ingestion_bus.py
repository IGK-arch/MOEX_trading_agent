"""Очередь новостных событий."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from pydantic import BaseModel, ConfigDict, Field  # type: ignore

    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False

if _HAS_PYDANTIC:

    class NormalizedNewsEvent(BaseModel):
        """Standardised event format across all parsers."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        source: str
        source_tier: str
        ts_utc: datetime
        headline: str
        body: str = ""
        url: str = ""
        tickers: list[str] = Field(default_factory=list)
        language: str = "ru"
        raw_payload: dict[str, Any] = Field(default_factory=dict)

        @property
        def is_time_critical(self) -> bool:
            """Is time critical."""
            return self.source_tier in ("S", "A")
else:

    @dataclass
    class NormalizedNewsEvent:  # type: ignore
        """Normalized News Event."""

        source: str = ""
        source_tier: str = "B"
        ts_utc: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
        headline: str = ""
        body: str = ""
        url: str = ""
        tickers: list = field(default_factory=list)
        language: str = "ru"
        raw_payload: dict = field(default_factory=dict)
        event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

        @property
        def is_time_critical(self) -> bool:
            """Is time critical."""
            return self.source_tier in ("S", "A")

class IngestionBus:
    """In-process pub/sub queue for news events.

    Phase 11.5: two-queue design.
      - `_priority_queue`: tier S events (sanctions, MOEX ISS halts, Fed press).
        Consumed first by NewsLLM's priority loop; bypasses the materiality
        filter to minimise reaction time.
      - `_queue`: everything else (regular news pipeline).
    """

    MAX_QUEUE_SIZE = 4000
    HIGH_WATERMARK = 3200
    PRIORITY_QUEUE_SIZE = 200

    def __init__(self) -> None:
        """Init."""
        self._queue: asyncio.Queue[NormalizedNewsEvent] = asyncio.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._priority_queue: asyncio.Queue[NormalizedNewsEvent] = asyncio.Queue(
            maxsize=self.PRIORITY_QUEUE_SIZE
        )

        self._recent: deque[NormalizedNewsEvent] = deque(maxlen=500)

        self._stats: dict[str, dict[str, int]] = {}
        self._dropped_count = 0
        self._processed_count = 0
        self._priority_processed = 0
        self._start_time = time.monotonic()

    async def publish_priority(self, event: NormalizedNewsEvent) -> bool:
        """
        Publish to the fast-path priority queue.

        Used by sanctions_parser and MOEX ISS halts — events that should
        bypass the regular ingestion cycle and trigger immediate signal
        generation. Falls back to the regular queue on overflow.
        """
        if event.source not in self._stats:
            self._stats[event.source] = {"in": 0, "dropped": 0, "priority": 0}
        self._stats[event.source]["in"] += 1
        self._stats[event.source].setdefault("priority", 0)
        self._stats[event.source]["priority"] += 1

        try:
            self._priority_queue.put_nowait(event)
            self._recent.append(event)
            return True
        except asyncio.QueueFull:
            logger.warning(
                "Priority queue full, falling back to regular",
                extra={"source": event.source, "tier": event.source_tier},
            )
            return await self.publish(event)

    async def consume_priority(self, timeout: float = 0.5) -> NormalizedNewsEvent | None:
        """Read from priority queue. Returns None on timeout."""
        try:
            event = await asyncio.wait_for(self._priority_queue.get(), timeout=timeout)
            self._priority_processed += 1
            return event
        except TimeoutError:
            return None

    async def publish(self, event: NormalizedNewsEvent) -> bool:
        """
        Push an event to the bus.
        Returns True if accepted, False if dropped due to backpressure.
        """

        if event.source not in self._stats:
            self._stats[event.source] = {"in": 0, "dropped": 0}
        self._stats[event.source]["in"] += 1

        if self._queue.qsize() >= self.HIGH_WATERMARK and not event.is_time_critical:
            self._stats[event.source]["dropped"] += 1
            self._dropped_count += 1
            logger.debug(
                "Bus: dropped event (backpressure)",
                extra={
                    "source": event.source,
                    "tier": event.source_tier,
                    "qsize": self._queue.qsize(),
                },
            )
            return False

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            if event.is_time_critical:
                try:
                    evicted = self._queue.get_nowait()
                    self._dropped_count += 1
                    if evicted.source in self._stats:
                        self._stats[evicted.source]["dropped"] = (
                            self._stats[evicted.source].get("dropped", 0) + 1
                        )
                    logger.warning(
                        "Bus: hard overflow but Tier S/A arrived — evicted oldest",
                        extra={
                            "evicted_source": evicted.source,
                            "evicted_tier": evicted.source_tier,
                            "incoming_source": event.source,
                            "incoming_tier": event.source_tier,
                        },
                    )
                    self._queue.put_nowait(event)
                    self._recent.append(event)
                    return True
                except asyncio.QueueEmpty:
                    pass
            self._stats[event.source]["dropped"] += 1
            self._dropped_count += 1
            logger.warning(
                "Bus: hard overflow, dropping event",
                extra={"source": event.source, "tier": event.source_tier},
            )
            return False

        self._recent.append(event)
        return True

    async def consume(self, timeout: float | None = None) -> NormalizedNewsEvent | None:
        """
        Pull one event from the bus. Returns None on timeout.
        """
        try:
            if timeout is None:
                event = await self._queue.get()
            else:
                event = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            self._processed_count += 1
            return event
        except TimeoutError:
            return None

    def get_recent(self, k: int = 50) -> list[NormalizedNewsEvent]:
        """Get most recent k events (for dashboard / morning_plan)."""
        return list(self._recent)[-k:]

    def stats(self) -> dict[str, Any]:
        """Aggregate stats for monitoring."""
        uptime = time.monotonic() - self._start_time
        return {
            "queue_size": self._queue.qsize(),
            "priority_queue_size": self._priority_queue.qsize(),
            "processed_total": self._processed_count,
            "priority_processed_total": self._priority_processed,
            "dropped_total": self._dropped_count,
            "uptime_sec": round(uptime, 0),
            "events_per_sec": round(self._processed_count / max(1, uptime), 3),
            "per_source": dict(self._stats),
        }

    def reset_stats(self) -> None:
        """Reset stats."""
        self._stats.clear()
        self._dropped_count = 0
        self._processed_count = 0
        self._start_time = time.monotonic()

_bus: IngestionBus | None = None

def get_bus() -> IngestionBus:
    """Get bus."""
    global _bus
    if _bus is None:
        _bus = IngestionBus()
    return _bus
