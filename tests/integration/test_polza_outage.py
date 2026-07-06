"""
tests/integration/test_polza_outage.py — Polza 401/outage → RuBERT local
sentiment fallback emits signals.

Scenario (Phase 26, v0.0.31): the Polza API key is revoked or returns 401
mid-trading. The bot must:
  * Set ``cfg.DISABLE_LLM = True`` (or be configured that way).
  * Keep ingesting news.
  * Fall back to ``app.news.local_sentiment.LocalSentimentScorer`` and still
    emit ``SignalSource.NEWS`` ``UnifiedSignal`` instances with
    ``detector="local_sentiment"``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

import app.config as cfg
from app.agents.news_llm import NewsLLM
from app.dispatcher.signal import Direction, SignalSource
from app.news.ingestion_bus import IngestionBus, NormalizedNewsEvent


def _noop_async(return_value=None):
    """Noop async."""

    async def _fn(*_a, **_k):
        """Fn."""
        return return_value

    return _fn


def _positive_news_event() -> NormalizedNewsEvent:
    """Positive news event."""
    return NormalizedNewsEvent(
        source="interfax",
        source_tier="A",
        ts_utc=datetime.now(tz=UTC),
        headline="Сбербанк объявил рекордные дивиденды и рост прибыли",
        body=(
            "Сбербанк объявил рекордные дивиденды по итогам года. "
            "Прибыль выросла на 35%, совет директоров одобрил выкуп акций. "
            "Рынок отреагировал ростом котировок."
        ),
        url="https://example.com/sber-dividends",
        tickers=["SBER"],
        language="ru",
        raw_payload={},
    )


class FakeHttpxClient401:
    """A fake httpx-like async client that always returns 401 Unauthorized.

    Used to prove that even if the real polza.ai endpoint is unreachable,
    no test makes a real network call — the LocalSentimentScorer path takes
    over because ``cfg.DISABLE_LLM`` is set before startup.
    """

    closed: bool = False

    async def post(self, *args, **kwargs):
        """Post."""
        raise RuntimeError("FakeHttpxClient401: simulated 401 — bot must NOT hit network")

    async def aclose(self):
        """Aclose."""
        self.closed = True


@pytest.mark.asyncio
async def test_polza_401_falls_back_to_local_sentiment(monkeypatch):
    """401 from Polza → DISABLE_LLM=True → local sentiment emits BUY signal."""
    monkeypatch.setenv("POLZA_API_KEY", "")
    monkeypatch.setattr(cfg, "DISABLE_LLM", True)
    monkeypatch.setattr(cfg, "NEWS_LOCAL_SENTIMENT_ENABLED", True)
    monkeypatch.setattr(cfg, "NEWS_LOCAL_SENT_THRESHOLD", 0.3)

    from app.llm.polza_client import get_polza_client

    polza = get_polza_client()
    polza._client = FakeHttpxClient401()  # type: ignore[attr-defined]

    bus = IngestionBus()
    nl = NewsLLM(bus=bus)
    nl.local_sentiment._force_keyword_only = True  # type: ignore[attr-defined]
    nl.local_sentiment._tried_load = True  # type: ignore[attr-defined]
    nl.local_sentiment._pipeline = None  # type: ignore[attr-defined]
    nl.history.record_event = _noop_async("reactive")  # type: ignore
    nl.history.update_analysis = _noop_async(None)  # type: ignore
    nl.history.save_context_snapshot = _noop_async(None)  # type: ignore
    nl.history.find_similar_cases = _noop_async({})  # type: ignore

    await nl.startup()
    nl.dedup.is_duplicate = lambda *_a, **_k: False  # type: ignore

    await bus.publish(_positive_news_event())

    deadline = asyncio.get_event_loop().time() + 2.0
    while asyncio.get_event_loop().time() < deadline:
        if nl._signal_buffer:
            break
        await asyncio.sleep(0.02)

    try:
        assert nl._signal_buffer, "Local sentiment fallback emitted no signal on Polza 401"
        sig = nl._signal_buffer[0]
        assert sig.source == SignalSource.NEWS
        assert sig.detector == "local_sentiment"
        assert sig.direction == Direction.BUY
        assert sig.metadata.get("fallback_mode") == "local_sentiment"
    finally:
        await nl.shutdown()


@pytest.mark.asyncio
async def test_polza_outage_with_local_sentiment_disabled_emits_nothing(monkeypatch):
    """If BOTH Polza is down AND local sentiment is disabled, news pipeline
    should stay inert (no signals, no crashes)."""
    monkeypatch.setattr(cfg, "DISABLE_LLM", True)
    monkeypatch.setattr(cfg, "NEWS_LOCAL_SENTIMENT_ENABLED", False)

    bus = IngestionBus()
    nl = NewsLLM(bus=bus)
    await nl.startup()
    nl.dedup.is_duplicate = lambda *_a, **_k: False  # type: ignore

    await bus.publish(_positive_news_event())
    await asyncio.sleep(0.3)

    try:
        assert len(nl._signal_buffer) == 0
        assert nl._consumer_task is None
    finally:
        await nl.shutdown()
