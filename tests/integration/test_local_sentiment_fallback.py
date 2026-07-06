"""News pipeline fallback via local sentiment when POLZA is disabled.

Verifies the Phase 26 (v0.0.31) safety net: when ``cfg.DISABLE_LLM=True``
and ``cfg.NEWS_LOCAL_SENTIMENT_ENABLED=True``, a material news event still
produces a UnifiedSignal — via the local sentiment scorer rather than the
LLM prompt path.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

import app.config as cfg
from app.agents.news_llm import NewsLLM
from app.dispatcher.signal import Direction, SignalSource
from app.news.ingestion_bus import IngestionBus, NormalizedNewsEvent


def _positive_event() -> NormalizedNewsEvent:
    """Positive event."""
    return NormalizedNewsEvent(
        source="interfax",
        source_tier="A",
        ts_utc=datetime.now(tz=UTC),
        headline="Сбербанк увеличил прибыль на 30% и выплатит рекордные дивиденды",
        body=(
            "Сбербанк объявил о рекордном росте прибыли по итогам квартала. "
            "Совет директоров рекомендовал выплатить дивиденды и одобрил выкуп акций. "
            "Рынок отреагировал ростом котировок."
        ),
        url="https://example.com/sber-record",
        tickers=["SBER"],
        language="ru",
        raw_payload={},
    )


def _negative_event() -> NormalizedNewsEvent:
    """Negative event."""
    return NormalizedNewsEvent(
        source="ofac_sdn",
        source_tier="S",
        ts_utc=datetime.now(tz=UTC),
        headline="OFAC ввёл санкции, запрет и штраф против Газпрома",
        body=(
            "Минфин США объявил о расширении санкционного списка. "
            "Введён запрет на операции, штраф $1 млрд, лицензия отозвана, "
            "активы заблокированы. Ожидается падение акций."
        ),
        url="https://example.com/sanctions",
        tickers=["GAZP"],
        language="ru",
        raw_payload={"jurisdiction": "US"},
    )


@pytest.mark.asyncio
async def test_disable_llm_with_local_sentiment_emits_buy_on_positive_event(
    monkeypatch,
):
    """Positive Russian news → BUY signal via local keyword scorer."""
    monkeypatch.setattr(cfg, "DISABLE_LLM", True)
    monkeypatch.setattr(cfg, "NEWS_LOCAL_SENTIMENT_ENABLED", True)
    monkeypatch.setattr(cfg, "NEWS_LOCAL_SENT_THRESHOLD", 0.3)

    bus = IngestionBus()
    nl = NewsLLM(bus=bus)
    nl.local_sentiment._force_keyword_only = True  # type: ignore[attr-defined]
    nl.local_sentiment._tried_load = True  # type: ignore[attr-defined]
    nl.local_sentiment._pipeline = None  # type: ignore[attr-defined]
    nl.history.record_event = _async_noop("reactive")  # type: ignore[assignment]
    nl.history.update_analysis = _async_noop(None)  # type: ignore[assignment]
    nl.history.save_context_snapshot = _async_noop(None)  # type: ignore[assignment]
    nl.history.find_similar_cases = _async_noop({})  # type: ignore[assignment]

    await nl.startup()
    nl.dedup.is_duplicate = lambda *_a, **_k: False  # type: ignore

    await bus.publish(_positive_event())

    deadline = asyncio.get_event_loop().time() + 2.0
    while asyncio.get_event_loop().time() < deadline:
        if nl._signal_buffer:
            break
        await asyncio.sleep(0.02)

    try:
        assert len(nl._signal_buffer) > 0, "Local sentiment produced no signal"
        sig = nl._signal_buffer[0]
        assert sig.source == SignalSource.NEWS
        assert sig.direction == Direction.BUY
        assert sig.detector == "local_sentiment"
        assert sig.metadata.get("fallback_mode") == "local_sentiment"
        assert 0.0 < sig.magnitude <= 1.0
        assert sig.metadata.get("local_sentiment_score", 0) > 0
    finally:
        await nl.shutdown()


@pytest.mark.asyncio
async def test_disable_llm_with_local_sentiment_emits_sell_on_negative_event(
    monkeypatch,
):
    """Sanctions-heavy news → SELL signal via local keyword scorer."""
    monkeypatch.setattr(cfg, "DISABLE_LLM", True)
    monkeypatch.setattr(cfg, "NEWS_LOCAL_SENTIMENT_ENABLED", True)
    monkeypatch.setattr(cfg, "NEWS_LOCAL_SENT_THRESHOLD", 0.3)

    bus = IngestionBus()
    nl = NewsLLM(bus=bus)
    nl.local_sentiment._force_keyword_only = True  # type: ignore[attr-defined]
    nl.local_sentiment._tried_load = True  # type: ignore[attr-defined]
    nl.local_sentiment._pipeline = None  # type: ignore[attr-defined]
    nl.history.record_event = _async_noop("sanctions")  # type: ignore[assignment]
    nl.history.update_analysis = _async_noop(None)  # type: ignore[assignment]
    nl.history.save_context_snapshot = _async_noop(None)  # type: ignore[assignment]
    nl.history.find_similar_cases = _async_noop({})  # type: ignore[assignment]

    await nl.startup()
    nl.dedup.is_duplicate = lambda *_a, **_k: False  # type: ignore

    await bus.publish_priority(_negative_event())

    deadline = asyncio.get_event_loop().time() + 2.0
    while asyncio.get_event_loop().time() < deadline:
        if nl._signal_buffer:
            break
        await asyncio.sleep(0.02)

    try:
        assert len(nl._signal_buffer) > 0, "No SELL signal from sanctions event"
        sig = nl._signal_buffer[0]
        assert sig.source == SignalSource.NEWS
        assert sig.direction == Direction.SELL
        assert sig.detector == "local_sentiment"
        assert sig.metadata.get("local_sentiment_score", 0) < 0
    finally:
        await nl.shutdown()


@pytest.mark.asyncio
async def test_disable_llm_without_local_sentiment_stays_inert(monkeypatch):
    """When both DISABLE_LLM and local sentiment are off, no signals appear."""
    monkeypatch.setattr(cfg, "DISABLE_LLM", True)
    monkeypatch.setattr(cfg, "NEWS_LOCAL_SENTIMENT_ENABLED", False)

    bus = IngestionBus()
    nl = NewsLLM(bus=bus)
    await nl.startup()
    nl.dedup.is_duplicate = lambda *_a, **_k: False  # type: ignore

    await bus.publish(_positive_event())
    await asyncio.sleep(0.3)

    try:
        assert len(nl._signal_buffer) == 0
        assert nl._consumer_task is None
    finally:
        await nl.shutdown()


def _async_noop(return_value):
    """Build an async function that returns ``return_value`` regardless of args."""

    async def _fn(*_args, **_kwargs):
        """Fn."""
        return return_value

    return _fn
