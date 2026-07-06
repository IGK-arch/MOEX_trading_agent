"""Unit tests for app.agents.consensus_compare."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.consensus_compare import (
    ComparisonResult,
    ConsensusComparator,
    _parse_compare_response,
)
from app.memory.rag_store import RAGStore
from app.news.consensus_rag import ConsensusEntry
from app.news.ingestion_bus import NormalizedNewsEvent


def _make_event(
    headline: str, body: str, tickers: list[str], source: str = "rss"
) -> NormalizedNewsEvent:
    """Make event."""
    return NormalizedNewsEvent(
        event_id=f"event_{headline[:20]}",
        source=source,
        source_tier="A",
        ts_utc=datetime.now(tz=UTC),
        headline=headline,
        body=body,
        url="",
        tickers=tickers,
        language="ru",
        raw_payload={},
    )


def _make_consensus(direction: str = "BUY", strength: float = 0.7) -> ConsensusEntry:
    """Make consensus."""
    return ConsensusEntry(
        ticker="SBER",
        direction=direction,
        strength=strength,
        key_themes=["дивиденды", "Q1"],
        rationale="Positive on dividends",
        expected_to_positive="дивидендный сюрприз",
        expected_to_negative="миссы по выручке",
        built_at=datetime.now(tz=UTC),
        n_news=8,
        backend="test",
    )


def test_parse_compare_response_matches_consensus():
    """Test parse compare response matches consensus."""
    cons = _make_consensus(direction="BUY")
    result = _parse_compare_response(
        ticker="SBER",
        parsed={
            "alignment": "matches_consensus",
            "direction": "BUY",
            "magnitude": 0.85,
            "rationale": "buyback подтверждает позитив",
        },
        cons=cons,
        n_similar=2,
        backend="gemini_via_polza",
    )
    assert isinstance(result, ComparisonResult)
    assert result.alignment == "matches_consensus"
    assert result.direction == "BUY"
    assert result.magnitude == pytest.approx(0.85)
    assert result.consensus_direction == "BUY"
    assert result.n_similar == 2


def test_parse_compare_response_contradicts_forces_flip_when_neutral_direction():
    """Test parse compare response contradicts forces flip when neutral direction."""
    cons = _make_consensus(direction="BUY")
    result = _parse_compare_response(
        ticker="SBER",
        parsed={"alignment": "contradicts", "direction": "NEUTRAL", "magnitude": 0.7},
        cons=cons,
        n_similar=1,
        backend="polza",
    )
    assert result.alignment == "contradicts"
    assert result.direction == "SELL"


def test_parse_compare_response_internal_consistency_guard():
    """Test parse compare response internal consistency guard."""
    cons = _make_consensus(direction="BUY")
    result = _parse_compare_response(
        ticker="SBER",
        parsed={"alignment": "contradicts", "direction": "BUY", "magnitude": 0.4},
        cons=cons,
        n_similar=0,
        backend="polza",
    )
    assert result.alignment == "neutral"


def test_parse_compare_response_handles_empty_payload():
    """Test parse compare response handles empty payload."""
    cons = _make_consensus(direction="BUY")
    result = _parse_compare_response(
        ticker="SBER",
        parsed={},
        cons=cons,
        n_similar=0,
        backend="polza",
    )
    assert result.alignment == "neutral"
    assert result.direction == "NEUTRAL"
    assert result.magnitude == 0.0


def test_parse_compare_response_clamps_magnitude():
    """Test parse compare response clamps magnitude."""
    cons = _make_consensus(direction="BUY")
    over = _parse_compare_response(
        ticker="SBER",
        parsed={"alignment": "matches_consensus", "direction": "BUY", "magnitude": 1.5},
        cons=cons,
        n_similar=0,
        backend="polza",
    )
    under = _parse_compare_response(
        ticker="SBER",
        parsed={"alignment": "matches_consensus", "direction": "BUY", "magnitude": -0.2},
        cons=cons,
        n_similar=0,
        backend="polza",
    )
    assert 0.0 <= over.magnitude <= 1.0
    assert 0.0 <= under.magnitude <= 1.0


@pytest.mark.asyncio
async def test_comparator_returns_neutral_when_no_consensus(tmp_path):
    """Test comparator returns neutral when no consensus."""
    rag = RAGStore(tmp_path)
    comp = ConsensusComparator(rag=rag, consensus_today={}, llm_backend="polza")
    event = _make_event("Сбербанк объявил buyback", "тело новости", ["SBER"])
    results = await comp.compare_event(event)
    assert "SBER" in results
    assert results["SBER"].alignment == "neutral"
    assert results["SBER"].direction == "NEUTRAL"


@pytest.mark.asyncio
async def test_comparator_calls_llm_and_returns_match(tmp_path):
    """Test comparator calls llm and returns match."""
    rag = RAGStore(tmp_path)
    rag.add_news(
        event_id="prev1",
        text="Сбербанк отчитался о росте прибыли на 18%",
        ts_utc=datetime.now(tz=UTC),
        tickers=["SBER"],
        source="rss",
        source_tier="A",
        headline="Сбербанк отчитался",
        body="полное тело",
    )
    cons = _make_consensus(direction="BUY", strength=0.7)
    comp = ConsensusComparator(
        rag=rag,
        consensus_today={"SBER": cons},
        llm_backend="gemini",
    )
    event = _make_event(
        "Сбербанк рекомендовал дивиденды 40 рублей за акцию",
        "Совет директоров рекомендовал дивиденды 40 рублей.",
        ["SBER"],
    )

    mock_llm = AsyncMock(
        return_value={
            "content": "{}",
            "parsed": {
                "alignment": "matches_consensus",
                "direction": "BUY",
                "magnitude": 0.8,
                "rationale": "дивиденды подтверждают консенсус",
            },
            "model": "google/gemini-2.5-flash",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_rub": 0.001,
            "cached": False,
            "backend": "gemini_via_polza",
        }
    )

    with patch("app.agents.consensus_compare.llm_chat_json", mock_llm):
        results = await comp.compare_event(event)

    assert mock_llm.await_count == 1
    assert results["SBER"].alignment == "matches_consensus"
    assert results["SBER"].direction == "BUY"
    assert comp.stats["matches"] == 1


@pytest.mark.asyncio
async def test_comparator_aggregate_signal_bumps_on_match(tmp_path):
    """Test comparator aggregate signal bumps on match."""
    rag = RAGStore(tmp_path)
    comp = ConsensusComparator(rag=rag, consensus_today={}, llm_backend="polza")
    cmp = {
        "SBER": ComparisonResult(
            ticker="SBER",
            alignment="matches_consensus",
            direction="BUY",
            magnitude=0.6,
            consensus_direction="BUY",
            consensus_strength=0.7,
        ),
    }
    direction, magnitude, _ = comp.aggregate_signal(
        comparisons=cmp,
        base_direction="BUY",
        base_magnitude=0.5,
    )
    assert direction == "BUY"
    assert magnitude == pytest.approx(0.9, rel=1e-3)


@pytest.mark.asyncio
async def test_comparator_aggregate_signal_flips_on_contradict(tmp_path):
    """Test comparator aggregate signal flips on contradict."""
    rag = RAGStore(tmp_path)
    comp = ConsensusComparator(rag=rag, consensus_today={}, llm_backend="polza")
    cmp = {
        "SBER": ComparisonResult(
            ticker="SBER",
            alignment="contradicts",
            direction="SELL",
            magnitude=0.7,
            consensus_direction="BUY",
            consensus_strength=0.7,
        ),
    }
    direction, magnitude, _ = comp.aggregate_signal(
        comparisons=cmp,
        base_direction="BUY",
        base_magnitude=0.5,
    )
    assert direction == "SELL"
    assert magnitude == pytest.approx(0.7, rel=1e-3)


@pytest.mark.asyncio
async def test_comparator_aggregate_signal_drops_all_neutral(tmp_path):
    """Test comparator aggregate signal drops all neutral."""
    rag = RAGStore(tmp_path)
    comp = ConsensusComparator(rag=rag, consensus_today={}, llm_backend="polza")
    cmp = {
        "SBER": ComparisonResult(
            ticker="SBER",
            alignment="neutral",
            direction="NEUTRAL",
            magnitude=0.0,
            consensus_direction="BUY",
            consensus_strength=0.7,
        ),
    }
    direction, magnitude, _ = comp.aggregate_signal(
        comparisons=cmp,
        base_direction="BUY",
        base_magnitude=0.5,
    )
    assert magnitude == 0.0


@pytest.mark.asyncio
async def test_comparator_update_consensus(tmp_path):
    """Test comparator update consensus."""
    rag = RAGStore(tmp_path)
    comp = ConsensusComparator(rag=rag, consensus_today={}, llm_backend="polza")
    assert comp.stats["consensus_tickers"] == 0
    cons = {"SBER": _make_consensus(direction="BUY")}
    comp.update_consensus(cons)
    assert comp.stats["consensus_tickers"] == 1


@pytest.mark.asyncio
async def test_comparator_empty_tickers_returns_empty(tmp_path):
    """Test comparator empty tickers returns empty."""
    rag = RAGStore(tmp_path)
    comp = ConsensusComparator(rag=rag, consensus_today={}, llm_backend="polza")
    event = _make_event("global headline", "", [])
    results = await comp.compare_event(event)
    assert results == {}
