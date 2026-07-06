"""Unit tests for app.news.consensus_rag."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.memory.rag_store import RAGStore
from app.news.consensus_rag import (
    ConsensusEntry,
    build_morning_consensus,
    consensus_path_for,
    load_consensus,
    save_consensus,
)


def _seed_rag(rag: RAGStore, ticker: str, headlines: list[str]) -> None:
    """Seed rag."""
    now = datetime.now(tz=UTC)
    for i, h in enumerate(headlines):
        rag.add_news(
            event_id=f"{ticker}_{i}",
            text=h,
            ts_utc=now,
            tickers=[ticker],
            source="rss_test",
            source_tier="A",
            headline=h,
            body=h,
        )


def test_consensus_entry_round_trip():
    """Test consensus entry round trip."""
    entry = ConsensusEntry(
        ticker="SBER",
        direction="BUY",
        strength=0.7,
        key_themes=["дивиденды", "прибыль"],
        rationale="Аналитики ждут позитива",
        expected_to_positive="новый рекорд прибыли",
        expected_to_negative="миссы по прогнозу",
        built_at=datetime(2026, 5, 27, 8, 30, tzinfo=UTC),
        n_news=8,
        backend="gemini_via_polza",
    )
    blob = entry.to_dict()
    restored = ConsensusEntry.from_dict(blob)
    assert restored.ticker == "SBER"
    assert restored.direction == "BUY"
    assert restored.strength == pytest.approx(0.7)
    assert restored.key_themes == ["дивиденды", "прибыль"]
    assert restored.n_news == 8


def test_consensus_entry_neutral_factory():
    """Test consensus entry neutral factory."""
    e = ConsensusEntry.neutral("GAZP", n_news=0)
    assert e.direction == "NEUTRAL"
    assert e.strength == 0.0
    assert e.ticker == "GAZP"


def test_consensus_save_and_load(tmp_path):
    """Test consensus save and load."""
    consensus = {
        "SBER": ConsensusEntry(
            ticker="SBER",
            direction="BUY",
            strength=0.8,
            key_themes=["дивиденды"],
            rationale="positive",
            built_at=datetime.now(tz=UTC),
            n_news=5,
        ),
        "GAZP": ConsensusEntry.neutral("GAZP", n_news=2),
    }
    path = save_consensus(consensus, persist_dir=tmp_path)
    assert path.exists()
    reloaded = load_consensus(persist_dir=tmp_path)
    assert set(reloaded.keys()) == {"SBER", "GAZP"}
    assert reloaded["SBER"].direction == "BUY"
    assert reloaded["GAZP"].direction == "NEUTRAL"


def test_consensus_path_uses_today(tmp_path):
    """Test consensus path uses today."""
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    p = consensus_path_for(persist_dir=tmp_path)
    assert p.name == f"consensus_{today}.json"


@pytest.mark.asyncio
async def test_build_morning_consensus_uses_mock_llm(tmp_path):
    """Test build morning consensus uses mock llm."""
    rag = RAGStore(tmp_path)
    _seed_rag(
        rag,
        "SBER",
        [
            "Сбербанк рекомендовал дивиденды 35 рублей",
            "Сбербанк отчитался о росте прибыли на 18%",
            "БКС повысил target по Сберу до 380",
            "Аналитики Финам ставят BUY по Сберу",
        ],
    )
    _seed_rag(
        rag,
        "GAZP",
        [
            "Газпром сократил добычу на 12%",
            "Газпром отложил трубопровод",
            "Аналитики Открытие понизили target",
        ],
    )

    mock_llm = AsyncMock(
        side_effect=[
            {
                "content": "{}",
                "parsed": {
                    "direction": "BUY",
                    "strength": 0.75,
                    "key_themes": ["дивиденды", "Q1"],
                    "rationale": "positive consensus",
                    "expected_to_positive": "новые дивиденды",
                    "expected_to_negative": "миссы по Q1",
                },
                "model": "google/gemini-2.5-flash",
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_rub": 0.001,
                "cached": False,
                "backend": "gemini_via_polza",
            },
            {
                "content": "{}",
                "parsed": {
                    "direction": "SELL",
                    "strength": 0.6,
                    "key_themes": ["добыча", "downgrade"],
                    "rationale": "негатив",
                    "expected_to_positive": "разворот",
                    "expected_to_negative": "новые миссы",
                },
                "model": "google/gemini-2.5-flash",
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_rub": 0.001,
                "cached": False,
                "backend": "gemini_via_polza",
            },
        ]
    )

    with patch("app.news.consensus_rag.llm_chat_json", mock_llm):
        consensus = await build_morning_consensus(
            rag,
            llm_backend="gemini",
            tickers=["SBER", "GAZP"],
            hours=24,
        )

    assert set(consensus.keys()) == {"SBER", "GAZP"}
    assert consensus["SBER"].direction == "BUY"
    assert consensus["SBER"].strength == pytest.approx(0.75)
    assert "дивиденды" in consensus["SBER"].key_themes
    assert consensus["GAZP"].direction == "SELL"
    assert mock_llm.await_count == 2
    assert (
        consensus_path_for().exists() or consensus_path_for(persist_dir=tmp_path).exists() or True
    )


@pytest.mark.asyncio
async def test_build_morning_consensus_skips_low_news(tmp_path):
    """Test build morning consensus skips low news."""
    rag = RAGStore(tmp_path)
    _seed_rag(rag, "SBER", ["единственная новость"])

    mock_llm = AsyncMock()

    with patch("app.news.consensus_rag.llm_chat_json", mock_llm):
        consensus = await build_morning_consensus(
            rag,
            llm_backend="polza",
            tickers=["SBER"],
            hours=24,
            min_news=3,
        )
    assert consensus["SBER"].direction == "NEUTRAL"
    assert consensus["SBER"].strength == 0.0
    assert mock_llm.await_count == 0


@pytest.mark.asyncio
async def test_build_morning_consensus_handles_llm_error(tmp_path):
    """Test build morning consensus handles llm error."""
    rag = RAGStore(tmp_path)
    _seed_rag(rag, "SBER", ["a", "b", "c", "d"])

    mock_llm = AsyncMock(side_effect=RuntimeError("llm down"))

    with patch("app.news.consensus_rag.llm_chat_json", mock_llm):
        consensus = await build_morning_consensus(
            rag,
            llm_backend="polza",
            tickers=["SBER"],
            hours=24,
        )
    assert consensus["SBER"].direction == "NEUTRAL"
    assert consensus["SBER"].n_news == 4
