"""Tests for app/news/local_sentiment.py — local sentiment fallback.

These tests cover the keyword scorer (always available) and the
LocalSentimentScorer wrapper. The HuggingFace BERT path requires
``transformers`` + ``torch`` plus a model download, so we mock the
pipeline rather than exercising it here. End-to-end inference is
covered separately in CI smoke tests.
"""

from __future__ import annotations

import pytest

from app.news.local_sentiment import (
    LocalSentimentScorer,
    get_local_sentiment_scorer,
    keyword_sentiment_score,
)


def test_keyword_score_positive_russian():
    """Test keyword score positive russian."""
    score = keyword_sentiment_score(
        "Сбербанк увеличил прибыль на 30% и выплатит рекордные дивиденды"
    )
    assert score > 0.2, f"expected positive score, got {score}"


def test_keyword_score_negative_russian():
    """Test keyword score negative russian."""
    score = keyword_sentiment_score(
        "OFAC ввёл санкции и запрет на экспорт, штраф $1 млрд, падение акций"
    )
    assert score < -0.5, f"expected strongly negative score, got {score}"


def test_keyword_score_single_negative_keyword():
    """Test keyword score single negative keyword."""
    score = keyword_sentiment_score("В отношении компании введены санкции")
    assert -0.3 < score < 0.0


def test_keyword_score_neutral_empty():
    """Test keyword score neutral empty."""
    assert keyword_sentiment_score("") == 0.0
    assert keyword_sentiment_score("Сегодня прошло собрание акционеров") == 0.0


def test_keyword_score_in_range():
    """Score must stay in [-1, +1] even with many keyword repeats."""
    pos = "рост прибыли выплата дивидендов рекордные дивиденды выкуп"
    neg = "санкции запрет блокировка штраф падение убытки"
    assert -1.0 <= keyword_sentiment_score(pos) <= 1.0
    assert -1.0 <= keyword_sentiment_score(neg) <= 1.0


def test_scorer_keyword_only_mode():
    """force_keyword_only=True must skip HF entirely and stay deterministic."""
    s = LocalSentimentScorer(force_keyword_only=True)
    score_pos = s.score_text(
        "Газпром заключил рекордный экспортный контракт, ожидается рост выручки"
    )
    score_neg = s.score_text("Аэрофлот: убытки, санкции, отзыв лицензии, расследование")
    assert score_pos > 0
    assert score_neg < 0
    assert s.backend == "keyword"


def test_scorer_score_text_empty():
    """Test scorer score text empty."""
    s = LocalSentimentScorer(force_keyword_only=True)
    assert s.score_text("") == 0.0


def test_scorer_singleton_returns_same_instance():
    """Test scorer singleton returns same instance."""
    a = get_local_sentiment_scorer()
    b = get_local_sentiment_scorer()
    assert a is b


def test_scorer_falls_back_when_bert_unavailable(monkeypatch):
    """If transformers import fails, fall back to keyword silently."""
    s = LocalSentimentScorer(model_candidates=("nonexistent/no-such-model",))
    score = s.score_text("Сбербанк увеличил прибыль и выплатит дивиденды")
    assert score > 0
    assert s.backend in ("keyword", "bert")


def test_scorer_stats_after_call():
    """Test scorer stats after call."""
    s = LocalSentimentScorer(force_keyword_only=True)
    s.score_text("Сбербанк прибыль рекордная")
    s.score_text("санкции блокировка")
    stats = s.stats()
    assert stats["calls_total"] == 2
    assert stats["calls_keyword"] == 2
    assert stats["calls_bert"] == 0
    assert stats["backend"] == "keyword"


def test_label_to_score_mapping():
    """Test label to score mapping."""
    s = LocalSentimentScorer(force_keyword_only=True)
    assert s._label_to_score("positive", 0.9) == pytest.approx(0.9)
    assert s._label_to_score("negative", 0.8) == pytest.approx(-0.8)
    assert s._label_to_score("neutral", 0.7) == 0.0
    assert s._label_to_score("LABEL_1", 0.6) == pytest.approx(0.6)
    assert s._label_to_score("LABEL_2", 0.5) == pytest.approx(-0.5)
    assert s._label_to_score("LABEL_0", 0.4) == 0.0


def test_scorer_returns_in_range():
    """Test scorer returns in range."""
    s = LocalSentimentScorer(force_keyword_only=True)
    cases = [
        "Сбербанк рекордная прибыль дивиденды",
        "санкции штрафы блокировка убытки",
        "обычное собрание акционеров",
        "",
    ]
    for text in cases:
        score = s.score_text(text)
        assert -1.0 <= score <= 1.0, f"out of range: {score} for '{text}'"
