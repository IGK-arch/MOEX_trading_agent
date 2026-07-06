"""Tests for historical news memory and enhanced news signal helpers."""

from app.agents.news_llm import NewsLLM
from app.news.history import SimilarCase, classify_event_type, get_history_store


def test_classify_event_type_dividend():
    """Test classify event type dividend."""
    assert classify_event_type("Сбербанк рекомендовал дивиденды 35 рублей") == "dividend"


def test_classify_event_type_sanctions_override():
    """Test classify event type sanctions override."""
    assert classify_event_type("Neutral headline", is_sanctions=True) == "sanctions"


def test_similar_cases_summary_positive_bias():
    """Test similar cases summary positive bias."""
    cases = [
        SimilarCase("a", "SBER", 0.9, "Dividend 1", "dividend", "BUY", "A", 0.012, 0.020, 0.030),
        SimilarCase("b", "SBER", 0.8, "Dividend 2", "dividend", "BUY", "S", 0.008, 0.015, 0.018),
    ]
    summary = get_history_store()._summarise_cases(cases)
    assert summary["n_cases"] == 2
    assert summary["bias_label"] == "BUY"
    assert summary["avg_ret_60m"] > 0
    assert summary["positive_rate_60m"] == 1.0


def test_news_magnitude_gets_boosted_by_history_and_ta():
    """Test news magnitude gets boosted by history and ta."""
    boosted = NewsLLM._finalise_magnitude(
        base_magnitude=0.55,
        source_tier="S",
        direction="BUY",
        historical_summary={"bias": 0.01, "bias_label": "BUY"},
        ta_direction="BUY",
        catboost_score=0.75,
    )
    faded = NewsLLM._finalise_magnitude(
        base_magnitude=0.55,
        source_tier="C",
        direction="BUY",
        historical_summary={"bias": -0.01, "bias_label": "SELL"},
        ta_direction="SELL",
        catboost_score=0.10,
    )
    assert boosted > faded
    assert 0.0 <= boosted <= 1.0


def test_news_trade_levels_have_positive_rr():
    """Test news trade levels have positive rr."""
    entry, stop, target = NewsLLM._build_trade_levels(
        direction="BUY",
        price=100.0,
        atr=2.0,
        expected_rr=1.8,
        entry_bias="market_now",
    )
    assert entry > stop
    assert target > entry

    entry_s, stop_s, target_s = NewsLLM._build_trade_levels(
        direction="SELL",
        price=100.0,
        atr=2.0,
        expected_rr=1.8,
        entry_bias="wait_breakout",
    )
    assert stop_s > entry_s
    assert target_s < entry_s
