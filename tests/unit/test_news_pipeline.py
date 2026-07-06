"""Test news pipeline: ticker tagger, material filter, dedup."""

from app.news.dedup import NewsDeduplicator
from app.news.material_filter import is_material
from app.news.ticker_tagger import get_ticker_tagger


def test_ticker_tagger_russian():
    """Test ticker tagger russian."""
    tagger = get_ticker_tagger()
    assert "SBER" in tagger.tag("Сбербанк объявил дивиденды")
    assert "GAZP" in tagger.tag("Газпром приостановил поставки газа")
    assert "GMKN" in tagger.tag("Норникель снизил прогноз")


def test_ticker_tagger_english():
    """Test ticker tagger english."""
    tagger = get_ticker_tagger()
    assert "SBER" in tagger.tag("Sberbank reports earnings")
    assert "PLZL" in tagger.tag("Polyus Gold wins auction")

    tags = tagger.tag("Iron ore prices crashed")
    assert "NLMK" in tags or "CHMF" in tags


def test_ticker_tagger_commodity_to_tickers():
    """Test ticker tagger commodity to tickers."""
    tagger = get_ticker_tagger()
    tags = tagger.tag("Brent crude rose 3% on OPEC+ cuts")
    assert "LKOH" in tags
    assert "ROSN" in tags


def test_material_filter_dividend():
    """Test material filter dividend."""
    is_mat, kws, always = is_material(
        "Сбербанк рекомендовал дивиденды 32 рубля на акцию",
        has_ticker=True,
    )
    assert is_mat
    assert "дивиденд" in kws
    assert not always


def test_material_filter_sanctions_always():
    """Test material filter sanctions always."""
    is_mat, kws, always = is_material(
        "OFAC adds entities to SDN list",
        has_ticker=False,
    )
    assert is_mat
    assert always


def test_material_filter_needs_ticker():
    """Без тикера и без ALWAYS-keyword → нематериально."""
    is_mat, kws, always = is_material(
        "Сегодня прошло обычное собрание акционеров",
        has_ticker=False,
    )
    assert not is_mat


def test_dedup_blocks_near_duplicates():
    """Test dedup blocks near duplicates."""
    d = NewsDeduplicator(threshold=0.75)

    assert not d.is_duplicate(
        "a1",
        "Сбербанк объявил рекордные дивиденды в размере тридцать два рубля на акцию по итогам года",
    )

    assert (
        d.is_duplicate(
            "a2",
            "Сбербанк объявил рекордные дивиденды в размере тридцать два рубля на акцию по итогам года",
        )
        is True
    )

    assert not d.is_duplicate("a3", "Газпром остановил экспорт газа в Европу после санкций")


def test_dedup_stats():
    """Test dedup stats."""
    d = NewsDeduplicator()
    d.is_duplicate("x1", "some text here")
    d.is_duplicate("x2", "another text here")
    stats = d.stats()
    assert stats["total_checks"] == 2
    assert stats["index_size"] == 2


def test_ticker_tagger_v060_new_tickers_russian():
    """Tinkoff / X5 / Алроса / Мосбиржа must all tag to MOEX symbols."""
    tagger = get_ticker_tagger()
    assert "T" in tagger.tag("Тинькофф запустил новый продукт"), (
        "Tinkoff/T-Bank must map to T (was empty in v0.5.x)"
    )
    assert "X5" in tagger.tag("X5 Retail сообщил о росте выручки"), "X5 must map to X5 ticker"
    assert "ALRS" in tagger.tag("Алроса увеличила продажи алмазов"), "Алроса must map to ALRS"
    assert "MOEX" in tagger.tag("Мосбиржа отчиталась о прибыли"), "Мосбиржа must map to MOEX"


def test_ticker_tagger_v060_new_tickers_english():
    """Test ticker tagger v060 new tickers english."""
    tagger = get_ticker_tagger()
    assert "T" in tagger.tag("Tinkoff Bank reports record earnings")
    assert "X5" in tagger.tag("X5 Retail Group announces buyback")
    assert "ALRS" in tagger.tag("Alrosa diamond sales beat estimates")
    assert "MOEX" in tagger.tag("Moscow Exchange volumes climbed")


def test_ticker_tagger_diamonds_commodity():
    """Diamond/diamonds/алмаз should pull ALRS via commodity map."""
    tagger = get_ticker_tagger()
    assert "ALRS" in tagger.tag("Global diamond prices surged on supply cuts")
    assert "ALRS" in tagger.tag("Мировые цены на алмазы выросли")


def test_keyword_direction_inference_bearish():
    """v0.6.0 — when LLM says NEUTRAL but headline has SELL keywords, infer SELL."""
    from app.agents.news_llm import NewsLLM

    out = NewsLLM._infer_direction_from_keywords(
        "Минфин ввёл штраф для Сбербанка за нарушение",
        "штраф и расследование",
        ["штраф"],
        is_sanctions=False,
    )
    assert out == "SELL"


def test_keyword_direction_inference_bullish():
    """Test keyword direction inference bullish."""
    from app.agents.news_llm import NewsLLM

    out = NewsLLM._infer_direction_from_keywords(
        "Сбербанк показал рекордную прибыль и объявил дивиденды",
        "выручка и прибыль выше прогноза",
        ["прибыл", "дивиденд"],
        is_sanctions=False,
    )
    assert out == "BUY"


def test_keyword_direction_inference_sanctions_always_sell():
    """Sanctions short-circuit to SELL regardless of LLM verdict."""
    from app.agents.news_llm import NewsLLM

    out = NewsLLM._infer_direction_from_keywords(
        "OFAC adds new entities to SDN list",
        "",
        ["ofac", "sdn list"],
        is_sanctions=True,
    )
    assert out == "SELL"


def test_keyword_direction_inference_ambiguous_returns_none():
    """Mixed bullish + bearish → keep NEUTRAL (return None → drop event)."""
    from app.agents.news_llm import NewsLLM

    out = NewsLLM._infer_direction_from_keywords(
        "Tatneft beat earnings but received a fine from regulators",
        "штраф и прибыль одновременно",
        ["штраф", "прибыл"],
        is_sanctions=False,
    )
    assert out is None


def test_keyword_direction_inference_no_keywords_returns_none():
    """Test keyword direction inference no keywords returns none."""
    from app.agents.news_llm import NewsLLM

    out = NewsLLM._infer_direction_from_keywords(
        "Какая-то нейтральная новость без катализаторов",
        "",
        [],
        is_sanctions=False,
    )
    assert out is None


def test_classify_event_type_sanctions():
    """Test classify event type sanctions."""
    from app.agents.news_llm import NewsLLM

    assert NewsLLM._classify_event_type("OFAC adds SBER to SDN list") == "sanctions"
    assert NewsLLM._classify_event_type("ЦБ объявил санкции против банка") == "sanctions"


def test_classify_event_type_earnings_buckets_distinct():
    """Sanity: classifier returns distinct buckets for distinct catalysts.

    Note: keyword order in `_classify_event_type` is sanctions → macro →
    commodity → earnings → guidance → other, so "Gazprom earnings" hits
    "газ" (commodity) first. We use a non-energy ticker in the earnings
    case to keep the test deterministic.
    """
    from app.agents.news_llm import NewsLLM

    assert NewsLLM._classify_event_type("Сбербанк отчитался за квартал") == "earnings"
    assert NewsLLM._classify_event_type("ЦБ повысил ключевую ставку") == "macro"
    assert NewsLLM._classify_event_type("Brent нефть подорожала на 3%") == "commodity"
