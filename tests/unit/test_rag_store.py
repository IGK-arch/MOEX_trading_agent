"""Unit tests for app.memory.rag_store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.memory.rag_store import RAGStore, _EmbedderFallback, _hash_embed


@pytest.fixture
def tmp_rag(tmp_path):
    """Tmp rag."""
    return RAGStore(tmp_path)


def _push(
    rag: RAGStore,
    event_id: str,
    text: str,
    ticker: str,
    *,
    age_hours: float = 0.0,
    source: str = "test",
    tier: str = "A",
) -> None:
    """Push."""
    ts = datetime.now(tz=UTC) - timedelta(hours=age_hours)
    rag.add_news(
        event_id=event_id,
        text=text,
        ts_utc=ts,
        tickers=[ticker],
        source=source,
        source_tier=tier,
        headline=text[:120],
        body=text,
    )


def test_rag_add_and_search_finds_top_hit(tmp_rag):
    """Test rag add and search finds top hit."""
    _push(tmp_rag, "a1", "Сбербанк рекомендовал дивиденды 35 рублей за акцию", "SBER")
    _push(tmp_rag, "a2", "Газпром снизил добычу газа на 12%", "GAZP")
    _push(tmp_rag, "a3", "Сбербанк отчитался о росте прибыли", "SBER")
    results = tmp_rag.search("Сбербанк дивиденды", tickers=["SBER"], top_k=3)
    assert len(results) >= 1
    assert results[0]["event_id"] in {"a1", "a3"}
    assert results[0]["score"] > 0.0


def test_rag_ticker_filter_excludes_other_tickers(tmp_rag):
    """Test rag ticker filter excludes other tickers."""
    _push(tmp_rag, "s1", "Сбербанк рекомендовал дивиденды", "SBER")
    _push(tmp_rag, "g1", "Газпром снизил добычу", "GAZP")
    sber_only = tmp_rag.search("новости по компании", tickers=["SBER"], top_k=5)
    assert all("SBER" in r["tickers"] for r in sber_only)
    gazp_only = tmp_rag.search("новости по компании", tickers=["GAZP"], top_k=5)
    assert all("GAZP" in r["tickers"] for r in gazp_only)


def test_rag_age_filter_excludes_stale(tmp_rag):
    """Test rag age filter excludes stale."""
    _push(tmp_rag, "fresh", "Сбербанк объявил buyback", "SBER", age_hours=1.0)
    _push(tmp_rag, "stale", "Сбербанк объявил buyback год назад", "SBER", age_hours=24 * 30)
    out = tmp_rag.search("Сбербанк buyback", tickers=["SBER"], top_k=10, max_age_hours=48)
    ids = [r["event_id"] for r in out]
    assert "fresh" in ids
    assert "stale" not in ids


def test_rag_get_recent_for_ticker_orders_desc(tmp_rag):
    """Test rag get recent for ticker orders desc."""
    _push(tmp_rag, "old", "Сбербанк отчитался", "SBER", age_hours=12.0)
    _push(tmp_rag, "new", "Сбербанк рекомендовал дивиденды", "SBER", age_hours=1.0)
    out = tmp_rag.get_recent_for_ticker("SBER", hours=24)
    assert len(out) == 2
    assert out[0]["event_id"] == "new"
    assert out[1]["event_id"] == "old"


def test_rag_dedup_by_event_id(tmp_rag):
    """Test rag dedup by event id."""
    _push(tmp_rag, "dup", "Сбербанк рекомендовал 35 рублей", "SBER")
    _push(tmp_rag, "dup", "Сбербанк РЕКОМЕНДОВАЛ 40 рублей", "SBER")
    assert len(tmp_rag) == 1
    rec = tmp_rag.get_recent_for_ticker("SBER", hours=24)[0]
    assert "40" in rec["body"] or "40" in rec["headline"]


def test_rag_prune_older_than(tmp_rag):
    """Test rag prune older than."""
    _push(tmp_rag, "fresh", "Сбербанк сегодня", "SBER", age_hours=2.0)
    _push(tmp_rag, "ancient", "Сбербанк давно", "SBER", age_hours=24 * 14)
    pruned = tmp_rag.prune_older_than(hours=24)
    assert pruned == 1
    ids = [r.event_id for r in tmp_rag._records]
    assert ids == ["fresh"]


def test_rag_persist_and_reload(tmp_path):
    """Test rag persist and reload."""
    rag = RAGStore(tmp_path)
    _push(rag, "p1", "Сбербанк объявил buyback", "SBER")
    _push(rag, "p2", "Газпром нарастил добычу", "GAZP")
    rag2 = RAGStore(tmp_path)
    assert len(rag2) == 2
    results = rag2.search("Сбербанк", tickers=["SBER"], top_k=3)
    assert results, "search after reload must return results"
    assert results[0]["event_id"] == "p1"


def test_hash_embed_is_l2_normalised():
    """Test hash embed is l2 normalised."""
    import numpy as np

    vec = _hash_embed("привет мир", dim=128)
    assert vec.shape == (128,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


def test_embedder_fallback_batch():
    """Test embedder fallback batch."""
    import numpy as np

    fb = _EmbedderFallback(dim=64)
    mat = fb.encode(["a", "b", "c"], normalize_embeddings=True)
    assert np.asarray(mat).shape == (3, 64)


def test_rag_search_empty_store_returns_empty(tmp_rag):
    """Test rag search empty store returns empty."""
    assert tmp_rag.search("anything", top_k=5) == []
    assert tmp_rag.get_recent_for_ticker("SBER", hours=24) == []


def test_rag_search_normalises_unicode_query(tmp_rag):
    """Test rag search normalises unicode query."""
    _push(tmp_rag, "u1", "Газпром санкции против труб", "GAZP")
    results = tmp_rag.search("санкции", tickers=["GAZP"], top_k=1)
    assert results
    assert results[0]["event_id"] == "u1"
