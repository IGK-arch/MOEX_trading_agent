"""Historical storage and similarity search for news events."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import app.config as cfg
from app.news.ingestion_bus import NormalizedNewsEvent
from app.utils.logging import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[\w\-]+", re.UNICODE)

EVENT_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sanctions": ("sanction", "санкц", "ofac", "sdn", "блокировк", "заморожен"),
    "dividend": ("dividend", "дивиденд", "buyback", "выкуп", "payout", "выплат"),
    "earnings_beat": ("beat estimate", "beat forecast", "лучше прогноза", "выше ожидан"),
    "earnings_miss": ("miss estimate", "miss forecast", "хуже прогноза", "ниже ожидан"),
    "earnings": ("earnings", "отчет", "отчёт", "profit", "revenue", "ebitda", "выручк", "прибыл"),
    "guidance_up": ("raised guidance", "повыш прогноз", "upgraded"),
    "guidance_down": ("cut guidance", "lowered guidance", "снизил прогноз", "downgraded"),
    "guidance": ("guidance", "outlook", "forecast", "прогноз"),
    "ipo": ("ipo", "first listing", "первичное размещен"),
    "spo": ("spo", "secondary offering", "вторичное размещен", "доразмещен"),
    "macro_rate": (
        "key rate",
        "ставка цб",
        "ставка ключевая",
        "rate decision",
        "fomc",
        "ecb meeting",
        "cbr meeting",
    ),
    "macro_inflation": ("cpi", "inflation", "инфляц", "ppi", "indice потребитель"),
    "macro_gdp": ("gdp", "ввп", "growth rate", "темп роста экономик"),
    "macro": (
        "recession",
        "stagflation",
        "стагфляц",
        "рецесс",
        "key rate",
        "ставка",
        "cpi",
        "inflation",
        "fomc",
        "ecb",
        "cbr",
    ),
    "commodity_oil": ("opec", "brent", "wti", "нефть", "oil price", "нефтяной"),
    "commodity_gas": ("gas price", "lng", "газ", "natural gas", "газовый"),
    "commodity_metals": ("nickel", "никел", "gold", "золот", "iron ore", "руд", "copper", "медь"),
    "commodity": ("opec", "oil", "нефть", "gas", "газ", "nickel", "gold"),
    "deal": ("merger", "acquisition", "m&a", "поглощ", "сделк", "приобрет"),
    "operations": ("accident", "авар", "halt", "suspension", "strike", "забаст", "взрыв", "пожар"),
    "management": ("ceo", "resignation", "отставк", "директор", "генеральн"),
    "rating": ("rating", "рейтинг", "moody's", "s&p", "fitch", "акра"),
}

REACTION_WINDOWS_MIN = (5, 15, 30, 60, 120)

def _utc_now_iso() -> str:
    """Utc now iso."""
    return datetime.now(tz=UTC).isoformat()

def _normalise_text(text: str) -> str:
    """Normalise text."""
    return " ".join(_WORD_RE.findall((text or "").lower()))

def _tokenise(text: str) -> set[str]:
    """Tokenise."""
    return set(_WORD_RE.findall((text or "").lower()))

def classify_event_type(text: str, is_sanctions: bool = False) -> str:
    """Classify event type."""
    if is_sanctions:
        return "sanctions"
    text_norm = (text or "").lower()
    for event_type, keywords in EVENT_TYPE_KEYWORDS.items():
        if any(k in text_norm for k in keywords):
            return event_type
    return "other"

@dataclass
class SimilarCase:
    """Similar Case."""

    event_id: str
    ticker: str
    score: float
    headline: str
    event_type: str
    direction: str
    source_tier: str
    ret_15m: float | None
    ret_60m: float | None
    ret_120m: float | None

class NewsHistoryStore:
    """SQLite-backed history for news events and market reactions."""

    def __init__(self, db_path: str | None = None) -> None:
        """Init."""
        self.db_path = db_path or str(cfg.DATA_DIR / "feeds.db")

    async def record_event(
        self,
        event: NormalizedNewsEvent,
        tickers: list[str],
        matched_keywords: list[str],
        is_material: bool,
        is_sanctions: bool,
    ) -> str:
        """Record event."""
        event_text = f"{event.headline} {event.body}".strip()
        event_type = classify_event_type(event_text, is_sanctions=is_sanctions)

        def _write() -> None:
            """Write."""
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO news_events
                    (event_id, ts_utc, source, source_tier, headline, body, url, language,
                     tickers_json, matched_keywords_json, event_type, is_material,
                     is_sanctions, text_norm, raw_payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.ts_utc.astimezone(UTC).isoformat(),
                        event.source,
                        event.source_tier,
                        event.headline,
                        event.body,
                        event.url,
                        event.language,
                        json.dumps(tickers, ensure_ascii=False),
                        json.dumps(matched_keywords, ensure_ascii=False),
                        event_type,
                        1 if is_material else 0,
                        1 if is_sanctions else 0,
                        _normalise_text(event_text),
                        json.dumps(event.raw_payload, ensure_ascii=False),
                        _utc_now_iso(),
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_write)
        return event_type

    async def update_analysis(
        self,
        event_id: str,
        direction: str,
        magnitude: float,
        horizon_min: int,
        reason: str,
    ) -> None:
        """Update analysis."""

        def _write() -> None:
            """Write."""
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE news_events
                    SET llm_direction=?, llm_magnitude=?, horizon_min=?, reason=?
                    WHERE event_id=?
                    """,
                    (direction, magnitude, horizon_min, reason[:300], event_id),
                )
                conn.commit()

        await asyncio.to_thread(_write)

    async def save_context_snapshot(
        self,
        event_id: str,
        ticker: str,
        context: dict[str, Any],
    ) -> None:
        """Save context snapshot."""

        def _write() -> None:
            """Write."""
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO news_context_snapshots
                    (event_id, ticker, price, atr, atr_pct, rsi, vol_z, ret_30m_pct,
                     regime, regime_proba_json, catboost_score, ta_pattern, ta_direction,
                     ta_expected_rr, historical_bias, retrieval_cases, context_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        ticker,
                        float(context.get("price", 0.0) or 0.0),
                        float(context.get("atr", 0.0) or 0.0),
                        float(context.get("atr_pct", 0.0) or 0.0),
                        float(context.get("rsi", 0.0) or 0.0),
                        float(context.get("vol_z", 0.0) or 0.0),
                        float(context.get("ret_30m_pct", 0.0) or 0.0),
                        str(context.get("regime", "unknown")),
                        json.dumps(context.get("regime_proba", {}), ensure_ascii=False),
                        float(context.get("catboost_score", 0.0) or 0.0),
                        str(context.get("ta_pattern", "")),
                        str(context.get("ta_direction", "")),
                        float(context.get("ta_expected_rr", 0.0) or 0.0),
                        float(context.get("historical_bias", 0.0) or 0.0),
                        int(context.get("retrieval_cases", 0) or 0),
                        json.dumps(context, ensure_ascii=False),
                        _utc_now_iso(),
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_write)

    async def save_reaction(
        self,
        event_id: str,
        ticker: str,
        window_min: int,
        price_t0: float,
        price_tn: float,
    ) -> None:
        """Save reaction."""
        ret_pct = ((price_tn / price_t0) - 1.0) if price_t0 > 0 else 0.0

        def _write() -> None:
            """Write."""
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO news_reactions
                    (event_id, ticker, window_min, price_t0, price_tn, return_pct, abs_move_pct, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        ticker,
                        window_min,
                        price_t0,
                        price_tn,
                        ret_pct,
                        abs(ret_pct),
                        _utc_now_iso(),
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_write)

    async def find_similar_cases(
        self,
        *,
        event: NormalizedNewsEvent,
        ticker: str,
        matched_keywords: list[str],
        event_type: str,
        is_sanctions: bool,
        regime: str,
        catboost_score: float,
        limit: int = 5,
        lookback_days: int = 365,
    ) -> dict[str, Any]:
        """Find similar cases."""
        current_tokens = _tokenise(f"{event.headline} {event.body}")
        current_keywords = {k.lower() for k in matched_keywords}
        current_ts = event.ts_utc.astimezone(UTC)
        cutoff_ts = current_ts - timedelta(days=lookback_days)

        def _read() -> list[dict[str, Any]]:
            """Read."""
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT e.event_id, e.ts_utc, e.source_tier, e.headline, e.event_type,
                           e.tickers_json, e.matched_keywords_json, e.is_sanctions,
                           e.llm_direction, c.regime, c.catboost_score,
                           MAX(CASE WHEN r.window_min = 15 THEN r.return_pct END) AS ret_15m,
                           MAX(CASE WHEN r.window_min = 60 THEN r.return_pct END) AS ret_60m,
                           MAX(CASE WHEN r.window_min = 120 THEN r.return_pct END) AS ret_120m
                    FROM news_events e
                    LEFT JOIN news_context_snapshots c
                      ON c.event_id = e.event_id AND c.ticker = ?
                    LEFT JOIN news_reactions r
                      ON r.event_id = e.event_id AND r.ticker = ?
                    -- Phase 16 (v0.0.12): JSON-aware LIKE pattern. The previous
                    -- '%TICKER%' matched SNGS inside SNGSP rows. JSON serialises
                    -- tickers with double quotes so '%"TICKER"%' is exact.
                    WHERE e.event_id != ?
                      AND e.ts_utc >= ?
                      AND e.tickers_json LIKE ?
                    GROUP BY e.event_id, e.ts_utc, e.source_tier, e.headline, e.event_type,
                             e.tickers_json, e.matched_keywords_json, e.is_sanctions,
                             e.llm_direction, c.regime, c.catboost_score
                    ORDER BY e.ts_utc DESC
                    LIMIT 200
                    """,
                    (
                        ticker,
                        ticker,
                        event.event_id,
                        cutoff_ts.isoformat(),
                        f'%"{ticker}"%',
                    ),
                ).fetchall()
                return [dict(r) for r in rows]

        rows = await asyncio.to_thread(_read)
        cases: list[SimilarCase] = []
        for row in rows:
            score = self._similarity_score(
                current_tokens=current_tokens,
                current_keywords=current_keywords,
                current_event_type=event_type,
                current_source_tier=event.source_tier,
                current_is_sanctions=is_sanctions,
                current_regime=regime,
                current_catboost_score=catboost_score,
                row=row,
            )
            if score <= 0.15:
                continue
            cases.append(
                SimilarCase(
                    event_id=str(row["event_id"]),
                    ticker=ticker,
                    score=score,
                    headline=str(row["headline"] or ""),
                    event_type=str(row["event_type"] or "other"),
                    direction=str(row["llm_direction"] or "NEUTRAL"),
                    source_tier=str(row["source_tier"] or "C"),
                    ret_15m=_to_float(row.get("ret_15m")),
                    ret_60m=_to_float(row.get("ret_60m")),
                    ret_120m=_to_float(row.get("ret_120m")),
                )
            )

        cases.sort(key=lambda c: c.score, reverse=True)
        top_cases = cases[:limit]
        return self._summarise_cases(top_cases)

    @staticmethod
    def _similarity_score(
        *,
        current_tokens: set[str],
        current_keywords: set[str],
        current_event_type: str,
        current_source_tier: str,
        current_is_sanctions: bool,
        current_regime: str,
        current_catboost_score: float,
        row: dict[str, Any],
    ) -> float:
        """Similarity score."""
        text_tokens = _tokenise(str(row.get("headline", "")))
        text_jaccard = _jaccard(current_tokens, text_tokens)
        hist_keywords = {
            k.lower()
            for k in json.loads(row.get("matched_keywords_json") or "[]")
            if isinstance(k, str)
        }
        keyword_overlap = _jaccard(current_keywords, hist_keywords)

        score = 0.0
        score += 0.25 * text_jaccard
        score += 0.15 * keyword_overlap
        if str(row.get("event_type") or "other") == current_event_type:
            score += 0.20
        if bool(row.get("is_sanctions")) == current_is_sanctions:
            score += 0.10
        if str(row.get("source_tier") or "") == current_source_tier:
            score += 0.05
        if str(row.get("regime") or "unknown") == current_regime:
            score += 0.10

        hist_cb = _to_float(row.get("catboost_score"))
        if hist_cb is not None and current_catboost_score > 0:
            score += 0.15 * max(0.0, 1.0 - min(1.0, abs(hist_cb - current_catboost_score)))

        return min(1.0, score)

    @staticmethod
    def _summarise_cases(cases: list[SimilarCase]) -> dict[str, Any]:
        """Summarise cases."""
        if not cases:
            return {
                "n_cases": 0,
                "bias": 0.0,
                "bias_label": "NEUTRAL",
                "avg_ret_15m": 0.0,
                "avg_ret_60m": 0.0,
                "positive_rate_15m": 0.0,
                "positive_rate_60m": 0.0,
                "top_cases": [],
                "summary_text": "No close historical analogs.",
            }

        def _avg(vals: list[float | None]) -> float:
            """Avg."""
            real = [v for v in vals if v is not None]
            return sum(real) / max(1, len(real))

        avg_15m = _avg([c.ret_15m for c in cases])
        avg_60m = _avg([c.ret_60m for c in cases])
        pos_15m = sum(1 for c in cases if (c.ret_15m or 0.0) > 0) / len(cases)
        pos_60m = sum(1 for c in cases if (c.ret_60m or 0.0) > 0) / len(cases)
        bias = (avg_15m * 0.4) + (avg_60m * 0.6)
        bias_label = "BUY" if bias > 0.002 else ("SELL" if bias < -0.002 else "NEUTRAL")
        summary_text = (
            f"{len(cases)} similar cases: avg15m={avg_15m:+.2%}, "
            f"avg60m={avg_60m:+.2%}, pos60m={pos_60m:.0%}, bias={bias_label}"
        )

        return {
            "n_cases": len(cases),
            "bias": bias,
            "bias_label": bias_label,
            "avg_ret_15m": avg_15m,
            "avg_ret_60m": avg_60m,
            "positive_rate_15m": pos_15m,
            "positive_rate_60m": pos_60m,
            "top_cases": [
                {
                    "event_id": c.event_id,
                    "score": round(c.score, 3),
                    "headline": c.headline[:120],
                    "event_type": c.event_type,
                    "direction": c.direction,
                    "ret_15m": c.ret_15m,
                    "ret_60m": c.ret_60m,
                    "ret_120m": c.ret_120m,
                }
                for c in cases
            ],
            "summary_text": summary_text,
        }

def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard."""
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def _to_float(value: Any) -> float | None:
    """To float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

_history_store: NewsHistoryStore | None = None

def get_history_store() -> NewsHistoryStore:
    """Get history store."""
    global _history_store
    if _history_store is None:
        _history_store = NewsHistoryStore()
    return _history_store
