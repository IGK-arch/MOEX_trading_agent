"""Per-ticker morning consensus builder.

Phase 27.4 (RAG Consensus). At 08:30 МСК (90 min before MOEX open) we
collate every news/analytics item indexed in the :class:`RAGStore` for
each of the 20 tickers in ``cfg.TICKERS`` over the last 24h and ask the
LLM (Gemini via polza by default, or deepseek-flash as fallback) to
produce a structured consensus::

    ConsensusEntry(
        ticker="SBER",
        direction="BUY",        # BUY | SELL | NEUTRAL
        strength=0.7,           # 0..1
        key_themes=["дивиденды", "выручка Q1"],
        rationale="Аналитики ждут позитива на дивидендах и Q1...",
        expected_to_positive="новый рост, прибыль, дивиденды",
        expected_to_negative="миссы, санкции, downgrade",
        built_at=datetime,
    )

The resulting per-ticker map is persisted under ``data/rag/consensus_
<YYYY-MM-DD>.json`` so it can be reloaded after a restart and re-used by
the reactive comparator.

Budget. 20 tickers × ~1 reasoning call = 20 LLM calls. On
``google/gemini-2.5-flash`` (₽4-8 each, polza relay) that's well under
₽200 per day — the morning cycle is by far the cheapest part of the
system. Each call is cached by polza so a same-day re-run is free.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app.config as cfg
from app.llm.gemini_client import llm_chat_json
from app.memory.rag_store import RAGStore
from app.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a sell-side strategist building a single-day directional "
    "consensus for a Moscow Exchange (MOEX) ticker. Output STRICT JSON "
    "ONLY — no prose, no markdown fences, no explanations outside the "
    "JSON object."
)

@dataclass
class ConsensusEntry:
    """Consensus Entry."""

    ticker: str
    direction: str = "NEUTRAL"
    strength: float = 0.0
    key_themes: list[str] = field(default_factory=list)
    rationale: str = ""
    expected_to_positive: str = ""
    expected_to_negative: str = ""
    built_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    n_news: int = 0
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        d = asdict(self)
        d["built_at"] = self.built_at.astimezone(UTC).isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConsensusEntry:
        """From dict."""
        built_at_raw = d.get("built_at")
        if isinstance(built_at_raw, str):
            built_at = datetime.fromisoformat(built_at_raw)
            if built_at.tzinfo is None:
                built_at = built_at.replace(tzinfo=UTC)
        elif isinstance(built_at_raw, datetime):
            built_at = built_at_raw
        else:
            built_at = datetime.now(tz=UTC)
        return cls(
            ticker=str(d.get("ticker") or "").upper(),
            direction=str(d.get("direction") or "NEUTRAL").upper(),
            strength=float(d.get("strength") or 0.0),
            key_themes=list(d.get("key_themes") or []),
            rationale=str(d.get("rationale") or ""),
            expected_to_positive=str(d.get("expected_to_positive") or ""),
            expected_to_negative=str(d.get("expected_to_negative") or ""),
            built_at=built_at,
            n_news=int(d.get("n_news") or 0),
            backend=str(d.get("backend") or ""),
        )

    @classmethod
    def neutral(cls, ticker: str, n_news: int = 0) -> ConsensusEntry:
        """Neutral."""
        return cls(ticker=ticker.upper(), direction="NEUTRAL", strength=0.0, n_news=n_news)

def _format_news_block(news: list[dict[str, Any]], limit: int = 12) -> str:
    """Render a compact news bullet list for the consensus prompt."""
    lines: list[str] = []
    for i, n in enumerate(news[:limit], 1):
        head = (n.get("headline") or "")[:200].replace("\n", " ")
        body = (n.get("body") or "")[:400].replace("\n", " ")
        ts = (n.get("ts_utc") or "")[:19]
        src = n.get("source") or "?"
        tier = n.get("source_tier") or "?"
        if body:
            lines.append(f"  {i}. [{src} · {tier} · {ts}] {head} — {body}")
        else:
            lines.append(f"  {i}. [{src} · {tier} · {ts}] {head}")
    return "\n".join(lines) or "  (no news in window)"

def _build_consensus_prompt(ticker: str, news: list[dict[str, Any]]) -> str:
    """Build consensus prompt."""
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    block = _format_news_block(news, limit=12)
    return (
        f"Дата: {today} (UTC).\n"
        f"Тикер: {ticker} (MOEX).\n\n"
        f"Все новости и аналитика за последние 24 часа по этому тикеру:\n"
        f"{block}\n\n"
        f"Задача: определи направление консенсуса аналитиков и рынка на сегодня "
        f"для {ticker} на горизонте торговой сессии MOEX (10:00-18:50 МСК).\n\n"
        f"Ответь СТРОГИМ JSON следующего вида (без markdown, без префиксов, без комментариев):\n"
        f"{{\n"
        f'  "direction": "BUY" | "SELL" | "NEUTRAL",\n'
        f'  "strength": 0..1,\n'
        f'  "key_themes": ["тема1", "тема2", "..."],\n'
        f'  "rationale": "1-3 предложения по-русски с объяснением",\n'
        f'  "expected_to_positive": "какие новости подтвердят bullish направление",\n'
        f'  "expected_to_negative": "какие новости подтвердят bearish направление"\n'
        f"}}\n\n"
        f"Если новостей слишком мало или они нейтральны — верни direction NEUTRAL, strength 0.0."
    )

def _parse_consensus_response(
    ticker: str,
    response_parsed: dict[str, Any],
    n_news: int,
    backend: str,
) -> ConsensusEntry:
    """Parse consensus response."""
    if not isinstance(response_parsed, dict) or not response_parsed:
        return ConsensusEntry.neutral(ticker, n_news=n_news)
    direction = str(response_parsed.get("direction") or "NEUTRAL").upper()
    if direction not in ("BUY", "SELL", "NEUTRAL"):
        direction = "NEUTRAL"
    try:
        strength = float(response_parsed.get("strength") or 0.0)
    except (TypeError, ValueError):
        strength = 0.0
    strength = max(0.0, min(1.0, strength))
    themes = response_parsed.get("key_themes") or []
    if not isinstance(themes, list):
        themes = []
    themes_clean = [str(t)[:80] for t in themes][:8]
    rationale = str(response_parsed.get("rationale") or "")[:400]
    exp_pos = str(response_parsed.get("expected_to_positive") or "")[:300]
    exp_neg = str(response_parsed.get("expected_to_negative") or "")[:300]
    return ConsensusEntry(
        ticker=ticker.upper(),
        direction=direction,
        strength=strength,
        key_themes=themes_clean,
        rationale=rationale,
        expected_to_positive=exp_pos,
        expected_to_negative=exp_neg,
        built_at=datetime.now(tz=UTC),
        n_news=n_news,
        backend=backend,
    )

def consensus_path_for(date: datetime | None = None, persist_dir: Path | None = None) -> Path:
    """Return the canonical persistence path for a day's consensus."""
    date = date or datetime.now(tz=UTC)
    persist_dir = persist_dir or cfg.RAG_PERSIST_DIR
    return Path(persist_dir) / f"consensus_{date.strftime('%Y-%m-%d')}.json"

def save_consensus(
    consensus: dict[str, ConsensusEntry],
    persist_dir: Path | None = None,
) -> Path:
    """Save consensus."""
    persist_dir = Path(persist_dir or cfg.RAG_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)
    path = consensus_path_for(persist_dir=persist_dir)
    payload = {k: v.to_dict() for k, v in consensus.items()}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path

def load_consensus(
    persist_dir: Path | None = None,
    date: datetime | None = None,
) -> dict[str, ConsensusEntry]:
    """Load consensus."""
    persist_dir = Path(persist_dir or cfg.RAG_PERSIST_DIR)
    path = consensus_path_for(date=date, persist_dir=persist_dir)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "consensus_rag: failed to load consensus",
            extra={"path": str(path), "error": str(exc)},
        )
        return {}
    out: dict[str, ConsensusEntry] = {}
    for k, v in (raw or {}).items():
        try:
            out[str(k).upper()] = ConsensusEntry.from_dict(v)
        except Exception:
            continue
    return out

async def build_morning_consensus(
    rag: RAGStore,
    llm_backend: str | None = None,
    tickers: Iterable[str] | None = None,
    hours: int | None = None,
    min_news: int | None = None,
) -> dict[str, ConsensusEntry]:
    """Iterate every ticker, fetch its 24h news, ask the LLM for a direction.

    Returns a dict ``{ticker: ConsensusEntry}``. Persists the result to
    ``data/rag/consensus_YYYY-MM-DD.json`` so a restarted bot can resume.

    Parameters
    ----------
    rag : RAGStore
        Index containing all morning-collected news.
    llm_backend : "gemini" | "polza"
        Which LLM route to use. Defaults to ``cfg.RAG_LLM_BACKEND``.
    tickers : iterable
        Which tickers to build for. Defaults to ``cfg.TICKERS``.
    hours : int
        Look-back window. Defaults to 24.
    min_news : int
        Skip tickers with fewer than this many news items (returns
        NEUTRAL). Defaults to ``cfg.CONSENSUS_MIN_NEWS_PER_TICKER``.
    """
    backend = (llm_backend or cfg.RAG_LLM_BACKEND or "polza").lower()
    tickers = list(tickers or cfg.TICKERS)
    hours = hours if hours is not None else 24
    min_news = min_news if min_news is not None else cfg.CONSENSUS_MIN_NEWS_PER_TICKER
    consensus: dict[str, ConsensusEntry] = {}

    for ticker in tickers:
        news = rag.get_recent_for_ticker(ticker, hours=hours)
        if len(news) < min_news:
            consensus[ticker] = ConsensusEntry.neutral(ticker, n_news=len(news))
            logger.debug(
                "consensus_rag: ticker skipped (insufficient news)",
                extra={"ticker": ticker, "n_news": len(news), "min": min_news},
            )
            continue
        prompt = _build_consensus_prompt(ticker, news)
        try:
            response = await llm_chat_json(
                prompt=prompt,
                backend=backend,
                system=_DEFAULT_SYSTEM_PROMPT,
                model=(
                    cfg.GEMINI_MODEL_REASONING if backend == "gemini" else cfg.POLZA_MODEL_REASONING
                ),
                max_tokens=600,
                temperature=0.2,
                purpose=f"consensus_morning_{ticker}",
            )
        except Exception as exc:
            logger.warning(
                "consensus_rag: LLM call failed",
                extra={"ticker": ticker, "error": str(exc), "backend": backend},
            )
            consensus[ticker] = ConsensusEntry.neutral(ticker, n_news=len(news))
            continue
        parsed = response.get("parsed") or {}
        entry = _parse_consensus_response(
            ticker=ticker,
            response_parsed=parsed,
            n_news=len(news),
            backend=str(response.get("backend") or backend),
        )
        consensus[ticker] = entry
        logger.info(
            "consensus_rag: built",
            extra={
                "ticker": ticker,
                "direction": entry.direction,
                "strength": round(entry.strength, 2),
                "n_news": entry.n_news,
                "themes": entry.key_themes[:4],
                "backend": entry.backend,
                "cost_rub": round(float(response.get("cost_rub") or 0.0), 4),
            },
        )

    try:
        path = save_consensus(consensus)
        logger.info(
            "consensus_rag: persisted",
            extra={
                "path": str(path),
                "n_tickers": len(consensus),
                "n_non_neutral": sum(1 for e in consensus.values() if e.direction != "NEUTRAL"),
            },
        )
    except Exception as exc:
        logger.warning(
            "consensus_rag: persist failed (in-memory consensus still valid)",
            extra={"error": str(exc)},
        )
    return consensus

__all__ = [
    "ConsensusEntry",
    "build_morning_consensus",
    "save_consensus",
    "load_consensus",
    "consensus_path_for",
]
