"""Reactive consensus comparator.

Phase 27.4 (RAG Consensus). During the trading session a new news event
arrives at the :class:`NewsLLM` consumer; once standard event-classification
+ ticker tagging are done, ``ConsensusComparator.compare_event`` is called
to align the event against the morning consensus.

The comparator returns a :class:`ComparisonResult` per ticker:

  * ``matches_consensus``   → caller bumps magnitude by
                              ``cfg.CONSENSUS_MATCH_MAGNITUDE_BUMP``
                              (default 1.5×, clipped to 1.0).
  * ``contradicts``         → caller flips the direction (reverse trade).
                              The intuition: a news that contradicts the
                              prevailing consensus is a high-magnitude
                              surprise; we want to enter on the unexpected
                              side.
  * ``neutral``             → caller drops the signal (no edge).

Latency budget: < 5 s per event. We make ONE LLM call (top-k context +
consensus + new news) so the cost is similar to a regular reactive news
prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import app.config as cfg
from app.llm.gemini_client import llm_chat_json
from app.memory.rag_store import RAGStore
from app.memory.trade_outcome_rag import filter_trade_outcomes, parse_outcome_from_body
from app.news.consensus_rag import ConsensusEntry
from app.news.ingestion_bus import NormalizedNewsEvent
from app.utils.logging import get_logger

logger = get_logger(__name__)

_SANCTIONS_CLASS_TOKENS = (
    "sanction",
    "sanctions",
    "sanctions_relief",
    "санкц",
)
_SANCTIONS_TEXT_TOKENS = (
    "ofac",
    "sdn list",
    "sdn-list",
    "санкционный список",
    "санкции",
    "санкции против",
    "новые санкции",
    "eu fsf",
    "ec sanctions",
    "uk ofsi",
    "ofsi",
    "frozen assets",
    "asset freeze",
    "заморозка активов",
    "санкционный пакет",
    "sanctions package",
    "sanctions list",
    "sanction package",
    "imposes sanctions",
    "impose sanctions",
    "sanction against",
    "sanctions against",
    "sanctioned by",
    "designates ",
    "designation by ofac",
)
_EARNINGS_CLASS_TOKENS = (
    "earnings_beat",
    "earnings_miss",
    "earnings",
)
_EARNINGS_TEXT_TOKENS = (
    "earnings",
    "чистая прибыль",
    "чистый убыток",
    "выручка q",
    "прибыль превысила",
    "превзошёл прогноз",
    "хуже консенсуса",
    "лучше консенсуса",
    "miss consensus",
    "beat consensus",
    "beats estimates",
    "missed estimates",
    "revenue beat",
    "revenue miss",
    "eps beat",
    "eps miss",
)

def _strength_floor() -> float:
    return float(getattr(cfg, "CONSENSUS_DROP_STRENGTH_FLOOR", 0.5))

def _earnings_min_magnitude() -> float:
    return float(getattr(cfg, "CONSENSUS_EARNINGS_MIN_MAGNITUDE", 0.4))

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower())

def is_sanctions_signal(
    *,
    classification: str = "",
    headline: str = "",
    body: str = "",
    matched_keywords: list[str] | None = None,
    source: str = "",
) -> bool:
    """Return True when the event looks like a sanctions story.

    Used to force-through the consensus filter — sanctions news is the
    single highest-edge category and we never want to drop it on a
    ``consensus is NEUTRAL`` technicality.
    """
    cls = (classification or "").strip().lower()
    if any(tok in cls for tok in _SANCTIONS_CLASS_TOKENS):
        return True
    src = (source or "").lower()
    if src.startswith(("ofac", "eu_fsf", "uk_ofsi")):
        return True
    for kw in matched_keywords or []:
        k = str(kw).lower()
        if "ofac" in k or "санкци" in k or "sdn" in k:
            return True
    text = _norm_text(f"{headline} {body}")
    return any(tok in text for tok in _SANCTIONS_TEXT_TOKENS)

def is_earnings_signal(
    *,
    classification: str = "",
    headline: str = "",
    body: str = "",
) -> bool:
    """Return True when the event looks like an earnings (beat/miss) story."""
    cls = (classification or "").strip().lower()
    if any(tok in cls for tok in _EARNINGS_CLASS_TOKENS):
        return True
    text = _norm_text(f"{headline} {body}")
    return any(tok in text for tok in _EARNINGS_TEXT_TOKENS)

_DEFAULT_SYSTEM_PROMPT = (
    "You are an intraday news-reaction analyst for MOEX equities. The "
    "user shows you the morning consensus, a handful of similar recent "
    "events, and a fresh news headline + body. Your job: classify whether "
    "the new news *matches*, *contradicts*, or is *neutral* to the "
    "consensus, then state the trading direction (BUY/SELL/NEUTRAL) and "
    "magnitude (0..1). Output STRICT JSON only."
)

@dataclass
class ComparisonResult:
    """Comparison Result."""

    ticker: str
    alignment: str = "neutral"
    direction: str = "NEUTRAL"
    magnitude: float = 0.0
    rationale: str = ""
    consensus_direction: str = "NEUTRAL"
    consensus_strength: float = 0.0
    n_similar: int = 0
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "ticker": self.ticker,
            "alignment": self.alignment,
            "direction": self.direction,
            "magnitude": self.magnitude,
            "rationale": self.rationale,
            "consensus_direction": self.consensus_direction,
            "consensus_strength": self.consensus_strength,
            "n_similar": self.n_similar,
            "backend": self.backend,
        }

def _format_similar_block(similar: list[dict[str, Any]]) -> str:
    """Format similar block."""
    if not similar:
        return "  (нет похожих новостей в окне 24ч)"
    lines: list[str] = []
    for i, n in enumerate(similar[:3], 1):
        head = (n.get("headline") or "")[:160].replace("\n", " ")
        ts = (n.get("ts_utc") or "")[:19]
        score = n.get("score")
        if isinstance(score, (float, int)):
            lines.append(f"  {i}. [{ts}] sim={float(score):.2f} — {head}")
        else:
            lines.append(f"  {i}. [{ts}] {head}")
    return "\n".join(lines)

def _build_compare_prompt(
    *,
    ticker: str,
    cons: ConsensusEntry,
    event: NormalizedNewsEvent,
    similar: list[dict[str, Any]],
) -> str:
    """Build compare prompt."""
    headline = (event.headline or "")[:300]
    body = (event.body or "")[:1500]
    similar_block = _format_similar_block(similar)
    cons_themes = ", ".join(cons.key_themes[:6]) or "(нет)"
    return (
        f"Тикер: {ticker}\n"
        f"\nКонсенсус на сегодня ({cons.built_at.strftime('%Y-%m-%d %H:%M UTC')}):\n"
        f"  direction = {cons.direction}, strength = {cons.strength:.2f}\n"
        f"  key_themes = {cons_themes}\n"
        f"  rationale = {cons.rationale[:300]}\n"
        f"  expected_to_positive = {cons.expected_to_positive[:200]}\n"
        f"  expected_to_negative = {cons.expected_to_negative[:200]}\n"
        f"\nПохожие новости за последние 24ч (top-3 из RAG):\n{similar_block}\n"
        f"\nНОВАЯ НОВОСТЬ:\n  headline: {headline}\n"
        f"  body: {body}\n"
        f"\nЗадача: определи интерпретацию.\n"
        f"Ответь СТРОГИМ JSON следующего вида:\n"
        f"{{\n"
        f'  "alignment": "matches_consensus" | "contradicts" | "neutral",\n'
        f'  "direction": "BUY" | "SELL" | "NEUTRAL",\n'
        f'  "magnitude": 0..1,\n'
        f'  "rationale": "1-2 предложения по-русски"\n'
        f"}}\n\n"
        f"Правила:\n"
        f"  - matches_consensus = новость подтверждает направление консенсуса.\n"
        f"  - contradicts = новость неожиданна и идёт ПРОТИВ консенсуса (запускаем reverse).\n"
        f"  - neutral = новость не несёт edge (либо общая, либо повторяет что-то старое).\n"
    )

def _parse_compare_response(
    ticker: str,
    parsed: dict[str, Any],
    cons: ConsensusEntry,
    n_similar: int,
    backend: str,
) -> ComparisonResult:
    """Parse compare response."""
    if not isinstance(parsed, dict) or not parsed:
        return ComparisonResult(
            ticker=ticker,
            alignment="neutral",
            direction="NEUTRAL",
            magnitude=0.0,
            rationale="LLM parse failed",
            consensus_direction=cons.direction,
            consensus_strength=cons.strength,
            n_similar=n_similar,
            backend=backend,
        )
    alignment = str(parsed.get("alignment") or "neutral").lower()
    if alignment not in ("matches_consensus", "contradicts", "neutral"):
        alignment = "neutral"
    direction = str(parsed.get("direction") or "NEUTRAL").upper()
    if direction not in ("BUY", "SELL", "NEUTRAL"):
        direction = "NEUTRAL"
    try:
        magnitude = float(parsed.get("magnitude") or 0.0)
    except (TypeError, ValueError):
        magnitude = 0.0
    magnitude = max(0.0, min(1.0, magnitude))
    rationale = str(parsed.get("rationale") or "")[:400]
    if alignment == "contradicts" and direction == cons.direction and direction != "NEUTRAL":
        alignment = "neutral"
    if alignment == "contradicts" and direction == "NEUTRAL":
        direction = (
            "SELL"
            if cons.direction == "BUY"
            else ("BUY" if cons.direction == "SELL" else "NEUTRAL")
        )
    return ComparisonResult(
        ticker=ticker,
        alignment=alignment,
        direction=direction,
        magnitude=magnitude,
        rationale=rationale,
        consensus_direction=cons.direction,
        consensus_strength=cons.strength,
        n_similar=n_similar,
        backend=backend,
    )

class ConsensusComparator:
    """Compares a new news event against the daily consensus.

    Owns the per-day ``consensus_today`` map and can be hot-swapped via
    :meth:`update_consensus` when the morning scheduler runs.
    """

    def __init__(
        self,
        rag: RAGStore,
        consensus_today: dict[str, ConsensusEntry] | None = None,
        llm_backend: str | None = None,
    ) -> None:
        """Init."""
        self.rag = rag
        self.consensus_today: dict[str, ConsensusEntry] = {
            k.upper(): v for k, v in (consensus_today or {}).items()
        }
        self.llm_backend = (llm_backend or cfg.RAG_LLM_BACKEND or "polza").lower()
        self._compares_done = 0
        self._matches = 0
        self._contradicts = 0
        self._neutrals = 0

    def update_consensus(self, consensus_today: dict[str, ConsensusEntry]) -> None:
        """Update consensus."""
        self.consensus_today = {k.upper(): v for k, v in (consensus_today or {}).items()}
        logger.info(
            "ConsensusComparator: consensus updated",
            extra={"n_tickers": len(self.consensus_today)},
        )

    async def compare_event(self, event: NormalizedNewsEvent) -> dict[str, ComparisonResult]:
        """Return per-ticker comparison results for the given event."""
        tickers = list(event.tickers or [])
        if not tickers:
            return {}
        results: dict[str, ComparisonResult] = {}
        for ticker in tickers:
            cons = self.consensus_today.get(ticker.upper())
            if cons is None or cons.direction == "NEUTRAL":
                results[ticker] = ComparisonResult(
                    ticker=ticker,
                    alignment="neutral",
                    direction="NEUTRAL",
                    magnitude=0.0,
                    rationale="no consensus for ticker",
                    consensus_direction=cons.direction if cons else "NEUTRAL",
                    consensus_strength=cons.strength if cons else 0.0,
                    n_similar=0,
                    backend=self.llm_backend,
                )
                self._neutrals += 1
                continue
            query = (event.headline or "") + " " + (event.body or "")
            try:
                similar = self.rag.search(
                    query_text=query,
                    tickers=[ticker],
                    top_k=cfg.RAG_TOP_K,
                    max_age_hours=24,
                )
            except Exception as exc:
                logger.debug(
                    "ConsensusComparator: rag.search failed",
                    extra={"ticker": ticker, "error": str(exc)},
                )
                similar = []
            prompt = _build_compare_prompt(
                ticker=ticker,
                cons=cons,
                event=event,
                similar=similar,
            )
            try:
                response = await llm_chat_json(
                    prompt=prompt,
                    backend=self.llm_backend,
                    system=_DEFAULT_SYSTEM_PROMPT,
                    model=(
                        cfg.GEMINI_MODEL_REACTIVE
                        if self.llm_backend == "gemini"
                        else cfg.POLZA_MODEL_REACTIVE
                    ),
                    max_tokens=400,
                    temperature=0.15,
                    purpose=f"consensus_compare_{ticker}",
                )
            except Exception as exc:
                logger.warning(
                    "ConsensusComparator: LLM call failed",
                    extra={"ticker": ticker, "error": str(exc)},
                )
                results[ticker] = ComparisonResult(
                    ticker=ticker,
                    alignment="neutral",
                    direction="NEUTRAL",
                    magnitude=0.0,
                    rationale=f"LLM error: {exc!s}",
                    consensus_direction=cons.direction,
                    consensus_strength=cons.strength,
                    n_similar=len(similar),
                    backend=self.llm_backend,
                )
                self._neutrals += 1
                continue
            parsed = response.get("parsed") or {}
            result = _parse_compare_response(
                ticker=ticker,
                parsed=parsed,
                cons=cons,
                n_similar=len(similar),
                backend=str(response.get("backend") or self.llm_backend),
            )
            results[ticker] = result
            self._compares_done += 1
            if result.alignment == "matches_consensus":
                self._matches += 1
            elif result.alignment == "contradicts":
                self._contradicts += 1
            else:
                self._neutrals += 1
            logger.info(
                "ConsensusComparator: result",
                extra={
                    "ticker": ticker,
                    "alignment": result.alignment,
                    "direction": result.direction,
                    "magnitude": round(result.magnitude, 2),
                    "consensus_direction": cons.direction,
                    "consensus_strength": round(cons.strength, 2),
                    "n_similar": len(similar),
                    "headline": (event.headline or "")[:120],
                    "backend": result.backend,
                },
            )
        return results

    def aggregate_signal(
        self,
        comparisons: dict[str, ComparisonResult],
        base_direction: str,
        base_magnitude: float,
    ) -> tuple[str, float, dict[str, ComparisonResult]]:
        """Aggregate comparisons into a final (direction, magnitude, dropped_set).

        Strategy: take the strongest non-neutral comparison; bump magnitude
        if it matches consensus, flip direction if it contradicts. If all
        are neutral, return (base_direction, 0.0, {}) → the caller drops
        the signal entirely.
        """
        if not comparisons:
            return base_direction, base_magnitude, {}
        non_neutral = {t: r for t, r in comparisons.items() if r.alignment != "neutral"}
        if not non_neutral:
            return base_direction, 0.0, {}
        top_ticker, top = max(non_neutral.items(), key=lambda kv: kv[1].magnitude)
        new_direction = top.direction
        new_magnitude = max(base_magnitude, top.magnitude)
        if top.alignment == "matches_consensus":
            new_magnitude *= cfg.CONSENSUS_MATCH_MAGNITUDE_BUMP
        elif top.alignment == "contradicts" and not cfg.CONSENSUS_CONTRADICT_REVERSE:
            new_direction = base_direction
        new_magnitude = max(0.0, min(1.0, new_magnitude))
        logger.debug(
            "ConsensusComparator: aggregated",
            extra={
                "top_ticker": top_ticker,
                "alignment": top.alignment,
                "base_direction": base_direction,
                "base_magnitude": round(base_magnitude, 2),
                "new_direction": new_direction,
                "new_magnitude": round(new_magnitude, 2),
            },
        )
        return new_direction, new_magnitude, comparisons

    def apply_consensus_to_signal(
        self,
        *,
        comparisons: dict[str, ComparisonResult],
        base_direction: str,
        base_magnitude: float,
        classification: str = "",
        headline: str = "",
        body: str = "",
        matched_keywords: list[str] | None = None,
        source: str = "",
        strength_floor: float | None = None,
        earnings_min_magnitude: float | None = None,
    ) -> tuple[str, float, bool, dict[str, Any]]:
        """Tuned consensus aggregation (v12).

        Replaces the legacy ``aggregate_signal``-then-drop pattern used by
        :class:`NewsLLM`. Returns
        ``(new_direction, new_magnitude, should_drop, debug)``.

        Rules
        -----
        1. **Sanctions force-through** — if :func:`is_sanctions_signal`
           matches, never drop; keep base direction/magnitude.
        2. **Earnings floor** — if :func:`is_earnings_signal` matches,
           guarantee ``magnitude >= earnings_min_magnitude`` even when
           consensus says neutral.
        3. **Default drop** — drop only when *all* per-ticker comparisons
           are neutral **AND** every consulted consensus has
           ``strength >= strength_floor``. A weak consensus (< floor) is
           not a strong enough signal to veto the LLM's verdict.
        4. **Non-neutral path** — at least one ``matches_consensus`` or
           ``contradicts`` → behave like the legacy
           :meth:`aggregate_signal`.
        """
        floor = float(strength_floor) if strength_floor is not None else _strength_floor()
        earnings_floor = (
            float(earnings_min_magnitude)
            if earnings_min_magnitude is not None
            else _earnings_min_magnitude()
        )
        debug: dict[str, Any] = {
            "rule": "default",
            "n_comparisons": len(comparisons),
            "n_non_neutral": 0,
            "max_consensus_strength": 0.0,
        }

        if is_sanctions_signal(
            classification=classification,
            headline=headline,
            body=body,
            matched_keywords=matched_keywords,
            source=source,
        ):
            debug["rule"] = "sanctions_force_through"
            debug["forced"] = True
            return base_direction, base_magnitude, False, debug

        if not comparisons:
            debug["rule"] = "no_comparator_results"
            return base_direction, base_magnitude, False, debug

        non_neutral = [r for r in comparisons.values() if r.alignment != "neutral"]
        debug["n_non_neutral"] = len(non_neutral)
        max_strength = max(
            (float(r.consensus_strength or 0.0) for r in comparisons.values()),
            default=0.0,
        )
        debug["max_consensus_strength"] = round(max_strength, 3)

        if is_earnings_signal(classification=classification, headline=headline, body=body):
            new_dir = base_direction
            new_mag = base_magnitude
            if non_neutral:
                top = max(non_neutral, key=lambda r: r.magnitude)
                new_dir = top.direction or base_direction
                new_mag = max(base_magnitude, top.magnitude)
                if top.alignment == "matches_consensus":
                    new_mag = new_mag * float(cfg.CONSENSUS_MATCH_MAGNITUDE_BUMP)
            new_mag = max(new_mag, earnings_floor)
            new_mag = max(0.0, min(1.0, new_mag))
            debug["rule"] = "earnings_floor"
            debug["earnings_floor_applied"] = earnings_floor
            return new_dir, new_mag, False, debug

        if not non_neutral:
            if max_strength >= floor:
                debug["rule"] = "default_drop_strong_consensus_neutral"
                return base_direction, 0.0, True, debug
            attenuated = max(0.10, base_magnitude * 0.85)
            debug["rule"] = "default_passthrough_weak_consensus"
            debug["attenuation"] = round(attenuated / max(base_magnitude, 1e-6), 3)
            return base_direction, attenuated, False, debug

        new_dir, new_mag, _ = self.aggregate_signal(
            comparisons=comparisons,
            base_direction=base_direction,
            base_magnitude=base_magnitude,
        )
        debug["rule"] = "aggregate_signal"
        return new_dir, new_mag, False, debug

    @property
    def stats(self) -> dict[str, int]:
        """Stats."""
        return {
            "compares_done": self._compares_done,
            "matches": self._matches,
            "contradicts": self._contradicts,
            "neutrals": self._neutrals,
            "consensus_tickers": len(self.consensus_today),
        }

    async def get_similar_past_trades(
        self,
        ticker: str,
        current_setup: dict[str, Any],
        k: int = 5,
        max_age_days: int = 30,
    ) -> list[dict[str, Any]]:
        """Find post-mortems of similar setups in the RAG store.

        ``current_setup`` is a dict describing the proposed trade — at a
        minimum it should carry ``direction``, ``hmm_regime``, ``detector``
        and ``tier``. The keys mirror the natural-language template used
        by :func:`app.memory.trade_outcome_rag._format_trade_text`, which
        makes the cosine similarity search hit the right records.

        Returns the (already filtered) trade-outcome records, each with a
        ``parsed_outcome`` field populated from the ``body`` payload.

        Never raises — returns ``[]`` on any internal failure.
        """
        if not ticker:
            return []
        direction = str(current_setup.get("direction") or "").upper() or "?"
        regime = str(current_setup.get("hmm_regime") or "unknown")
        detector = str(current_setup.get("detector") or "n/a")
        tier = str(current_setup.get("tier") or "?")
        query = (
            f"Ticker {ticker.upper()} {direction} в {regime}-режиме, "
            f"detector {detector}, tier {tier}"
        )
        try:
            raw = self.rag.search(
                query_text=query,
                tickers=[ticker.upper()],
                top_k=max(1, k * 2),
                max_age_hours=int(max_age_days) * 24,
            )
        except Exception as exc:
            logger.debug(
                "ConsensusComparator: rag.search for historical trades failed",
                extra={"ticker": ticker, "error": str(exc)},
            )
            return []
        outcomes = filter_trade_outcomes(raw)
        enriched: list[dict[str, Any]] = []
        for rec in outcomes[:k]:
            parsed = parse_outcome_from_body(str(rec.get("body") or ""))
            rec_out = dict(rec)
            rec_out["parsed_outcome"] = parsed
            if "outcome" in parsed:
                rec_out["outcome"] = parsed["outcome"]
            if "pnl_pct" in parsed:
                rec_out["pnl_pct"] = parsed["pnl_pct"]
            enriched.append(rec_out)
        return enriched

    @staticmethod
    def historical_edge_multiplier(
        similar_trades: list[dict[str, Any]],
        *,
        win_bonus: float = 1.3,
        loss_penalty: float = 0.7,
        win_threshold: float = 0.7,
        loss_threshold: float = 0.3,
        min_samples: int = 3,
    ) -> float:
        """Compute a magnitude multiplier from historical win-rate.

        Pure function (no I/O) so it's trivial to unit-test:

        * If we have fewer than ``min_samples`` historical hits, return
          ``1.0`` (no signal).
        * If historical WR ≥ ``win_threshold`` → return ``win_bonus``.
        * If historical WR ≤ ``loss_threshold`` → return ``loss_penalty``.
        * Otherwise the edge is mixed — return ``1.0``.

        ``similar_trades`` records are expected to carry an ``outcome``
        field (``"win"``/``"loss"``/``"flat"``) at the top level, as
        produced by :meth:`get_similar_past_trades`.
        """
        if not similar_trades:
            return 1.0
        wins = sum(1 for s in similar_trades if str(s.get("outcome")) == "win")
        losses = sum(1 for s in similar_trades if str(s.get("outcome")) == "loss")
        n_decisive = wins + losses
        if n_decisive < max(1, min_samples):
            return 1.0
        wr = wins / n_decisive
        if wr >= win_threshold:
            return float(win_bonus)
        if wr <= loss_threshold:
            return float(loss_penalty)
        return 1.0

    async def apply_historical_edge(
        self,
        ticker: str,
        current_setup: dict[str, Any],
        magnitude: float,
        k: int = 5,
    ) -> tuple[float, dict[str, Any]]:
        """Convenience: fetch similar trades + scale magnitude in one call.

        Returns ``(new_magnitude, debug)`` where ``debug`` carries the
        win-rate, sample count and applied multiplier for logging.
        """
        try:
            base = float(magnitude)
        except (TypeError, ValueError):
            base = 0.0
        similar = await self.get_similar_past_trades(
            ticker=ticker,
            current_setup=current_setup,
            k=k,
        )
        mult = self.historical_edge_multiplier(similar)
        new_mag = max(0.0, min(1.0, base * mult))
        wins = sum(1 for s in similar if str(s.get("outcome")) == "win")
        losses = sum(1 for s in similar if str(s.get("outcome")) == "loss")
        n = wins + losses
        wr = (wins / n) if n else 0.0
        debug = {
            "n_similar": len(similar),
            "n_decisive": n,
            "wins": wins,
            "losses": losses,
            "wr": wr,
            "multiplier": mult,
            "base_magnitude": base,
            "new_magnitude": new_mag,
        }
        if mult != 1.0:
            logger.info(
                "ConsensusComparator: historical edge applied",
                extra={"ticker": ticker, **debug},
            )
        return new_mag, debug

__all__ = [
    "ConsensusComparator",
    "ComparisonResult",
    "is_sanctions_signal",
    "is_earnings_signal",
]
