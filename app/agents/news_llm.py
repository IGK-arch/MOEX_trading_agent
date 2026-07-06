"""News-driven сигналы через polza.ai."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app.config as cfg
from app.agents.base import BaseAdapter
from app.agents.consensus_compare import is_sanctions_signal as _is_sanctions_signal
from app.agents.hmm_regime import get_hmm_detector
from app.agents.ta_indicators import compute_all
from app.agents.ta_trader import get_ta_trader
from app.data.candle_store import get_candle_store
from app.dispatcher.signal import Direction, SignalSource, UnifiedSignal
from app.llm.polza_client import get_polza_client
from app.news.dedup import get_deduplicator
from app.news.history import REACTION_WINDOWS_MIN, get_history_store
from app.news.ingestion_bus import IngestionBus, NormalizedNewsEvent, get_bus
from app.news.local_sentiment import LocalSentimentScorer, get_local_sentiment_scorer
from app.news.material_filter import is_material
from app.news.ticker_tagger import get_ticker_tagger
from app.utils.logging import get_logger, get_trace_id

logger = get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "news" / "prompts"

try:
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

class NewsLLM(BaseAdapter):
    """News-driven signal generator backed by polza.ai LLMs."""

    name = "NEWS"

    def __init__(
        self,
        bus: IngestionBus | None = None,
        max_signals_per_poll: int = 5,
        consume_timeout: float = 0.05,
    ) -> None:
        """Init."""
        super().__init__()
        self.bus = bus or get_bus()
        self.max_signals_per_poll = max_signals_per_poll
        self.consume_timeout = consume_timeout
        self.polza = get_polza_client()
        self.dedup = get_deduplicator()
        self.tagger = get_ticker_tagger()
        self.history = get_history_store()
        self.candle_store = get_candle_store()
        self.hmm = get_hmm_detector()
        self.ta = get_ta_trader()
        self.local_sentiment: LocalSentimentScorer = get_local_sentiment_scorer()

        self._signal_buffer: deque[UnifiedSignal] = deque(maxlen=200)
        self._consumer_task: asyncio.Task | None = None
        self._priority_consumer_task: asyncio.Task | None = None
        self._dispatcher_trigger: Any = None
        self._reaction_tasks: set[asyncio.Task] = set()

        self._prompt_reactive: str = ""
        self._prompt_sanctions: str = ""
        self._prompt_reactive_v2: str = ""
        self._prompt_sanctions_v2: str = ""
        self._prompt_reactive_v3: str = ""

        self._poll_count = 0
        self._signal_count = 0
        self._events_consumed = 0
        self._llm_calls = 0
        self._drops_no_text = 0
        self._drops_dedup = 0
        self._drops_not_material = 0
        self._drops_disable_llm = 0
        self._drops_llm_neutral = 0
        self._drops_low_magnitude = 0
        self._drops_no_affected = 0
        self._fallbacks_keyword_signal = 0

        self.comparator: Any = None
        self._consensus_matches = 0
        self._consensus_contradicts = 0
        self._consensus_neutral_drops = 0

    async def startup(self) -> None:
        """Startup."""
        if cfg.DISABLE_LLM and not cfg.NEWS_LOCAL_SENTIMENT_ENABLED:
            logger.warning(
                "NewsLLM: DISABLE_LLM=1 and local sentiment disabled — "
                "news pipeline inert (no consumers, no LLM calls)"
            )
            self._started = True
            return
        if cfg.DISABLE_LLM:
            logger.info("NewsLLM: DISABLE_LLM=1 — using local sentiment fallback (POLZA bypass)")
        else:
            if not self.polza._started:
                await self.polza.startup()

        try:
            with open(PROMPTS_DIR / "reactive_v1.txt", encoding="utf-8") as f:
                self._prompt_reactive = f.read()
            with open(PROMPTS_DIR / "sanctions_v1.txt", encoding="utf-8") as f:
                self._prompt_sanctions = f.read()
        except Exception as exc:
            logger.error("NewsLLM: v1 prompt load failed", extra={"error": str(exc)})
        try:
            v2_react = PROMPTS_DIR / "reactive_v2_dkcot.txt"
            if v2_react.exists():
                self._prompt_reactive_v2 = v2_react.read_text(encoding="utf-8")
            v2_sanc = PROMPTS_DIR / "sanctions_v2_dkcot.txt"
            if v2_sanc.exists():
                self._prompt_sanctions_v2 = v2_sanc.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("NewsLLM: v2_dkcot prompts not loaded", extra={"error": str(exc)})
        try:
            v3_react = PROMPTS_DIR / "reactive_v3.txt"
            if v3_react.exists():
                self._prompt_reactive_v3 = v3_react.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("NewsLLM: v3 prompt not loaded", extra={"error": str(exc)})

        self._consumer_task = asyncio.create_task(self._consume_loop(), name="news_consumer")
        if cfg.SANCTIONS_PUSH_MODE:
            self._priority_consumer_task = asyncio.create_task(
                self._priority_consume_loop(), name="news_priority_consumer"
            )
        self._started = True
        logger.info(
            "NewsLLM started",
            extra={
                "prompts_loaded": bool(self._prompt_reactive),
                "v2_dkcot_loaded": bool(self._prompt_reactive_v2),
                "v3_loaded": bool(self._prompt_reactive_v3),
                "prompt_version": cfg.NEWS_PROMPT_VERSION,
                "sanctions_push_mode": cfg.SANCTIONS_PUSH_MODE,
                "polza_started": self.polza._started,
            },
        )

    async def shutdown(self) -> None:
        """Shutdown."""
        for task in (self._consumer_task, self._priority_consumer_task):
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        if self._reaction_tasks:
            for task in list(self._reaction_tasks):
                task.cancel()
            await asyncio.gather(*self._reaction_tasks, return_exceptions=True)
        self._started = False
        logger.info(
            "NewsLLM stopped",
            extra={
                "events_consumed": self._events_consumed,
                "llm_calls": self._llm_calls,
                "signals_generated": self._signal_count,
                "drops_no_text": self._drops_no_text,
                "drops_dedup": self._drops_dedup,
                "drops_not_material": self._drops_not_material,
                "drops_disable_llm": self._drops_disable_llm,
                "drops_llm_neutral": self._drops_llm_neutral,
                "drops_low_magnitude": self._drops_low_magnitude,
                "drops_no_affected": self._drops_no_affected,
                "fallbacks_keyword_signal": self._fallbacks_keyword_signal,
                "consensus_matches": self._consensus_matches,
                "consensus_contradicts": self._consensus_contradicts,
                "consensus_neutral_drops": self._consensus_neutral_drops,
            },
        )

    def set_dispatcher_trigger(self, trigger: Any) -> None:
        """Wire an asyncio.Event-like trigger; priority loop sets it on high-impact events."""
        self._dispatcher_trigger = trigger

    def attach_comparator(self, comparator: Any) -> None:
        """Wire the Phase 27.4 :class:`ConsensusComparator` instance.

        Once attached, ``_process_event`` calls
        ``comparator.compare_event(event)`` after the LLM has produced a
        base direction/magnitude. Matching news boost the magnitude by
        ``cfg.CONSENSUS_MATCH_MAGNITUDE_BUMP``, contradicting news may
        reverse direction, neutral news are dropped entirely (we want
        only news with edge to survive to the dispatcher).
        """
        self.comparator = comparator
        logger.info(
            "NewsLLM: consensus comparator attached",
            extra={"backend": getattr(comparator, "llm_backend", "unknown")},
        )

    async def poll(self) -> list[UnifiedSignal]:
        """Return buffered signals; LLM work happens in the background consumer."""
        signals: list[UnifiedSignal] = []
        while self._signal_buffer and len(signals) < self.max_signals_per_poll:
            signals.append(self._signal_buffer.popleft())
        self._poll_count += 1
        return signals

    async def _consume_loop(self) -> None:
        """Consume loop.

        Cycle-5: when :data:`cfg.NEWS_LLM_BATCH_ENABLED` is True the loop drains
        the bus and groups same-feed events into windows of up to
        :data:`cfg.NEWS_LLM_BATCH_MAX_SIZE` for parallel ``_process_event``
        dispatch via ``asyncio.gather``. This trims the wall-clock latency
        proportional to the batch size and improves prompt-cache amortisation
        (shared context-block prefix). Sanctions and priority push events are
        NEVER batched — they flow through ``_priority_consume_loop``.

        When the flag is False (default) the loop's behaviour is identical to
        the pre-cycle-5 implementation: one event ⇒ one ``_process_event``.

        NOTE: this does NOT yet merge multiple events into a single LLM prompt
        with an array of results — that merge requires a new prompt template +
        response parser and is left as a TODO for cycle-6 (see
        ``data/training_cache/research_cycle5.md`` insight #7 + #8 for rationale).
        """
        while True:
            event = await self.bus.consume(timeout=1.0)
            if event is None:
                continue

            if cfg.NEWS_LLM_BATCH_ENABLED and not (
                event.source_tier == "S"
                or event.source.startswith(("ofac", "eu_fsf", "uk_ofsi"))
            ):
                batch = await self._collect_batch(seed_event=event)
                try:
                    results = await asyncio.gather(
                        *(self._process_event(e) for e in batch),
                        return_exceptions=True,
                    )
                except asyncio.CancelledError:
                    raise
                for ev, res in zip(batch, results, strict=False):
                    if isinstance(res, BaseException):
                        logger.error(
                            "NewsLLM: batched event processing failed",
                            extra={"event_source": ev.source, "error": str(res)},
                        )
                        continue
                    for signal in res:
                        self._signal_buffer.append(signal)
                        self._signal_count += 1
                    self._events_consumed += 1
                continue

            try:
                signals = await self._process_event(event)
                for signal in signals:
                    self._signal_buffer.append(signal)
                    self._signal_count += 1
                self._events_consumed += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "NewsLLM: event processing failed",
                    extra={"event_source": event.source, "error": str(exc)},
                )

    async def _collect_batch(
        self, seed_event: NormalizedNewsEvent
    ) -> list[NormalizedNewsEvent]:
        """Cycle-5: pull up to ``NEWS_LLM_BATCH_MAX_SIZE - 1`` additional events
        from the SAME RSS feed source as ``seed_event`` over a window of up to
        ``NEWS_LLM_BATCH_WAIT_MS`` milliseconds.

        Stops early once the buffer hits the size cap. Priority/sanctions events
        are filtered out — they should never block in this loop's batch.
        Returns the batch with ``seed_event`` at index 0.
        """
        batch: list[NormalizedNewsEvent] = [seed_event]
        deadline = asyncio.get_running_loop().time() + (cfg.NEWS_LLM_BATCH_WAIT_MS / 1000.0)
        max_extra = max(0, cfg.NEWS_LLM_BATCH_MAX_SIZE - 1)

        while len(batch) <= max_extra:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            timeout = min(1.0, remaining)
            next_event = await self.bus.consume(timeout=timeout)
            if next_event is None:
                continue
            if next_event.source_tier == "S" or next_event.source.startswith(
                ("ofac", "eu_fsf", "uk_ofsi")
            ):
                try:
                    await self.bus.publish(next_event)
                except Exception:
                    pass
                continue
            if next_event.source != seed_event.source:
                try:
                    await self.bus.publish(next_event)
                except Exception:
                    pass
                break
            batch.append(next_event)

        return batch

    async def _priority_consume_loop(self) -> None:
        """Fast-path consumer for sanctions & MOEX halts."""
        while True:
            event = await self.bus.consume_priority(timeout=1.0)
            if event is None:
                continue
            try:
                signals = await self._process_event(event)
                for s in signals:
                    s.metadata.setdefault("priority", True)
                    self._signal_buffer.appendleft(s)
                    self._signal_count += 1
                self._events_consumed += 1
                if signals and self._dispatcher_trigger is not None:
                    with contextlib.suppress(Exception):
                        self._dispatcher_trigger.set()
                if signals:
                    logger.info(
                        "Priority signal pushed",
                        extra={
                            "source": event.source,
                            "tier": event.source_tier,
                            "n_signals": len(signals),
                        },
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "NewsLLM priority processing failed",
                    extra={"event_source": event.source, "error": str(exc)},
                )

    async def _process_event(self, event: NormalizedNewsEvent) -> list[UnifiedSignal]:
        """Process event."""

        if cfg.DISABLE_LLM and not cfg.NEWS_LOCAL_SENTIMENT_ENABLED:
            self._drops_disable_llm += 1
            return []
        full_text = f"{event.headline} {event.body}".strip()
        if not full_text:
            self._drops_no_text += 1
            return []

        if self.dedup.is_duplicate(event.event_id, full_text):
            self._drops_dedup += 1
            return []

        tickers = self.tagger.tag(full_text)
        event.tickers = tickers

        is_sanctions = event.source_tier == "S" and event.source.startswith(
            ("ofac", "eu_fsf", "uk_ofsi")
        )
        is_mat, matched_kw, is_always = is_material(full_text, has_ticker=bool(tickers))
        if not (is_mat or is_sanctions):
            self._drops_not_material += 1
            return []

        if not tickers:
            tickers = ["SBER", "GAZP", "LKOH", "ROSN"] if is_sanctions else ["SBER", "GAZP", "LKOH"]

        if cfg.DISABLE_LLM:
            return await self._process_event_local(
                event=event,
                full_text=full_text,
                tickers=tickers,
                is_sanctions=is_sanctions,
                is_mat=is_mat,
                matched_kw=matched_kw,
                is_always=is_always,
            )

        event_type = await self.history.record_event(
            event=event,
            tickers=tickers,
            matched_keywords=matched_kw,
            is_material=is_mat or is_sanctions,
            is_sanctions=is_sanctions,
        )

        ticker_contexts: dict[str, dict[str, Any]] = {}
        for ticker in tickers:
            ticker_contexts[ticker] = await self._build_ticker_context(
                ticker=ticker,
                event=event,
                matched_keywords=matched_kw,
                event_type=event_type,
                is_sanctions=is_sanctions,
            )

        use_v2 = cfg.NEWS_PROMPT_VERSION == "v2_dkcot"
        use_v3 = cfg.NEWS_PROMPT_VERSION == "v3"
        if is_sanctions:
            tpl = (
                self._prompt_sanctions_v2
                if (use_v2 and self._prompt_sanctions_v2)
                else self._prompt_sanctions
            )
            prompt = self._render(
                tpl,
                {
                    "headline": event.headline,
                    "body": event.body[:2000],
                    "jurisdiction": event.raw_payload.get("jurisdiction", "?"),
                    "ts": event.ts_utc.isoformat(),
                    "source": event.source,
                    "tickers_whitelist": ", ".join(cfg.TICKERS),
                },
            )
            model = cfg.POLZA_MODEL_SANCTIONS
        else:
            if use_v3 and self._prompt_reactive_v3:
                tpl = self._prompt_reactive_v3
            elif use_v2 and self._prompt_reactive_v2:
                tpl = self._prompt_reactive_v2
            else:
                tpl = self._prompt_reactive
            prompt = self._render(
                tpl,
                {
                    "headline": event.headline,
                    "body": event.body[:2000],
                    "tickers": ", ".join(tickers),
                    "tickers_whitelist": ", ".join(cfg.TICKERS),
                    "source": event.source,
                    "source_tier": event.source_tier,
                    "ts": event.ts_utc.isoformat(),
                },
            )
            model = cfg.POLZA_MODEL_REACTIVE

        prompt = prompt + "\n\n" + self._build_context_block(ticker_contexts)

        cache_ttl = (
            cfg.PROMPT_CACHE_TTL_SECONDS
            if (is_sanctions or event.source_tier == "S")
            else cfg.PROMPT_CACHE_TTL_SECONDS_LONG
        )
        self._llm_calls += 1
        sanctions_max_tokens = (
            cfg.NEWS_LLM_SANCTIONS_MAX_TOKENS if is_sanctions else cfg.NEWS_LLM_REACTIVE_MAX_TOKENS
        )
        try:
            llm_response = await self.polza.chat_json(
                messages=[
                    {"role": "system", "content": "You output strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                purpose="news_sanctions" if is_sanctions else "news_reactive",
                max_tokens=sanctions_max_tokens,
                cache_ttl_seconds=cache_ttl,
            )
        except Exception as exc:
            logger.warning(
                "NewsLLM: LLM call failed",
                extra={"source": event.source, "error": str(exc)},
            )
            return []

        if not isinstance(llm_response, dict):
            return []

        parsed = llm_response.get("parsed")
        payload = parsed if isinstance(parsed, dict) and parsed else llm_response

        direction_str = str(payload.get("direction", "NEUTRAL")).upper()
        if direction_str not in ("BUY", "SELL", "NEUTRAL"):
            direction_str = "NEUTRAL"
        magnitude = max(0.0, min(1.0, float(payload.get("magnitude", 0.5) or 0.5)))
        classification = str(payload.get("classification", "") or "").strip()
        if direction_str == "NEUTRAL":
            implied_dir = self._infer_direction_from_keywords(
                event.headline, event.body, matched_kw, is_sanctions
            )
            if implied_dir is None:
                self._drops_llm_neutral += 1
                return []
            direction_str = implied_dir
            magnitude = max(magnitude, 0.5)
            self._fallbacks_keyword_signal += 1
            logger.info(
                "NewsLLM: keyword fallback (LLM NEUTRAL, keywords decisive)",
                extra={
                    "source": event.source,
                    "headline": event.headline[:120],
                    "implied_direction": direction_str,
                    "matched_keywords": matched_kw,
                    "trace_id": get_trace_id(),
                },
            )
        if magnitude < cfg.NEWS_LLM_MAGNITUDE_FLOOR:
            self._drops_low_magnitude += 1
            return []

        affected = payload.get("affected_tickers", tickers)
        if not isinstance(affected, list):
            affected = tickers
        affected = [ticker.upper() for ticker in affected if ticker.upper() in cfg.TICKERS]
        if not affected:
            affected = [t for t in tickers if t in cfg.TICKERS]
        if not affected:
            self._drops_no_affected += 1
            return []

        horizon_min = int(payload.get("horizon_min", 60) or 60)
        horizon_min = max(5, min(300, horizon_min))
        reason = str(payload.get("reason", ""))[:200]
        entry_bias = str(payload.get("entry_bias", "market_now") or "market_now").lower()
        expected_rr_llm = float(payload.get("expected_rr", 0.0) or 0.0)

        is_sanctions_broad = is_sanctions or _is_sanctions_signal(
            classification=classification,
            headline=event.headline or "",
            body=event.body or "",
            matched_keywords=matched_kw,
            source=event.source,
        )

        consensus_meta: dict[str, Any] = {}
        if cfg.RAG_CONSENSUS_ENABLED and self.comparator is not None and not is_sanctions_broad:
            event.tickers = list(affected)
            try:
                cmp_results = await self.comparator.compare_event(event)
            except Exception as exc:
                logger.warning(
                    "NewsLLM: comparator failed (continuing without)",
                    extra={"event_id": event.event_id, "error": str(exc)},
                )
                cmp_results = {}
            new_direction, new_magnitude, should_drop, dbg = (
                self.comparator.apply_consensus_to_signal(
                    comparisons=cmp_results,
                    base_direction=direction_str,
                    base_magnitude=magnitude,
                    classification=classification,
                    headline=event.headline or "",
                    body=event.body or "",
                    matched_keywords=matched_kw,
                    source=event.source,
                )
            )
            if should_drop:
                self._consensus_neutral_drops += 1
                logger.info(
                    "NewsLLM: dropped by consensus (all neutral, strong consensus)",
                    extra={
                        "event_id": event.event_id,
                        "headline": event.headline[:120],
                        "tickers": affected,
                        "consensus_rule": dbg.get("rule"),
                        "max_consensus_strength": dbg.get("max_consensus_strength"),
                    },
                )
                return []
            non_neutral = [r for r in cmp_results.values() if r.alignment != "neutral"]
            if any(r.alignment == "matches_consensus" for r in non_neutral):
                self._consensus_matches += 1
            if any(r.alignment == "contradicts" for r in non_neutral):
                self._consensus_contradicts += 1
            direction_str = new_direction
            magnitude = new_magnitude
            consensus_meta = {
                "rag_consensus_applied": True,
                "rag_rule": dbg.get("rule"),
                "rag_alignment": ",".join(sorted({r.alignment for r in non_neutral}))
                or "all_neutral",
                "rag_per_ticker": {t: r.to_dict() for t, r in cmp_results.items()},
            }
        elif is_sanctions_broad and not is_sanctions:
            consensus_meta = {
                "rag_consensus_applied": False,
                "rag_rule": "sanctions_force_through_semantic",
            }
            logger.info(
                "NewsLLM: sanctions force-through (semantic)",
                extra={
                    "event_id": event.event_id,
                    "headline": event.headline[:120],
                    "classification": classification,
                    "tickers": affected,
                },
            )

        await self.history.update_analysis(
            event.event_id,
            direction=direction_str,
            magnitude=magnitude,
            horizon_min=horizon_min,
            reason=reason,
        )

        signals: list[UnifiedSignal] = []
        for ticker in affected:
            ctx = ticker_contexts.get(ticker, {})
            hist = ctx.get("historical_summary", {})
            event_type = self._classify_event_type(event.headline or "", event.body or "")
            tod_bucket = self._time_of_day_bucket(event.ts_utc)
            final_magnitude = self._finalise_magnitude(
                base_magnitude=magnitude,
                source_tier=event.source_tier,
                direction=direction_str,
                historical_summary=hist,
                ta_direction=str(ctx.get("ta_direction", "")),
                catboost_score=float(ctx.get("catboost_score", 0.0) or 0.0),
                event_type=event_type,
                tod_bucket=tod_bucket,
            )
            price = float(ctx.get("price", 0.0) or 0.0)
            atr = float(ctx.get("atr", 0.0) or 0.0)
            expected_rr = self._estimate_expected_rr(
                expected_rr_llm=expected_rr_llm,
                final_magnitude=final_magnitude,
                historical_bias=float(hist.get("bias", 0.0) or 0.0),
                ta_expected_rr=float(ctx.get("ta_expected_rr", 0.0) or 0.0),
            )
            entry, stop, target = self._build_trade_levels(
                direction=direction_str,
                price=price,
                atr=atr,
                expected_rr=expected_rr,
                entry_bias=entry_bias,
            )

            await self.history.save_context_snapshot(
                event.event_id,
                ticker,
                {
                    **ctx,
                    "historical_bias": float(hist.get("bias", 0.0) or 0.0),
                    "retrieval_cases": int(hist.get("n_cases", 0) or 0),
                },
            )

            if price > 0:
                self._schedule_reaction_tracking(event.event_id, ticker, price)

            signals.append(
                UnifiedSignal(
                    source=SignalSource.NEWS,
                    detector=event.source,
                    ticker=ticker,
                    direction=Direction(direction_str),
                    magnitude=final_magnitude,
                    raw_confidence=magnitude,
                    horizon_min=horizon_min,
                    price=price,
                    entry_level=entry,
                    stop_level=stop,
                    target_level=target,
                    expected_rr=expected_rr,
                    atr=atr,
                    metadata={
                        "news_event_id": event.event_id,
                        "headline": event.headline[:150],
                        "url": event.url,
                        "tier": event.source_tier,
                        "event_type": event_type,
                        "matched_keywords": matched_kw,
                        "is_sanctions": is_sanctions,
                        "always_material": is_always,
                        "reason": reason,
                        "entry_bias": entry_bias,
                        "historical_summary": hist,
                        "market_context": {
                            "regime": ctx.get("regime", "unknown"),
                            "rsi": ctx.get("rsi", 0.0),
                            "vol_z": ctx.get("vol_z", 0.0),
                            "ret_30m_pct": ctx.get("ret_30m_pct", 0.0),
                            "ta_pattern": ctx.get("ta_pattern", ""),
                            "ta_direction": ctx.get("ta_direction", ""),
                            "catboost_score": ctx.get("catboost_score", 0.0),
                        },
                        **(
                            {
                                "rag_consensus": consensus_meta.get("rag_per_ticker", {}).get(
                                    ticker, {}
                                ),
                                "rag_alignment": consensus_meta.get("rag_alignment", ""),
                            }
                            if consensus_meta
                            else {}
                        ),
                    },
                )
            )

        try:
            from app.memory.rag_store import get_rag_store

            rag = get_rag_store()
            rag.add_news(
                event_id=event.event_id,
                text=f"{event.headline}\n{event.body}".strip(),
                ts_utc=event.ts_utc,
                tickers=affected or tickers,
                source=event.source,
                source_tier=event.source_tier,
                headline=event.headline,
                body=event.body,
            )
        except Exception as exc:
            logger.debug(
                "NewsLLM: RAG index failed (non-fatal)",
                extra={"event_id": event.event_id, "error": str(exc)},
            )

        logger.info(
            "News signal generated",
            extra={
                "source": event.source,
                "headline": event.headline[:100],
                "direction": direction_str,
                "magnitude": round(magnitude, 2),
                "tickers": affected,
                "horizon_min": horizon_min,
                "is_sanctions": is_sanctions,
                "trace_id": get_trace_id(),
            },
        )
        return signals

    async def _process_event_local(
        self,
        *,
        event: NormalizedNewsEvent,
        full_text: str,
        tickers: list[str],
        is_sanctions: bool,
        is_mat: bool,
        matched_kw: list[str],
        is_always: bool,
    ) -> list[UnifiedSignal]:
        """POLZA-down branch: emit signals from local sentiment scoring."""
        score = self.local_sentiment.score_text(full_text)
        if abs(score) < cfg.NEWS_LOCAL_SENT_THRESHOLD:
            return []

        direction_str = "BUY" if score > 0 else "SELL"
        base_magnitude = max(0.0, min(1.0, abs(score) * cfg.NEWS_LOCAL_SENT_MAG_MULT))
        horizon_min = 60

        try:
            event_type = await self.history.record_event(
                event=event,
                tickers=tickers,
                matched_keywords=matched_kw,
                is_material=is_mat or is_sanctions,
                is_sanctions=is_sanctions,
            )
        except Exception as exc:
            logger.debug(
                "NewsLLM(local): record_event failed, continuing",
                extra={"event_id": event.event_id, "error": str(exc)},
            )
            event_type = "sanctions" if is_sanctions else "reactive"

        ticker_contexts: dict[str, dict[str, Any]] = {}
        for ticker in tickers:
            try:
                ticker_contexts[ticker] = await self._build_ticker_context(
                    ticker=ticker,
                    event=event,
                    matched_keywords=matched_kw,
                    event_type=event_type,
                    is_sanctions=is_sanctions,
                )
            except Exception as exc:
                logger.debug(
                    "NewsLLM(local): ticker context failed",
                    extra={"ticker": ticker, "error": str(exc)},
                )
                ticker_contexts[ticker] = {
                    "ticker": ticker,
                    "price": 0.0,
                    "atr": 0.0,
                    "historical_summary": {},
                }

        try:
            await self.history.update_analysis(
                event.event_id,
                direction=direction_str,
                magnitude=base_magnitude,
                horizon_min=horizon_min,
                reason=f"local_sentiment score={score:+.2f}",
            )
        except Exception as exc:
            logger.debug(
                "NewsLLM(local): update_analysis failed",
                extra={"event_id": event.event_id, "error": str(exc)},
            )

        event_type = self._classify_event_type(event.headline or "", event.body or "")
        tod_bucket = self._time_of_day_bucket(event.ts_utc)

        signals: list[UnifiedSignal] = []
        for ticker in tickers:
            ctx = ticker_contexts.get(ticker, {})
            hist = ctx.get("historical_summary", {})
            final_magnitude = self._finalise_magnitude(
                base_magnitude=base_magnitude,
                source_tier=event.source_tier,
                direction=direction_str,
                historical_summary=hist,
                ta_direction=str(ctx.get("ta_direction", "")),
                catboost_score=float(ctx.get("catboost_score", 0.0) or 0.0),
                event_type=event_type,
                tod_bucket=tod_bucket,
            )
            price = float(ctx.get("price", 0.0) or 0.0)
            atr = float(ctx.get("atr", 0.0) or 0.0)
            expected_rr = self._estimate_expected_rr(
                expected_rr_llm=0.0,
                final_magnitude=final_magnitude,
                historical_bias=float(hist.get("bias", 0.0) or 0.0),
                ta_expected_rr=float(ctx.get("ta_expected_rr", 0.0) or 0.0),
            )
            entry, stop, target = self._build_trade_levels(
                direction=direction_str,
                price=price,
                atr=atr,
                expected_rr=expected_rr,
                entry_bias="market_now",
            )

            with contextlib.suppress(Exception):
                await self.history.save_context_snapshot(
                    event.event_id,
                    ticker,
                    {
                        **ctx,
                        "historical_bias": float(hist.get("bias", 0.0) or 0.0),
                        "retrieval_cases": int(hist.get("n_cases", 0) or 0),
                        "local_sentiment_score": score,
                    },
                )

            if price > 0:
                self._schedule_reaction_tracking(event.event_id, ticker, price)

            signals.append(
                UnifiedSignal(
                    source=SignalSource.NEWS,
                    detector="local_sentiment",
                    ticker=ticker,
                    direction=Direction(direction_str),
                    magnitude=final_magnitude,
                    raw_confidence=abs(score),
                    horizon_min=horizon_min,
                    price=price,
                    entry_level=entry,
                    stop_level=stop,
                    target_level=target,
                    expected_rr=expected_rr,
                    atr=atr,
                    metadata={
                        "news_event_id": event.event_id,
                        "headline": event.headline[:150],
                        "url": event.url,
                        "tier": event.source_tier,
                        "event_type": event_type,
                        "matched_keywords": matched_kw,
                        "is_sanctions": is_sanctions,
                        "always_material": is_always,
                        "reason": f"local_sentiment score={score:+.2f}",
                        "entry_bias": "market_now",
                        "fallback_mode": "local_sentiment",
                        "local_sentiment_score": score,
                        "local_sentiment_backend": self.local_sentiment.backend,
                        "local_sentiment_model": self.local_sentiment.model_name,
                        "historical_summary": hist,
                        "market_context": {
                            "regime": ctx.get("regime", "unknown"),
                            "rsi": ctx.get("rsi", 0.0),
                            "vol_z": ctx.get("vol_z", 0.0),
                            "ret_30m_pct": ctx.get("ret_30m_pct", 0.0),
                            "ta_pattern": ctx.get("ta_pattern", ""),
                            "ta_direction": ctx.get("ta_direction", ""),
                            "catboost_score": ctx.get("catboost_score", 0.0),
                        },
                    },
                )
            )

        logger.info(
            "News signal generated (local sentiment)",
            extra={
                "source": event.source,
                "headline": event.headline[:100],
                "direction": direction_str,
                "score": round(score, 3),
                "magnitude": round(base_magnitude, 2),
                "tickers": tickers,
                "is_sanctions": is_sanctions,
                "backend": self.local_sentiment.backend,
                "trace_id": get_trace_id(),
            },
        )
        return signals

    async def _build_ticker_context(
        self,
        *,
        ticker: str,
        event: NormalizedNewsEvent,
        matched_keywords: list[str],
        event_type: str,
        is_sanctions: bool,
    ) -> dict[str, Any]:
        """Build ticker context."""
        context: dict[str, Any] = {
            "ticker": ticker,
            "price": 0.0,
            "atr": 0.0,
            "atr_pct": 0.0,
            "rsi": 0.0,
            "vol_z": 0.0,
            "ret_30m_pct": 0.0,
            "regime": self.hmm.current_label,
            "regime_proba": {},
            "catboost_score": 0.0,
            "ta_pattern": "",
            "ta_direction": "",
            "ta_expected_rr": 0.0,
            "historical_summary": {},
        }

        df = self.candle_store.get(ticker, 5)
        if _HAS_PANDAS and isinstance(df, pd.DataFrame) and not df.empty:
            try:
                indicators = compute_all(df, atr_period=14)
                atr_series = indicators.get("atr")
                rsi_series = indicators.get("rsi")
                vol_z_series = indicators.get("vol_z")

                price = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0
                atr = (
                    float(atr_series.iloc[-1])
                    if hasattr(atr_series, "iloc") and len(atr_series)
                    else 0.0
                )
                rsi = (
                    float(rsi_series.iloc[-1])
                    if hasattr(rsi_series, "iloc") and len(rsi_series)
                    else 0.0
                )
                vol_z = (
                    float(vol_z_series.iloc[-1])
                    if hasattr(vol_z_series, "iloc") and len(vol_z_series)
                    else 0.0
                )
                ret_30m = 0.0
                if len(df) >= 7 and "close" in df.columns:
                    ret_30m = float(df["close"].iloc[-1]) / float(df["close"].iloc[-7]) - 1.0

                context.update(
                    {
                        "price": price,
                        "atr": atr,
                        "atr_pct": (atr / price) if price > 0 else 0.0,
                        "rsi": rsi,
                        "vol_z": vol_z,
                        "ret_30m_pct": ret_30m,
                        "regime_proba": self.hmm.predict_proba_last(df),
                    }
                )
            except Exception as exc:
                logger.debug(
                    "NewsLLM market context failed", extra={"ticker": ticker, "error": str(exc)}
                )

        try:
            ta_context = await self.ta.get_context_snapshot(ticker)
            if isinstance(ta_context, dict):
                context.update(ta_context)
        except Exception as exc:
            logger.debug("NewsLLM TA context failed", extra={"ticker": ticker, "error": str(exc)})

        try:
            historical_summary = await self.history.find_similar_cases(
                event=event,
                ticker=ticker,
                matched_keywords=matched_keywords,
                event_type=event_type,
                is_sanctions=is_sanctions,
                regime=str(context.get("regime", "unknown")),
                catboost_score=float(context.get("catboost_score", 0.0) or 0.0),
            )
            context["historical_summary"] = historical_summary
        except Exception as exc:
            logger.debug(
                "NewsLLM history lookup failed", extra={"ticker": ticker, "error": str(exc)}
            )
            context["historical_summary"] = {}

        return context

    @staticmethod
    def _build_context_block(ticker_contexts: dict[str, dict[str, Any]]) -> str:
        """Build context block."""
        lines = [
            "ADDITIONAL MARKET CONTEXT:",
            "Use this to decide whether the textual signal should be amplified, faded, or rejected.",
        ]
        for ticker, ctx in ticker_contexts.items():
            hist = ctx.get("historical_summary", {})
            lines.append(
                f"- {ticker}: price={float(ctx.get('price', 0.0) or 0.0):.2f}, "
                f"atr_pct={float(ctx.get('atr_pct', 0.0) or 0.0):.2%}, "
                f"rsi={float(ctx.get('rsi', 0.0) or 0.0):.1f}, "
                f"vol_z={float(ctx.get('vol_z', 0.0) or 0.0):+.2f}, "
                f"ret30m={float(ctx.get('ret_30m_pct', 0.0) or 0.0):+.2%}, "
                f"regime={ctx.get('regime', 'unknown')}"
            )
            lines.append(
                f"  TA context: pattern={ctx.get('ta_pattern', '') or 'none'}, "
                f"direction={ctx.get('ta_direction', '') or 'none'}, "
                f"catboost={float(ctx.get('catboost_score', 0.0) or 0.0):.2f}, "
                f"ta_rr={float(ctx.get('ta_expected_rr', 0.0) or 0.0):.2f}"
            )
            lines.append(
                f"  Historical analogs: n={int(hist.get('n_cases', 0) or 0)}, "
                f"avg15m={float(hist.get('avg_ret_15m', 0.0) or 0.0):+.2%}, "
                f"avg60m={float(hist.get('avg_ret_60m', 0.0) or 0.0):+.2%}, "
                f"pos60m={float(hist.get('positive_rate_60m', 0.0) or 0.0):.0%}, "
                f"bias={hist.get('bias_label', 'NEUTRAL')}"
            )

        lines.extend(
            [
                "Return strict JSON with existing keys plus these keys:",
                '{ "expected_rr": float >= 0.8, "entry_bias": "market_now|wait_pullback|wait_breakout|avoid" }',
                "Raise magnitude only when text, market context, and historical analogs align.",
                "If historical analogs strongly contradict the text, lower magnitude or return NEUTRAL.",
            ]
        )
        return "\n".join(lines)

    _BEARISH_KEYWORDS = (
        "санкц",
        "sanction",
        "ofac",
        "sdn",
        "запрет",
        "ban",
        "штраф",
        "fine",
        "investigation",
        "расследован",
        "приостанов",
        "halt",
        "suspension",
        "downgrade",
        "пониж",
        "убыт",
        "loss",
        "отмен",
        "cancel",
        "guidance_down",
        "снижен прогноз",
        "warning",
        "увольнен",
        "resignation",
        "отставк",
        "missed",
        "хуже прогноза",
    )
    _BULLISH_KEYWORDS = (
        "прибыл",
        "profit",
        "выручк",
        "revenue",
        "ebitda",
        "beat",
        "дивиденд",
        "dividend",
        "buyback",
        "выкуп",
        "upgrade",
        "повыс",
        "rating_up",
        "guidance_up",
        "лучше прогноза",
        "рекорд",
        "record",
        "успех",
        "discovery",
        "месторожден",
    )

    @classmethod
    def _infer_direction_from_keywords(
        cls,
        headline: str,
        body: str,
        matched_kw: list[str],
        is_sanctions: bool,
    ) -> str | None:
        """Return BUY/SELL implied by matched keywords, or None if ambiguous."""
        if is_sanctions:
            return "SELL"
        text = (headline + " " + body + " " + " ".join(matched_kw or [])).lower()
        bearish_hit = any(k in text for k in cls._BEARISH_KEYWORDS)
        bullish_hit = any(k in text for k in cls._BULLISH_KEYWORDS)
        if bearish_hit and not bullish_hit:
            return "SELL"
        if bullish_hit and not bearish_hit:
            return "BUY"
        return None

    @staticmethod
    def _classify_event_type(headline: str, body: str = "") -> str:
        """Lightweight keyword classifier — sanctions/macro/commodity/earnings/guidance/other."""
        text = (headline + " " + body).lower()
        if any(
            k in text for k in ("санкц", "sanction", "ofac", "sdn", "blocked", "блок", "ограничен")
        ):
            return "sanctions"
        if any(
            k in text
            for k in (
                "ставк",
                "rate",
                "цб ",
                "cbr",
                "inflation",
                "инфляц",
                "gdp",
                "ввп",
                "fomc",
                "ecb",
                "ецб",
            )
        ):
            return "macro"
        if any(
            k in text
            for k in (
                "brent",
                "wti",
                "нефт",
                "oil",
                "газ",
                "gas",
                "gold",
                "золот",
                "никел",
                "алюмин",
                "сталь",
                "коммоди",
            )
        ):
            return "commodity"
        if any(
            k in text
            for k in (
                "earnings",
                "квартал",
                "отчётнос",
                "отчетн",
                "прибыл",
                "выручк",
                "ebitda",
                "ipo",
                "финансов",
            )
        ):
            return "earnings"
        if any(
            k in text
            for k in ("дивиденд", "dividend", "прогноз", "guidance", "forecast", "buyback", "выкуп")
        ):
            return "guidance"
        return "other"

    @staticmethod
    def _time_of_day_bucket(ts_utc: datetime | None = None) -> str:
        """MSK hour bucket — premarket/morning/midday/afternoon/afterhours."""
        from datetime import datetime as _dt

        if ts_utc is None:
            ts_utc = _dt.now(tz=UTC)
        msk_hour = (ts_utc.hour + 3) % 24
        if msk_hour < 10:
            return "premarket"
        if msk_hour < 13:
            return "morning"
        if msk_hour < 17:
            return "midday"
        if msk_hour < 19:
            return "afternoon"
        return "afterhours"

    @staticmethod
    def _finalise_magnitude(
        *,
        base_magnitude: float,
        source_tier: str,
        direction: str,
        historical_summary: dict[str, Any],
        ta_direction: str,
        catboost_score: float,
        event_type: str = "other",
        tod_bucket: str = "midday",
    ) -> float:
        """Finalise magnitude."""
        source_mult = {"S": 1.25, "A": 1.12, "B": 1.0, "C": 0.92}.get(source_tier, 1.0)
        ta_mult = 1.0
        if ta_direction:
            ta_mult = 1.12 if ta_direction == direction else 0.88

        hist_mult = 1.0
        hist_bias = float(historical_summary.get("bias", 0.0) or 0.0)
        hist_label = str(historical_summary.get("bias_label", "NEUTRAL"))
        if hist_label == direction:
            hist_mult += min(0.18, abs(hist_bias) * 8.0)
        elif hist_label not in ("", "NEUTRAL"):
            hist_mult -= min(0.18, abs(hist_bias) * 8.0)

        catboost_mult = 0.95 + min(0.20, max(0.0, catboost_score) * 0.25)
        event_mult = cfg.NEWS_EVENT_TYPE_MAG_MULT.get(event_type, 1.0)
        tod_mult = cfg.NEWS_TIME_OF_DAY_MULT.get(tod_bucket, 1.0)
        final = (
            base_magnitude
            * source_mult
            * ta_mult
            * hist_mult
            * catboost_mult
            * event_mult
            * tod_mult
        )
        return max(0.0, min(1.0, final))

    @staticmethod
    def _estimate_expected_rr(
        *,
        expected_rr_llm: float,
        final_magnitude: float,
        historical_bias: float,
        ta_expected_rr: float,
    ) -> float:
        """Estimate expected rr."""
        if expected_rr_llm > 0:
            rr = expected_rr_llm
        elif ta_expected_rr > 0:
            rr = ta_expected_rr
        else:
            rr = 1.0 + final_magnitude
        rr += min(0.4, abs(historical_bias) * 10.0)
        return max(0.8, min(3.5, rr))

    @staticmethod
    def _build_trade_levels(
        *,
        direction: str,
        price: float,
        atr: float,
        expected_rr: float,
        entry_bias: str,
    ) -> tuple[float, float, float]:
        """Build trade levels."""
        if price <= 0:
            return 0.0, 0.0, 0.0

        atr = atr if atr > 0 else price * 0.008
        if direction == "BUY":
            entry = price if entry_bias != "wait_pullback" else max(0.0, price - atr * 0.25)
            stop = max(0.0, entry - atr * 1.2)
            target = entry + (entry - stop) * expected_rr
        else:
            entry = price if entry_bias != "wait_breakout" else price - atr * 0.15
            stop = entry + atr * 1.2
            target = max(0.0, entry - (stop - entry) * expected_rr)
        return entry, stop, target

    def _schedule_reaction_tracking(self, event_id: str, ticker: str, price_t0: float) -> None:
        """Schedule reaction tracking."""
        for window_min in REACTION_WINDOWS_MIN:
            task = asyncio.create_task(
                self._capture_reaction_later(event_id, ticker, price_t0, window_min),
                name=f"news_reaction_{ticker}_{window_min}",
            )
            self._reaction_tasks.add(task)
            task.add_done_callback(self._reaction_tasks.discard)

    async def _capture_reaction_later(
        self,
        event_id: str,
        ticker: str,
        price_t0: float,
        window_min: int,
    ) -> None:
        """Capture reaction later."""
        try:
            await asyncio.sleep(window_min * 60)
            latest_price = self.candle_store.get_last_price(ticker, interval=1)
            if latest_price is None:
                latest_price = self.candle_store.get_last_price(ticker, interval=5)
            if latest_price is None:
                return
            await self.history.save_reaction(
                event_id, ticker, window_min, price_t0, float(latest_price)
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "News reaction capture failed",
                extra={
                    "event_id": event_id,
                    "ticker": ticker,
                    "window_min": window_min,
                    "error": str(exc),
                },
            )

    @staticmethod
    def _render(template: str, vars: dict[str, Any]) -> str:
        """Render."""
        out = template
        for key, value in vars.items():
            out = out.replace("{{" + key + "}}", str(value))
        return out

_news_llm: NewsLLM | None = None

def get_news_llm() -> NewsLLM:
    """Get news llm."""
    global _news_llm
    if _news_llm is None:
        _news_llm = NewsLLM()
    return _news_llm
