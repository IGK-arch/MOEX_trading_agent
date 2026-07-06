"""v1.3.1 — Hourly Market Mood Scan.

Background-задача: каждый час получает текущее состояние (HMM regime,
открытые позиции, dd, недавние news, top tickers) и просит LLM
(gemini-3.1-flash-lite) выдать единый "mood snapshot":

    {
      "verdict": "bullish" | "bearish" | "sideways" | "risk_off",
      "confidence": 0..1,
      "size_multiplier": 0.5..1.5,
      "reasoning": "..."
    }

Результат пишется в /data/market_mood.json для:
  * morning_plan (контекст утреннего брифинга)
  * adaptive sizing (умножается на текущий regime multiplier)

Не блокирует hot-path Dispatcher cycle — чистый background.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from typing import Any

import app.config as cfg
from app.llm.polza_client import get_polza_client
from app.utils.logging import get_logger
from app.utils.sessions import is_trading_open

logger = get_logger(__name__)

MOOD_PATH = cfg.DATA_DIR / "market_mood.json"

class MarketMoodScanner:
    """Background scanner: hourly LLM mood snapshot."""

    def __init__(self) -> None:
        """Init."""
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._last_mood: dict[str, Any] = {}
        self.polza = get_polza_client()

    async def start(self) -> None:
        """Spawn the periodic background task."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop(), name="market_mood_scanner")
        logger.info(
            "MarketMoodScanner started",
            extra={"interval_sec": cfg.MARKET_MOOD_INTERVAL_SEC},
        )

    async def stop(self) -> None:
        """Stop the loop."""
        self._stop_event.set()
        if self._task is not None:
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._task, timeout=3.0)
        logger.info("MarketMoodScanner stopped")

    async def _loop(self) -> None:
        """Periodic loop."""
        while not self._stop_event.is_set():
            try:
                if is_trading_open():
                    await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("MarketMoodScanner cycle failed", extra={"error": str(exc)})
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=cfg.MARKET_MOOD_INTERVAL_SEC,
                )

    async def _scan_once(self) -> None:
        """Один проход: собрать контекст → LLM → записать в JSON."""
        if cfg.DISABLE_LLM:
            return

        ctx = await self._build_context()
        prompt = self._build_prompt(ctx)
        try:
            response = await self.polza.chat_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты — макро-аналитик MOEX. Кратко оцени текущий market mood. "
                            "Отвечай СТРОГО JSON без markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=cfg.POLZA_MODEL_REACTIVE,
                purpose="market_mood_scan",
                max_tokens=cfg.MARKET_MOOD_MAX_TOKENS,
            )
        except Exception as exc:
            logger.warning("MarketMoodScanner LLM failed", extra={"error": str(exc)})
            return

        parsed = response.get("parsed") or {}
        if not parsed:
            return

        verdict = str(parsed.get("verdict") or "sideways").lower()
        if verdict not in ("bullish", "bearish", "sideways", "risk_off"):
            verdict = "sideways"
        try:
            confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0.5)))
            size_mult = max(0.5, min(1.5, float(parsed.get("size_multiplier") or 1.0)))
        except (TypeError, ValueError):
            confidence, size_mult = 0.5, 1.0
        reasoning = str(parsed.get("reasoning") or "")[:500]

        mood = {
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "size_multiplier": round(size_mult, 3),
            "reasoning": reasoning,
            "scanned_at_utc": datetime.now(tz=UTC).isoformat(),
            "context_summary": ctx,
        }
        self._last_mood = mood

        try:
            MOOD_PATH.parent.mkdir(parents=True, exist_ok=True)
            MOOD_PATH.write_text(json.dumps(mood, ensure_ascii=False, indent=2))
        except Exception as exc:
            logger.warning("Mood persist failed", extra={"error": str(exc)})

        logger.info(
            "Market mood scanned",
            extra={
                "verdict": verdict,
                "confidence": confidence,
                "size_multiplier": size_mult,
                "cost_rub": round(response.get("cost_rub") or 0.0, 5),
            },
        )

    async def _build_context(self) -> dict[str, Any]:
        """Собрать компактный контекст: HMM regime, top positions, recent news."""
        ctx: dict[str, Any] = {}
        try:
            from app.agents.hmm_regime import get_hmm_detector

            ctx["hmm_regime"] = get_hmm_detector().current_label
        except Exception:
            ctx["hmm_regime"] = "unknown"

        try:
            from app.risk.circuit_breakers import get_circuit_breaker
            from app.risk.position_book import get_position_book

            book = get_position_book()
            cb = get_circuit_breaker()
            ctx["open_positions"] = len(book.positions)
            ctx["cash_balance_rub"] = round(book.cash_balance, 0)
            ctx["current_drawdown_pct"] = round(cb.state.current_drawdown_pct * 100, 2)
            ctx["daily_pnl_rub"] = round(cb.state.daily_pnl_rub, 0)
        except Exception:
            pass

        try:
            from app.news.ingestion_bus import get_ingestion_bus

            bus = get_ingestion_bus()
            recent = list(bus.recent_buffer)[-5:] if hasattr(bus, "recent_buffer") else []
            ctx["recent_news_headlines"] = [
                str(getattr(e, "headline", "") or "")[:120] for e in recent
            ]
        except Exception:
            ctx["recent_news_headlines"] = []

        return ctx

    @staticmethod
    def _build_prompt(ctx: dict[str, Any]) -> str:
        """Сформировать промпт по контексту."""
        headlines = ctx.get("recent_news_headlines", [])
        headlines_block = "\n".join(f"  - {h}" for h in headlines) if headlines else "  (нет свежих новостей в буфере)"
        return (
            f"Текущее состояние рынка MOEX (timestamp: {datetime.now(tz=UTC).isoformat()}):\n"
            f"  - HMM режим: {ctx.get('hmm_regime', 'unknown')}\n"
            f"  - Открытых позиций: {ctx.get('open_positions', 0)}\n"
            f"  - Cash balance: {ctx.get('cash_balance_rub', 0)} ₽\n"
            f"  - Текущая просадка: {ctx.get('current_drawdown_pct', 0)}%\n"
            f"  - Дневной P&L: {ctx.get('daily_pnl_rub', 0)} ₽\n"
            f"  - Свежие новости (топ-5):\n{headlines_block}\n\n"
            f"Оцени market mood и верни JSON:\n"
            f"{{\n"
            f'  "verdict": "bullish" | "bearish" | "sideways" | "risk_off",\n'
            f'  "confidence": 0..1,\n'
            f'  "size_multiplier": 0.5..1.5 (множитель к size: 1.0 нейтрально, 0.5 урезать, 1.5 раздуть),\n'
            f'  "reasoning": "1-2 предложения по-русски"\n'
            f"}}"
        )

    @property
    def last_mood(self) -> dict[str, Any]:
        """Return last mood snapshot dict (in-memory)."""
        return dict(self._last_mood)

_scanner: MarketMoodScanner | None = None

def get_market_mood_scanner() -> MarketMoodScanner:
    """Singleton accessor."""
    global _scanner
    if _scanner is None:
        _scanner = MarketMoodScanner()
    return _scanner

def load_last_mood() -> dict[str, Any]:
    """Read latest persisted mood from disk (used by morning_plan)."""
    if not MOOD_PATH.exists():
        return {}
    try:
        return json.loads(MOOD_PATH.read_text())
    except Exception:
        return {}

__all__ = ["MarketMoodScanner", "get_market_mood_scanner", "load_last_mood"]
