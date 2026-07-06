"""polza.ai LLM клиент (OpenAI-совместимый)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import app.config as cfg
from app.utils.logging import get_logger, get_trace_id

logger = get_logger(__name__)

try:
    from openai import AsyncOpenAI  # type: ignore

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    logger.warning("openai package not installed — LLM calls will be mocked")

try:
    import aiosqlite  # type: ignore

    _HAS_AIOSQLITE = True
except ImportError:
    _HAS_AIOSQLITE = False

REASONING_MODEL_PATTERNS: tuple[str, ...] = (
    "deepseek/deepseek-r1",
    "qwen/qwen3.7-max",
    "qwen/qwen3-max-thinking",
    "deepseek/deepseek-v4-pro",
)

def _is_reasoning_model(model: str) -> bool:
    """Return True if model exposes an OpenRouter-style ``reasoning`` channel."""
    if not model:
        return False
    m = model.lower()
    return any(m.startswith(p) for p in REASONING_MODEL_PATTERNS)

MODEL_PRICES_INPUT_RUB_PER_M: dict[str, float] = {
    "deepseek/deepseek-v4-pro": 408.46,
    "deepseek/deepseek-v4-flash": 12.85,
    "deepseek/deepseek-r1-0528": 41.0,
    "deepseek/deepseek-r1-distill-llama-70b": 2.75,
    "deepseek/deepseek-r1-distill-qwen-32b": 5.0,
    "qwen/qwen3.7-max": 114.0,
    "qwen/qwen3-max": 100.0,
    "qwen/qwen3-max-thinking": 120.0,
    "qwen/qwen3.5-plus-20260420": 50.0,
    "qwen/qwen3.5-plus-02-15": 45.0,
    "qwen/qwen3-embedding-8b": 0.92,
    "qwen/qwen3-embedding-4b": 0.5,
    "google/gemini-3.5-flash": 137.0,
    "google/gemini-3.1-flash-lite": 12.0,
    "google/gemini-embedding-001": 13.73,
    "google/gemini-embedding-2-preview": 18.31,
}

MODEL_PRICES_OUTPUT_RUB_PER_M: dict[str, float] = {
    "deepseek/deepseek-v4-pro": 504.84,
    "deepseek/deepseek-v4-flash": 25.70,
    "deepseek/deepseek-r1-0528": 197.0,
    "deepseek/deepseek-r1-distill-llama-70b": 10.07,
    "deepseek/deepseek-r1-distill-qwen-32b": 15.0,
    "qwen/qwen3.7-max": 343.0,
    "qwen/qwen3-max": 300.0,
    "qwen/qwen3-max-thinking": 360.0,
    "qwen/qwen3.5-plus-20260420": 150.0,
    "qwen/qwen3.5-plus-02-15": 130.0,
    "qwen/qwen3-embedding-8b": 0.0,
    "qwen/qwen3-embedding-4b": 0.0,
    "google/gemini-3.5-flash": 824.0,
    "google/gemini-3.1-flash-lite": 30.0,
    "google/gemini-embedding-001": 0.0,
    "google/gemini-embedding-2-preview": 0.0,
}

class PolzaClient:
    """Async polza.ai client with budget tracking, prompt caching, model fallback."""

    BASE_URL = cfg.POLZA_BASE_URL

    def __init__(self) -> None:
        """Init."""
        self._client: Any = None
        self._db_path = cfg.DATA_DIR / "decisions.db"
        self._lock = asyncio.Lock()
        self._started: bool = False

        self._auth_disabled_until: float = 0.0
        self._auth_failure_count: int = 0
        self._auth_log_next_ts: float = 0.0
        self._auth_log_suppressed: int = 0

    async def startup(self) -> None:
        """Startup."""
        api_key = os.getenv("POLZA_API_KEY", "")
        if not api_key or cfg.DISABLE_LLM:
            cfg.DISABLE_LLM = True
            self._started = True
            logger.warning(
                "Polza disabled (no POLZA_API_KEY or DISABLE_LLM=1) — "
                "running in deterministic mode (keyword-based news, "
                "no LLM reasoning)",
            )
            return

        if not _HAS_OPENAI:
            cfg.DISABLE_LLM = True
            self._started = True
            logger.warning("openai package not installed — Polza disabled")
            return

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            timeout=60.0,
        )
        self._started = True
        logger.info("Polza client started", extra={"base_url": self.BASE_URL})

    async def shutdown(self) -> None:
        """Shutdown."""
        self._started = False
        logger.info("Polza client stopped")

    async def _get_cumulative_cost(self) -> float:
        """Read total spent from budget_log table."""
        if not _HAS_AIOSQLITE:
            return 0.0
        try:
            async with (
                aiosqlite.connect(self._db_path) as db,
                db.execute("SELECT MAX(cumulative_rub) FROM budget_log") as cur,
            ):
                row = await cur.fetchone()
                return float(row[0] or 0.0)
        except Exception:
            return 0.0

    async def _get_daily_cost(self) -> float:
        """Read today's spending from budget_log."""
        if not _HAS_AIOSQLITE:
            return 0.0
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        try:
            async with (
                aiosqlite.connect(self._db_path) as db,
                db.execute(
                    "SELECT COALESCE(SUM(cost_rub), 0) FROM budget_log WHERE ts LIKE ?",
                    (f"{today}%",),
                ) as cur,
            ):
                row = await cur.fetchone()
                return float(row[0] or 0.0)
        except Exception:
            return 0.0

    async def _log_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_rub: float,
        purpose: str = "",
    ) -> None:
        """Write token usage and cost to budget_log."""
        if not _HAS_AIOSQLITE:
            return
        cumulative = await self._get_cumulative_cost() + cost_rub
        ts = datetime.now(tz=UTC).isoformat()
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """INSERT INTO budget_log
                       (ts, model, input_tokens, output_tokens, cost_rub, cumulative_rub, purpose)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (ts, model, input_tokens, output_tokens, cost_rub, cumulative, purpose),
                )
                await db.commit()
        except Exception as exc:
            logger.error("Failed to log LLM usage", extra={"error": str(exc)})

        logger.info(
            "LLM usage logged",
            extra={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_rub": round(cost_rub, 5),
                "cumulative_rub": round(cumulative, 3),
                "purpose": purpose,
            },
        )

    def _select_model(self, requested_model: str) -> str:
        """Apply budget-based model fallback."""

        return requested_model

    async def _check_budget_and_select(self, requested_model: str) -> str:
        """Check budget and return appropriate model (with fallback)."""
        cumulative = await self._get_cumulative_cost()
        daily = await self._get_daily_cost()

        if cumulative >= cfg.POLZA_BUDGET_HARD_STOP_RUB:
            logger.critical(
                "POLZA HARD STOP: budget exhausted",
                extra={"cumulative_rub": cumulative, "limit": cfg.POLZA_BUDGET_HARD_STOP_RUB},
            )
            raise RuntimeError(f"Polza.ai budget hard stop: {cumulative:.2f} ₽ spent")

        if daily >= cfg.POLZA_DAILY_SOFT_LIMIT_RUB:
            logger.error(
                "Daily LLM budget limit hit — possible infinite loop bug!",
                extra={"daily_rub": daily, "limit": cfg.POLZA_DAILY_SOFT_LIMIT_RUB},
            )
            return cfg.POLZA_MODEL_FALLBACK

        if cumulative >= cfg.POLZA_BUDGET_SOFT_LIMIT_RUB:
            logger.warning(
                "Polza soft limit: switching to cheapest fallback",
                extra={"cumulative_rub": cumulative},
            )
            return cfg.POLZA_MODEL_FALLBACK

        return requested_model

    def _cache_key(self, model: str, messages: list[dict]) -> str:
        """Cache key."""
        content = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        return hashlib.sha1(content.encode()).hexdigest()

    async def _get_cached(self, cache_key: str) -> dict | None:
        """Get cached."""
        if not _HAS_AIOSQLITE:
            return None
        now_str = datetime.now(tz=UTC).isoformat()
        try:
            async with (
                aiosqlite.connect(self._db_path) as db,
                db.execute(
                    "SELECT response_json FROM prompt_cache WHERE cache_key = ? AND expires_at > ?",
                    (cache_key, now_str),
                ) as cur,
            ):
                row = await cur.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception:
            pass
        return None

    async def _save_cache(
        self,
        cache_key: str,
        model: str,
        response: dict,
        ttl_seconds: int | None = None,
    ) -> None:
        """Save cache.

        ``ttl_seconds`` defaults to :data:`cfg.PROMPT_CACHE_TTL_SECONDS` (4 h).
        Callers can pass :data:`cfg.PROMPT_CACHE_TTL_SECONDS_LONG` (24 h) for
        low-volatility content (cycle-5: non-sanctions, non-tier-S news).
        """
        if not _HAS_AIOSQLITE:
            return
        ttl = ttl_seconds if ttl_seconds is not None else cfg.PROMPT_CACHE_TTL_SECONDS
        now = datetime.now(tz=UTC)
        expires = (now + timedelta(seconds=ttl)).isoformat()
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO prompt_cache "
                    "(cache_key, model, response_json, created_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (cache_key, model, json.dumps(response), now.isoformat(), expires),
                )
                await db.commit()
        except Exception as exc:
            logger.debug("Cache save failed", extra={"error": str(exc)})

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = cfg.POLZA_MODEL_REACTIVE,
        max_tokens: int = 500,
        temperature: float = 0.1,
        response_format: dict | None = None,
        use_cache: bool = True,
        purpose: str = "",
        use_reasoning: bool = False,
        cache_ttl_seconds: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        include_reasoning: bool = False,
        response_schema: dict | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request to polza.ai.

        Args:
            cache_ttl_seconds: optional override for the prompt-cache row TTL on
                writes. ``None`` (default) keeps the historical 4-hour behaviour
                via :data:`cfg.PROMPT_CACHE_TTL_SECONDS`. Pass
                :data:`cfg.PROMPT_CACHE_TTL_SECONDS_LONG` for low-volatility news
                (cycle-5).
            tools: function-calling tools list (OpenAI schema). Forwarded to API.
            tool_choice: ``"none"`` | ``"auto"`` | ``{"type":"function","function":...}``.
            include_reasoning: when True and model supports CoT, return the
                ``reasoning`` field in the result dict (for logs/audit).
            response_schema: JSON-schema dict for structured outputs. When set,
                ``response_format`` is auto-built as
                ``{"type": "json_schema", "json_schema": {...}}``.
            reasoning_effort: "low" | "medium" | "high" — passed through as
                ``reasoning={"effort": ...}`` for OpenRouter reasoning models.

        Returns:
            dict: keys include content, reasoning (if requested), model,
            input_tokens, output_tokens, reasoning_tokens, cost_rub, cached,
            tool_calls.
        """
        if not self._client:
            logger.warning("Polza client not initialized — returning mock response")
            return {
                "content": '{"direction":"NEUTRAL","magnitude":0.0,"reason":"LLM unavailable"}',
                "model": model,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_rub": 0.0,
                "cached": False,
            }

        now_ts = time.monotonic()
        if now_ts < self._auth_disabled_until:
            logger.debug(
                "Polza auth circuit open — returning NEUTRAL stub",
                extra={
                    "purpose": purpose,
                    "seconds_remaining": round(self._auth_disabled_until - now_ts, 1),
                },
            )
            return {
                "content": '{"direction":"NEUTRAL","magnitude":0.0,"reason":"polza auth disabled"}',
                "model": model,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_rub": 0.0,
                "cached": False,
                "auth_disabled": True,
            }

        async with self._lock:
            model = await self._check_budget_and_select(model)

        if use_cache:
            cache_key = self._cache_key(model, messages)
            cached = await self._get_cached(cache_key)
            if cached:
                logger.debug(
                    "Prompt cache HIT",
                    extra={"cache_key": cache_key[:8], "model": model, "purpose": purpose},
                )
                return {**cached, "cached": True}

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": response_schema,
            }
        elif response_format:
            kwargs["response_format"] = response_format

        if _is_reasoning_model(model) and (use_reasoning or reasoning_effort):
            reasoning_payload: dict[str, Any] = {}
            if reasoning_effort in ("low", "medium", "high"):
                reasoning_payload["effort"] = reasoning_effort
            kwargs["reasoning"] = reasoning_payload

        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

        start_ms = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(**kwargs),
                timeout=cfg.POLZA_REQUEST_TIMEOUT_SEC,
            )

            self._auth_failure_count = 0
        except TimeoutError:
            logger.warning(
                "Polza chat() timed out",
                extra={
                    "model": model,
                    "purpose": purpose,
                    "timeout_sec": cfg.POLZA_REQUEST_TIMEOUT_SEC,
                },
            )
            return {
                "content": '{"direction":"NEUTRAL","magnitude":0.0,"reason":"polza timeout"}',
                "model": model,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_rub": 0.0,
                "cached": False,
                "timeout": True,
            }
        except Exception as exc:
            err_msg = str(exc)
            is_auth_error = (
                "401" in err_msg
                or "UNAUTHORIZED" in err_msg.upper()
                or "INVALID_API_KEY" in err_msg.upper()
                or "Некорректный API ключ" in err_msg
            )
            if is_auth_error:
                self._auth_failure_count += 1

                back_off_sec = min(1800.0, 60.0 * (5 ** (self._auth_failure_count - 1)))
                self._auth_disabled_until = time.monotonic() + back_off_sec

                now_mono = time.monotonic()
                first_three = self._auth_failure_count <= 3
                hour_window_open = now_mono >= self._auth_log_next_ts
                if first_three or hour_window_open:
                    emit = logger.error if self._auth_failure_count == 1 else logger.warning
                    emit(
                        "Polza auth failed — suspending LLM calls",
                        extra={
                            "model": model,
                            "purpose": purpose,
                            "back_off_sec": back_off_sec,
                            "auth_failure_count": self._auth_failure_count,
                            "suppressed_since_last": self._auth_log_suppressed,
                            "trace_id": get_trace_id(),
                        },
                    )
                    self._auth_log_suppressed = 0
                    self._auth_log_next_ts = now_mono + 3600.0
                else:
                    self._auth_log_suppressed += 1

                return {
                    "content": '{"direction":"NEUTRAL","magnitude":0.0,"reason":"polza auth failed"}',
                    "model": model,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_rub": 0.0,
                    "cached": False,
                    "auth_failed": True,
                }
            logger.error(
                "Polza API error",
                extra={
                    "model": model,
                    "error": err_msg,
                    "purpose": purpose,
                    "trace_id": get_trace_id(),
                },
            )
            raise

        elapsed_ms = round((time.monotonic() - start_ms) * 1000)

        choice = response.choices[0]
        message = choice.message
        content = (getattr(message, "content", None) or "") or ""
        reasoning_text = getattr(message, "reasoning", None) or ""

        raw_tool_calls = getattr(message, "tool_calls", None) or []
        tool_calls_out: list[dict[str, Any]] = []
        for tc in raw_tool_calls:
            fn = getattr(tc, "function", None)
            tool_calls_out.append(
                {
                    "id": getattr(tc, "id", ""),
                    "type": getattr(tc, "type", "function"),
                    "name": getattr(fn, "name", "") if fn else "",
                    "arguments": getattr(fn, "arguments", "") if fn else "",
                }
            )

        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)
        reasoning_tokens = 0
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning_tokens = (
                getattr(details, "reasoning_tokens", 0) or 0
                if not isinstance(details, dict)
                else int(details.get("reasoning_tokens") or 0)
            )

        cost_rub = getattr(usage, "cost_rub", None) or (
            (input_tokens / 1_000_000) * self._get_input_price(model)
            + (output_tokens / 1_000_000) * self._get_output_price(model)
        )

        logger.info(
            "Polza LLM call",
            extra={
                "model": model,
                "purpose": purpose,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "cost_rub": round(cost_rub, 5),
                "latency_ms": elapsed_ms,
                "finish_reason": choice.finish_reason,
                "trace_id": get_trace_id(),
                "tool_calls_n": len(tool_calls_out),
            },
        )

        await self._log_usage(model, input_tokens, output_tokens, cost_rub, purpose)

        result: dict[str, Any] = {
            "content": content,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cost_rub": cost_rub,
            "cached": False,
            "tool_calls": tool_calls_out,
            "finish_reason": choice.finish_reason,
        }
        if include_reasoning and reasoning_text:
            result["reasoning"] = reasoning_text

        if use_cache:
            await self._save_cache(cache_key, model, result, ttl_seconds=cache_ttl_seconds)

        return result

    @staticmethod
    def _get_input_price(model: str) -> float:
        """Return input price in RUB per million tokens.

        Lookup в :data:`MODEL_PRICES_INPUT_RUB_PER_M`. Для незнакомой модели
        используется консервативный fallback 100 ₽/M (предполагаем medium-tier).
        """
        return MODEL_PRICES_INPUT_RUB_PER_M.get(model, 100.0)

    @staticmethod
    def _get_output_price(model: str) -> float:
        """Return output price in RUB per million tokens. См. _get_input_price."""
        return MODEL_PRICES_OUTPUT_RUB_PER_M.get(model, 100.0)

    async def embeddings(
        self,
        inputs: str | list[str],
        model: str = cfg.POLZA_MODEL_EMBEDDING,
        purpose: str = "rag_embedding",
    ) -> list[list[float]]:
        """Получить эмбеддинги через polza.ai /embeddings endpoint.

        Args:
            inputs: текст или список текстов.
            model: ID модели (по умолчанию qwen3-embedding-8b — 4096-d).
            purpose: метка для логов.

        Returns:
            Список векторов (по одному на текст). Пустой список при ошибке.
        """
        if not self._client or cfg.DISABLE_LLM:
            return []
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        if not texts:
            return []
        start_ms = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                self._client.embeddings.create(model=model, input=texts),
                timeout=cfg.POLZA_REQUEST_TIMEOUT_SEC,
            )
        except TimeoutError:
            logger.warning("Polza embeddings() timed out", extra={"model": model, "n": len(texts)})
            return []
        except Exception as exc:
            logger.warning(
                "Polza embeddings() failed",
                extra={"model": model, "n": len(texts), "error": str(exc)[:200]},
            )
            return []
        elapsed_ms = round((time.monotonic() - start_ms) * 1000)
        vectors = [list(d.embedding) for d in resp.data]
        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        cost = (input_tokens / 1_000_000.0) * self._get_input_price(model)
        logger.info(
            "Polza embeddings",
            extra={
                "model": model,
                "purpose": purpose,
                "n_inputs": len(texts),
                "input_tokens": input_tokens,
                "cost_rub": round(cost, 5),
                "latency_ms": elapsed_ms,
                "dim": len(vectors[0]) if vectors else 0,
            },
        )
        await self._log_usage(model, input_tokens, 0, cost, purpose)
        return vectors

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str = cfg.POLZA_MODEL_REACTIVE,
        max_tokens: int = 400,
        purpose: str = "",
        cache_ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Like chat() but automatically parses JSON from content.

        ``cache_ttl_seconds`` is forwarded to :meth:`chat` — see its docstring.
        """

        if messages and not any("json" in m.get("content", "").lower() for m in messages):
            messages = messages.copy()
            messages[-1] = {
                **messages[-1],
                "content": messages[-1]["content"] + "\n\nОтвечай строго в JSON.",
            }

        result = await self.chat(
            messages,
            model=model,
            max_tokens=max_tokens,
            purpose=purpose,
            use_cache=True,
            cache_ttl_seconds=cache_ttl_seconds,
        )

        content = result["content"]
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            parsed = json.loads(content)
        except Exception as exc:
            logger.warning(
                "JSON parse failed",
                extra={"model": model, "content_preview": content[:100], "error": str(exc)},
            )
            parsed = {}

        return {**result, "parsed": parsed}

_polza_client: PolzaClient | None = None

def get_polza_client() -> PolzaClient:
    """Get polza client."""
    global _polza_client
    if _polza_client is None:
        _polza_client = PolzaClient()
    return _polza_client
