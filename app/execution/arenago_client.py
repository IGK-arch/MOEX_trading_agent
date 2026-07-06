"""Клиент ArenaGo paper-trading API."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import app.config as cfg
from app.dispatcher.signal import ArenaGoError
from app.utils.logging import get_logger, get_trace_id

logger = get_logger(__name__)

ARENAGO_CIRCUIT_BREAK_SEC: float = 60.0
ARENAGO_5XX_RETRY_ATTEMPTS: int = 3
ARENAGO_5XX_BACKOFF_BASE: float = 0.5
ARENAGO_5XX_BACKOFF_MAX: float = 4.0

try:
    import httpx  # type: ignore

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    import aiosqlite  # type: ignore

    _HAS_AIOSQLITE = True
except ImportError:
    _HAS_AIOSQLITE = False

class SubmitResult:
    """Result from ArenaGo submit_order endpoint."""

    __slots__ = (
        "success",
        "message",
        "order_value",
        "price",
        "quantity",
        "remaining_cash",
        "decision_id",
        "arena_error",
    )

    def __init__(
        self,
        success: bool,
        message: str,
        order_value: float = 0.0,
        price: float = 0.0,
        quantity: int = 0,
        remaining_cash: float = 0.0,
        decision_id: str = "",
    ) -> None:
        """Init."""
        self.success = success
        self.message = message
        self.order_value = order_value
        self.price = price
        self.quantity = quantity
        self.remaining_cash = remaining_cash
        self.decision_id = decision_id
        self.arena_error = self._parse_error(message) if not success else None

    @staticmethod
    def _parse_error(message: str) -> ArenaGoError:
        """Map a broker error message to ArenaGoError enum.

        Args:
            message: broker error text
        Returns:
            ArenaGoError: classified error
        """
        msg_upper = message.upper()
        if "MARKET CLOSED" in msg_upper:
            return ArenaGoError.MARKET_CLOSED
        if "NOT VALID SECID" in msg_upper:
            return ArenaGoError.NOT_VALID_SECID
        if "INSUFFICIENT CASH" in msg_upper:
            return ArenaGoError.INSUFFICIENT_CASH
        if "DAILY TRADE LIMIT" in msg_upper or "HAS REACHED" in msg_upper:
            return ArenaGoError.DAILY_TRADE_LIMIT
        return ArenaGoError.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        """Return serializable dict.

        Returns:
            dict[str, Any]: dict representation
        """
        return {
            "success": self.success,
            "message": self.message,
            "order_value": self.order_value,
            "price": self.price,
            "quantity": self.quantity,
            "remaining_cash": self.remaining_cash,
            "decision_id": self.decision_id,
            "arena_error": self.arena_error.value if self.arena_error else None,
        }

class ArenaGoClient:
    """Async ArenaGo API client."""

    BASE_URL = cfg.ARENAGO_BASE_URL

    def __init__(self) -> None:
        """Init."""
        self._client: Any = None
        self._api_key = ""
        self._bot_name = ""
        self._db_path = cfg.DATA_DIR / "decisions.db"
        self._daily_trade_count = 0
        self._started = False
        self._circuit_open_until: float = 0.0
        self._consecutive_5xx: int = 0
        self._last_known_cash: float | None = None
        self._reauth_attempts: int = 0
        self._submit_lock = asyncio.Lock()
        self._last_submit_ts: float = 0.0
        self._min_submit_pause_sec: float = float(
            os.getenv("ARENAGO_MIN_SUBMIT_PAUSE_SEC", "0.6")
        )

    def _positions_endpoint(self) -> tuple[str, dict[str, str] | None]:
        """Return (url, params) for the positions endpoint per URL variant.

        v1 (default): GET /api/positions?bot=<bot_name>   — legacy
        v2 (docs):    GET /api/positions/<bot_name>       — path-param

        Returns:
            tuple[str, dict[str, str] | None]: full URL and optional query params
        """
        if cfg.ARENAGO_URL_VARIANT == "v2" and self._bot_name:
            return f"{self.BASE_URL}/api/positions/{quote(self._bot_name, safe='')}", None
        params = {"bot": self._bot_name} if self._bot_name else None
        return f"{self.BASE_URL}/api/positions", params

    def _trades_endpoint(self) -> tuple[str, dict[str, str] | None]:
        """Return (url, params) for the trades endpoint per URL variant.

        v1 (default): GET /api/trades?bot=<bot_name>   — legacy
        v2 (docs):    GET /api/trades/<bot_name>       — path-param

        Returns:
            tuple[str, dict[str, str] | None]: full URL and optional query params
        """
        if cfg.ARENAGO_URL_VARIANT == "v2" and self._bot_name:
            return f"{self.BASE_URL}/api/trades/{quote(self._bot_name, safe='')}", None
        params = {"bot": self._bot_name} if self._bot_name else None
        return f"{self.BASE_URL}/api/trades", params

    async def startup(self) -> None:
        """Initialise HTTP client and probe broker bots endpoint."""
        self._api_key = os.getenv("SANDBOX_API_KEY", "")
        self._bot_name = os.getenv("ARENAGO_BOT_NAME", cfg.ARENAGO_BOT_NAME)

        if not self._api_key:
            logger.error("SANDBOX_API_KEY not set — ArenaGo client will not work")
            return

        if not _HAS_HTTPX:
            logger.error("httpx not installed")
            return

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.HTTP_TIMEOUT),
            headers={
                "Authorization": self._api_key,
                "Content-Type": "application/json",
                "User-Agent": "MoexML-Trader/0.1",
            },
        )
        self._started = True

        bots = await self.get_bots()
        if bots:
            bot = next((b for b in bots if b.get("name") == self._bot_name), bots[0])
            cash = bot.get("cash_balance", 0)
            logger.info(
                "ArenaGo подключён",
                extra={
                    "bot": self._bot_name,
                    "cash_balance": cash,
                    "available_bots": [b["name"] for b in bots],
                },
            )
        else:
            logger.warning("ArenaGo startup: could not fetch bots")

    async def shutdown(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
        logger.info("ArenaGo client stopped")

    def _circuit_is_open(self) -> bool:
        """Return True when local breaker is tripped.

        Returns:
            bool: breaker open state
        """
        return time.monotonic() < self._circuit_open_until

    def _circuit_seconds_remaining(self) -> float:
        """Return seconds left on the open breaker.

        Returns:
            float: seconds (>=0)
        """
        return max(0.0, self._circuit_open_until - time.monotonic())

    def _trip_circuit(self, reason: str) -> None:
        """Open the breaker for ARENAGO_CIRCUIT_BREAK_SEC.

        Args:
            reason: human-readable trip reason
        """
        self._circuit_open_until = time.monotonic() + ARENAGO_CIRCUIT_BREAK_SEC
        logger.critical(
            "ArenaGo circuit breaker tripped — pausing submits",
            extra={
                "reason": reason,
                "consecutive_5xx": self._consecutive_5xx,
                "break_for_sec": ARENAGO_CIRCUIT_BREAK_SEC,
            },
        )

    def _reset_circuit(self) -> None:
        """Reset breaker counters after a successful call."""
        if self._consecutive_5xx or self._circuit_open_until:
            logger.info(
                "ArenaGo circuit breaker reset (broker reachable again)",
                extra={"prev_consecutive_5xx": self._consecutive_5xx},
            )
        self._consecutive_5xx = 0
        self._circuit_open_until = 0.0

    async def _reauth_on_401(self) -> bool:
        """Re-read API key from env and rebuild client.

        Returns:
            bool: True if reauth succeeded
        """
        if self._reauth_attempts >= 5:
            logger.critical(
                "ArenaGo re-auth attempts exhausted — likely permanent 401",
                extra={"attempts": self._reauth_attempts},
            )
            return False
        self._reauth_attempts += 1
        fresh_key = os.getenv("SANDBOX_API_KEY", "")
        if not fresh_key:
            logger.error(
                "ArenaGo got 401 but SANDBOX_API_KEY is empty — cannot re-auth",
                extra={"attempt": self._reauth_attempts},
            )
            return False
        try:
            if self._client:
                await self._client.aclose()
        except Exception:
            pass
        self._api_key = fresh_key
        if _HAS_HTTPX:
            self._client = httpx.AsyncClient(  # type: ignore[union-attr]
                timeout=httpx.Timeout(cfg.HTTP_TIMEOUT),
                headers={
                    "Authorization": self._api_key,
                    "Content-Type": "application/json",
                    "User-Agent": "MoexML-Trader/0.1",
                },
            )
        logger.warning(
            "ArenaGo re-authenticated after 401",
            extra={"attempt": self._reauth_attempts},
        )
        return True

    def _check_cash_drift(self, new_cash: float) -> None:
        """Log when cash drops unexpectedly between calls.

        Args:
            new_cash: latest cash value
        """
        if self._last_known_cash is None:
            self._last_known_cash = new_cash
            return
        delta = new_cash - self._last_known_cash
        if delta < -50_000:
            logger.critical(
                "ArenaGo cash drop detected — possible silent broker action",
                extra={
                    "previous_cash": self._last_known_cash,
                    "new_cash": new_cash,
                    "delta": delta,
                    "trace_id": get_trace_id(),
                },
            )
        self._last_known_cash = new_cash

    async def _is_already_executed(self, decision_id: str) -> SubmitResult | None:
        """Return cached result if decision was already executed.

        Args:
            decision_id: id to look up
        Returns:
            SubmitResult | None: cached result or None
        """
        if not _HAS_AIOSQLITE:
            return None
        try:
            async with (
                aiosqlite.connect(self._db_path) as db,
                db.execute(
                    "SELECT arena_response_json FROM decisions "
                    "WHERE decision_id = ? AND executed_bool = 1",
                    (decision_id,),
                ) as cur,
            ):
                row = await cur.fetchone()
                if row and row[0]:
                    cached = json.loads(row[0])
                    logger.info(
                        "Order idempotency hit (skipping duplicate)",
                        extra={"decision_id": decision_id},
                    )
                    return SubmitResult(
                        success=cached.get("success", True),
                        message=cached.get("message", "cached"),
                        order_value=cached.get("order_value", 0),
                        price=cached.get("price", 0),
                        quantity=cached.get("quantity", 0),
                        remaining_cash=cached.get("remaining_cash", 0),
                        decision_id=decision_id,
                    )
        except Exception as exc:
            logger.warning("Idempotency check failed", extra={"error": str(exc)})
        return None

    async def _mark_executed(self, decision_id: str, result: SubmitResult) -> None:
        """Mark decision as executed in decisions.db.

        Args:
            decision_id: decision id
            result: broker result to cache
        """
        if not _HAS_AIOSQLITE:
            return
        now_str = datetime.now(tz=UTC).isoformat()
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    "UPDATE decisions SET executed_bool=1, arena_response_json=?, "
                    "executed_at=? WHERE decision_id=?",
                    (json.dumps(result.to_dict()), now_str, decision_id),
                )
                await db.commit()
        except Exception as exc:
            logger.error("Failed to mark decision executed", extra={"error": str(exc)})

    async def submit_order(
        self,
        direction: str,
        ticker: str,
        quantity: int,
        decision_id: str,
        bot: str | None = None,
        is_exit: bool = False,
    ) -> SubmitResult:
        """Submit a market order with retries and circuit breaker.

        Args:
            direction: BUY or SELL
            ticker: instrument code
            quantity: lot count
            decision_id: unique decision id
            bot: bot name (defaults to env)
            is_exit: True if this is an exit order
        Returns:
            SubmitResult: order outcome
        """
        if not self._client:
            raise RuntimeError("ArenaGoClient not started")

        async with self._submit_lock:
            now_ts = asyncio.get_event_loop().time()
            delta = now_ts - self._last_submit_ts
            if delta < self._min_submit_pause_sec:
                await asyncio.sleep(self._min_submit_pause_sec - delta)
            self._last_submit_ts = asyncio.get_event_loop().time()

        if self._circuit_is_open():
            remaining = self._circuit_seconds_remaining()
            logger.warning(
                "ArenaGo circuit breaker OPEN — skipping submit",
                extra={
                    "decision_id": decision_id,
                    "ticker": ticker,
                    "circuit_remaining_sec": round(remaining, 1),
                },
            )
            return SubmitResult(
                success=False,
                message=f"CIRCUIT_BREAKER_OPEN: {remaining:.1f}s remaining",
                decision_id=decision_id,
            )

        if self._daily_trade_count >= cfg.ARENAGO_DAILY_TRADE_LIMIT:
            logger.critical(
                "ArenaGo daily trade HARD limit reached — rejecting",
                extra={
                    "count": self._daily_trade_count,
                    "limit": cfg.ARENAGO_DAILY_TRADE_LIMIT,
                    "is_exit": is_exit,
                },
            )
            return SubmitResult(
                success=False,
                message=f"LOCAL_HARD_LIMIT: {self._daily_trade_count}/{cfg.ARENAGO_DAILY_TRADE_LIMIT}",
                decision_id=decision_id,
            )
        if self._daily_trade_count >= cfg.ARENAGO_DAILY_TRADE_ENTRY_HALT and not is_exit:
            logger.critical(
                "ArenaGo daily trade ENTRY HALT — allowing exits only",
                extra={
                    "count": self._daily_trade_count,
                    "entry_halt": cfg.ARENAGO_DAILY_TRADE_ENTRY_HALT,
                    "hard_limit": cfg.ARENAGO_DAILY_TRADE_LIMIT,
                },
            )
            return SubmitResult(
                success=False,
                message=(
                    f"ENTRY_HALT_AT_{self._daily_trade_count}"
                    f"/{cfg.ARENAGO_DAILY_TRADE_LIMIT}: exits-only mode"
                ),
                decision_id=decision_id,
            )
        if self._daily_trade_count >= cfg.ARENAGO_DAILY_TRADE_SLOWDOWN:
            logger.warning(
                "Daily trade limit near (slowdown threshold) — high-confidence only",
                extra={
                    "count": self._daily_trade_count,
                    "slowdown_threshold": cfg.ARENAGO_DAILY_TRADE_SLOWDOWN,
                    "entry_halt": cfg.ARENAGO_DAILY_TRADE_ENTRY_HALT,
                    "is_exit": is_exit,
                },
            )
        elif self._daily_trade_count >= cfg.ARENAGO_DAILY_TRADE_SOFT_LIMIT:
            logger.warning(
                "Daily trade limit approaching",
                extra={
                    "count": self._daily_trade_count,
                    "limit": cfg.ARENAGO_DAILY_TRADE_SOFT_LIMIT,
                },
            )

        cached = await self._is_already_executed(decision_id)
        if cached:
            return cached

        arena_direction = {"BUY": "B", "SELL": "S"}.get(direction.upper(), "B")
        if direction.upper() not in ("BUY", "SELL"):
            logger.error(
                "Invalid direction",
                extra={"direction": direction, "decision_id": decision_id},
            )
            return SubmitResult(
                success=False,
                message=f"Invalid direction: {direction}",
                decision_id=decision_id,
            )

        bot_name = bot or self._bot_name
        body = {
            "direction": arena_direction,
            "secid": ticker.upper(),
            "quantity": quantity,
            "bot": bot_name,
        }

        start_ms = time.monotonic()
        last_exc: Exception | None = None
        net_reset_replayed: bool = False
        reauth_replayed: bool = False
        response = None

        for attempt in range(1, ARENAGO_5XX_RETRY_ATTEMPTS + 2):
            cached = await self._is_already_executed(decision_id)
            if cached:
                return cached
            try:
                response = await self._client.post(
                    f"{self.BASE_URL}/api/submit_order",
                    json=body,
                )
            except Exception as exc:
                last_exc = exc
                if not net_reset_replayed:
                    net_reset_replayed = True
                    logger.warning(
                        "ArenaGo connection error — resubmitting once (idempotent via decision_id)",
                        extra={
                            "decision_id": decision_id,
                            "ticker": ticker,
                            "error": str(exc),
                            "attempt": attempt,
                        },
                    )
                    await asyncio.sleep(0.25)
                    continue
                logger.error(
                    "ArenaGo submit_order HTTP error (after replay)",
                    extra={
                        "decision_id": decision_id,
                        "ticker": ticker,
                        "error": str(exc),
                        "trace_id": get_trace_id(),
                    },
                )
                self._consecutive_5xx += 1
                if self._consecutive_5xx >= ARENAGO_5XX_RETRY_ATTEMPTS:
                    self._trip_circuit(f"network_error: {exc.__class__.__name__}")
                return SubmitResult(
                    success=False,
                    message=f"NETWORK_ERROR: {str(exc)[:100]}",
                    decision_id=decision_id,
                )

            status = getattr(response, "status_code", 0)

            if status == 401 and not reauth_replayed:
                reauth_replayed = True
                if await self._reauth_on_401():
                    continue
                return SubmitResult(
                    success=False,
                    message="HTTP_401_REAUTH_FAILED",
                    decision_id=decision_id,
                )

            if 500 <= status <= 599:
                self._consecutive_5xx += 1
                if attempt <= ARENAGO_5XX_RETRY_ATTEMPTS:
                    backoff = min(
                        ARENAGO_5XX_BACKOFF_BASE * (2 ** (attempt - 1)),
                        ARENAGO_5XX_BACKOFF_MAX,
                    )
                    logger.warning(
                        "ArenaGo 5xx — retrying after backoff",
                        extra={
                            "decision_id": decision_id,
                            "ticker": ticker,
                            "status": status,
                            "attempt": attempt,
                            "backoff_sec": backoff,
                        },
                    )
                    await asyncio.sleep(backoff)
                    continue
                self._trip_circuit(f"5xx_exhausted: status={status}")
                return SubmitResult(
                    success=False,
                    message=f"HTTP_{status}_RETRIES_EXHAUSTED",
                    decision_id=decision_id,
                )

            break

        if response is None:
            logger.error(
                "ArenaGo submit_order: no response after retry loop",
                extra={
                    "decision_id": decision_id,
                    "ticker": ticker,
                    "last_error": str(last_exc) if last_exc else "?",
                },
            )
            return SubmitResult(
                success=False,
                message=f"NO_RESPONSE: {str(last_exc)[:80] if last_exc else 'unknown'}",
                decision_id=decision_id,
            )

        elapsed_ms = round((time.monotonic() - start_ms) * 1000)

        try:
            response.raise_for_status()
        except Exception as exc:
            logger.error(
                "ArenaGo submit_order вернул не-2xx",
                extra={
                    "ticker": ticker,
                    "decision_id": decision_id,
                    "status": getattr(response, "status_code", "?"),
                    "body_excerpt": (getattr(response, "text", "") or "")[:200],
                    "error": str(exc),
                },
            )
            return SubmitResult(
                success=False,
                message=f"HTTP_{getattr(response, 'status_code', 'ERR')}: {str(exc)[:100]}",
                decision_id=decision_id,
            )

        body_text = getattr(response, "text", "") or ""
        if not body_text.strip():
            logger.error(
                "ArenaGo submit_order EMPTY response body",
                extra={
                    "ticker": ticker,
                    "decision_id": decision_id,
                    "status": getattr(response, "status_code", "?"),
                },
            )
            return SubmitResult(
                success=False,
                message="EMPTY_RESPONSE_BODY",
                decision_id=decision_id,
            )

        try:
            data = response.json()
        except Exception as exc:
            logger.error(
                "ArenaGo submit_order non-JSON response",
                extra={
                    "ticker": ticker,
                    "decision_id": decision_id,
                    "body_excerpt": body_text[:200],
                    "error": str(exc),
                },
            )
            return SubmitResult(
                success=False,
                message=f"INVALID_JSON: {str(exc)[:100]}",
                decision_id=decision_id,
            )

        if not isinstance(data, dict):
            logger.error(
                "ArenaGo submit_order returned non-dict JSON",
                extra={
                    "ticker": ticker,
                    "decision_id": decision_id,
                    "type": type(data).__name__,
                    "body_excerpt": body_text[:200],
                },
            )
            return SubmitResult(
                success=False,
                message="MALFORMED_JSON_NOT_DICT",
                decision_id=decision_id,
            )

        result = SubmitResult(
            success=bool(data.get("success", False)),
            message=str(data.get("message", "")),
            order_value=float(data.get("order_value", 0) or 0),
            price=float(data.get("price", 0) or 0),
            quantity=int(data.get("quantity", quantity) or quantity),
            remaining_cash=float(data.get("remaining_cash", 0) or 0),
            decision_id=decision_id,
        )

        if result.success:
            self._reset_circuit()
            self._daily_trade_count += 1
            self._check_cash_drift(result.remaining_cash)
            await self._mark_executed(decision_id, result)
            logger.info(
                "Ордер исполнен",
                extra={
                    "ticker": ticker,
                    "direction": direction,
                    "quantity": quantity,
                    "price": result.price,
                    "order_value": result.order_value,
                    "remaining_cash": result.remaining_cash,
                    "decision_id": decision_id,
                    "latency_ms": elapsed_ms,
                    "trace_id": get_trace_id(),
                },
            )
        else:
            logger.error(
                "Order rejected",
                extra={
                    "ticker": ticker,
                    "direction": direction,
                    "decision_id": decision_id,
                    "arena_error": result.arena_error.value if result.arena_error else "?",
                    "message": result.message,
                    "latency_ms": elapsed_ms,
                },
            )

        return result

    async def get_trades(self) -> list[dict[str, Any]]:
        """Fetch today's trades for our bot.

        Returns:
            list[dict[str, Any]]: trades or empty list
        """
        if not self._client:
            return []
        try:
            url, params = self._trades_endpoint()
            r = await self._client.get(url, params=params)
            r.raise_for_status()
            return r.json() if isinstance(r.json(), list) else []
        except Exception as exc:
            logger.debug(
                "get_trades unavailable (expected on ArenaGo sandbox)", extra={"error": str(exc)}
            )
            return []

    async def get_positions(self) -> list[dict[str, Any]]:
        """Fetch current open positions.

        Returns:
            list[dict[str, Any]]: positions or empty list
        """
        if not self._client:
            return []
        try:
            url, params = self._positions_endpoint()
            r = await self._client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.debug(
                "get_positions unavailable (expected on ArenaGo sandbox)", extra={"error": str(exc)}
            )
            return []

    async def get_positions_safe(
        self,
        *,
        max_retries: int = 5,
        base_backoff_sec: float = 1.0,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return (positions, confirmed) tuple with retry-with-backoff.

        v0.19.4: при 404/transient errors делаем до max_retries попыток с
        exponential backoff. Если ВСЕ попытки провалились — CRITICAL log,
        возвращаем (empty, False) — local state preserved.

        v0.19.6 (filter-audit, 27 May 2026): когда `/api/positions` отдаёт
        404 (поведение ArenaGo sandbox на нескольких ботах), пробуем
        FALLBACK через `/api/trades` (накапливаем cumulative qty per ticker).
        Если и trades упали — пробуем cached `recovery_state.json` (если он
        не старше 1 часа). Это гасит "DD 82% feedback loop" когда
        positions API ломается, но реальные позиции на бирже есть.
        flag `confirmed`:
          - True  → данные взяты прямо из `/api/positions`
          - True  → данные восстановлены из `/api/trades` (FIFO)
          - False → возвращаем последний снимок из recovery_state (стейл)

        Args:
            max_retries: количество попыток (1..N)
            base_backoff_sec: базовая задержка (растёт ×2 каждую попытку)
        Returns:
            tuple[list[dict[str, Any]], bool]: (positions, broker_confirmed)
        """
        if not self._client:
            return [], False

        try:
            from app.risk.broker_health_monitor import get_broker_health_monitor

            health = get_broker_health_monitor()
        except Exception:  # pragma: no cover
            health = None

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                url, params = self._positions_endpoint()
                r = await self._client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, list):
                    logger.warning(
                        "get_positions_safe: non-list body, treating as transient",
                        extra={"type": type(data).__name__, "attempt": attempt},
                    )
                    last_exc = ValueError("non-list body")
                    if attempt < max_retries:
                        await asyncio.sleep(base_backoff_sec * (2 ** (attempt - 1)))
                    continue
                if attempt > 1:
                    logger.info(
                        "get_positions_safe: recovered after retries",
                        extra={"attempt": attempt, "n_positions": len(data)},
                    )
                if health is not None:
                    health.on_positions_success()
                return data, True
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    backoff = base_backoff_sec * (2 ** (attempt - 1))
                    logger.warning(
                        "get_positions_safe: attempt failed, retrying",
                        extra={
                            "attempt": attempt,
                            "max_retries": max_retries,
                            "backoff_sec": backoff,
                            "error": str(exc),
                        },
                    )
                    await asyncio.sleep(backoff)
        logger.critical(
            "get_positions_safe: ALL retries failed — trying trades+recovery fallback",
            extra={
                "max_retries": max_retries,
                "last_error": str(last_exc),
                "bot": self._bot_name,
            },
        )
        if health is not None:
            health.on_positions_fail()

        try:
            trades = await self.get_trades()
        except Exception as exc:
            logger.warning("Fallback get_trades failed", extra={"error": str(exc)})
            trades = []
        if trades:
            reconstructed = self._reconstruct_positions_from_trades(trades)
            if reconstructed:
                logger.warning(
                    "get_positions_safe: rebuilt positions from /api/trades",
                    extra={"n_positions": len(reconstructed), "source": "trades_fallback"},
                )
                if health is not None:
                    health.on_positions_success()
                return reconstructed, True

        cached, age_sec = self._load_recovery_positions()
        if cached:
            stale = age_sec > 3600.0
            if stale:
                logger.critical(
                    "get_positions_safe: using STALE recovery_state cache",
                    extra={
                        "n_positions": len(cached),
                        "age_sec": round(age_sec, 1),
                        "stale_threshold_sec": 3600.0,
                    },
                )
            else:
                logger.warning(
                    "get_positions_safe: using fresh recovery_state cache",
                    extra={"n_positions": len(cached), "age_sec": round(age_sec, 1)},
                )
            return cached, False

        return [], False

    @staticmethod
    def _reconstruct_positions_from_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Aggregate cumulative net qty + VWAP per ticker from a trade list.

        Args:
            trades: list of broker trade dicts (must carry secid/direction/
                    quantity/price OR `ticker`/`side`/`qty`/`price`).

        Returns:
            list[dict[str, Any]]: ArenaGo-shaped positions
                (secid, position, average_price, bot) for tickers with
                net |qty| > 0. Empty when no usable rows.
        """
        agg: dict[str, dict[str, float]] = {}
        for t in trades or []:
            ticker = str(t.get("secid") or t.get("ticker") or t.get("symbol") or "").upper()
            if not ticker:
                continue
            direction = str(t.get("direction") or t.get("side") or "").upper()
            qty = int(t.get("quantity") or t.get("qty") or 0)
            if qty <= 0:
                continue
            price = float(t.get("price") or 0.0)
            if direction in ("B", "BUY", "ПОКУПКА", "LONG"):
                signed = qty
                buy_qty = qty
                buy_notional = qty * price
            elif direction in ("S", "SELL", "ПРОДАЖА", "SHORT"):
                signed = -qty
                buy_qty = 0
                buy_notional = 0.0
            else:
                continue
            row = agg.setdefault(
                ticker,
                {"net_qty": 0.0, "buy_qty": 0.0, "buy_notional": 0.0},
            )
            row["net_qty"] += signed
            row["buy_qty"] += buy_qty
            row["buy_notional"] += buy_notional
        out: list[dict[str, Any]] = []
        for ticker, row in agg.items():
            net = int(row["net_qty"])
            if net == 0:
                continue
            vwap = row["buy_notional"] / row["buy_qty"] if row["buy_qty"] > 0 else 0.0
            out.append(
                {
                    "secid": ticker,
                    "position": net,
                    "average_price": vwap,
                    "bot": "",
                }
            )
        return out

    def _load_recovery_positions(self) -> tuple[list[dict[str, Any]], float]:
        """Read open_positions from recovery_state.json with age in seconds.

        Returns:
            tuple[list[dict[str, Any]], float]: positions and age_sec.
                Empty list + age=inf when cache unreadable or empty.
        """
        try:
            path = cfg.RECOVERY_STATE_PATH
            if not path.exists():
                return [], float("inf")
            with open(path, encoding="utf-8") as fh:
                state = json.load(fh)
            saved_ts = float(state.get("last_save_ts_utc", 0) or 0)
            age_sec = max(0.0, time.time() - saved_ts) if saved_ts else float("inf")
            raw = state.get("open_positions") or []
            out: list[dict[str, Any]] = []
            for p in raw:
                ticker = str(p.get("ticker") or p.get("secid") or "").upper()
                qty = int(p.get("quantity") or p.get("position") or 0)
                price = float(p.get("avg_price") or p.get("average_price") or 0.0)
                if not ticker or qty == 0:
                    continue
                out.append(
                    {
                        "secid": ticker,
                        "position": qty,
                        "average_price": price,
                        "bot": str(p.get("bot", "")),
                    }
                )
            return out, age_sec
        except Exception as exc:
            logger.debug("recovery_state load failed", extra={"error": str(exc)})
            return [], float("inf")

    async def get_bots(self) -> list[dict[str, Any]]:
        """Fetch bots and cash balances.

        Returns:
            list[dict[str, Any]]: bots or empty list
        """
        if not self._client:
            return []
        try:
            r = await self._client.get(f"{self.BASE_URL}/api/bots")
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.error("get_bots failed", extra={"error": str(exc)})
            return []

    async def get_cash_balance(self) -> float:
        """Return current cash balance for our bot.

        Returns:
            float: cash in RUB
        """
        bots = await self.get_bots()
        bot = next(
            (b for b in bots if b.get("name") == self._bot_name),
            bots[0] if bots else {},
        )
        return float(bot.get("cash_balance", 0.0))

    def reset_daily_counter(self) -> None:
        """Reset trade counter at midnight."""
        logger.info(
            "Daily trade counter reset",
            extra={"previous_count": self._daily_trade_count},
        )
        self._daily_trade_count = 0

_arenago_client: ArenaGoClient | None = None

def get_arenago_client() -> ArenaGoClient:
    """Return process-wide ArenaGoClient singleton.

    Returns:
        ArenaGoClient: shared instance
    """
    global _arenago_client
    if _arenago_client is None:
        _arenago_client = ArenaGoClient()
    return _arenago_client
