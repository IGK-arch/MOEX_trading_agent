"""Retry с экспоненциальной задержкой."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

BUSINESS_ERRORS: frozenset[str] = frozenset(
    [
        "INSUFFICIENT_CASH",
        "NOT VALID SECID",
        "MARKET CLOSED",
        "BOT",
    ]
)

def _is_business_error(exc: Exception) -> bool:
    """Return True if this error should NOT be retried."""
    msg = str(exc).upper()
    return any(be in msg for be in BUSINESS_ERRORS)

def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    backoff_max: float = 10.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator: retry async function with exponential backoff.

    Args:
        max_attempts: max number of attempts (including first try).
        backoff_base: base sleep seconds (doubles each retry).
        backoff_max: maximum sleep cap in seconds.
        exceptions: exception types to catch and retry on.
    Returns:
        Callable: decorated function.
    """

    def decorator(func: F) -> F:
        """Decorator."""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper."""
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if _is_business_error(exc):
                        logger.debug(
                            "Business error (no retry)",
                            extra={"fn": func.__name__, "error": str(exc)},
                        )
                        raise
                    if attempt == max_attempts:
                        logger.error(
                            "Max retries exhausted",
                            extra={
                                "fn": func.__name__,
                                "attempts": attempt,
                                "error": str(exc),
                            },
                        )
                        raise
                    sleep = min(backoff_base * (2 ** (attempt - 1)), backoff_max)
                    logger.warning(
                        "Retrying after error",
                        extra={
                            "fn": func.__name__,
                            "attempt": attempt,
                            "sleep_secs": sleep,
                            "error": str(exc),
                        },
                    )
                    await asyncio.sleep(sleep)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator

class RateLimiter:
    """Token-bucket rate limiter for API calls."""

    def __init__(
        self,
        requests_per_second: float | None = None,
        requests_per_minute: float | None = None,
    ) -> None:
        """Init."""
        if requests_per_second is not None:
            self._interval = 1.0 / requests_per_second
        elif requests_per_minute is not None:
            self._interval = 60.0 / requests_per_minute
        else:
            self._interval = 0.0

        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def acquire(self) -> None:
        """Wait if needed to respect rate limit."""
        if self._interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

class DailyQuotaTracker:
    """Track daily API quota usage."""

    def __init__(self, daily_limit: int, name: str = "api") -> None:
        """Init."""
        self._limit = daily_limit
        self._name = name
        self._count = 0
        self._reset_date = ""

    def _check_reset(self) -> None:
        """Check reset."""
        today = time.strftime("%Y-%m-%d")
        if today != self._reset_date:
            self._count = 0
            self._reset_date = today

    def try_acquire(self) -> bool:
        """Return True and increment counter if quota available."""
        self._check_reset()
        if self._count >= self._limit:
            logger.warning(
                "Daily quota exhausted",
                extra={"api": self._name, "count": self._count, "limit": self._limit},
            )
            return False
        self._count += 1
        remaining = self._limit - self._count
        if remaining <= self._limit * 0.1:
            logger.warning(
                "Daily quota low",
                extra={"api": self._name, "remaining": remaining, "limit": self._limit},
            )
        return True

    @property
    def remaining(self) -> int:
        """Remaining."""
        self._check_reset()
        return max(0, self._limit - self._count)
