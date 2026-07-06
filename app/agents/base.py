"""Базовый класс адаптера BaseAdapter."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from app.dispatcher.signal import UnifiedSignal
from app.utils.logging import get_logger

logger = get_logger(__name__)

class BaseAdapter(ABC):
    """Abstract base class for all trading model adapters."""

    name: str = "base"

    def __init__(self) -> None:
        """Init."""
        self._started = False
        self._call_count = 0
        self._error_count = 0

    @abstractmethod
    async def startup(self) -> None:
        """Initialize: load ML models, connect to data sources, warm caches."""
        ...

    @abstractmethod
    async def poll(self) -> list[UnifiedSignal]:
        """Run one dispatcher cycle. Returns: list[UnifiedSignal] (may be empty)."""
        ...

    async def shutdown(self) -> None:
        """Optional cleanup (close connections, flush buffers)."""
        logger.info("Adapter shutdown", extra={"adapter": self.name})

    async def safe_poll(self, timeout: float = 0.5) -> list[UnifiedSignal]:
        """Wrapper called by Dispatcher; catches exceptions and enforces timeout.

        Args:
            timeout: max seconds for poll().
        Returns:
            list[UnifiedSignal]: empty on error/timeout.
        """
        import time

        start = time.monotonic()
        try:
            signals = await asyncio.wait_for(self.poll(), timeout=timeout)
            elapsed = time.monotonic() - start
            self._call_count += 1
            if elapsed > timeout * 0.8:
                logger.warning(
                    "Adapter poll slow",
                    extra={
                        "adapter": self.name,
                        "elapsed_ms": round(elapsed * 1000),
                        "timeout_ms": round(timeout * 1000),
                    },
                )
            else:
                logger.debug(
                    "Adapter poll OK",
                    extra={
                        "adapter": self.name,
                        "signals": len(signals),
                        "elapsed_ms": round(elapsed * 1000),
                    },
                )
            return signals
        except TimeoutError:
            self._error_count += 1
            logger.error(
                "Adapter poll timed out",
                extra={"adapter": self.name, "timeout_ms": round(timeout * 1000)},
            )
            return []
        except Exception as exc:
            self._error_count += 1
            logger.error(
                "Adapter poll exception",
                extra={"adapter": self.name, "error": str(exc)},
                exc_info=True,
            )
            return []

    @property
    def stats(self) -> dict[str, Any]:
        """Stats."""
        return {
            "adapter": self.name,
            "calls": self._call_count,
            "errors": self._error_count,
            "error_rate": self._error_count / max(1, self._call_count),
        }
