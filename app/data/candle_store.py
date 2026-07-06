"""In-memory кэш свечей."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

SUPPORTED_INTERVALS: tuple[int, ...] = (1, 5, 10, 60, 24)

MAX_CANDLES = 200

class CandleStore:
    """Thread-safe in-memory candle cache."""

    def __init__(self) -> None:
        """Init."""

        self._store: dict[int, dict[str, Any]] = {
            iv: {} for iv in SUPPORTED_INTERVALS
        }

        self._last_update: dict[tuple[str, int], datetime] = {}

        self._locks: dict[tuple[str, int], asyncio.Lock] = {}

        self._update_count: int = 0
        self._miss_count: int = 0

    def _get_lock(self, ticker: str, interval: int) -> asyncio.Lock:
        """Get lock."""
        key = (ticker, interval)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def update(
        self,
        ticker: str,
        interval: int,
        df: "pd.DataFrame | list",
    ) -> None:
        """Atomically replace candles for a (ticker, interval). Trims to MAX_CANDLES."""
        if interval not in self._store:
            logger.warning(
                "CandleStore: unsupported interval, ignoring",
                extra={"ticker": ticker, "interval": interval},
            )
            return

        ticker = ticker.upper()

        async with self._get_lock(ticker, interval):
            if _HAS_PANDAS and isinstance(df, pd.DataFrame):
                if len(df) > MAX_CANDLES:
                    df = df.tail(MAX_CANDLES).reset_index(drop=True)
                self._store[interval][ticker] = df
                rows = len(df)
            else:
                data = list(df)[-MAX_CANDLES:]
                self._store[interval][ticker] = data
                rows = len(data)

            self._last_update[(ticker, interval)] = datetime.now(tz=timezone.utc)
            self._update_count += 1

        logger.debug(
            "CandleStore updated",
            extra={"ticker": ticker, "interval": interval, "rows": rows},
        )

    def get(
        self,
        ticker: str,
        interval: int = 5,
    ) -> "pd.DataFrame | list":
        """Return cached candles for (ticker, interval). Empty DataFrame if missing."""
        ticker = ticker.upper()

        if interval not in self._store:
            self._miss_count += 1
            return pd.DataFrame() if _HAS_PANDAS else []

        result = self._store[interval].get(ticker)
        if result is None:
            self._miss_count += 1
            return pd.DataFrame() if _HAS_PANDAS else []

        return result

    def get_last_price(self, ticker: str, interval: int = 5) -> float | None:
        """Return the most recent close price, or None if no data."""
        df = self.get(ticker, interval)
        if _HAS_PANDAS and isinstance(df, pd.DataFrame):
            if df.empty or "close" not in df.columns:
                return None
            return float(df["close"].iloc[-1])
        if isinstance(df, list) and df:
            row = df[-1]
            return float(row.get("close", 0)) or None
        return None

    def age_seconds(self, ticker: str, interval: int = 5) -> float:
        """Seconds since the last update. Returns inf if never updated."""
        key = (ticker.upper(), interval)
        ts = self._last_update.get(key)
        if ts is None:
            return float("inf")
        return (datetime.now(tz=timezone.utc) - ts).total_seconds()

    def is_stale(self, ticker: str, interval: int = 5, max_age: int = 120) -> bool:
        """True if candles are older than max_age seconds."""
        return self.age_seconds(ticker, interval) > max_age

    def all_tickers_for_interval(self, interval: int = 5) -> list[str]:
        """Return tickers that have data for this interval."""
        return list(self._store.get(interval, {}).keys())

    def stats(self) -> dict:
        """Return cache statistics for monitoring."""
        populated: dict[int, int] = {}
        for iv in SUPPORTED_INTERVALS:
            populated[iv] = sum(
                1 for v in self._store[iv].values()
                if (isinstance(v, list) and v) or
                   (_HAS_PANDAS and isinstance(v, pd.DataFrame) and not v.empty)
            )
        return {
            "update_count": self._update_count,
            "miss_count": self._miss_count,
            "populated": populated,
            "tickers_5m": populated.get(5, 0),
        }

    async def warm_up(
        self,
        tickers: list[str] | None = None,
        intervals: tuple[int, ...] = (1, 5),
    ) -> None:
        """Pre-populate the store by fetching candles from ISS."""
        from app.data.iss_client import get_iss_client

        if tickers is None:
            tickers = cfg.TICKERS

        iss = get_iss_client()
        if not iss._started:
            await iss.startup()

        logger.info(
            "CandleStore warm-up start",
            extra={"tickers": len(tickers), "intervals": intervals},
        )

        for interval in intervals:
            results = await iss.get_candles_multi(tickers, interval=interval)
            for ticker, df in results.items():
                if (
                    (_HAS_PANDAS and isinstance(df, pd.DataFrame) and not df.empty)
                    or (isinstance(df, list) and df)
                ):
                    await self.update(ticker, interval, df)

        s = self.stats()
        logger.info(
            "CandleStore warm-up done",
            extra={
                "populated_5m": s["tickers_5m"],
                "populated_1m": s["populated"].get(1, 0),
            },
        )

_candle_store: CandleStore | None = None

def get_candle_store() -> CandleStore:
    """Get candle store."""
    global _candle_store
    if _candle_store is None:
        _candle_store = CandleStore()
    return _candle_store
