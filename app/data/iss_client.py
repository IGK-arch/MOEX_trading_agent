"""Клиент MOEX ISS (свечи + sitenews)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any
from urllib.parse import quote_plus

import app.config as cfg
from app.utils.async_cache import SingleFlight, TTLCache
from app.utils.logging import get_logger, get_trace_id
from app.utils.retry import with_retry, RateLimiter

logger = get_logger(__name__)

_ISS_CANDLES_SHORT_CACHE: TTLCache = TTLCache(ttl_seconds=60.0, max_entries=1024)
_ISS_CANDLES_LONG_CACHE: TTLCache = TTLCache(ttl_seconds=300.0, max_entries=1024)
_ISS_CANDLES_FLIGHT: SingleFlight = SingleFlight()

def _pick_candles_cache(interval: int) -> TTLCache:
    """Short cache for intraday; long cache for hourly+ bars."""
    return _ISS_CANDLES_LONG_CACHE if interval >= 60 else _ISS_CANDLES_SHORT_CACHE

try:
    import httpx  # type: ignore
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

INTERVAL_1M = 1
INTERVAL_5M = 5
INTERVAL_10M = 10
INTERVAL_60M = 60
INTERVAL_24H = 24

class ISSClient:
    """Async MOEX ISS client for candle data (public endpoints, no auth)."""

    BASE_URL = cfg.ISS_BASE_URL
    BOARD = cfg.ISS_BOARD

    def __init__(self) -> None:
        """Init."""
        self._client: Any = None
        self._rate_limiter = RateLimiter(requests_per_second=5)
        self._semaphore = asyncio.Semaphore(10)
        self._started = False

    async def startup(self) -> None:
        """Startup."""
        if not _HAS_HTTPX:
            logger.error("httpx not installed — ISS client unavailable")
            return
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.HTTP_TIMEOUT),
            limits=httpx.Limits(
                max_keepalive_connections=cfg.HTTP_KEEPALIVE_CONNECTIONS,
                max_connections=cfg.HTTP_MAX_CONNECTIONS,
            ),
            headers={"User-Agent": "MoexML-Trader/0.1 (kirill.vygovov@yandex.ru)"},
        )
        self._started = True
        logger.info("ISS client started", extra={"base_url": self.BASE_URL})

    async def shutdown(self) -> None:
        """Shutdown."""
        if self._client:
            await self._client.aclose()
            self._started = False
        logger.info("ISS client stopped")

    async def get_candles(
        self,
        ticker: str,
        interval: int = INTERVAL_5M,
        from_dt: datetime | None = None,
        till_dt: datetime | None = None,
        limit: int = 5000,
        page_size: int = 500,
    ) -> "pd.DataFrame | list[dict]":
        """Fetch OHLCV candles from ISS. Paginates automatically.

        Args:
            ticker: MOEX SECID.
            interval: candle interval in minutes (1, 5, 10, 60, 24).
            from_dt: start datetime UTC.
            till_dt: end datetime UTC.
            limit: max rows total.
            page_size: ISS page size.
        Returns:
            pd.DataFrame | list[dict]: candles.
        """
        cache_eligible = from_dt is None and till_dt is None and limit == 5000

        if cache_eligible:
            cache = _pick_candles_cache(interval)
            key = ("candles", ticker.upper(), interval, page_size)
            hit = cache.get(key)
            if hit is not None:
                return hit

            return await _ISS_CANDLES_FLIGHT.do(
                key,
                lambda: self._get_candles_with_cache(
                    ticker, interval, page_size, limit, cache, key
                ),
            )

        return await self._get_candles_impl(
            ticker, interval, from_dt, till_dt, limit, page_size
        )

    async def _get_candles_with_cache(
        self,
        ticker: str,
        interval: int,
        page_size: int,
        limit: int,
        cache: TTLCache,
        key: tuple,
    ) -> "pd.DataFrame | list[dict]":
        """Get candles with cache."""
        second_look = cache.get(key)
        if second_look is not None:
            return second_look

        result = await self._get_candles_impl(
            ticker, interval, None, None, limit, page_size
        )

        is_empty = False
        if _HAS_PANDAS and isinstance(result, pd.DataFrame):
            is_empty = result.empty
        elif isinstance(result, list):
            is_empty = len(result) == 0
        if not is_empty:
            cache.set(key, result)
        return result

    @with_retry(max_attempts=3, backoff_base=0.5)
    async def _get_candles_impl(
        self,
        ticker: str,
        interval: int = INTERVAL_5M,
        from_dt: datetime | None = None,
        till_dt: datetime | None = None,
        limit: int = 5000,
        page_size: int = 500,
    ) -> "pd.DataFrame | list[dict]":
        """Underlying ISS fetch — public callers should use `get_candles`."""
        if not self._client:
            raise RuntimeError("ISSClient not started — call startup() first")

        now = datetime.now(tz=timezone.utc)
        if till_dt is None:
            till_dt = now
        if from_dt is None:
            if interval <= 1:
                from_dt = till_dt - timedelta(hours=2)
            elif interval <= 10:
                from_dt = till_dt - timedelta(hours=24)
            elif interval <= 60:
                from_dt = till_dt - timedelta(days=5)
            else:
                from_dt = till_dt - timedelta(days=60)

        from_enc = quote_plus(from_dt.strftime("%Y-%m-%d %H:%M:%S"))
        till_enc = quote_plus(till_dt.strftime("%Y-%m-%d %H:%M:%S"))

        base_url = (
            f"{self.BASE_URL}/engines/stock/markets/shares/boards/{self.BOARD}"
            f"/securities/{ticker}/candles.json"
        )

        all_rows: list[list[Any]] = []
        columns: list[str] = []
        start = 0
        pages = 0

        while start < limit:
            full_url = (
                f"{base_url}?from={from_enc}&till={till_enc}"
                f"&interval={interval}&start={start}"
            )

            async with self._semaphore:
                await self._rate_limiter.acquire()
                t0 = asyncio.get_event_loop().time()
                response = await self._client.get(full_url)
                elapsed_ms = round((asyncio.get_event_loop().time() - t0) * 1000)

            response.raise_for_status()
            data = response.json()
            candles_section = data.get("candles", {})
            page_cols: list[str] = candles_section.get("columns", [])
            page_rows: list[list[Any]] = candles_section.get("data", [])

            if not columns and page_cols:
                columns = page_cols
            pages += 1
            logger.debug(
                "ISS candles page",
                extra={
                    "ticker": ticker, "interval": interval,
                    "start": start, "page_rows": len(page_rows),
                    "latency_ms": elapsed_ms, "page": pages,
                },
            )

            if not page_rows:
                break

            all_rows.extend(page_rows)

            if len(page_rows) < page_size:
                break

            start += page_size

        if not all_rows:
            logger.debug(
                "ISS candles: no data",
                extra={"ticker": ticker, "from": from_enc, "till": till_enc,
                       "interval": interval},
            )
            if _HAS_PANDAS:
                return pd.DataFrame(columns=columns)
            return []

        if _HAS_PANDAS:
            df = pd.DataFrame(all_rows, columns=columns)
            for col in ("begin", "end"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            if "begin" in df.columns:
                df = df.sort_values("begin").drop_duplicates(subset=["begin"]).reset_index(drop=True)
            if len(df) > limit:
                df = df.tail(limit).reset_index(drop=True)
            return df
        else:
            records = [dict(zip(columns, row)) for row in all_rows]
            return records[-limit:]

    async def get_candles_multi(
        self,
        tickers: list[str],
        interval: int = INTERVAL_5M,
        from_dt: datetime | None = None,
        till_dt: datetime | None = None,
    ) -> "dict[str, pd.DataFrame | list]":
        """Fetch candles for multiple tickers concurrently.

        Returns:
            dict[str, pd.DataFrame | list]: ticker -> DataFrame.
        """
        tasks = {
            ticker: asyncio.create_task(
                self.get_candles(ticker, interval, from_dt, till_dt)
            )
            for ticker in tickers
        }

        results: dict[str, Any] = {}
        for ticker, task in tasks.items():
            try:
                results[ticker] = await task
            except Exception as exc:
                logger.error(
                    "Candle fetch failed",
                    extra={"ticker": ticker, "error": str(exc)},
                )
                if _HAS_PANDAS:
                    results[ticker] = pd.DataFrame()
                else:
                    results[ticker] = []

        logger.info(
            "Multi-ticker candles fetched",
            extra={
                "tickers": len(tickers),
                "success": sum(1 for v in results.values() if len(v) > 0),
            },
        )
        return results

    async def get_securities_info(self, ticker: str) -> dict[str, Any]:
        """Fetch security metadata (lot size, step, etc.)."""
        if not self._client:
            raise RuntimeError("ISSClient not started")

        url = f"{self.BASE_URL}/securities/{ticker}.json"
        params = {"iss.meta": "off", "iss.json": "extended"}

        try:
            async with self._semaphore:
                await self._rate_limiter.acquire()
                response = await self._client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning(
                "ISS get_security_info failed",
                extra={"ticker": ticker, "error": str(exc)},
            )
            return {}

        desc_section = data.get("description", {})
        if isinstance(desc_section, list) and desc_section:
            desc_section = desc_section[0]

        cols = desc_section.get("columns", [])
        rows = desc_section.get("data", [])

        info: dict[str, Any] = {}
        for row in rows:
            record = dict(zip(cols, row))
            info[record.get("name", "")] = record.get("value")

        return info

    async def get_sitenews(self, limit: int = 20) -> list[dict[str, Any]]:
        """Fetch MOEX site news (Tier S source, no auth)."""
        if not self._client:
            raise RuntimeError("ISSClient not started")

        url = f"{self.BASE_URL}/sitenews.json"
        params = {"start": 0, "limit": limit}

        try:
            async with self._semaphore:
                await self._rate_limiter.acquire()
                response = await self._client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning("ISS get_sitenews failed", extra={"error": str(exc)})
            return []

        news_section = data.get("sitenews", {})
        columns = news_section.get("columns", [])
        rows = news_section.get("data", [])

        return [dict(zip(columns, row)) for row in rows]

_iss_client: ISSClient | None = None

def get_iss_client() -> ISSClient:
    """Get iss client."""
    global _iss_client
    if _iss_client is None:
        _iss_client = ISSClient()
    return _iss_client

def get_candles_caches() -> dict[str, TTLCache]:
    """Expose both candle TTL caches for monitoring / tests."""
    return {
        "short": _ISS_CANDLES_SHORT_CACHE,
        "long": _ISS_CANDLES_LONG_CACHE,
    }

def clear_candles_caches() -> None:
    """Wipe both TTL caches — used in tests."""
    _ISS_CANDLES_SHORT_CACHE.clear()
    _ISS_CANDLES_LONG_CACHE.clear()
