"""Клиент AlgoPack с ISS fallback."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

import app.config as cfg
from app.utils.async_cache import SingleFlight, TTLCache
from app.utils.logging import get_logger, get_trace_id
from app.utils.retry import with_retry

logger = get_logger(__name__)

_OBSTATS_CACHE: TTLCache = TTLCache(ttl_seconds=30.0, max_entries=512)
_TRADESTATS_CACHE: TTLCache = TTLCache(ttl_seconds=30.0, max_entries=512)
_OBSTATS_FLIGHT: SingleFlight = SingleFlight()
_TRADESTATS_FLIGHT: SingleFlight = SingleFlight()

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

class AlgoPackClient:
    """Async AlgoPack client (falls back to ISS approximations if JWT auth fails)."""

    BASE_URL = cfg.ALGOPACK_BASE_URL

    def __init__(self) -> None:
        """Init."""
        self._client: Any = None
        self._token: str = ""
        self._semaphore = asyncio.Semaphore(cfg.ALGOPACK_SEMAPHORE_LIMIT)
        self._auth_failed = False
        self._started = False

    async def startup(self) -> None:
        """Startup."""
        import os
        self._token = os.getenv("ALGOPACK_TOKEN", "")
        if not self._token:
            logger.warning("ALGOPACK_TOKEN not set — using ISS fallback only")
            self._auth_failed = True
        else:
            logger.info("AlgoPack token loaded")

        if not _HAS_HTTPX:
            logger.error("httpx not installed — AlgoPack client unavailable")
            return

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.HTTP_TIMEOUT),
            limits=httpx.Limits(
                max_keepalive_connections=cfg.HTTP_KEEPALIVE_CONNECTIONS,
                max_connections=cfg.HTTP_MAX_CONNECTIONS,
            ),
            headers={
                "User-Agent": "MoexML-Trader/0.1",
                "Authorization": f"Bearer {self._token}",
            },
        )
        self._started = True

        if self._token:
            ok = await self._check_auth()
            if not ok:
                logger.warning(
                    "AlgoPack JWT auth failed (401/403). "
                    "Falling back to ISS candle-based OFI approximation. "
                    "To fix: refresh token at data.moex.com and update ALGOPACK_TOKEN in .env",
                    extra={"token_prefix": self._token[:20] + "..."},
                )
                self._auth_failed = True

        logger.info(
            "AlgoPack client started",
            extra={"premium": not self._auth_failed},
        )

    async def _check_auth(self) -> bool:
        """Quick auth check — try to fetch 1 row of obstats."""
        try:
            today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            url = f"{self.BASE_URL}/eq/obstats.json"
            async with self._semaphore:
                r = await self._client.get(
                    url,
                    params={"date": today, "secid": "SBER", "limit": 1},
                    timeout=5.0,
                )
            if r.status_code == 200:
                return True
            logger.debug(
                "AlgoPack auth check failed",
                extra={"status": r.status_code, "body": r.text[:100]},
            )
            return False
        except Exception as exc:
            logger.debug("AlgoPack auth check error", extra={"error": str(exc)})
            return False

    async def shutdown(self) -> None:
        """Shutdown."""
        if self._client:
            await self._client.aclose()
        logger.info("AlgoPack client stopped")

    async def get_obstats(
        self,
        ticker: str,
        date: str | None = None,
    ) -> "pd.DataFrame | list[dict]":
        """Fetch orderbook statistics for a ticker.

        Returns:
            pd.DataFrame | list[dict]: obstats rows.
        """
        if date is None:
            date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        cache_key = ("obstats", ticker.upper(), date)
        cached = _OBSTATS_CACHE.get(cache_key)
        if cached is not None:
            return cached

        async def _runner() -> "pd.DataFrame | list[dict]":
            """Runner."""
            second_look = _OBSTATS_CACHE.get(cache_key)
            if second_look is not None:
                return second_look
            result = await self._fetch_obstats(ticker, date, cache_key)
            self._maybe_store(_OBSTATS_CACHE, cache_key, result)
            return result

        return await _OBSTATS_FLIGHT.do(cache_key, _runner)

    async def _fetch_obstats(
        self,
        ticker: str,
        date: str,
        cache_key: tuple,
    ) -> "pd.DataFrame | list[dict]":
        """Underlying obstats fetch — uncached. Wrapped by `get_obstats`."""
        if self._auth_failed:
            return await self._fallback_ofi(ticker)

        try:
            url = f"{self.BASE_URL}/eq/obstats.json"
            async with self._semaphore:
                r = await self._client.get(
                    url,
                    params={"date": date, "secid": ticker},
                )
            if r.status_code in (401, 403):
                self._auth_failed = True
                logger.warning(
                    "AlgoPack auth failed during request, switching to fallback",
                    extra={"ticker": ticker, "status": r.status_code},
                )
                return await self._fallback_ofi(ticker)

            r.raise_for_status()
            return self._parse_response(r.json(), "obstats")

        except Exception as exc:
            logger.error(
                "AlgoPack obstats error",
                extra={"ticker": ticker, "error": str(exc)},
            )
            return await self._fallback_ofi(ticker)

    @staticmethod
    def _maybe_store(cache: TTLCache, key: tuple, value: Any) -> None:
        """Cache non-empty results only."""
        if value is None:
            return
        empty = getattr(value, "empty", None)
        if empty is True:
            return
        if isinstance(value, (list, tuple)) and len(value) == 0:
            return
        cache.set(key, value)

    async def get_tradestats(
        self,
        ticker: str,
        date: str | None = None,
    ) -> "pd.DataFrame | list[dict]":
        """Fetch trade statistics (aggressive buys/sells).

        Returns:
            pd.DataFrame | list[dict]: tradestats rows.
        """
        if date is None:
            date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        cache_key = ("tradestats", ticker.upper(), date)
        cached = _TRADESTATS_CACHE.get(cache_key)
        if cached is not None:
            return cached

        async def _runner() -> "pd.DataFrame | list[dict]":
            """Runner."""
            second_look = _TRADESTATS_CACHE.get(cache_key)
            if second_look is not None:
                return second_look
            result = await self._fetch_tradestats(ticker, date, cache_key)
            self._maybe_store(_TRADESTATS_CACHE, cache_key, result)
            return result

        return await _TRADESTATS_FLIGHT.do(cache_key, _runner)

    async def _fetch_tradestats(
        self,
        ticker: str,
        date: str,
        cache_key: tuple,
    ) -> "pd.DataFrame | list[dict]":
        """Underlying tradestats fetch — wrapped by `get_tradestats`."""
        if self._auth_failed:
            return await self._fallback_tradestats(ticker)

        try:
            url = f"{self.BASE_URL}/eq/tradestats.json"
            async with self._semaphore:
                r = await self._client.get(
                    url,
                    params={"date": date, "secid": ticker},
                )
            if r.status_code in (401, 403):
                self._auth_failed = True
                return await self._fallback_tradestats(ticker)

            r.raise_for_status()
            return self._parse_response(r.json(), "tradestats")

        except Exception as exc:
            logger.error(
                "AlgoPack tradestats error",
                extra={"ticker": ticker, "error": str(exc)},
            )
            return await self._fallback_tradestats(ticker)

    def _parse_response(self, data: dict, section_name: str) -> "pd.DataFrame | list[dict]":
        """Parse ISS-style {section: {columns: [...], data: [...]}} response."""
        section = data.get(section_name, {})
        columns = section.get("columns", [])
        rows = section.get("data", [])
        if _HAS_PANDAS:
            if not rows:
                return pd.DataFrame(columns=columns)
            return pd.DataFrame(rows, columns=columns)
        return [dict(zip(columns, row)) for row in rows]

    async def _fallback_ofi(
        self, ticker: str
    ) -> "pd.DataFrame | list[dict]":
        """Approximate OFI from free ISS candle data."""
        from app.data.iss_client import get_iss_client
        iss = get_iss_client()
        if not iss._started:
            await iss.startup()

        now = datetime.now(tz=timezone.utc)
        df = await iss.get_candles(ticker, interval=5, till_dt=now)

        if _HAS_PANDAS and isinstance(df, pd.DataFrame) and not df.empty:
            df["price_range"] = df["high"] - df["low"]
            df["price_range"] = df["price_range"].replace(0, float("nan"))
            df["ofi_approx"] = (df["close"] - df["open"]) / df["price_range"] * df["volume"]
            df["imbalance_vol_bbo"] = df["ofi_approx"] / (df["volume"] + 1)
            df["imbalance_vol_bbo"] = df["imbalance_vol_bbo"].clip(-1, 1)
            df["vol_b"] = df["volume"] * (df["ofi_approx"].clip(0, None) / (df["volume"] + 1))
            df["vol_s"] = df["volume"] - df["vol_b"]
            df["source"] = "iss_approx"
            return df
        return df

    async def _fallback_tradestats(
        self, ticker: str
    ) -> "pd.DataFrame | list[dict]":
        """Return candle-based trade direction approximation."""
        return await self._fallback_ofi(ticker)

_algopack_client: AlgoPackClient | None = None

def get_algopack_client() -> AlgoPackClient:
    """Get algopack client."""
    global _algopack_client
    if _algopack_client is None:
        _algopack_client = AlgoPackClient()
    return _algopack_client

def get_obstats_cache() -> TTLCache:
    """Expose the obstats TTL cache (used by dashboard / tests)."""
    return _OBSTATS_CACHE

def get_tradestats_cache() -> TTLCache:
    """Expose the tradestats TTL cache."""
    return _TRADESTATS_CACHE

def clear_algopack_caches() -> None:
    """Wipe both caches — invoked by tests."""
    _OBSTATS_CACHE.clear()
    _TRADESTATS_CACHE.clear()
