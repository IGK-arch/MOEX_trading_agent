"""
app/news/data_feeds/fred_client.py — FRED (St. Louis Fed) data client.

One API key gives access to:
  - Commodities: Brent (DCOILBRENTEU), WTI (DCOILWTICO),
                 Henry Hub gas (DHHNGSP), Gold (GOLDPMGBD228NLBM),
                 Iron ore (PIORECRUSDM), Nickel (PNICKUSDM), Copper (PCOPPUSDM)
  - Volatility:  VIX (VIXCLS)
  - Rates:       UST 2Y (DGS2), 10Y (DGS10), DXY (DTWEXBGS)

Endpoint: https://api.stlouisfed.org/fred/series/observations?series_id=...&api_key=...

Quota: 120 req/min — easily within. We poll once per day after EU/US close.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from app.utils.logging import get_logger
from app.utils.retry import RateLimiter

logger = get_logger(__name__)

try:
    import httpx  # type: ignore

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

FRED_BASE = "https://api.stlouisfed.org/fred"

FRED_SERIES: dict[str, str] = {
    "brent": "DCOILBRENTEU",
    "wti": "DCOILWTICO",
    "henry_hub_gas": "DHHNGSP",
    "gold": "GOLDPMGBD228NLBM",
    "iron_ore": "PIORECRUSDM",
    "nickel": "PNICKUSDM",
    "copper": "PCOPPUSDM",
    "vix": "VIXCLS",
    "ust_2y": "DGS2",
    "ust_10y": "DGS10",
    "dxy": "DTWEXBGS",
}

class FREDClient:
    """Async FRED API client."""

    def __init__(self) -> None:
        """Init."""
        self._client: Any = None
        self._api_key = ""
        self._rate_limiter = RateLimiter(requests_per_second=2)
        self._cache: dict[str, list[dict]] = {}

    async def startup(self) -> None:
        """Startup."""
        self._api_key = os.getenv("FRED_API_KEY", "")
        if not self._api_key:
            logger.warning("FRED_API_KEY not set — FRED client disabled")
            return
        if not _HAS_HTTPX:
            logger.error("httpx not installed")
            return
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(15.0))
        logger.info("FRED client started")

    async def shutdown(self) -> None:
        """Shutdown."""
        if self._client:
            await self._client.aclose()

    async def get_series(
        self,
        series_id: str,
        days_back: int = 30,
        limit: int = 30,
    ) -> list[dict]:
        """Fetch observations for one series."""
        if not self._client or not self._api_key:
            return []

        till_dt = datetime.now(tz=UTC)
        from_dt = till_dt - timedelta(days=days_back)

        await self._rate_limiter.acquire()
        try:
            r = await self._client.get(
                f"{FRED_BASE}/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": self._api_key,
                    "file_type": "json",
                    "observation_start": from_dt.strftime("%Y-%m-%d"),
                    "observation_end": till_dt.strftime("%Y-%m-%d"),
                    "limit": limit,
                    "sort_order": "desc",
                },
            )
            r.raise_for_status()
            data = r.json()
            return data.get("observations", [])
        except Exception as exc:
            logger.error("FRED series fetch failed", extra={"series": series_id, "error": str(exc)})
            return []

    async def get_latest_value(self, name: str) -> float | None:
        """Convenience: get the most recent value of a known series by short name."""
        series_id = FRED_SERIES.get(name)
        if not series_id:
            logger.warning("FRED: unknown series", extra={"name": name})
            return None
        obs = await self.get_series(series_id, limit=5)
        for o in obs:
            v = o.get("value", ".")
            if v not in (".", "", None):
                try:
                    return float(v)
                except Exception:
                    continue
        return None

    async def fetch_all_latest(self) -> dict[str, float]:
        """Fetch latest value for every tracked series. Returns dict name → value."""
        out: dict[str, float] = {}
        tasks = [self.get_latest_value(name) for name in FRED_SERIES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for name, res in zip(FRED_SERIES.keys(), results, strict=False):
            if isinstance(res, float):
                out[name] = res
        logger.info("FRED batch fetch", extra={"fetched": len(out), "total": len(FRED_SERIES)})
        return out

_fred: FREDClient | None = None

def get_fred_client() -> FREDClient:
    """Get fred client."""
    global _fred
    if _fred is None:
        _fred = FREDClient()
    return _fred
