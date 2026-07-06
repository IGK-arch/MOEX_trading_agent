"""
app/news/data_feeds/cbr_client.py — Central Bank of Russia daily XML.

Endpoint: https://www.cbr.ru/scripts/XML_daily.asp
Returns: ~54 currency rates in cp1251 XML.

Usage:
    cbr = get_cbr_client()
    rates = await cbr.fetch_daily()

    usd_rub = await cbr.get_rate("USD")

Also fetches CBR key rate (separate page).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import httpx  # type: ignore

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    import xml.etree.ElementTree as ET

    _HAS_XML = True
except ImportError:
    _HAS_XML = False

CBR_DAILY_URL = "https://www.cbr.ru/scripts/XML_daily.asp"
CBR_KEY_RATE_URL = "https://www.cbr.ru/eng/key-indicators/"

class CBRClient:
    """Daily ₽ exchange rates + key rate from CBR."""

    def __init__(self) -> None:
        """Init."""
        self._client: Any = None
        self._cached_rates: dict[str, float] = {}
        self._cache_date: str = ""

    async def startup(self) -> None:
        """Startup."""
        if not _HAS_HTTPX:
            logger.error("httpx not installed — CBR client unavailable")
            return
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.HTTP_TIMEOUT),
            headers={"User-Agent": "MoexML-Trader/0.1"},
        )

    async def shutdown(self) -> None:
        """Shutdown."""
        if self._client:
            await self._client.aclose()

    async def fetch_daily(self) -> dict[str, float]:
        """
        Fetch today's CBR rates. Returns {currency_code: rate_in_rub}.
        Cached for the rest of the day (CBR rate is daily-fixed).
        """
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        if self._cache_date == today and self._cached_rates:
            return self._cached_rates

        if not self._client:
            return {}

        try:
            r = await self._client.get(CBR_DAILY_URL)
            r.raise_for_status()
        except Exception as exc:
            logger.warning("CBR fetch failed", extra={"error": str(exc)})
            return self._cached_rates or {}

        try:
            content = r.content.decode("cp1251", errors="ignore")
        except Exception:
            content = r.text

        try:
            root = ET.fromstring(content)
        except Exception as exc:
            logger.error("CBR XML parse failed", extra={"error": str(exc)})
            return self._cached_rates or {}

        rates: dict[str, float] = {}
        for valute in root.findall("Valute"):
            char_code = (valute.findtext("CharCode") or "").strip()
            nominal_s = (valute.findtext("Nominal") or "1").strip()
            value_s = (valute.findtext("Value") or "0").replace(",", ".").strip()
            try:
                nominal = float(nominal_s)
                value = float(value_s)
                if nominal > 0:
                    rates[char_code] = value / nominal
            except Exception:
                continue

        rates["RUB"] = 1.0

        self._cached_rates = rates
        self._cache_date = today
        logger.info("CBR rates fetched", extra={"currencies": len(rates), "date": today})
        return rates

    async def get_rate(self, currency: str) -> float | None:
        """Get rate of `currency` in RUB (e.g. 'USD' → 92.13)."""
        rates = await self.fetch_daily()
        return rates.get(currency.upper())

_cbr_client: CBRClient | None = None

def get_cbr_client() -> CBRClient:
    """Get cbr client."""
    global _cbr_client
    if _cbr_client is None:
        _cbr_client = CBRClient()
    return _cbr_client
