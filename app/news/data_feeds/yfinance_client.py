"""
app/news/data_feeds/yfinance_client.py — Yahoo Finance (free, 15-min delayed).

Used for commodities, FX, and global volatility — primary low-cost source.

Tickers we track:
  BZ=F  — Brent crude
  CL=F  — WTI crude
  NG=F  — Natural Gas (Henry Hub)
  GC=F  — Gold
  HG=F  — Copper
  PA=F  — Palladium
  SI=F  — Silver
  ^VIX  — CBOE Volatility Index
  ^DXY  — Dollar Index (not always available; use DX-Y.NYB or DXY.US)
  RUB=X — USDRUB
  CNYRUB=X — CNYRUB
  EURRUB=X — EURRUB

yfinance is synchronous — we call it in run_in_executor to avoid blocking.
"""

from __future__ import annotations

import asyncio
import time

from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import yfinance as yf  # type: ignore

    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

try:
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

TRACKED_SYMBOLS: list[str] = [
    "BZ=F",
    "CL=F",
    "NG=F",
    "GC=F",
    "HG=F",
    "PA=F",
    "SI=F",
    "^VIX",
    "DX-Y.NYB",
    "RUB=X",
    "CNYRUB=X",
    "EURRUB=X",
]

class YFinanceClient:
    """Thin async-wrapped batch fetcher."""

    def __init__(self, cache_ttl_sec: int = 900) -> None:
        """Init."""
        self.cache_ttl_sec = cache_ttl_sec
        self._cache: dict[str, tuple[float, dict]] = {}

    async def get_quote(self, symbol: str) -> dict | None:
        """Get last quote for one symbol with TTL cache."""
        if not _HAS_YFINANCE:
            return None

        now = time.monotonic()
        cached = self._cache.get(symbol)
        if cached and now - cached[0] < self.cache_ttl_sec:
            return cached[1]

        loop = asyncio.get_event_loop()
        try:
            data = await loop.run_in_executor(None, self._fetch_sync, symbol)
            if data:
                self._cache[symbol] = (now, data)
            return data
        except Exception as exc:
            logger.warning("yfinance fetch failed", extra={"symbol": symbol, "error": str(exc)})
            return None

    @staticmethod
    def _fetch_sync(symbol: str) -> dict | None:
        """Synchronous yfinance call — runs in executor."""
        if not _HAS_YFINANCE:
            return None
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1d")
            if hist.empty:
                return None
            last = hist.iloc[-1]
            return {
                "symbol": symbol,
                "close": float(last["Close"]),
                "open": float(last["Open"]),
                "high": float(last["High"]),
                "low": float(last["Low"]),
                "volume": float(last["Volume"]) if not pd.isna(last["Volume"]) else 0,
                "ts": str(hist.index[-1]),
            }
        except Exception:
            return None

    async def fetch_all(
        self,
        symbols: list[str] | None = None,
    ) -> dict[str, dict]:
        """Fetch all tracked symbols concurrently. Returns {symbol: quote_dict}."""
        if symbols is None:
            symbols = TRACKED_SYMBOLS
        tasks = [self.get_quote(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, dict] = {}
        for s, r in zip(symbols, results, strict=False):
            if isinstance(r, dict):
                out[s] = r
        logger.info("yfinance batch fetch", extra={"fetched": len(out), "requested": len(symbols)})
        return out

_yf_client: YFinanceClient | None = None

def get_yfinance_client() -> YFinanceClient:
    """Get yfinance client."""
    global _yf_client
    if _yf_client is None:
        _yf_client = YFinanceClient()
    return _yf_client
