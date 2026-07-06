"""moexalgo SuperCandles async wrapper."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import app.config as cfg
from app.utils.async_cache import SingleFlight, TTLCache
from app.utils.logging import get_logger

logger = get_logger(__name__)

_SUPER_CACHE: TTLCache = TTLCache(ttl_seconds=60.0, max_entries=512)
_SUPER_FLIGHT: SingleFlight = SingleFlight()

try:
    from moexalgo import Ticker  # type: ignore
    _HAS_MOEXALGO = True
except ImportError:
    _HAS_MOEXALGO = False

try:
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

_DB_PATH = cfg.DATA_DIR / "feeds.db"
_CACHE_TTL_INTRADAY_SEC = 30
_CACHE_TTL_HISTORICAL_SEC = 3600

def _init_db() -> None:
    """Ensure feeds.db has the supercandles cache table."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_DB_PATH)) as cn:
        cn.execute("""
            CREATE TABLE IF NOT EXISTS supercandles_cache (
                ticker  TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                interval INTEGER NOT NULL,
                cached_at INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (ticker, trade_date, interval)
            )
        """)

def _cache_lookup(ticker: str, dt: str, interval: int) -> "pd.DataFrame | None":
    """Return cached DataFrame if fresh, else None."""
    if not _HAS_PANDAS:
        return None
    now = int(time.time())
    try:
        with sqlite3.connect(str(_DB_PATH)) as cn:
            row = cn.execute(
                "SELECT cached_at, payload_json FROM supercandles_cache "
                "WHERE ticker=? AND trade_date=? AND interval=?",
                (ticker, dt, interval),
            ).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    cached_at, payload = row

    today_iso = date.today().isoformat()
    ttl = _CACHE_TTL_INTRADAY_SEC if dt >= today_iso else _CACHE_TTL_HISTORICAL_SEC
    if now - cached_at > ttl:
        return None
    try:
        df = pd.read_json(payload, orient="split")
        return df
    except Exception:
        return None

def _cache_store(ticker: str, dt: str, interval: int, df: "pd.DataFrame") -> None:
    """Cache store."""
    if not _HAS_PANDAS or df is None or len(df) == 0:
        return
    try:
        payload = df.to_json(orient="split", date_format="iso")
        with sqlite3.connect(str(_DB_PATH)) as cn:
            cn.execute(
                "INSERT OR REPLACE INTO supercandles_cache "
                "(ticker, trade_date, interval, cached_at, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (ticker, dt, interval, int(time.time()), payload),
            )
    except (sqlite3.Error, Exception) as exc:
        logger.debug("supercandles cache store failed", extra={"error": str(exc)})

def _fetch_sync(ticker: str, dt: str, interval: int) -> "pd.DataFrame | None":
    """Synchronous moexalgo call. Run inside asyncio.to_thread."""
    if not _HAS_MOEXALGO:
        return None
    try:
        t = Ticker(ticker)

        df = t.candles(date=dt, period=interval)
        if df is None:
            return None

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(list(df))
        return df
    except Exception as exc:
        logger.debug(
            "moexalgo Ticker.candles failed",
            extra={"ticker": ticker, "date": dt, "interval": interval,
                   "error": str(exc)},
        )
        return None

async def get_supercandles(
    ticker: str,
    *,
    trade_date: str | None = None,
    interval: int = 5,
) -> "pd.DataFrame | None":
    """Return SuperCandles DataFrame for one ticker x one trading day.

    Args:
        ticker: MOEX SECID.
        trade_date: ISO date "YYYY-MM-DD"; defaults to today UTC.
        interval: candle period in minutes.
    Returns:
        pd.DataFrame | None: candles or None on failure.
    """
    if not _HAS_MOEXALGO or not _HAS_PANDAS:
        return None
    _init_db()
    if trade_date is None:
        trade_date = datetime.now(tz=timezone.utc).date().isoformat()

    key = (ticker.upper(), trade_date, interval)

    cached_mem = _SUPER_CACHE.get(key)
    if cached_mem is not None:
        return cached_mem

    return await _SUPER_FLIGHT.do(
        key,
        lambda: _fetch_supercandles_uncached(ticker, trade_date, interval, key),
    )

async def _fetch_supercandles_uncached(
    ticker: str,
    trade_date: str,
    interval: int,
    key: tuple,
) -> "pd.DataFrame | None":
    """Network + sqlite path — invoked by the single-flight leader."""
    second_look = _SUPER_CACHE.get(key)
    if second_look is not None:
        return second_look

    cached_sql = _cache_lookup(ticker, trade_date, interval)
    if cached_sql is not None:
        _SUPER_CACHE.set(key, cached_sql)
        return cached_sql

    df = await asyncio.to_thread(_fetch_sync, ticker, trade_date, interval)
    if df is not None and len(df) > 0:
        _cache_store(ticker, trade_date, interval, df)
        _SUPER_CACHE.set(key, df)
    return df

def get_supercandles_cache() -> TTLCache:
    """Expose the in-memory cache for dashboard / tests."""
    return _SUPER_CACHE

def clear_supercandles_cache() -> None:
    """Wipe the in-memory layer — invoked by tests."""
    _SUPER_CACHE.clear()

__all__ = [
    "get_supercandles",
    "get_supercandles_cache",
    "clear_supercandles_cache",
]
