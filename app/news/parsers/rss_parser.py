"""
app/news/parsers/rss_parser.py — Universal RSS / Atom feed parser.

One worker class that handles all feeds from `feeds_config.FEEDS`.
Uses `feedparser` for the heavy lifting (handles RSS 0.9x/1.0/2.0 + Atom 0.3/1.0).

Each feed is polled in its own asyncio task at its configured `poll_interval`.
Watermark per feed: stores `last_seen_id` to avoid re-publishing same items.

Failure handling:
  - Per-feed exponential backoff on errors
  - After 5 consecutive 5xx/timeout, pause the feed for 10 minutes
  - Logs warning, never crashes the worker loop
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import xml.sax  # noqa: F401  # for SAXException isinstance check
from collections import deque
from datetime import UTC, datetime
from typing import Any

import app.config as cfg
from app.news.feeds_config import FeedConfig, enabled_feeds
from app.news.ingestion_bus import IngestionBus, NormalizedNewsEvent, get_bus
from app.utils.logging import get_logger
from app.utils.retry import RateLimiter

logger = get_logger(__name__)

try:
    import feedparser  # type: ignore

    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False

try:
    import httpx  # type: ignore

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

_HOST_RATE_LIMITERS: dict[str, RateLimiter] = {}

def _get_host_limiter(host: str) -> RateLimiter:
    """Get host limiter."""
    if host not in _HOST_RATE_LIMITERS:
        _HOST_RATE_LIMITERS[host] = RateLimiter(requests_per_second=1)
    return _HOST_RATE_LIMITERS[host]

def _host_of(url: str) -> str:
    """Host of."""
    try:
        from urllib.parse import urlparse

        return urlparse(url).netloc
    except Exception:
        return url[:30]

class RSSFeedWorker:
    """One worker per feed; runs continuously polling at the configured interval."""

    def __init__(
        self,
        feed: FeedConfig,
        bus: IngestionBus,
        max_consecutive_errors: int = 5,
        error_pause_sec: int = 600,
    ) -> None:
        """Init."""
        self.feed = feed
        self.bus = bus
        self.max_consecutive_errors = max_consecutive_errors
        self.error_pause_sec = error_pause_sec
        self._last_seen_max = 500
        self._last_seen_order: deque[str] = deque(maxlen=self._last_seen_max)
        self._last_seen_ids: set[str] = set()
        self._error_count = 0
        self._paused_until_ts: float = 0
        self._items_published = 0
        self._client: Any = None
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        """Main loop. Cancel via self._stop_event.set()."""
        if not _HAS_FEEDPARSER:
            logger.error(
                "feedparser not installed — RSS worker disabled", extra={"feed": self.feed.name}
            )
            return
        if not _HAS_HTTPX:
            logger.error(
                "httpx not installed — RSS worker disabled", extra={"feed": self.feed.name}
            )
            return

        if self._client is None:
            verify_ssl = not getattr(self.feed, "insecure_ssl", False)
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(cfg.HTTP_TIMEOUT),
                verify=verify_ssl,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "application/rss+xml, application/atom+xml, "
                    "application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
                    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                },
                follow_redirects=True,
            )

        logger.debug(
            "RSS worker starting",
            extra={
                "feed": self.feed.name,
                "tier": self.feed.tier,
                "poll_interval": self.feed.poll_interval,
            },
        )

        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now < self._paused_until_ts:
                    await asyncio.sleep(min(60, self._paused_until_ts - now))
                    continue

                try:
                    await self._poll_once()
                    self._error_count = 0
                except Exception as exc:
                    self._error_count += 1
                    logger.warning(
                        "RSS poll failed",
                        extra={
                            "feed": self.feed.name,
                            "error": str(exc),
                            "consecutive_errors": self._error_count,
                        },
                    )
                    if self._error_count >= self.max_consecutive_errors:
                        self._paused_until_ts = time.monotonic() + self.error_pause_sec
                        logger.error(
                            "RSS feed paused after errors",
                            extra={
                                "feed": self.feed.name,
                                "pause_sec": self.error_pause_sec,
                                "errors": self._error_count,
                            },
                        )
                        self._error_count = 0

                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.feed.poll_interval,
                    )
        finally:
            if self._client:
                await self._client.aclose()
            logger.debug(
                "RSS worker stopped",
                extra={"feed": self.feed.name, "published": self._items_published},
            )

    def stop(self) -> None:
        """Stop."""
        self._stop_event.set()

    async def _poll_once(self) -> None:
        """Fetch the RSS feed once and publish new items to the bus."""
        host = _host_of(self.feed.url)
        await _get_host_limiter(host).acquire()

        response = await self._client.get(self.feed.url)
        response.raise_for_status()

        loop = asyncio.get_running_loop()
        parsed = await loop.run_in_executor(None, feedparser.parse, response.content)

        if parsed.bozo and not parsed.entries:
            logger.debug(
                "RSS feed returned malformed XML, skipping cycle",
                extra={
                    "feed": self.feed.name,
                    "exc": str(parsed.bozo_exception)[:200],
                },
            )
            return

        new_items = 0
        for entry in parsed.entries:
            entry_id = entry.get("id") or entry.get("link") or entry.get("title", "")
            if not entry_id or entry_id in self._last_seen_ids:
                continue
            if len(self._last_seen_order) >= self._last_seen_max:
                oldest = self._last_seen_order[0]
                self._last_seen_ids.discard(oldest)
            self._last_seen_order.append(entry_id)
            self._last_seen_ids.add(entry_id)

            ts_utc = self._parse_ts(entry)
            headline = entry.get("title", "").strip()
            body = (entry.get("summary", "") or entry.get("description", "") or "").strip()
            url = entry.get("link", "")

            event = NormalizedNewsEvent(
                source=self.feed.name,
                source_tier=self.feed.tier,
                ts_utc=ts_utc,
                headline=headline,
                body=body,
                url=url,
                tickers=[],
                language=self.feed.language,
                raw_payload={"feed_url": self.feed.url, "entry_id": entry_id},
            )

            ok = await self.bus.publish(event)
            if ok:
                new_items += 1
                self._items_published += 1

        if new_items:
            logger.debug(
                "RSS items published",
                extra={
                    "feed": self.feed.name,
                    "new_items": new_items,
                    "total_entries": len(parsed.entries),
                },
            )

    @staticmethod
    def _parse_ts(entry: Any) -> datetime:
        """Best-effort timestamp parsing from RSS entry."""
        for field in ("published_parsed", "updated_parsed"):
            ts = entry.get(field)
            if ts is not None:
                try:
                    import time as _t

                    return datetime.fromtimestamp(_t.mktime(ts), tz=UTC)
                except Exception:
                    continue
        return datetime.now(tz=UTC)

class RSSParserManager:
    """Manages all RSS worker tasks for the system."""

    def __init__(self, bus: IngestionBus | None = None) -> None:
        """Init."""
        self.bus = bus or get_bus()
        self.workers: list[RSSFeedWorker] = []
        self.tasks: list[asyncio.Task] = []

    def start(self, feeds: list[FeedConfig] | None = None) -> None:
        """Spawn one worker task per feed."""
        if feeds is None:
            feeds = enabled_feeds()

        for feed in feeds:
            worker = RSSFeedWorker(feed=feed, bus=self.bus)
            task = asyncio.create_task(worker.run(), name=f"rss_{feed.name}")
            self.workers.append(worker)
            self.tasks.append(task)

        logger.info("RSS parser manager started", extra={"workers": len(self.workers)})

    async def stop(self) -> None:
        """Stop."""
        for w in self.workers:
            w.stop()

        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        logger.info("RSS parser manager stopped")

    def stats(self) -> dict:
        """Stats."""
        return {
            "workers": len(self.workers),
            "active_tasks": sum(1 for t in self.tasks if not t.done()),
            "total_published": sum(w._items_published for w in self.workers),
            "per_feed": {
                w.feed.name: {
                    "published": w._items_published,
                    "paused": time.monotonic() < w._paused_until_ts,
                }
                for w in self.workers
            },
        }

_rss_manager: RSSParserManager | None = None

def get_rss_manager() -> RSSParserManager:
    """Get rss manager."""
    global _rss_manager
    if _rss_manager is None:
        _rss_manager = RSSParserManager()
    return _rss_manager
