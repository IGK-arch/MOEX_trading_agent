"""
app/news/parsers/moex_iss_parser.py — MOEX ISS sitenews watcher.

Polls https://iss.moex.com/iss/sitenews.json every 30s.
Uses the existing ISSClient for proper rate limiting + retry.

ISS sitenews structure (from live test):
  {id, tag, title, published_at, modified_at}
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime

from app.data.iss_client import get_iss_client
from app.news.ingestion_bus import IngestionBus, NormalizedNewsEvent, get_bus
from app.utils.logging import get_logger

logger = get_logger(__name__)

class MoexISSParser:
    """MOEX exchange news watcher (Tier S)."""

    def __init__(
        self,
        bus: IngestionBus | None = None,
        poll_interval: int = 30,
        limit_per_poll: int = 20,
    ) -> None:
        """Init."""
        self.bus = bus or get_bus()
        self.poll_interval = poll_interval
        self.limit_per_poll = limit_per_poll
        self._iss = get_iss_client()
        self._seen_ids: set[str] = set()
        self._max_seen = 500
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._published = 0
        self._initialized = False

    async def start(self) -> None:
        """Start."""
        if not self._iss._started:
            await self._iss.startup()
        self._task = asyncio.create_task(self._loop(), name="moex_iss_news")
        logger.info("MoexISSParser started", extra={"poll_sec": self.poll_interval})

    async def stop(self) -> None:
        """Stop."""
        self._stop_event.set()
        if self._task:
            await asyncio.gather(self._task, return_exceptions=True)
        logger.info("MoexISSParser stopped", extra={"published": self._published})

    async def _loop(self) -> None:
        """Loop."""
        while not self._stop_event.is_set():
            try:
                await self._poll_once()
            except Exception as exc:
                logger.warning("MOEX ISS news poll failed", extra={"error": str(exc)})
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval)

    async def _poll_once(self) -> None:
        """Poll once."""
        news = await self._iss.get_sitenews(limit=self.limit_per_poll)

        if not self._initialized:
            self._seen_ids = {str(n.get("id")) for n in news if n.get("id") is not None}
            self._initialized = True
            logger.info("MOEX ISS news baseline", extra={"items": len(self._seen_ids)})
            return

        for item in news:
            iid = str(item.get("id", ""))
            if not iid or iid in self._seen_ids:
                continue
            self._seen_ids.add(iid)

            if len(self._seen_ids) > self._max_seen:
                self._seen_ids = set(list(self._seen_ids)[-self._max_seen // 2 :])

            ts = item.get("published_at", "")
            try:
                ts_utc = (
                    datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if ts
                    else datetime.now(tz=UTC)
                )
                if ts_utc.tzinfo is None:
                    ts_utc = ts_utc.replace(tzinfo=UTC)
            except Exception:
                ts_utc = datetime.now(tz=UTC)

            event = NormalizedNewsEvent(
                source="moex_iss_sitenews",
                source_tier="S",
                ts_utc=ts_utc,
                headline=item.get("title", ""),
                body="",
                url=f"https://www.moex.com/n{iid}",
                tickers=[],
                language="ru",
                raw_payload={"id": iid, "tag": item.get("tag", "")},
            )
            ok = await self.bus.publish(event)
            if ok:
                self._published += 1

_moex_parser: MoexISSParser | None = None

def get_moex_iss_parser() -> MoexISSParser:
    """Get moex iss parser."""
    global _moex_parser
    if _moex_parser is None:
        _moex_parser = MoexISSParser()
    return _moex_parser
