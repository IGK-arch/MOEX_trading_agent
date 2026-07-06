"""
app/news/parsers/sanctions_parser.py — OFAC + EU + UK sanctions watcher.

Sources:
  - OFAC SDN XML: https://www.treasury.gov/ofac/downloads/sdn.xml (XML, large)
  - OFAC recent actions: HTML page, lighter polling
  - EU FSF RSS: easier feed
  - UK OFSI Atom: gov.uk feed

Strategy:
  - Poll EU FSF RSS + UK OFSI every 60s (light)
  - Poll OFAC recent actions HTML every 60s
  - Poll OFAC SDN XML every 5min (heavy but authoritative)
  - For OFAC SDN: maintain a snapshot diff — any new entry mentioning Russia/RU →
    high-priority Tier S event with bypass material_filter
  - All events use source_tier="S"
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app.config as cfg
from app.news.feeds_config import SANCTIONS_SOURCES
from app.news.ingestion_bus import IngestionBus, NormalizedNewsEvent, get_bus
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import httpx  # type: ignore

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    import feedparser  # type: ignore

    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False

RU_MARKERS = re.compile(
    r"\b(russia|russian federation|rosneft|gazprom|sberbank|vtb|moscow|kremlin|"
    r"россия|российск|москв|кремль)\b",
    re.IGNORECASE,
)

class SanctionsParser:
    """Multi-source sanctions watcher."""

    def __init__(
        self,
        bus: IngestionBus | None = None,
        ofac_xml_poll_sec: int = 300,
        rss_poll_sec: int = 60,
        state_file: Path | None = None,
    ) -> None:
        """Init."""
        self.bus = bus or get_bus()
        self.ofac_xml_poll_sec = ofac_xml_poll_sec
        self.rss_poll_sec = rss_poll_sec
        self.state_file = state_file or (cfg.DATA_DIR / "sanctions_state.json")
        self._client: Any = None
        self._last_sdn_hashes: set[str] = set()
        self._last_eu_ids: set[str] = set()
        self._last_uk_ids: set[str] = set()
        self._last_ofac_action_ids: set[str] = set()
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._published = 0
        self._initialized = False

    async def start(self) -> None:
        """Start."""
        if not _HAS_HTTPX or not _HAS_FEEDPARSER:
            logger.error("SanctionsParser: httpx or feedparser not installed")
            return

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={"User-Agent": "MoexML-Trader/0.1 (sanctions monitor)"},
            follow_redirects=True,
        )

        self._tasks.append(asyncio.create_task(self._eu_fsf_loop(), name="sanctions_eu"))
        self._tasks.append(asyncio.create_task(self._uk_ofsi_loop(), name="sanctions_uk"))
        self._tasks.append(asyncio.create_task(self._ofac_sdn_loop(), name="sanctions_ofac_sdn"))

        logger.info("SanctionsParser started", extra={"tasks": len(self._tasks)})

    async def stop(self) -> None:
        """Stop."""
        self._stop_event.set()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._client:
            await self._client.aclose()
        logger.info("SanctionsParser stopped", extra={"published": self._published})

    async def _eu_fsf_loop(self) -> None:
        """Eu fsf loop."""
        while not self._stop_event.is_set():
            try:
                await self._poll_eu_fsf()
            except Exception as exc:
                logger.warning("EU FSF poll failed", extra={"error": str(exc)})
            await self._wait(self.rss_poll_sec)

    async def _poll_eu_fsf(self) -> None:
        """Poll eu fsf."""
        url = SANCTIONS_SOURCES["eu_fsf_rss"]
        try:
            r = await self._client.get(url)
            r.raise_for_status()
        except Exception:
            return

        loop = asyncio.get_event_loop()
        parsed = await loop.run_in_executor(None, feedparser.parse, r.content)

        for entry in parsed.entries:
            eid = entry.get("id") or entry.get("link") or entry.get("title", "")
            if not eid or eid in self._last_eu_ids:
                continue
            self._last_eu_ids.add(eid)

            text = entry.get("title", "") + " " + entry.get("summary", "")
            if not RU_MARKERS.search(text):
                continue

            await self._publish(
                source="eu_fsf",
                headline=entry.get("title", ""),
                body=entry.get("summary", ""),
                url=entry.get("link", ""),
                jurisdiction="EU",
            )

    async def _uk_ofsi_loop(self) -> None:
        """Uk ofsi loop."""
        while not self._stop_event.is_set():
            try:
                await self._poll_uk_ofsi()
            except Exception as exc:
                logger.warning("UK OFSI poll failed", extra={"error": str(exc)})
            await self._wait(self.rss_poll_sec)

    async def _poll_uk_ofsi(self) -> None:
        """Poll uk ofsi."""
        url = SANCTIONS_SOURCES["uk_ofsi_atom"]
        try:
            r = await self._client.get(url)
            r.raise_for_status()
        except Exception:
            return

        loop = asyncio.get_event_loop()
        parsed = await loop.run_in_executor(None, feedparser.parse, r.content)

        for entry in parsed.entries:
            eid = entry.get("id") or entry.get("link") or entry.get("title", "")
            if not eid or eid in self._last_uk_ids:
                continue
            self._last_uk_ids.add(eid)

            text = entry.get("title", "") + " " + entry.get("summary", "")
            if not RU_MARKERS.search(text):
                continue

            await self._publish(
                source="uk_ofsi",
                headline=entry.get("title", ""),
                body=entry.get("summary", ""),
                url=entry.get("link", ""),
                jurisdiction="UK",
            )

    async def _ofac_sdn_loop(self) -> None:
        """Ofac sdn loop."""
        while not self._stop_event.is_set():
            try:
                await self._poll_ofac_sdn()
            except Exception as exc:
                logger.warning("OFAC SDN poll failed", extra={"error": str(exc)})
            await self._wait(self.ofac_xml_poll_sec)

    async def _poll_ofac_sdn(self) -> None:
        """Pull OFAC SDN XML and diff vs last hash set."""
        url = SANCTIONS_SOURCES["ofac_sdn_xml"]
        try:
            r = await self._client.get(url)
            r.raise_for_status()
        except Exception:
            return

        xml_bytes = r.content

        if not self._initialized:
            self._last_sdn_hashes = set(self._extract_entry_hashes(xml_bytes))
            self._initialized = True
            logger.info(
                "OFAC SDN baseline established", extra={"entries": len(self._last_sdn_hashes)}
            )
            return

        new_hashes = set(self._extract_entry_hashes(xml_bytes))
        added = new_hashes - self._last_sdn_hashes
        removed = self._last_sdn_hashes - new_hashes
        self._last_sdn_hashes = new_hashes

        if not added and not removed:
            return

        xml_str = xml_bytes.decode("utf-8", errors="ignore")
        ru_keyword_count = len(RU_MARKERS.findall(xml_str))

        await self._publish(
            source="ofac_sdn_diff",
            headline=f"OFAC SDN list updated: +{len(added)} / -{len(removed)} entries",
            body=f"OFAC SDN diff detected. RU markers in current list: {ru_keyword_count}. "
            f"Review at https://ofac.treasury.gov/recent-actions",
            url="https://ofac.treasury.gov/recent-actions",
            jurisdiction="US",
            extra={"added": len(added), "removed": len(removed), "ru_mentions": ru_keyword_count},
        )

    @staticmethod
    def _extract_entry_hashes(xml_bytes: bytes) -> list[str]:
        """Cheap entry hashing — split on </sdnEntry> and hash each block."""
        text = xml_bytes.decode("utf-8", errors="ignore")
        chunks = text.split("</sdnEntry>")
        hashes = []
        for chunk in chunks:
            if "<sdnEntry>" in chunk:
                entry_text = chunk.split("<sdnEntry>", 1)[1]
                h = hashlib.sha1(entry_text.encode("utf-8", errors="ignore")).hexdigest()
                hashes.append(h)
        return hashes

    async def _publish(
        self,
        source: str,
        headline: str,
        body: str,
        url: str,
        jurisdiction: str,
        extra: dict | None = None,
    ) -> None:
        """Publish."""
        event = NormalizedNewsEvent(
            source=source,
            source_tier="S",
            ts_utc=datetime.now(tz=UTC),
            headline=headline,
            body=body,
            url=url,
            tickers=[],
            language="en",
            raw_payload={"jurisdiction": jurisdiction, **(extra or {})},
        )

        import app.config as _cfg

        if getattr(_cfg, "SANCTIONS_PUSH_MODE", True) and hasattr(self.bus, "publish_priority"):
            ok = await self.bus.publish_priority(event)
        else:
            ok = await self.bus.publish(event)
        if ok:
            self._published += 1
            logger.warning(
                "Sanctions event published",
                extra={
                    "source": source,
                    "headline": headline[:100],
                    "jurisdiction": jurisdiction,
                    "priority": getattr(_cfg, "SANCTIONS_PUSH_MODE", True),
                },
            )

    async def _wait(self, seconds: int) -> None:
        """Wait."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)

_sanctions_parser: SanctionsParser | None = None

def get_sanctions_parser() -> SanctionsParser:
    """Get sanctions parser."""
    global _sanctions_parser
    if _sanctions_parser is None:
        _sanctions_parser = SanctionsParser()
    return _sanctions_parser
