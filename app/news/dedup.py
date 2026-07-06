"""MinHash LSH дедупликация новостей."""

from __future__ import annotations

import re
import time
from collections import deque

from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from datasketch import MinHash, MinHashLSH  # type: ignore

    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False

_WORD_RE = re.compile(r"\w+", re.UNICODE)

def _shingle(text: str, k: int = 3) -> set[str]:
    """3-word shingles, lowercased, joined with space."""
    words = [word.lower() for word in _WORD_RE.findall(text)]
    if len(words) < k:
        return set(words)
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}

class NewsDeduplicator:
    """Near-duplicate detector with TTL-based eviction."""

    def __init__(
        self,
        threshold: float | None = None,
        num_perm: int | None = None,
        ttl_seconds: int | None = None,
        max_size: int | None = None,
    ) -> None:
        """Init."""
        import app.config as _cfg

        self.threshold = float(
            threshold if threshold is not None else getattr(_cfg, "DEDUP_JACCARD_THRESHOLD", 0.85)
        )
        self.num_perm = int(
            num_perm if num_perm is not None else getattr(_cfg, "DEDUP_NUM_PERM", 128)
        )
        self.ttl_seconds = int(
            ttl_seconds if ttl_seconds is not None else getattr(_cfg, "DEDUP_TTL_SEC", 86400)
        )
        self.max_size = int(
            max_size if max_size is not None else getattr(_cfg, "DEDUP_MAX_SIZE", 10000)
        )
        self._fallback_texts: dict[str, set[str]] = {}

        if _HAS_DATASKETCH:
            self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        else:
            self._lsh = None
            logger.warning("datasketch not installed - using simple dedup fallback")

        self._index: deque[tuple[str, float]] = deque()
        self._key_set: set[str] = set()
        self._duplicates_blocked = 0
        self._total_checks = 0

    def is_duplicate(self, event_id: str, text: str) -> bool:
        """Return True when text is too similar to a recent item."""
        self._total_checks += 1
        shingles = _shingle(text)
        if not shingles:
            return False

        if self._lsh is None:
            for existing in self._fallback_texts.values():
                union = shingles | existing
                score = (len(shingles & existing) / len(union)) if union else 0.0
                if score >= self.threshold:
                    self._duplicates_blocked += 1
                    return True
            self._fallback_texts[event_id] = shingles
            self._index.append((event_id, time.time()))
            self._key_set.add(event_id)
            if len(self._index) > self.max_size:
                self._evict_oldest(self.max_size // 4)
            return False

        mh = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            mh.update(shingle.encode("utf-8"))

        matches = self._lsh.query(mh)
        if matches:
            self._duplicates_blocked += 1
            logger.debug(
                "Duplicate news blocked",
                extra={"event_id": event_id, "matches": list(matches)[:3]},
            )
            return True

        try:
            self._lsh.insert(event_id, mh)
            self._index.append((event_id, time.time()))
            self._key_set.add(event_id)
        except ValueError:
            pass

        if len(self._index) > self.max_size:
            self._evict_oldest(self.max_size // 4)

        return False

    def prune(self) -> int:
        """Remove entries older than ttl_seconds. Returns count removed."""
        cutoff = time.time() - self.ttl_seconds
        removed = 0
        while self._index and self._index[0][1] < cutoff:
            event_id, _ = self._index.popleft()
            removed += self._remove_entry(event_id)
        if removed > 0:
            logger.info("NewsDeduplicator pruned", extra={"removed": removed})
        return removed

    def _evict_oldest(self, n: int) -> None:
        """Evict the oldest n entries."""
        for _ in range(min(n, len(self._index))):
            event_id, _ = self._index.popleft()
            self._remove_entry(event_id)

    def _remove_entry(self, event_id: str) -> int:
        """Remove entry."""
        if self._lsh is None:
            self._fallback_texts.pop(event_id, None)
            self._key_set.discard(event_id)
            return 1
        try:
            self._lsh.remove(event_id)
            self._key_set.discard(event_id)
            return 1
        except KeyError:
            return 0

    def stats(self) -> dict[str, float]:
        """Stats."""
        return {
            "total_checks": self._total_checks,
            "duplicates_blocked": self._duplicates_blocked,
            "dup_rate": round(self._duplicates_blocked / max(1, self._total_checks), 3),
            "index_size": len(self._index),
            "threshold": self.threshold,
        }

_dedup: NewsDeduplicator | None = None

def get_deduplicator() -> NewsDeduplicator:
    """Get deduplicator."""
    global _dedup
    if _dedup is None:
        _dedup = NewsDeduplicator()
    return _dedup
