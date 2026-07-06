"""RAG vector store for news/analytics — used for morning consensus + reactive
similarity search.

Phase 27.4 (RAG Consensus). Persists news embeddings under data/rag/ and
exposes a small async-friendly API:

  * ``add_news`` — index a normalized news event (sync, called from the
    NewsLLM consumer loop or from build_morning_consensus pre-fetch).
  * ``search`` — top-k cosine-similarity hits, optionally filtered by
    ticker(s) and recency (hours).
  * ``get_recent_for_ticker`` — flat list of recent (event_id, headline,
    body, ts_utc) tuples for the morning consensus prompt builder.
  * ``prune_older_than`` — drop entries older than N hours, called from the
    nightly memory-decay scheduler.

Embedder
--------
We try ``sentence-transformers`` (multilingual MiniLM, 384-d) first; if it
is unavailable in the runtime environment (no GPU pod, missing wheel) we
fall back to a hash-based bag-of-words pseudo-embedder. The fallback is
intentionally deterministic — it lets us preserve the API contract (cosine
sim, top-k) even without the ML dependency installed. Quality is
materially worse than real sentence transformers, but the RAG layer still
adds value over a pure keyword search.

Backend
-------
ChromaDB is the first-choice backend (already in pyproject deps). When
``chromadb`` is missing OR fails to initialise (e.g. SQLite WAL conflict
inside the test sandbox) we transparently fall back to JSONL+numpy:
``data/rag/news_index.jsonl`` (metadata) + ``data/rag/embeddings.npy``
(numpy matrix, one row per indexed event). Both paths are atomic on
flush.

The class is intentionally synchronous — embedding + search are sub-50 ms
for the 50-200 entry working set we expect during a trading session;
wrapping every call in asyncio.to_thread would add more overhead than it
saves. The morning consensus task awaits these calls anyway via a
``run_in_executor``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

_EMBED_DIM_FALLBACK = 256
_WORD_RE = re.compile(r"[\w\-]+", re.UNICODE)

@dataclass
class RAGRecord:
    """RAGRecord."""

    event_id: str
    text: str
    headline: str
    body: str
    ts_utc: datetime
    tickers: list[str]
    source: str
    source_tier: str

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "event_id": self.event_id,
            "text": self.text,
            "headline": self.headline,
            "body": self.body,
            "ts_utc": self.ts_utc.astimezone(UTC).isoformat(),
            "tickers": list(self.tickers),
            "source": self.source,
            "source_tier": self.source_tier,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RAGRecord:
        """From dict."""
        ts = d.get("ts_utc")
        if isinstance(ts, str):
            ts_dt = datetime.fromisoformat(ts)
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=UTC)
        elif isinstance(ts, datetime):
            ts_dt = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
        else:
            ts_dt = datetime.now(tz=UTC)
        return cls(
            event_id=str(d["event_id"]),
            text=str(d.get("text") or ""),
            headline=str(d.get("headline") or ""),
            body=str(d.get("body") or ""),
            ts_utc=ts_dt,
            tickers=[str(t).upper() for t in (d.get("tickers") or [])],
            source=str(d.get("source") or ""),
            source_tier=str(d.get("source_tier") or "C"),
        )

def _hash_embed(text: str, dim: int = _EMBED_DIM_FALLBACK) -> np.ndarray:
    """Deterministic hashed bag-of-words embedding (TF-style, L2-normalised).

    Each token of the input is hashed into one of ``dim`` buckets and the
    bucket counter is incremented. We then L2-normalise the resulting
    vector so that cosine similarity reduces to a dot product. This is a
    quick-and-dirty stand-in for sentence-transformers when the latter
    isn't available in the runtime environment.
    """
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec
    toks = _WORD_RE.findall(text.lower())
    if not toks:
        return vec
    for tok in toks:
        idx = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=4).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec

class _EmbedderFallback:
    """No-dep deterministic embedder used when sentence-transformers is
    missing. Returns ``np.ndarray`` of shape ``(dim,)`` (single text) or
    ``(n, dim)`` (batch). API matches ``SentenceTransformer.encode``.
    """

    def __init__(self, dim: int = _EMBED_DIM_FALLBACK) -> None:
        """Init."""
        self.dim = dim
        self.backend = "hash_fallback"

    def encode(self, text: str | list[str], normalize_embeddings: bool = True) -> np.ndarray:
        """Encode."""
        if isinstance(text, str):
            return _hash_embed(text, dim=self.dim)
        out = np.zeros((len(text), self.dim), dtype=np.float32)
        for i, t in enumerate(text):
            out[i] = _hash_embed(t, dim=self.dim)
        return out

def _load_embedder() -> tuple[Any, str, int]:
    """Lazy-load embedder. Returns (model, backend_name, dim)."""
    model_id = cfg.RAG_EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        try:
            model = SentenceTransformer(model_id)
            probe = model.encode("test", normalize_embeddings=True)
            return model, f"sentence-transformers/{model_id}", int(np.asarray(probe).shape[-1])
        except Exception as exc:
            logger.warning(
                "RAG: sentence-transformers load failed, falling back to hash embedder",
                extra={"model": model_id, "error": str(exc)},
            )
    except ImportError:
        logger.info(
            "RAG: sentence-transformers not installed — using hash fallback "
            "(quality degraded but RAG contract preserved)",
            extra={"requested_model": model_id},
        )
    return _EmbedderFallback(), "hash_fallback", _EMBED_DIM_FALLBACK

class RAGStore:
    """Vector store for news + analytics events.

    Thread-safe (single coarse lock around mutating ops). Designed for the
    ~10k-event/day working set of a single trading day: an in-memory numpy
    matrix is plenty.
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        """Init."""
        self.persist_dir = Path(persist_dir) if persist_dir else cfg.RAG_PERSIST_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records: list[RAGRecord] = []
        self._embeddings: np.ndarray | None = None
        self._index_path = self.persist_dir / "news_index.jsonl"
        self._embed_path = self.persist_dir / "embeddings.npy"
        self.embedder, self.embedder_backend, self.embed_dim = _load_embedder()
        self._chroma: Any = None
        self._try_chroma()
        self._load_persisted()
        logger.info(
            "RAGStore initialised",
            extra={
                "persist_dir": str(self.persist_dir),
                "embedder": self.embedder_backend,
                "embed_dim": self.embed_dim,
                "records_loaded": len(self._records),
                "chroma": bool(self._chroma),
            },
        )

    def _try_chroma(self) -> None:
        """Try to attach chromadb (does not affect API contract)."""
        if os.getenv("RAG_DISABLE_CHROMADB", "0") == "1":
            return
        try:
            import chromadb  # type: ignore

            client = chromadb.PersistentClient(path=str(self.persist_dir / "chroma"))
            self._chroma = client.get_or_create_collection(
                name="news_rag",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.debug(
                "RAG: chromadb not used (will fallback to JSONL+numpy)",
                extra={"error": str(exc)},
            )
            self._chroma = None

    def _load_persisted(self) -> None:
        """Load persisted."""
        with self._lock:
            self._records = []
            if self._index_path.exists():
                try:
                    with self._index_path.open("r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                self._records.append(RAGRecord.from_dict(json.loads(line)))
                            except Exception:
                                continue
                except Exception as exc:
                    logger.warning(
                        "RAG: failed to load persisted index",
                        extra={"path": str(self._index_path), "error": str(exc)},
                    )
            if self._embed_path.exists() and self._records:
                try:
                    arr = np.load(self._embed_path)
                    if arr.shape[0] == len(self._records) and arr.shape[1] == self.embed_dim:
                        self._embeddings = arr.astype(np.float32, copy=False)
                    else:
                        logger.info(
                            "RAG: embedding shape mismatch, regenerating",
                            extra={
                                "arr_shape": list(arr.shape),
                                "records": len(self._records),
                                "embed_dim": self.embed_dim,
                            },
                        )
                        self._regenerate_embeddings_unlocked()
                except Exception as exc:
                    logger.warning(
                        "RAG: failed to load embeddings, regenerating",
                        extra={"error": str(exc)},
                    )
                    self._regenerate_embeddings_unlocked()
            elif self._records:
                self._regenerate_embeddings_unlocked()

    def _flush_unlocked(self) -> None:
        """Persist current state. Caller MUST hold ``self._lock``."""
        try:
            tmp = self._index_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for rec in self._records:
                    fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            tmp.replace(self._index_path)
            if self._embeddings is not None:
                tmp_npy = self._embed_path.with_name(self._embed_path.name + ".tmp")
                with tmp_npy.open("wb") as fh:
                    np.save(fh, self._embeddings)
                tmp_npy.replace(self._embed_path)
        except Exception as exc:
            logger.warning(
                "RAG: flush failed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )

    def _regenerate_embeddings_unlocked(self) -> None:
        """Regenerate embeddings unlocked."""
        if not self._records:
            self._embeddings = np.zeros((0, self.embed_dim), dtype=np.float32)
            return
        texts = [r.text or r.headline for r in self._records]
        try:
            mat = self.embedder.encode(texts, normalize_embeddings=True)
            self._embeddings = np.asarray(mat, dtype=np.float32)
        except Exception as exc:
            logger.warning(
                "RAG: embedder.encode failed, using hash fallback",
                extra={"error": str(exc)},
            )
            fb = _EmbedderFallback(dim=self.embed_dim)
            self._embeddings = np.asarray(
                fb.encode(texts, normalize_embeddings=True), dtype=np.float32
            )

    def add_news(
        self,
        event_id: str,
        text: str,
        ts_utc: datetime,
        tickers: Iterable[str],
        source: str = "",
        source_tier: str = "C",
        headline: str = "",
        body: str = "",
    ) -> None:
        """Index a single news event. Idempotent on ``event_id``."""
        if not event_id:
            return
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.replace(tzinfo=UTC)
        clean_text = (text or "").strip()
        if not clean_text:
            clean_text = f"{headline} {body}".strip()
        if not clean_text:
            return
        tickers_norm = sorted({str(t).upper() for t in (tickers or []) if t})
        rec = RAGRecord(
            event_id=str(event_id),
            text=clean_text[:8000],
            headline=(headline or clean_text[:160])[:200],
            body=(body or "")[:8000],
            ts_utc=ts_utc,
            tickers=tickers_norm,
            source=source or "",
            source_tier=source_tier or "C",
        )
        with self._lock:
            existing_idx = next(
                (i for i, r in enumerate(self._records) if r.event_id == rec.event_id),
                None,
            )
            try:
                emb = np.asarray(
                    self.embedder.encode(rec.text, normalize_embeddings=True),
                    dtype=np.float32,
                )
            except Exception as exc:
                logger.warning(
                    "RAG: encode failed on add_news",
                    extra={"event_id": event_id, "error": str(exc)},
                )
                emb = _hash_embed(rec.text, dim=self.embed_dim)
            emb = emb.reshape(-1)
            if emb.shape[0] != self.embed_dim:
                emb = _hash_embed(rec.text, dim=self.embed_dim)
            if existing_idx is not None:
                self._records[existing_idx] = rec
                if self._embeddings is not None:
                    self._embeddings[existing_idx] = emb
            else:
                self._records.append(rec)
                if self._embeddings is None or self._embeddings.size == 0:
                    self._embeddings = emb.reshape(1, -1)
                else:
                    self._embeddings = np.vstack([self._embeddings, emb.reshape(1, -1)])
            if self._chroma is not None:
                try:
                    self._chroma.upsert(
                        ids=[rec.event_id],
                        embeddings=[emb.tolist()],
                        metadatas=[
                            {
                                "tickers": ",".join(rec.tickers),
                                "ts_utc": rec.ts_utc.isoformat(),
                                "source": rec.source,
                                "source_tier": rec.source_tier,
                            }
                        ],
                        documents=[rec.text[:2000]],
                    )
                except Exception as exc:
                    logger.debug(
                        "RAG: chroma upsert failed",
                        extra={"event_id": event_id, "error": str(exc)},
                    )
            self._flush_unlocked()

    def search(
        self,
        query_text: str,
        tickers: list[str] | None = None,
        top_k: int = 5,
        max_age_hours: int = 48,
    ) -> list[dict[str, Any]]:
        """Top-k similar news, optionally filtered by ticker and recency.

        Returns a list of dicts with keys: ``event_id``, ``headline``,
        ``body``, ``ts_utc`` (ISO string), ``tickers``, ``source``,
        ``source_tier``, ``score`` (cosine in [-1, 1], higher = more similar).
        """
        if not query_text or not query_text.strip():
            return []
        with self._lock:
            if not self._records or self._embeddings is None or self._embeddings.size == 0:
                return []
            try:
                q = np.asarray(
                    self.embedder.encode(query_text, normalize_embeddings=True),
                    dtype=np.float32,
                ).reshape(-1)
            except Exception:
                q = _hash_embed(query_text, dim=self.embed_dim)
            if q.shape[0] != self.embed_dim:
                q = _hash_embed(query_text, dim=self.embed_dim)
            now = datetime.now(tz=UTC)
            cutoff = now - timedelta(hours=max(1, max_age_hours))
            ticker_filter = {t.upper() for t in tickers} if tickers else None
            mask = np.zeros(len(self._records), dtype=bool)
            for i, rec in enumerate(self._records):
                if rec.ts_utc < cutoff:
                    continue
                if ticker_filter is not None:
                    if not ticker_filter.intersection(rec.tickers):
                        continue
                mask[i] = True
            if not mask.any():
                return []
            emb_subset = self._embeddings[mask]
            scores = emb_subset @ q
            order = np.argsort(-scores)[: max(1, top_k)]
            idxs_in_records = np.where(mask)[0]
            results: list[dict[str, Any]] = []
            for rank, j in enumerate(order):
                global_idx = int(idxs_in_records[int(j)])
                rec = self._records[global_idx]
                results.append(
                    {
                        "event_id": rec.event_id,
                        "headline": rec.headline,
                        "body": rec.body[:600],
                        "ts_utc": rec.ts_utc.isoformat(),
                        "tickers": list(rec.tickers),
                        "source": rec.source,
                        "source_tier": rec.source_tier,
                        "score": float(scores[int(j)]),
                        "rank": rank + 1,
                    }
                )
            return results

    def get_recent_for_ticker(self, ticker: str, hours: int = 24) -> list[dict[str, Any]]:
        """Return all news for a ticker within ``hours`` of now (UTC),
        ordered most-recent first."""
        if not ticker:
            return []
        ticker_u = ticker.upper()
        now = datetime.now(tz=UTC)
        cutoff = now - timedelta(hours=max(1, hours))
        with self._lock:
            out: list[dict[str, Any]] = []
            for rec in self._records:
                if rec.ts_utc < cutoff:
                    continue
                if ticker_u not in rec.tickers:
                    continue
                out.append(
                    {
                        "event_id": rec.event_id,
                        "headline": rec.headline,
                        "body": rec.body[:600],
                        "ts_utc": rec.ts_utc.isoformat(),
                        "tickers": list(rec.tickers),
                        "source": rec.source,
                        "source_tier": rec.source_tier,
                    }
                )
            out.sort(key=lambda r: r["ts_utc"], reverse=True)
            return out

    def prune_older_than(self, hours: int = 168) -> int:
        """Drop records older than ``hours``. Returns number of pruned entries."""
        cutoff = datetime.now(tz=UTC) - timedelta(hours=max(1, hours))
        with self._lock:
            keep_idx = [i for i, r in enumerate(self._records) if r.ts_utc >= cutoff]
            pruned = len(self._records) - len(keep_idx)
            if pruned <= 0:
                return 0
            new_records = [self._records[i] for i in keep_idx]
            if self._embeddings is not None and self._embeddings.size > 0:
                if keep_idx:
                    self._embeddings = self._embeddings[keep_idx]
                else:
                    self._embeddings = np.zeros((0, self.embed_dim), dtype=np.float32)
            removed_ids = [
                self._records[i].event_id
                for i in range(len(self._records))
                if i not in set(keep_idx)
            ]
            self._records = new_records
            if self._chroma is not None and removed_ids:
                with contextlib.suppress(Exception):
                    self._chroma.delete(ids=removed_ids)
            self._flush_unlocked()
            logger.info(
                "RAG: pruned old records", extra={"pruned": pruned, "kept": len(new_records)}
            )
            return pruned

    def __len__(self) -> int:
        """Len."""
        with self._lock:
            return len(self._records)

    def __repr__(self) -> str:
        """Repr."""
        return (
            f"RAGStore(records={len(self._records)}, embedder={self.embedder_backend}, "
            f"dim={self.embed_dim}, persist_dir={self.persist_dir})"
        )

_rag_store: RAGStore | None = None
_rag_lock = threading.Lock()

def get_rag_store() -> RAGStore:
    """Process-wide singleton."""
    global _rag_store
    with _rag_lock:
        if _rag_store is None:
            _rag_store = RAGStore(cfg.RAG_PERSIST_DIR)
        return _rag_store

__all__ = ["RAGStore", "RAGRecord", "get_rag_store"]
