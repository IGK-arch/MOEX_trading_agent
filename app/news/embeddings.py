"""
app/news/embeddings.py — Optional semantic similarity via sentence-transformers.

Phase 13. The NewsHistoryStore default Jaccard similarity is fast but
shallow ("Газпром снизил дивиденды" and "Сбербанк объявил buyback" are
very different events but share tokens like "снизил", "объявил").

This module provides a graceful upgrade path: if a sentence-transformers
model is available locally, we use cosine similarity over dense embeddings;
otherwise, the caller falls back to Jaccard automatically.

**Why opt-in.** sentence-transformers + a multilingual model adds ~400 MB
to the Docker image and 50-150 ms per encode. On a CPU-only k8s pod with
4 vCPU it's noticeable; we don't want to force it on every team.

**How to enable.** Set `cfg.NEWS_EMBEDDINGS_ENABLED = True` (env
`NEWS_EMBEDDINGS_ENABLED=1`) and install
`sentence-transformers>=2.2,<3` and a model. The recommended choices,
in order of preference for MOEX use:

  1. **ai-forever/FRIDA** (best RU + EN, ~1.5 GB) — top quality but big
  2. **intfloat/multilingual-e5-small** (~470 MB) — good RU support, much
     smaller; preferred default
  3. **ai-forever/sbert_large_nlu_ru** (~1.6 GB) — RU only, classic baseline

The model is loaded lazily on first call. Encodes are sync (run in a
thread when called from async code) but cached in memory by full-text
SHA1 so repeat queries on the same headline are free.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Any

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

_MODEL_LOCK = threading.Lock()
_MODEL: Any = None
_MODEL_NAME: str = ""
_CACHE: dict[str, np.ndarray] = {}
_CACHE_MAX = 4096

def _model_id() -> str:
    """Resolve which model to load, with env override + sensible default."""
    return os.getenv(
        "NEWS_EMBEDDINGS_MODEL",
        "intfloat/multilingual-e5-small",
    )

def is_enabled() -> bool:
    """Return True if embeddings are configured AND importable."""
    if not getattr(cfg, "NEWS_EMBEDDINGS_ENABLED", False):
        return False
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False

def _load_model_once() -> Any:
    """Lazy load + cache the model under a lock. Returns None on failure."""
    global _MODEL, _MODEL_NAME
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            logger.warning("sentence-transformers not installed — embeddings disabled")
            return None
        model_id = _model_id()
        try:
            logger.info("Loading news embeddings model", extra={"model": model_id})
            _MODEL = SentenceTransformer(model_id)
            _MODEL_NAME = model_id
            logger.info("News embeddings model loaded", extra={"model": model_id})
            return _MODEL
        except Exception as exc:
            logger.error(
                "News embeddings model load failed", extra={"model": model_id, "error": str(exc)}
            )
            return None

def _cache_key(text: str) -> str:
    """Cache key."""
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def encode(text: str) -> np.ndarray | None:
    """
    Return the (normalised) embedding for a single text, or None when
    embeddings are unavailable. Synchronous — wrap with asyncio.to_thread
    when calling from an event loop.
    """
    if not _HAS_NUMPY or not is_enabled() or not text:
        return None
    key = _cache_key(text)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    model = _load_model_once()
    if model is None:
        return None
    try:
        vec = model.encode(text, normalize_embeddings=True)
        arr = np.asarray(vec, dtype=np.float32)
        if len(_CACHE) > _CACHE_MAX:
            for k in list(_CACHE.keys())[: _CACHE_MAX // 4]:
                _CACHE.pop(k, None)
        _CACHE[key] = arr
        return arr
    except Exception as exc:
        logger.warning("Embeddings encode failed", extra={"error": str(exc)})
        return None

def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Cosine similarity for unit-norm vectors. Returns 0.0 when input invalid."""
    if not _HAS_NUMPY or a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

def similarity(text_a: str, text_b: str) -> float:
    """Convenience: encode both + cosine. Returns 0.0 if embeddings disabled."""
    if not is_enabled():
        return 0.0
    va = encode(text_a)
    vb = encode(text_b)
    return cosine(va, vb)

__all__ = ["is_enabled", "encode", "cosine", "similarity"]
