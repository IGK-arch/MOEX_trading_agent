"""Phase 27.9 — Meta-classifier wrapper with per-session models.

Wraps an existing :class:`app.agents.meta_classifier.MetaClassifier`
instance. When a session-specific CatBoost model is available on disk
(``data/models/meta_session_<label>.cbm``) the wrapper routes scoring
calls to that model; otherwise it transparently falls back to the base
classifier.

The wrapper purposely does **not** mutate the base classifier — it only
delegates calls. This keeps the rest of the system safe while a parallel
agent (#77) is refactoring ``meta_classifier.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import app.config as cfg
from app.agents.meta_classifier import MetaClassifier, MetaContext
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from catboost import CatBoostClassifier  # type: ignore

    _HAS_CATBOOST = True
except ImportError:  # pragma: no cover
    _HAS_CATBOOST = False

try:
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _HAS_PANDAS = False

_SESSION_MODEL_LABELS: tuple[str, ...] = (
    "morning_open",
    "morning",
    "midday",
    "closing",
    "evening",
)

@dataclass
class SessionModelEntry:
    """One per-session CatBoost model entry."""

    label: str
    path: Path
    model: Any
    feature_names: list[str] | None = None

class MetaPerSessionWrapper:
    """Delegating meta-classifier with per-session model overrides.

    The wrapper is duck-typed against :class:`MetaClassifier`: ``.score()``
    and ``.score_batch()`` accept the same arguments. Callers pass an
    optional ``session_label`` to route to the appropriate model.
    """

    def __init__(
        self,
        base: MetaClassifier | None = None,
        models_dir: Path | None = None,
    ) -> None:
        """Init."""
        self.base: MetaClassifier = base or MetaClassifier()
        if not getattr(self.base, "_loaded", False):
            try:
                self.base.startup()
            except Exception:  # pragma: no cover - defensive
                pass
        self.models_dir: Path = models_dir or (cfg.DATA_DIR / "models")
        self.session_models: dict[str, SessionModelEntry] = {}
        self._try_load_session_models()

    def _try_load_session_models(self) -> None:
        """Scan ``models_dir`` for ``meta_session_<label>.cbm`` files."""
        if not _HAS_CATBOOST:
            logger.info(
                "CatBoost not installed — per-session meta wrapper falls back to base",
            )
            return
        if not self.models_dir.exists():
            return
        for label in _SESSION_MODEL_LABELS:
            path = self.models_dir / f"meta_session_{label}.cbm"
            if not path.exists():
                continue
            try:
                model = CatBoostClassifier()
                model.load_model(str(path))
                try:
                    feature_names = list(model.feature_names_)
                except Exception:
                    feature_names = None
                self.session_models[label] = SessionModelEntry(
                    label=label,
                    path=path,
                    model=model,
                    feature_names=feature_names,
                )
                logger.info(
                    "Per-session meta model loaded",
                    extra={"session": label, "path": str(path)},
                )
            except Exception as exc:
                logger.warning(
                    "Per-session meta model load failed",
                    extra={"session": label, "error": str(exc)},
                )

    def has_session_model(self, session_label: str | None) -> bool:
        """Whether a dedicated model exists for ``session_label``."""
        if not session_label:
            return False
        return session_label in self.session_models

    def score(
        self,
        decision: Any,
        context: MetaContext,
        session_label: str | None = None,
    ) -> float:
        """Score one decision; route to session model when available."""
        if session_label and session_label in self.session_models:
            try:
                return self._score_with_entry(
                    self.session_models[session_label],
                    decision,
                    context,
                )
            except Exception as exc:
                logger.warning(
                    "Session meta scoring failed — fallback to base",
                    extra={"session": session_label, "error": str(exc)},
                )
        return self.base.score(decision, context)

    def score_batch(
        self,
        decisions: list[Any],
        contexts: list[MetaContext],
        session_labels: Iterable[str | None] | None = None,
    ) -> list[float]:
        """Score a batch of decisions, dispatching per session."""
        if not decisions:
            return []
        if len(decisions) != len(contexts):
            raise ValueError("decisions and contexts must have same length")
        labels_list: list[str | None]
        if session_labels is None:
            labels_list = [None] * len(decisions)
        else:
            labels_list = list(session_labels)
            if len(labels_list) != len(decisions):
                raise ValueError("session_labels length mismatch")

        out: list[float] = [0.0] * len(decisions)

        per_session: dict[str | None, list[int]] = {}
        for i, lbl in enumerate(labels_list):
            key = lbl if lbl and lbl in self.session_models else None
            per_session.setdefault(key, []).append(i)

        for key, idxs in per_session.items():
            sub_dec = [decisions[i] for i in idxs]
            sub_ctx = [contexts[i] for i in idxs]
            if key is None:
                sub_scores = self.base.score_batch(sub_dec, sub_ctx)
            else:
                entry = self.session_models[key]
                try:
                    sub_scores = self._score_batch_with_entry(
                        entry,
                        sub_dec,
                        sub_ctx,
                    )
                except Exception as exc:
                    logger.warning(
                        "Session meta batch scoring failed — fallback",
                        extra={"session": key, "error": str(exc)},
                    )
                    sub_scores = self.base.score_batch(sub_dec, sub_ctx)
            for j, i in enumerate(idxs):
                out[i] = float(sub_scores[j])
        return out

    def _score_with_entry(
        self,
        entry: SessionModelEntry,
        decision: Any,
        context: MetaContext,
    ) -> float:
        """Score with entry."""
        return self._score_batch_with_entry(entry, [decision], [context])[0]

    def _score_batch_with_entry(
        self,
        entry: SessionModelEntry,
        decisions: list[Any],
        contexts: list[MetaContext],
    ) -> list[float]:
        """Score batch with entry."""
        feats = [
            MetaClassifier.build_features(d, c) for d, c in zip(decisions, contexts, strict=False)
        ]
        cols = entry.feature_names or list(feats[0].keys())
        if _HAS_PANDAS:
            rows = [[f.get(c, 0.0) for c in cols] for f in feats]
            X = pd.DataFrame(rows, columns=cols)
            proba = entry.model.predict_proba(X)
            return [float(p[1]) for p in proba]
        rows = [[f.get(c, 0.0) for c in cols] for f in feats]
        proba = entry.model.predict_proba(rows)
        return [float(p[1]) for p in proba]

_wrapper: MetaPerSessionWrapper | None = None

def get_meta_per_session_wrapper() -> MetaPerSessionWrapper:
    """Module-level lazy accessor, mirroring ``get_meta_classifier``."""
    global _wrapper
    if _wrapper is None:
        _wrapper = MetaPerSessionWrapper()
    return _wrapper

__all__ = [
    "MetaPerSessionWrapper",
    "SessionModelEntry",
    "get_meta_per_session_wrapper",
]
