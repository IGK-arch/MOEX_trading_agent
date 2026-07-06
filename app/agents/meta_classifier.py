"""Secondary CatBoost meta-classifier ("trade or skip").

v1 model: `data/models/catboost_meta.cbm` (28 features, binary outcome).
v2 model: `data/models/meta_v2.cbm` (Phase 27.8 — 100 features, 4-class
          outcome or regression on pnl_pct).

The classifier transparently uses v2 when it's loaded on startup; otherwise
falls back to v1 → heuristic. Signature of `score()` is preserved so
callers (dispatcher / aggregator) don't change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore

    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

META_V2_MODEL_PATH = cfg.DATA_DIR / "models" / "meta_v2.cbm"
META_V2_METRICS_PATH = cfg.DATA_DIR / "models" / "meta_v2.metrics.json"

META_NUMERIC_FEATURES: list[str] = [
    "combined_magnitude",
    "expected_rr",
    "n_signals",
    "n_sources_unique",
    "confluence_mult",
    "ofi",
    "kyles_lambda",
    "vpin",
    "vol_z",
    "spread_bbo_bps",
    "current_dd_pct",
    "daily_pnl_pct",
    "n_open_positions",
    "n_trades_today",
    "winning_streak",
    "losing_streak",
    "atr_pct",
    "hour_of_day",
    "minutes_to_close",
]

META_SOURCE_ONEHOT: list[str] = [
    "src_TA",
    "src_NEWS",
    "src_ANOMALY",
    "src_PAIR",
    "src_MEAN_REV",
]

META_REGIME_ONEHOT: list[str] = [
    "regime_trending",
    "regime_mean_reverting",
    "regime_crisis",
]

META_DIRECTION_ONEHOT: list[str] = ["dir_BUY", "dir_SELL"]

META_FEATURE_COLUMNS: list[str] = (
    META_NUMERIC_FEATURES + META_SOURCE_ONEHOT + META_REGIME_ONEHOT + META_DIRECTION_ONEHOT
)

@dataclass
class MetaContext:
    """Runtime context for meta-scoring."""

    ofi: float = 0.0
    kyles_lambda: float = 0.0
    vpin: float = 0.0
    vol_z: float = 0.0
    spread_bbo_bps: float = 0.0
    atr_pct: float = 0.0

    regime: str = "unknown"

    current_dd_pct: float = 0.0
    daily_pnl_pct: float = 0.0
    n_open_positions: int = 0
    n_trades_today: int = 0
    winning_streak: int = 0
    losing_streak: int = 0

    hour_of_day: int = 12
    minutes_to_close: int = 360

    extras: dict[str, Any] = field(default_factory=dict)

class MetaClassifier:
    """Secondary classifier — decides whether to actually execute a Decision.

    Loading order on startup():
      1. meta_v2.cbm (preferred — 100 features, 4-class or regression)
      2. catboost_meta.cbm (legacy — 28 features, binary)
      3. heuristic (always available, no ML)
    """

    def __init__(self, model_path: Path | None = None) -> None:
        """Init."""
        self.model: Any = None
        self.model_path = model_path or (cfg.DATA_DIR / "models" / "catboost_meta.cbm")
        self._loaded = False
        self._model_feature_names: list[str] | None = None
        self.model_v2: Any = None
        self.model_v2_mode: str = "classification"
        self.feature_extractor_v2: Any = None
        self._model_v2_feature_names: list[str] | None = None

    def startup(self) -> bool:
        """Try to load v2 first; fall back to v1; finally heuristic.

        Returns True iff any ML model loaded successfully.
        """
        if not _HAS_CATBOOST:
            logger.info("CatBoost not installed — meta running in heuristic mode")
            return False
        v2_loaded = self._try_load_v2()
        if not self.model_path.exists():
            logger.info(
                "Meta v1 model not found — heuristic mode unless v2 is loaded",
                extra={"path": str(self.model_path)},
            )
            return v2_loaded
        try:
            model = CatBoostClassifier()
            model.load_model(str(self.model_path))
            self.model = model
            self._loaded = True
            try:
                self._model_feature_names = list(model.feature_names_)
            except Exception:
                self._model_feature_names = None
            logger.info(
                "Meta v1 model loaded",
                extra={
                    "path": str(self.model_path),
                    "n_features": (
                        len(self._model_feature_names) if self._model_feature_names else None
                    ),
                },
            )
            return True
        except Exception as exc:
            logger.error("Meta v1 model load failed", extra={"error": str(exc)})
            return v2_loaded

    def _try_load_v2(self) -> bool:
        """Attempt to load meta_v2.cbm. Returns True on success."""
        if not META_V2_MODEL_PATH.exists():
            return False
        mode = "classification"
        if META_V2_METRICS_PATH.exists():
            try:
                metrics = json.loads(META_V2_METRICS_PATH.read_text())
                mode = str(metrics.get("mode", "classification")).lower()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "meta_v2 metrics.json unreadable, defaulting to classification",
                    extra={"error": str(exc)},
                )
        try:
            m = CatBoostRegressor() if mode == "regression" else CatBoostClassifier()
            m.load_model(str(META_V2_MODEL_PATH))
            self.model_v2 = m
            self.model_v2_mode = mode
            try:
                self._model_v2_feature_names = list(m.feature_names_)
            except Exception:
                self._model_v2_feature_names = None
            from app.training.feature_extractor import FeatureExtractor

            self.feature_extractor_v2 = FeatureExtractor()
            logger.info(
                "meta_v2 loaded",
                extra={
                    "mode": mode,
                    "path": str(META_V2_MODEL_PATH),
                    "n_features": (
                        len(self._model_v2_feature_names) if self._model_v2_feature_names else None
                    ),
                },
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("meta_v2 load failed, falling back to v1", extra={"error": str(exc)})
            self.model_v2 = None
            self.feature_extractor_v2 = None
            return False

    def _inference_columns(self) -> list[str]:
        """Return columns to use at inference (model.feature_names_ if known)."""
        if self._model_feature_names:
            return self._model_feature_names
        return META_FEATURE_COLUMNS

    @staticmethod
    def build_features(decision: Any, context: MetaContext) -> dict[str, float]:
        """Build a single-row feature dict from a Decision + MetaContext.

        Returns:
            dict[str, float]: feature row.
        """
        from app.dispatcher.signal import Direction, SignalSource

        signals = decision.signals or []
        sources = [s.source for s in signals]
        unique_sources = len(set(sources))

        if getattr(cfg, "CONFLUENCE_TIERED_BOOST", True):
            if unique_sources >= 4:
                confluence_mult = 2.0
            elif unique_sources == 3:
                confluence_mult = 1.7
            elif unique_sources == 2:
                confluence_mult = 1.3
            else:
                confluence_mult = 1.0
        else:
            if unique_sources >= 3:
                confluence_mult = 2.0
            elif unique_sources == 2:
                confluence_mult = 1.5
            else:
                confluence_mult = 1.0

        feat: dict[str, float] = {
            "combined_magnitude": float(decision.combined_magnitude),
            "expected_rr": float(min(10.0, decision.expected_rr)),
            "n_signals": float(len(signals)),
            "n_sources_unique": float(unique_sources),
            "confluence_mult": float(confluence_mult),
            "ofi": float(context.ofi),
            "kyles_lambda": float(context.kyles_lambda),
            "vpin": float(context.vpin),
            "vol_z": float(context.vol_z),
            "spread_bbo_bps": float(context.spread_bbo_bps),
            "current_dd_pct": float(context.current_dd_pct),
            "daily_pnl_pct": float(context.daily_pnl_pct),
            "n_open_positions": float(context.n_open_positions),
            "n_trades_today": float(context.n_trades_today),
            "winning_streak": float(context.winning_streak),
            "losing_streak": float(context.losing_streak),
            "atr_pct": float(context.atr_pct),
            "hour_of_day": float(context.hour_of_day),
            "minutes_to_close": float(context.minutes_to_close),
        }

        for src in (
            SignalSource.TA,
            SignalSource.NEWS,
            SignalSource.ANOMALY,
            SignalSource.PAIR,
            SignalSource.MEAN_REV,
        ):
            feat[f"src_{src.value}"] = 1.0 if src in sources else 0.0

        feat["regime_trending"] = 1.0 if context.regime == "trending" else 0.0
        feat["regime_mean_reverting"] = 1.0 if context.regime == "mean_reverting" else 0.0
        feat["regime_crisis"] = 1.0 if context.regime == "crisis" else 0.0

        feat["dir_BUY"] = 1.0 if decision.direction == Direction.BUY else 0.0
        feat["dir_SELL"] = 1.0 if decision.direction == Direction.SELL else 0.0
        return feat

    def score(self, decision: Any, context: MetaContext) -> float:
        """Return P(trade is profitable) in [0, 1].

        Preference order:
            v2 (4-class or regression) → v1 (binary) → heuristic.
        Latency target: < 15ms (the v2 hot path is well under that — the
        FeatureExtractor reports ~250µs/featurize on production-shape data;
        CatBoost predict_proba on a single 100-col row is ~1-3ms).
        """
        if self.model_v2 is not None and self.feature_extractor_v2 is not None and _HAS_PANDAS:
            try:
                broker_state = {
                    "ofi": context.ofi,
                    "vpin": context.vpin,
                    "kyle_lambda": context.kyles_lambda,
                    "vol_z": context.vol_z,
                    "atr_at_entry": context.atr_pct,
                    "current_dd_pct": context.current_dd_pct,
                    "daily_pnl_pct": context.daily_pnl_pct,
                    "n_open_positions": context.n_open_positions,
                    "regime": context.regime,
                    **(context.extras or {}),
                }
                feats = self.feature_extractor_v2.featurize(
                    decision,
                    broker_state,
                    ts_at_entry=datetime.now(tz=UTC),
                )
                cols = self._model_v2_feature_names or self.feature_extractor_v2.feature_columns
                row = [feats.get(c, 0.0) for c in cols]
                X = pd.DataFrame([row], columns=cols)
                if self.model_v2_mode == "regression":
                    pred = float(self.model_v2.predict(X)[0])
                    return float(1.0 / (1.0 + np.exp(-pred * 200)))
                proba = self.model_v2.predict_proba(X)[0]
                classes = list(self.model_v2.classes_)
                p_win = 0.0
                for cls_name, p in zip(classes, proba, strict=False):
                    if str(cls_name) in ("small_win", "big_win"):
                        p_win += float(p)
                return float(max(0.0, min(1.0, p_win)))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "meta_v2 predict failed, falling back to v1/heuristic",
                    extra={"error": str(exc)},
                )

        feat = self.build_features(decision, context)
        if self._loaded and self.model is not None and _HAS_PANDAS:
            try:
                cols = self._inference_columns()
                row = [feat.get(c, 0.0) for c in cols]
                X = pd.DataFrame([row], columns=cols)
                proba = self.model.predict_proba(X)[0, 1]
                return float(proba)
            except Exception as exc:
                logger.warning("Meta v1 predict failed, fallback", extra={"error": str(exc)})
        return self._heuristic_score(feat)

    def score_batch(
        self,
        decisions: list[Any],
        contexts: list[MetaContext],
    ) -> list[float]:
        """Score batch."""
        if not decisions:
            return []
        if len(decisions) != len(contexts):
            raise ValueError("decisions and contexts must have same length")
        if self.model_v2 is not None and self.feature_extractor_v2 is not None and _HAS_PANDAS:
            try:
                feats_v2 = []
                now = datetime.now(tz=UTC)
                for d, ctx in zip(decisions, contexts, strict=False):
                    bs = {
                        "ofi": ctx.ofi,
                        "vpin": ctx.vpin,
                        "kyle_lambda": ctx.kyles_lambda,
                        "vol_z": ctx.vol_z,
                        "atr_at_entry": ctx.atr_pct,
                        "current_dd_pct": ctx.current_dd_pct,
                        "daily_pnl_pct": ctx.daily_pnl_pct,
                        "n_open_positions": ctx.n_open_positions,
                        "regime": ctx.regime,
                        **(ctx.extras or {}),
                    }
                    feats_v2.append(self.feature_extractor_v2.featurize(d, bs, now))
                cols = self._model_v2_feature_names or self.feature_extractor_v2.feature_columns
                rows = [[f.get(c, 0.0) for c in cols] for f in feats_v2]
                X = pd.DataFrame(rows, columns=cols)
                if self.model_v2_mode == "regression":
                    preds = self.model_v2.predict(X)
                    return [float(1.0 / (1.0 + np.exp(-p * 200))) for p in preds]
                proba = self.model_v2.predict_proba(X)
                classes = list(self.model_v2.classes_)
                out: list[float] = []
                for row_proba in proba:
                    p_win = 0.0
                    for cls_name, p in zip(classes, row_proba, strict=False):
                        if str(cls_name) in ("small_win", "big_win"):
                            p_win += float(p)
                    out.append(float(max(0.0, min(1.0, p_win))))
                return out
            except Exception as exc:  # noqa: BLE001
                logger.warning("meta_v2 batch predict failed, fallback", extra={"error": str(exc)})
        feats = [self.build_features(d, c) for d, c in zip(decisions, contexts, strict=False)]
        if self._loaded and self.model is not None and _HAS_PANDAS:
            try:
                cols = self._inference_columns()
                rows = [[f.get(c, 0.0) for c in cols] for f in feats]
                X = pd.DataFrame(rows, columns=cols)
                proba = self.model.predict_proba(X)[:, 1]
                return [float(p) for p in proba]
            except Exception as exc:
                logger.warning("Meta batch predict failed", extra={"error": str(exc)})
        return [self._heuristic_score(f) for f in feats]

    @staticmethod
    def _heuristic_score(feat: dict[str, float]) -> float:
        """Heuristic score for when no trained model is available."""
        base = 0.40

        conf_mult = feat.get("confluence_mult", 1.0)
        conf_bonus = (conf_mult - 1.0) * 0.20

        mag = feat.get("combined_magnitude", 0.0)
        mag_bonus = mag * 0.20

        rr = feat.get("expected_rr", 1.0)
        rr_bonus = min(0.10, max(-0.05, (rr - 1.0) * 0.05))

        ofi = feat.get("ofi", 0.0)
        dir_buy = feat.get("dir_BUY", 0.0)
        dir_sell = feat.get("dir_SELL", 0.0)

        if dir_buy > 0.5:
            ofi_bonus = min(0.08, max(-0.08, ofi * 0.4))
        elif dir_sell > 0.5:
            ofi_bonus = min(0.08, max(-0.08, -ofi * 0.4))
        else:
            ofi_bonus = 0.0

        vpin = feat.get("vpin", 0.0)
        vpin_penalty = -0.10 if vpin > 0.45 else 0.0

        regime_bonus = 0.0
        if feat.get("regime_crisis", 0) > 0.5:
            regime_bonus = -0.10
        elif feat.get("regime_trending", 0) > 0.5:
            regime_bonus = 0.03

        dd = feat.get("current_dd_pct", 0.0)
        dd_penalty = -min(0.15, max(0.0, dd) * 1.5)
        losing = feat.get("losing_streak", 0.0)
        streak_penalty = -min(0.10, losing * 0.04)
        winning = feat.get("winning_streak", 0.0)
        streak_bonus = min(0.05, winning * 0.02)

        score = (
            base
            + conf_bonus
            + mag_bonus
            + rr_bonus
            + ofi_bonus
            + vpin_penalty
            + regime_bonus
            + dd_penalty
            + streak_penalty
            + streak_bonus
        )
        return max(0.05, min(0.95, score))

def adaptive_meta_min_proba(current_dd_pct: float) -> float:
    """v1.0.3 — tighten the meta gate as the book bleeds.

    Returns the per-decision meta-probability threshold to use right now. At
    shallow drawdowns we trade with the configured baseline (``cfg.META_MIN_PROBA``).
    As drawdown deepens we step the threshold up to admit only progressively
    higher-confidence trades, capped at ``base + 0.15`` so the bot doesn't
    starve itself out of recovery.

    The function is intentionally pure: no I/O, no globals beyond reading the
    current config snapshot. Callers that override ``cfg.META_MIN_PROBA`` at
    runtime (e.g. ``turnover_tracker``) compose with this transparently because
    we read ``cfg.META_MIN_PROBA`` at call time, not at import time.

    Args:
        current_dd_pct: current peak-to-trough drawdown expressed as a positive
            fraction (e.g. ``0.045`` for 4.5%). Negative or NaN values are
            treated as zero.

    Returns:
        float: meta probability floor to apply at the gate, in the same scale
            as ``meta_score`` (``[0, 1]``).
    """
    base = float(cfg.META_MIN_PROBA)
    if not getattr(cfg, "ADAPTIVE_META_THRESHOLD", True):
        return base
    try:
        dd = float(current_dd_pct)
    except (TypeError, ValueError):
        return base
    if dd != dd:
        return base
    dd = max(0.0, dd)
    if dd < 0.02:
        return base
    if dd < 0.04:
        return base + 0.05
    if dd < 0.06:
        return base + 0.10
    return base + 0.15

_meta: MetaClassifier | None = None

def get_meta_classifier() -> MetaClassifier:
    """Get meta classifier."""
    global _meta
    if _meta is None:
        _meta = MetaClassifier()
        _meta.startup()
    return _meta

__all__ = [
    "MetaClassifier",
    "MetaContext",
    "META_FEATURE_COLUMNS",
    "META_NUMERIC_FEATURES",
    "META_SOURCE_ONEHOT",
    "META_REGIME_ONEHOT",
    "META_DIRECTION_ONEHOT",
    "adaptive_meta_min_proba",
    "get_meta_classifier",
]
