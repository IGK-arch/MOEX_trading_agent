"""
app/training/online_retrain.py — Online incremental retraining for meta_v2.

Phase 27.8 (meta_v2). When `threshold_new_trades` (default 50) new
trades have been written to trades.db since the last retrain, this
module:

  1. Loads the current meta_v2 model.
  2. Builds the latest dataset slice via DatasetBuilder.
  3. Fits CatBoost with `init_model=current` so new gradient steps
     start from the existing model (incremental fit).
  4. Runs a guard check: hold-out AUC (or MAE) must not degrade by
     more than `MAX_AUC_DEGRADATION` (default 0.02). If it does, we
     KEEP the current model and only log a warning.
  5. Atomically swaps: rename current → .bak, rename new → current.

Integration in main.py
----------------------
The runner is NOT started automatically (this module does not import
main.py — that would be a circular import). The dispatcher orchestrator
must add:

    from app.training.online_retrain import start_background_loop
    asyncio.create_task(start_background_loop(interval_seconds=60))

Once #75 (evening_pipeline) wires this in, online retrain runs every
60 s, checks trades.db for new entries, and triggers an incremental
fit when the threshold is hit.

Concurrency
-----------
The retrain runs in a separate asyncio task; the main loop is NEVER
blocked. CatBoost fit IS CPU-bound, but on the moderate 100-feature
dataset sizes we see (~hundreds of trades after a couple months),
the full retrain finishes in a few seconds. For very large datasets
we'd push the fit into a process pool — out of scope for v1.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import time
from datetime import UTC, datetime
from typing import Any

try:
    import numpy as np  # noqa: F401  # type: ignore
    import pandas as pd  # type: ignore
    from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore

    _READY = True
except ImportError:  # pragma: no cover
    _READY = False

import app.config as cfg
from app.training.dataset_builder import (
    NON_FEATURE_COLS,
    OUTCOME_4CLASS_ORDER,
    DatasetBuilder,
)
from app.training.feature_extractor import FeatureExtractor
from app.utils.logging import get_logger

logger = get_logger(__name__)

MAX_AUC_DEGRADATION = 0.02

META_V2_MODEL_PATH = cfg.DATA_DIR / "models" / "meta_v2.cbm"
META_V2_BACKUP_PATH = cfg.DATA_DIR / "models" / "meta_v2.cbm.bak"
META_V2_METRICS_PATH = cfg.DATA_DIR / "models" / "meta_v2.metrics.json"
RETRAIN_STATE_PATH = cfg.DATA_DIR / "models" / "meta_v2.retrain_state.json"

class OnlineRetrainer:
    """
    Incremental retrain loop for the meta_v2 model.

    Parameters
    ----------
    threshold_new_trades : int
        After this many NEW trades since the last retrain, fire the
        incremental fit. Default 50.
    min_initial_trades : int
        Below this we don't even bother — too little signal to learn
        anything robust. Default 50.
    mode : str
        "classification" (4-class) or "regression" (pnl_pct).
        Detected from the loaded model when possible.
    """

    def __init__(
        self,
        threshold_new_trades: int = 50,
        min_initial_trades: int = 50,
        mode: str = "classification",
    ) -> None:
        """Init."""
        self.threshold_new_trades = threshold_new_trades
        self.min_initial_trades = min_initial_trades
        self.mode = mode
        self._last_trades_count: int = self._load_retrain_state()
        self._is_running: bool = False
        self._n_retrains_total: int = 0
        self._n_retrains_accepted: int = 0
        self._n_retrains_rejected: int = 0
        self._last_run_at: datetime | None = None
        self._last_status: str = "idle"

    def stats(self) -> dict[str, Any]:
        """Stats."""
        return {
            "last_trades_count": self._last_trades_count,
            "is_running": self._is_running,
            "n_retrains_total": self._n_retrains_total,
            "n_retrains_accepted": self._n_retrains_accepted,
            "n_retrains_rejected": self._n_retrains_rejected,
            "last_run_at": self._last_run_at.isoformat() if self._last_run_at else None,
            "last_status": self._last_status,
        }

    async def check_and_retrain(self) -> bool:
        """
        Called periodically. Returns True if a retrain was triggered.
        """
        if not _READY:
            return False
        if self._is_running:
            return False

        current = self._count_trades()
        delta = current - self._last_trades_count
        if current < self.min_initial_trades:
            return False
        if delta < self.threshold_new_trades:
            return False

        logger.info(
            "Online retrain triggered",
            extra={
                "current_trades": current,
                "delta": delta,
                "threshold": self.threshold_new_trades,
            },
        )
        self._is_running = True
        try:
            await self._retrain_incremental()
            self._last_trades_count = current
            self._save_retrain_state()
            return True
        except Exception as exc:  # noqa: BLE001
            self._last_status = f"error: {exc}"
            logger.error("Online retrain failed", extra={"error": str(exc)})
            return False
        finally:
            self._is_running = False
            self._last_run_at = datetime.now(tz=UTC)

    async def _retrain_incremental(self) -> None:
        """
        Build dataset, fit on it (with init_model when v2 already exists),
        validate against current model on the last 20% holdout, swap iff
        the new model doesn't degrade.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._retrain_incremental_sync)

    def _retrain_incremental_sync(self) -> None:
        """Retrain incremental sync."""
        self._n_retrains_total += 1
        t0 = time.monotonic()
        builder = DatasetBuilder(
            trades_db=cfg.DATA_DIR / "trades.db",
            decisions_db=cfg.DATA_DIR / "decisions.db",
            fe=FeatureExtractor(),
        )
        df = builder.collect_all(days_back=365)
        df = df.dropna(subset=["pnl_pct"]).reset_index(drop=True)
        if len(df) < self.min_initial_trades:
            self._last_status = "skipped_too_few"
            return

        df = builder.compute_labels(df)
        feature_cols = [
            c for c in df.columns if c not in NON_FEATURE_COLS and c in builder.fe.feature_columns
        ]
        n = len(df)
        split = max(10, int(n * 0.8))
        df_tr = df.iloc[:split]
        df_te = df.iloc[split:]

        if self.mode == "classification":
            model_new = CatBoostClassifier(
                iterations=150,
                depth=6,
                learning_rate=0.05,
                loss_function="MultiClass",
                class_names=OUTCOME_4CLASS_ORDER,
                auto_class_weights="Balanced",
                l2_leaf_reg=3.0,
                random_seed=42,
                verbose=0,
            )
            target_col = "outcome_4class"
        else:
            model_new = CatBoostRegressor(
                iterations=150,
                depth=6,
                learning_rate=0.05,
                loss_function="MAE",
                l2_leaf_reg=3.0,
                random_seed=42,
                verbose=0,
            )
            target_col = "pnl_pct"

        init_model = None
        if META_V2_MODEL_PATH.exists():
            try:
                if self.mode == "classification":
                    init_model = CatBoostClassifier()
                else:
                    init_model = CatBoostRegressor()
                init_model.load_model(str(META_V2_MODEL_PATH))
            except Exception as exc:  # noqa: BLE001
                logger.warning("init_model load failed", extra={"error": str(exc)})
                init_model = None

        try:
            model_new.fit(
                df_tr[feature_cols].astype(float),
                df_tr[target_col],
                sample_weight=df_tr["sample_weight"].astype(float).to_numpy(),
                init_model=init_model,
                verbose=False,
            )
        except Exception:
            model_new.fit(
                df_tr[feature_cols].astype(float),
                df_tr[target_col],
                sample_weight=df_tr["sample_weight"].astype(float).to_numpy(),
                verbose=False,
            )

        new_score = _safe_score(model_new, df_te, feature_cols, target_col, self.mode)
        old_score = (
            _safe_score(init_model, df_te, feature_cols, target_col, self.mode)
            if init_model is not None
            else (new_score - 1.0)
        )

        accepted = (init_model is None) or (new_score >= old_score - MAX_AUC_DEGRADATION)
        if accepted:
            self._swap_model(
                model_new,
                metrics={
                    "trained_at": datetime.now(tz=UTC).isoformat(),
                    "mode": self.mode,
                    "n_samples": int(len(df)),
                    "holdout_old": round(float(old_score), 4),
                    "holdout_new": round(float(new_score), 4),
                    "incremental": True,
                    "elapsed_sec": round(time.monotonic() - t0, 2),
                },
            )
            self._n_retrains_accepted += 1
            self._last_status = f"accepted (new={new_score:.4f}, old={old_score:.4f})"
            logger.info(
                "Online retrain accepted",
                extra={"old_score": old_score, "new_score": new_score},
            )
        else:
            self._n_retrains_rejected += 1
            self._last_status = (
                f"rejected new={new_score:.4f}<old={old_score:.4f}-{MAX_AUC_DEGRADATION}"
            )
            logger.warning(
                "Online retrain rejected (degradation)",
                extra={"old_score": old_score, "new_score": new_score},
            )

    def _count_trades(self) -> int:
        """Cheap COUNT(*) of trades. Returns 0 if DB missing."""
        path = cfg.DATA_DIR / "trades.db"
        if not path.exists():
            return 0
        try:
            with sqlite3.connect(str(path)) as cn:
                row = cn.execute("SELECT COUNT(*) FROM trades").fetchone()
                return int(row[0]) if row else 0
        except sqlite3.OperationalError:
            return 0

    def _swap_model(self, new_model, metrics: dict) -> None:
        """Atomic swap: backup old → .bak, write new → main."""
        META_V2_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if META_V2_MODEL_PATH.exists():
            try:
                shutil.copy(META_V2_MODEL_PATH, META_V2_BACKUP_PATH)
            except Exception as exc:  # noqa: BLE001
                logger.warning("backup failed", extra={"error": str(exc)})
        tmp = META_V2_MODEL_PATH.with_suffix(".cbm.tmp")
        new_model.save_model(str(tmp))
        tmp.replace(META_V2_MODEL_PATH)
        existing_metrics: dict = {}
        if META_V2_METRICS_PATH.exists():
            try:
                existing_metrics = json.loads(META_V2_METRICS_PATH.read_text())
            except Exception:
                existing_metrics = {}
        existing_metrics.update(metrics)
        with open(META_V2_METRICS_PATH, "w") as f:
            json.dump(existing_metrics, f, indent=2)

    def _load_retrain_state(self) -> int:
        """Load retrain state."""
        if not RETRAIN_STATE_PATH.exists():
            return 0
        try:
            data = json.loads(RETRAIN_STATE_PATH.read_text())
            return int(data.get("last_trades_count", 0))
        except Exception:
            return 0

    def _save_retrain_state(self) -> None:
        """Save retrain state."""
        RETRAIN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_trades_count": self._last_trades_count,
            "last_run_at": datetime.now(tz=UTC).isoformat(),
            "n_retrains_total": self._n_retrains_total,
            "n_retrains_accepted": self._n_retrains_accepted,
            "n_retrains_rejected": self._n_retrains_rejected,
        }
        with open(RETRAIN_STATE_PATH, "w") as f:
            json.dump(data, f, indent=2)

def _safe_score(
    model,
    df_te: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    mode: str,
) -> float:
    """Compute holdout score (AUC for classification, -MAE for regression)."""
    if model is None or len(df_te) < 5:
        return 0.0
    X_te = df_te[feature_cols].astype(float)
    y_te = df_te[target_col]
    try:
        if mode == "classification":
            proba = model.predict_proba(X_te)
            try:
                from sklearn.metrics import roc_auc_score

                return float(
                    roc_auc_score(
                        y_te.values,
                        proba,
                        multi_class="ovr",
                        average="macro",
                        labels=list(model.classes_),
                    )
                )
            except Exception:
                preds = model.predict(X_te).flatten()
                return float((preds.astype(str) == y_te.values).mean())
        else:
            from sklearn.metrics import mean_absolute_error

            return -float(mean_absolute_error(y_te.values, model.predict(X_te)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("_safe_score failed", extra={"error": str(exc)})
        return 0.0

_global_retrainer: OnlineRetrainer | None = None

def get_retrainer() -> OnlineRetrainer:
    """Process-wide singleton."""
    global _global_retrainer
    if _global_retrainer is None:
        _global_retrainer = OnlineRetrainer()
    return _global_retrainer

async def start_background_loop(interval_seconds: int = 60) -> None:
    """
    Background coroutine. Should be added to main.py as:

        asyncio.create_task(start_background_loop())

    Loops forever, calling check_and_retrain() every `interval_seconds`.
    Errors are logged but never raised — the loop survives partial
    failures.
    """
    retrainer = get_retrainer()
    logger.info(
        "Online retrain loop started",
        extra={
            "interval_seconds": interval_seconds,
            "threshold_new_trades": retrainer.threshold_new_trades,
        },
    )
    while True:
        try:
            await retrainer.check_and_retrain()
        except asyncio.CancelledError:
            logger.info("Online retrain loop cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Online retrain tick error", extra={"error": str(exc)})
        await asyncio.sleep(interval_seconds)

__all__ = [
    "OnlineRetrainer",
    "get_retrainer",
    "start_background_loop",
    "MAX_AUC_DEGRADATION",
    "META_V2_MODEL_PATH",
    "META_V2_BACKUP_PATH",
    "META_V2_METRICS_PATH",
    "RETRAIN_STATE_PATH",
]
