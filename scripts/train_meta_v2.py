"""
scripts/train_meta_v2.py — Train meta-classifier v2 (Phase 27.8).

Improvements over scripts/train_meta.py (v1):
  - 100 features (vs 28 in v1) via FeatureExtractor (one-hot for ticker,
    detector, source, tier, regime; numeric microstructure + time of day)
  - Multi-class outcome (big_win / small_win / small_loss / big_loss)
    OR regression on pnl_pct — selectable via --mode
  - sample_weight by |pnl_pct| so the gradient cares more about the rare
    big winners/losers than the dense ±0.5% noise around commission
  - PurgedKFold cross-validation (Phase 11.1) for honest scores
  - Walk-forward last-20% holdout — emulates the live retrain trigger
  - Atomic model swap with backup (.bak) — safe to call from cron / online
    retrain loop

Cold-start: when there are fewer than --min-samples decisions in
decisions.db + trades.db, exits with code 0 and prints a notice (so
nightly cron is idempotent).

Usage
-----
    python3 scripts/train_meta_v2.py [--days 365] [--mode classification|regression] \\
                                       [--min-samples 50] [--no-online]

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app.config as cfg  # noqa: E402
from app.training.cross_validation import PurgedKFold  # noqa: E402
from app.training.dataset_builder import (  # noqa: E402
    NON_FEATURE_COLS,
    OUTCOME_4CLASS_ORDER,
    DatasetBuilder,
)
from app.training.feature_extractor import FeatureExtractor  # noqa: E402
from app.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore

    _READY = True
except ImportError as exc:  # pragma: no cover
    print(f"Missing dependencies: {exc}")
    _READY = False

META_V2_MODEL_PATH = cfg.DATA_DIR / "models" / "meta_v2.cbm"
META_V2_BACKUP_PATH = cfg.DATA_DIR / "models" / "meta_v2.cbm.bak"
META_V2_METRICS_PATH = cfg.DATA_DIR / "models" / "meta_v2.metrics.json"

def _parse_args() -> argparse.Namespace:
    """Parse args."""
    p = argparse.ArgumentParser(description="Train meta-classifier v2")
    p.add_argument(
        "--days", type=int, default=365, help="Only train on decisions from the last N days"
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Skip training if fewer than this many labeled trades",
    )
    p.add_argument(
        "--mode",
        choices=["classification", "regression"],
        default="classification",
        help="classification: 4-class outcome; regression: predict pnl_pct",
    )
    p.add_argument(
        "--no-online",
        action="store_true",
        help="Disable online incremental hint (purely offline training)",
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Use precomputed dataset parquet/pkl instead of rebuilding",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(META_V2_MODEL_PATH),
        help="Path to write the trained model (default: data/models/meta_v2.cbm)",
    )
    return p.parse_args()

def _build_or_load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    """Build or load dataset."""
    if args.dataset_path:
        path = Path(args.dataset_path)
        if not path.exists():
            print(f"--dataset-path not found: {path}")
            sys.exit(2)
        builder = DatasetBuilder(
            trades_db=cfg.DATA_DIR / "trades.db",
            decisions_db=cfg.DATA_DIR / "decisions.db",
            fe=FeatureExtractor(),
        )
        return builder.load(path)
    fe = FeatureExtractor()
    builder = DatasetBuilder(
        trades_db=cfg.DATA_DIR / "trades.db",
        decisions_db=cfg.DATA_DIR / "decisions.db",
        fe=fe,
    )
    return builder.collect_all(days_back=args.days)

def _train_classification(
    df: pd.DataFrame,
    feature_cols: list[str],
    cv_splits: int,
) -> tuple[object, dict, list[float]]:
    """Train classification."""
    X = df[feature_cols].astype(float)
    y = df["outcome_4class"].astype(str)
    w = df["sample_weight"].astype(float).clip(lower=0.1).to_numpy()

    cv_scores: list[float] = []
    n = len(X)
    if n >= 50 and cv_splits >= 2:
        t1 = np.arange(n) + 1
        kf = PurgedKFold(n_splits=cv_splits, embargo_pct=0.01)
        for tr_idx, te_idx in kf.split(n_samples=n, samples_t1=t1):
            if len(tr_idx) < 20 or len(te_idx) < 5:
                continue
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            w_tr = w[tr_idx]
            fold = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                loss_function="MultiClass",
                class_names=OUTCOME_4CLASS_ORDER,
                auto_class_weights="Balanced",
                l2_leaf_reg=3.0,
                random_seed=42,
                verbose=0,
            )
            fold.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            proba = fold.predict_proba(X_te)
            try:
                from sklearn.metrics import roc_auc_score

                sorted(set(y_te) | set(OUTCOME_4CLASS_ORDER))
                cv_scores.append(
                    float(
                        roc_auc_score(
                            y_te.values,
                            proba,
                            multi_class="ovr",
                            average="macro",
                            labels=list(fold.classes_),
                        )
                    )
                )
            except Exception:
                preds = fold.predict(X_te).flatten()
                cv_scores.append(float((preds.astype(str) == y_te.values).mean()))

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        class_names=OUTCOME_4CLASS_ORDER,
        auto_class_weights="Balanced",
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
    )
    return model, {"target": "outcome_4class"}, cv_scores

def _train_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    cv_splits: int,
) -> tuple[object, dict, list[float]]:
    """Train regression."""
    X = df[feature_cols].astype(float)
    y = df["pnl_pct"].astype(float)
    w = df["sample_weight"].astype(float).clip(lower=0.1).to_numpy()

    cv_scores: list[float] = []
    n = len(X)
    if n >= 50 and cv_splits >= 2:
        t1 = np.arange(n) + 1
        kf = PurgedKFold(n_splits=cv_splits, embargo_pct=0.01)
        for tr_idx, te_idx in kf.split(n_samples=n, samples_t1=t1):
            if len(tr_idx) < 20 or len(te_idx) < 5:
                continue
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            w_tr = w[tr_idx]
            fold = CatBoostRegressor(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                loss_function="MAE",
                l2_leaf_reg=3.0,
                random_seed=42,
                verbose=0,
            )
            fold.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            try:
                from sklearn.metrics import mean_absolute_error

                cv_scores.append(-float(mean_absolute_error(y_te, fold.predict(X_te))))
            except Exception:
                cv_scores.append(0.0)

    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MAE",
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
    )
    return model, {"target": "pnl_pct"}, cv_scores

def _walk_forward(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    is_classification: bool,
) -> float:
    """Refit on first 80%, score on last 20%."""
    n = len(df)
    split = int(n * 0.8)
    if split < 10 or n - split < 5:
        model.fit(
            df[feature_cols].astype(float),
            df[target_col],
            sample_weight=df["sample_weight"].astype(float).to_numpy(),
            verbose=False,
        )
        return 0.0
    df_tr = df.iloc[:split]
    df_te = df.iloc[split:]
    model.fit(
        df_tr[feature_cols].astype(float),
        df_tr[target_col],
        sample_weight=df_tr["sample_weight"].astype(float).to_numpy(),
        verbose=False,
    )
    if is_classification:
        proba = model.predict_proba(df_te[feature_cols].astype(float))
        try:
            from sklearn.metrics import roc_auc_score

            present_classes = list(model.classes_)
            return float(
                roc_auc_score(
                    df_te[target_col].values,
                    proba,
                    multi_class="ovr",
                    average="macro",
                    labels=present_classes,
                )
            )
        except Exception:
            preds = model.predict(df_te[feature_cols].astype(float)).flatten()
            return float((preds.astype(str) == df_te[target_col].values).mean())
    else:
        from sklearn.metrics import mean_absolute_error

        return -float(
            mean_absolute_error(
                df_te[target_col].values,
                model.predict(df_te[feature_cols].astype(float)),
            )
        )

def _save_model_and_metrics(
    model,
    output: Path,
    mode: str,
    metrics: dict,
    feature_cols: list[str],
) -> None:
    """Atomic save with .bak backup."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        backup = META_V2_BACKUP_PATH
        try:
            import shutil

            shutil.copy(output, backup)
            logger.info("Backup created", extra={"backup": str(backup)})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Backup failed", extra={"error": str(exc)})

    tmp = output.with_suffix(".cbm.tmp")
    model.save_model(str(tmp))
    tmp.replace(output)
    logger.info("Model saved", extra={"path": str(output)})

    try:
        importances = list(zip(feature_cols, model.get_feature_importance().tolist(), strict=False))
        importances.sort(key=lambda kv: kv[1], reverse=True)
        top10 = {k: round(float(v), 4) for k, v in importances[:10]}
    except Exception:
        top10 = {}

    metrics_out = {
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "mode": mode,
        "feature_count": len(feature_cols),
        "feature_importance_top10": top10,
        **metrics,
    }
    with open(META_V2_METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info("Metrics saved", extra={"path": str(META_V2_METRICS_PATH)})

def main() -> int:
    """Main."""
    setup_logging()
    args = _parse_args()

    if not _READY:
        return 1

    t_start = time.monotonic()
    print(f"Loading dataset (days={args.days})...")
    df = _build_or_load_dataset(args)

    df = df.dropna(subset=["pnl_pct"]).reset_index(drop=True)
    if len(df) < args.min_samples:
        print(
            f"Not enough labeled samples: {len(df)} < {args.min_samples} — "
            f"meta_v2 not trained. Exit 0 (heuristic / v1 stays active)."
        )
        return 0

    builder = DatasetBuilder(
        trades_db=cfg.DATA_DIR / "trades.db",
        decisions_db=cfg.DATA_DIR / "decisions.db",
        fe=FeatureExtractor(),
    )
    df = builder.compute_labels(df)
    cb = builder.compute_class_balance(df)
    print(f"Dataset: {len(df)} labeled trades")
    print(
        f"Class balance: big_loss={cb['counts']['big_loss']} "
        f"small_loss={cb['counts']['small_loss']} "
        f"small_win={cb['counts']['small_win']} "
        f"big_win={cb['counts']['big_win']}"
    )
    if cb["is_imbalanced"]:
        print(f"  WARN: imbalance_ratio = {cb['imbalance_ratio']:.2f} (>2.0)")
    print(f"  P(any win) = {cb['pos_rate']:.3f}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    expected = set(FeatureExtractor().feature_columns)
    feature_cols = [c for c in feature_cols if c in expected]

    cv_splits = min(5, max(2, len(df) // 20))
    print(f"PurgedKFold cv_splits={cv_splits}, embargo_pct=0.01")

    is_classification = args.mode == "classification"
    if is_classification:
        model, meta, cv_scores = _train_classification(df, feature_cols, cv_splits)
        target_col = "outcome_4class"
    else:
        model, meta, cv_scores = _train_regression(df, feature_cols, cv_splits)
        target_col = "pnl_pct"

    cv_mean = float(np.mean(cv_scores)) if cv_scores else 0.0
    cv_std = float(np.std(cv_scores)) if cv_scores else 0.0
    print(f"CV mean: {cv_mean:.4f}  std: {cv_std:.4f}  (n_folds={len(cv_scores)})")

    holdout_score = _walk_forward(model, df, feature_cols, target_col, is_classification)
    print(f"Walk-forward (last 20%) holdout: {holdout_score:.4f}")

    metrics = {
        "n_samples": int(len(df)),
        "cv_n_folds": int(len(cv_scores)),
        "cv_mean": round(cv_mean, 4),
        "cv_std": round(cv_std, 4),
        "holdout_score": round(float(holdout_score), 4),
        "class_balance": cb,
        "online_enabled": (not args.no_online),
        **meta,
    }
    _save_model_and_metrics(
        model,
        Path(args.output),
        args.mode,
        metrics,
        feature_cols,
    )

    elapsed = time.monotonic() - t_start
    print(f"Total time: {elapsed:.1f}s")
    return 0

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
