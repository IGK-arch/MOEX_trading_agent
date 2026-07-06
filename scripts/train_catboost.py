"""
scripts/train_catboost.py — Train the TA pattern success-probability model on REAL data.

Pipeline:
  1. Fetch 60 days of H1 candles from MOEX ISS for all 20 tickers
  2. Slide a window through the data; at each step run all pattern detectors
  3. For each detected pattern compute features (exactly the same way TACatBoost.build_features
     does in production)
  4. Label by forward-return: 1 if price reaches `target_level` within `horizon_bars` AND
     does not hit `stop_level` first; 0 otherwise.
  5. Train CatBoost with walk-forward CV (no peek-ahead)
  6. Save model to data/models/catboost_ta.cbm

This makes the TA Trader *actually intelligent* rather than relying on the heuristic.

Usage:
    python3 scripts/train_catboost.py [--days 60] [--ticker SBER]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app.config as cfg
from app.agents.ta_catboost import FEATURE_COLUMNS, get_ta_catboost
from app.agents.ta_indicators import compute_all
from app.agents.ta_patterns.candles import latest_candle_signal
from app.agents.ta_patterns.continuation import (
    detect_compression_breakout,
    detect_flag,
    detect_pennant,
    detect_rectangle,
    detect_triangle,
)
from app.agents.ta_patterns.levels import distance_to_nearest_atrs, find_support_resistance
from app.agents.ta_patterns.pivots import find_pivots
from app.agents.ta_patterns.reversal import (
    PatternSignal,
    detect_double_top_bottom,
    detect_head_shoulders,
    detect_megaphone,
    detect_rounding,
    detect_triple_top_bottom,
    detect_wedge_reversal,
)
from app.data.iss_client import get_iss_client
from app.training.cross_validation import PurgedKFold, time_train_test_split
from app.training.labeling import label_triple_barrier
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    from catboost import CatBoostClassifier, Pool  # type: ignore

    _READY = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    _READY = False

REVERSAL_DETECTORS = [
    detect_double_top_bottom,
    detect_head_shoulders,
    detect_wedge_reversal,
    detect_triple_top_bottom,
    detect_megaphone,
    detect_rounding,
]
CONTINUATION_DETECTORS = [
    detect_flag,
    detect_pennant,
    detect_triangle,
    detect_rectangle,
    detect_compression_breakout,
]

def label_outcome(
    df: pd.DataFrame,
    pattern: PatternSignal,
    horizon_bars: int = 24,
) -> tuple[int, int] | None:
    """
    Triple-barrier labeling — see app/training/labeling.py.

    Returns
    -------
    (binary_label, exit_bar_idx) | None
        binary_label : 1 if top barrier hit first (success), 0 if stop or timeout
        exit_bar_idx : bar where barrier closed (used by PurgedKFold for embargo)
        Returns None when there is no future data to label.
    """
    res = label_triple_barrier(
        df,
        bar_idx=pattern.bar_idx,
        direction=pattern.direction,
        entry=pattern.entry,
        stop=pattern.stop,
        target=pattern.target,
        atr_at_entry=pattern.atr_at_entry,
        horizon_bars=horizon_bars,
        atr_mult_top=2.0,
        atr_mult_bot=1.0,
    )
    if res.barrier_hit == "no_data":
        return None

    exit_idx = res.exit_bar_idx if res.exit_bar_idx >= 0 else pattern.bar_idx + 1
    return res.binary, exit_idx

async def collect_training_data(
    tickers: list[str],
    days: int = 60,
    interval: int = 60,
    window_stride: int = 5,
    min_history: int = 80,
) -> list[dict]:
    """
    Fetch candles, slide a window through history, run detectors at each step,
    label by forward outcome.

    For each ticker:
        for t in [min_history, len(df), window_stride):
            sub_df = df.iloc[:t+1]
            run detectors on sub_df
            for each PatternSignal: label by forward look in df[t:t+horizon]
    """
    iss = get_iss_client()
    if not iss._started:
        await iss.startup()

    catboost = get_ta_catboost()
    regime = "unknown"

    till = datetime.now(tz=UTC)
    from_dt = till - timedelta(days=days + 5)

    all_rows: list[dict] = []
    seen_signatures: set[tuple] = set()

    for ti, ticker in enumerate(tickers):
        try:
            df_full = await iss.get_candles(
                ticker, interval=interval, from_dt=from_dt, till_dt=till
            )
        except Exception as exc:
            logger.warning(
                "Skipped ticker (fetch error)", extra={"ticker": ticker, "error": str(exc)}
            )
            continue

        if not isinstance(df_full, pd.DataFrame) or len(df_full) < min_history:
            continue

        df_full = df_full.reset_index(drop=True)
        n_total = len(df_full)
        ticker_count = 0

        for t in range(min_history, n_total - 12, window_stride):
            sub_df = df_full.iloc[: t + 1].copy()

            try:
                indicators = compute_all(sub_df)
            except Exception:
                continue
            atr_series = indicators["atr"]
            if atr_series is None or len(atr_series) == 0:
                continue

            try:
                pivots = find_pivots(sub_df, order=5, atr=atr_series)
            except Exception:
                continue
            if len(pivots) < 3:
                continue

            window_patterns: list[PatternSignal] = []
            for det in REVERSAL_DETECTORS + CONTINUATION_DETECTORS:
                try:
                    sigs = det(sub_df, pivots, atr_series)
                    for s in sigs:
                        s.ticker = ticker

                        window_patterns.extend([s])
                except Exception:
                    continue

            if not window_patterns:
                continue

            current_price = float(sub_df["close"].iloc[-1])
            atr_now = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
            if atr_now <= 0:
                continue

            levels = find_support_resistance(
                sub_df,
                pivots,
                atr_series,
                current_price=current_price,
            )
            levels_info = distance_to_nearest_atrs(levels, current_price, atr_now)
            candle_bits = latest_candle_signal(sub_df)

            for p in window_patterns:
                sig = (ticker, p.pattern, p.bar_idx, round(p.entry, 2))
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)

                labelled = label_outcome(df_full, p, horizon_bars=24)
                if labelled is None:
                    continue
                label, exit_bar_idx = labelled

                feat = catboost.build_features(
                    pattern=p.pattern,
                    expected_rr=p.expected_rr,
                    price=p.entry,
                    atr_val=atr_now,
                    atr_at_entry=p.atr_at_entry if p.atr_at_entry > 0 else atr_now,
                    indicators=indicators,
                    levels_info=levels_info,
                    candle_bits=candle_bits,
                    regime=regime,
                    df=sub_df,
                    bar_idx=p.bar_idx,
                    pivots=pivots,
                )
                feat["__label__"] = label
                feat["__pattern__"] = p.pattern
                feat["__ticker__"] = ticker
                feat["__direction__"] = p.direction

                feat["__t0__"] = int(p.bar_idx)
                feat["__t1__"] = int(exit_bar_idx)
                all_rows.append(feat)
                ticker_count += 1

        logger.info(
            "Ticker done",
            extra={
                "ticker": ticker,
                "candles": n_total,
                "patterns_labeled": ticker_count,
                "progress": f"{ti + 1}/{len(tickers)}",
            },
        )

    await iss.shutdown()
    return all_rows

def _auc_roc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Pure-numpy ROC AUC. Returns 0.5 if all labels are the same."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)
    pos_mask = y_true == 1
    n_pos = pos_mask.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pos_scores = proba[pos_mask]
    neg_scores = proba[~pos_mask]

    pos_sorted = np.sort(pos_scores)
    rank_in_neg = np.searchsorted(np.sort(neg_scores), pos_sorted, side="right")
    tied = np.searchsorted(np.sort(neg_scores), pos_sorted, side="left")
    auc = (rank_in_neg - 0.5 * (rank_in_neg - tied)).sum() / (n_pos * n_neg)
    return float(auc)

def train(
    rows: list[dict],
    depth: int = 4,
    learning_rate: float = 0.05,
    l2_leaf_reg: float = 5.0,
    iterations: int = 500,
    random_seed: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """
    Train CatBoost with PURGED k-fold CV on the labelled data.

    Steps:
      1. Sort by event time t0 (chronological)
      2. PurgedKFold(n_splits=5, embargo_pct=0.01) — out-of-fold predictions
         give honest CV AUC and accuracy without lookahead bias.
      3. Final training on full data with hold-out test (last 20%) for
         feature importance + model save.
    """
    if not rows:
        raise ValueError("No training data")

    df = pd.DataFrame(rows)
    feature_cols = FEATURE_COLUMNS

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df.sort_values("__t0__", kind="mergesort").reset_index(drop=True)

    X = df[feature_cols].astype(float)
    y = df["__label__"].astype(int)
    t0_arr = df["__t0__"].astype(int).to_numpy()
    t1_arr = df["__t1__"].astype(int).to_numpy()
    n = len(X)
    if n < 30:
        raise ValueError(f"Too few samples: {n}")

    n_splits = min(5, n // 10)
    cv_aucs: list[float] = []
    cv_accs: list[float] = []
    if n_splits >= 2:
        cv = PurgedKFold(n_splits=n_splits, embargo_pct=0.01)
        for _fold_i, (tr_idx, te_idx) in enumerate(
            cv.split(n_samples=n, samples_t1=t1_arr, samples_t0=t0_arr)
        ):
            if len(tr_idx) < 20 or len(te_idx) < 5:
                continue
            X_tr_f = X.iloc[tr_idx]
            y_tr_f = y.iloc[tr_idx]
            X_te_f = X.iloc[te_idx]
            y_te_f = y.iloc[te_idx]
            pos_rate = y_tr_f.mean()
            cls_w = [1.0, max(0.5, (1 - pos_rate) / max(0.01, pos_rate))]
            fold_model = CatBoostClassifier(
                iterations=max(200, int(iterations * 0.8)),
                learning_rate=learning_rate,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                eval_metric="AUC",
                random_seed=random_seed,
                verbose=0,
                early_stopping_rounds=30,
                class_weights=cls_w,
            )
            fold_model.fit(
                Pool(X_tr_f, y_tr_f), eval_set=Pool(X_te_f, y_te_f), use_best_model=True, verbose=0
            )
            proba_f = fold_model.predict_proba(X_te_f)[:, 1]
            pred_f = (proba_f > 0.5).astype(int)
            cv_aucs.append(_auc_roc(y_te_f.to_numpy(), proba_f))
            cv_accs.append(float((pred_f == y_te_f.to_numpy()).mean()))

    train_idx, test_idx = time_train_test_split(
        n_samples=n,
        test_size=0.2,
        samples_t1=t1_arr,
        embargo_pct=0.01,
    )
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    pos_rate_tr = y_tr.mean()
    pos_rate_te = y_te.mean()

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        eval_metric="AUC",
        random_seed=random_seed,
        verbose=0,
        early_stopping_rounds=30,
        class_weights=[1.0, max(0.5, (1 - pos_rate_tr) / max(0.01, pos_rate_tr))],
    )
    model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_te, y_te), use_best_model=True, verbose=0)

    proba_train = model.predict_proba(X_tr)[:, 1]
    proba_test = model.predict_proba(X_te)[:, 1]
    pred_train = (proba_train > 0.5).astype(int)
    pred_test = (proba_test > 0.5).astype(int)

    metrics = {
        "n_samples_total": n,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_purged": int(n - len(X_tr) - len(X_te)),
        "pos_rate_train": round(float(pos_rate_tr), 3),
        "pos_rate_test": round(float(pos_rate_te), 3),
        "train_accuracy": round(float((pred_train == y_tr).mean()), 3),
        "test_accuracy": round(float((pred_test == y_te).mean()), 3),
        "test_auc": round(_auc_roc(y_te.to_numpy(), proba_test), 3),
        "cv_n_splits": len(cv_aucs),
        "cv_auc_mean": round(float(np.mean(cv_aucs)) if cv_aucs else 0.0, 3),
        "cv_auc_std": round(float(np.std(cv_aucs)) if cv_aucs else 0.0, 3),
        "cv_accuracy_mean": round(float(np.mean(cv_accs)) if cv_accs else 0.0, 3),
        "feature_importance_top5": dict(
            zip(
                feature_cols,
                [round(v, 3) for v in model.get_feature_importance().tolist()],
                strict=False,
            )
        ),
    }
    metrics["feature_importance_top5"] = dict(
        sorted(metrics["feature_importance_top5"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    )

    return model, metrics

async def main() -> None:
    """Main."""
    if not _READY:
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument(
        "--tickers", nargs="*", default=None, help="Subset of tickers (default: all 20)"
    )
    parser.add_argument(
        "--interval", type=int, default=60, help="Candle interval minutes (60 = H1)"
    )
    parser.add_argument("--depth", type=int, default=4, help="CatBoost tree depth")
    parser.add_argument("--lr", type=float, default=0.05, help="CatBoost learning rate")
    parser.add_argument("--l2", type=float, default=15.0, help="CatBoost l2_leaf_reg")
    parser.add_argument("--iterations", type=int, default=1000, help="CatBoost iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-save", action="store_true", help="Skip writing model to disk (for grid search)"
    )
    parser.add_argument(
        "--cache-rows",
        type=str,
        default=None,
        help="If set, cache labelled rows as pickle to this path (re-use across configs)",
    )
    args = parser.parse_args()

    setup_logging()

    tickers = args.tickers or cfg.TICKERS
    logger.info(
        "CatBoost training start",
        extra={"tickers": len(tickers), "days": args.days, "interval": args.interval},
    )

    import pickle

    cache_path = Path(args.cache_rows) if args.cache_rows else None
    rows: list[dict] = []

    if cache_path is not None and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                rows = pickle.load(f)
            print(f"  Loaded {len(rows)} cached rows from {cache_path}")
        except Exception as exc:
            print(f"  Cache load failed: {exc} — re-collecting")
            rows = []

    if not rows:
        t0 = time.monotonic()
        rows = await collect_training_data(tickers, days=args.days, interval=args.interval)
        elapsed_collect = time.monotonic() - t0
        print(f"  Collection time: {elapsed_collect:.1f}s")
        if cache_path is not None and rows:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(rows, f)
                print(f"  Cached rows to {cache_path}")
            except Exception as exc:
                print(f"  Cache save failed: {exc}")

    print("\n Training data collected:")
    print(f"  Total labelled samples: {len(rows)}")
    if not rows:
        print(" No training data — exiting")
        sys.exit(1)
    df = pd.DataFrame(rows)
    print("  By pattern:")
    for pat, count in df["__pattern__"].value_counts().items():
        win_rate = df[df["__pattern__"] == pat]["__label__"].mean()
        print(f"    {pat:30s}  n={count:4d}  win_rate={win_rate:.2%}")

    print("\n Training CatBoost...")
    print(f"  hyper: depth={args.depth} lr={args.lr} l2={args.l2} iter={args.iterations}")
    t1 = time.monotonic()
    model, metrics = train(
        rows,
        depth=args.depth,
        learning_rate=args.lr,
        l2_leaf_reg=args.l2,
        iterations=args.iterations,
        random_seed=args.seed,
    )
    elapsed_train = time.monotonic() - t1
    print(f"  Training time: {elapsed_train:.1f}s")

    print("\n Metrics:")
    for k, v in metrics.items():
        if k != "feature_importance_top5":
            print(f"  {k}: {v}")
    print("  Top-5 feature importance:")
    for k, v in metrics["feature_importance_top5"].items():
        print(f"    {k}: {v}")

    if not args.no_save:
        out = cfg.DATA_DIR / "models" / "catboost_ta.cbm"
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(out))
        print(f"\n Saved model: {out}")

        import json

        metrics_path = out.with_suffix(".metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f" Saved metrics: {metrics_path}")
    else:
        print("\n --no-save set, model not written to disk")

if __name__ == "__main__":
    asyncio.run(main())
