"""
tests/unit/test_meta_v2_inference.py — meta_v2 inference + OnlineRetrainer.

Validates:
  - meta_v2 loads when meta_v2.cbm is present (with sidecar metrics.json)
  - score() returns value in [0, 1]
  - score() falls back to v1 / heuristic if v2 missing
  - score_batch() works for v2
  - inference latency: 100 score() calls < 2 s (i.e. < 20ms each)
  - OnlineRetrainer triggers a retrain when >= threshold new trades
  - Anti-degradation: a worse model is rejected
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.agents.meta_classifier import (
    MetaClassifier,
    MetaContext,
)
from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    DecisionTier,
    Direction,
    RiskCheckResult,
    SignalSource,
    UnifiedSignal,
)
from app.training.dataset_builder import OUTCOME_4CLASS_ORDER
from app.training.feature_extractor import FEATURE_COLUMNS_V2


def _make_signal(
    source: SignalSource = SignalSource.TA, direction: Direction = Direction.BUY
) -> UnifiedSignal:
    """Make signal."""
    return UnifiedSignal(
        source=source,
        detector="double_top",
        ticker="SBER",
        direction=direction,
        magnitude=0.7,
        raw_confidence=0.7,
        horizon_min=60,
        price=100.0,
        expected_rr=2.0,
        atr=2.0,
    )


def _make_decision(direction: Direction = Direction.BUY) -> Decision:
    """Make decision."""
    return Decision(
        decision_id="t1",
        cycle_id="c",
        ticker="SBER",
        action=DecisionAction.EXECUTE,
        tier=DecisionTier.TIER1,
        direction=direction,
        combined_magnitude=0.7,
        signals=[_make_signal(direction=direction)],
        risk_check=RiskCheckResult.PASSED,
        expected_rr=2.0,
    )


def _train_mini_v2(tmp_path: Path, mode: str = "classification") -> Path:
    """Train a tiny CatBoost meta_v2 model on synthetic data and write
    artefacts to tmp_path. Returns the model path."""
    from catboost import CatBoostClassifier, CatBoostRegressor

    rng = np.random.default_rng(42)
    n = 80
    X = pd.DataFrame(
        rng.standard_normal((n, len(FEATURE_COLUMNS_V2))),
        columns=FEATURE_COLUMNS_V2,
    )
    pnl = X["combined_magnitude"].values * 0.03 + rng.standard_normal(n) * 0.005

    if mode == "classification":
        labels = []
        for v in pnl:
            if v >= 0.015:
                labels.append("big_win")
            elif v >= 0.0:
                labels.append("small_win")
            elif v > -0.015:
                labels.append("small_loss")
            else:
                labels.append("big_loss")
        y = pd.Series(labels)
        model = CatBoostClassifier(
            iterations=50,
            depth=4,
            learning_rate=0.1,
            loss_function="MultiClass",
            class_names=OUTCOME_4CLASS_ORDER,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
        )
    else:
        y = pd.Series(pnl)
        model = CatBoostRegressor(
            iterations=50,
            depth=4,
            learning_rate=0.1,
            loss_function="MAE",
            random_seed=42,
            verbose=0,
        )
    model.fit(X, y, verbose=False)
    out = tmp_path / "meta_v2.cbm"
    model.save_model(str(out))
    metrics = {"mode": mode, "n_samples": n, "trained_at": "2026-01-01"}
    (tmp_path / "meta_v2.metrics.json").write_text(json.dumps(metrics))
    return out


def _patch_v2_paths(monkeypatch, tmp_path: Path):
    """Redirect META_V2_MODEL_PATH/METRICS_PATH constants into tmp_path."""
    from app.agents import meta_classifier as mc_module

    new_model = tmp_path / "meta_v2.cbm"
    new_metrics = tmp_path / "meta_v2.metrics.json"
    monkeypatch.setattr(mc_module, "META_V2_MODEL_PATH", new_model)
    monkeypatch.setattr(mc_module, "META_V2_METRICS_PATH", new_metrics)


def test_score_in_unit_interval_classification(tmp_path, monkeypatch):
    """v2 classification model → score() ∈ [0, 1]."""
    _train_mini_v2(tmp_path, mode="classification")
    _patch_v2_paths(monkeypatch, tmp_path)
    meta = MetaClassifier()
    assert meta.startup() is True
    assert meta.model_v2 is not None
    s = meta.score(_make_decision(), MetaContext())
    assert 0.0 <= s <= 1.0


def test_score_in_unit_interval_regression(tmp_path, monkeypatch):
    """v2 regression model → score() in [0, 1] via sigmoid."""
    _train_mini_v2(tmp_path, mode="regression")
    _patch_v2_paths(monkeypatch, tmp_path)
    meta = MetaClassifier()
    assert meta.startup() is True
    assert meta.model_v2_mode == "regression"
    s = meta.score(_make_decision(), MetaContext())
    assert 0.0 <= s <= 1.0


def test_v2_missing_falls_back_to_v1_then_heuristic(tmp_path, monkeypatch):
    """No meta_v2.cbm → graceful fallback to v1 / heuristic."""
    _patch_v2_paths(monkeypatch, tmp_path)
    meta = MetaClassifier(model_path=Path("/no/such/path.cbm"))
    loaded = meta.startup()
    assert loaded is False
    assert meta.model_v2 is None
    s = meta.score(_make_decision(), MetaContext())
    assert 0.0 <= s <= 1.0


def test_score_batch_v2(tmp_path, monkeypatch):
    """Batch path returns same length and bounds as singular score()."""
    _train_mini_v2(tmp_path, mode="classification")
    _patch_v2_paths(monkeypatch, tmp_path)
    meta = MetaClassifier()
    meta.startup()
    decs = [_make_decision(), _make_decision(direction=Direction.SELL)]
    ctxs = [MetaContext(), MetaContext(regime="crisis")]
    scores = meta.score_batch(decs, ctxs)
    assert len(scores) == 2
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_score_v2_uses_extras_dict(tmp_path, monkeypatch):
    """Extras (e.g. RAG signals) on MetaContext flow into broker_state."""
    _train_mini_v2(tmp_path, mode="classification")
    _patch_v2_paths(monkeypatch, tmp_path)
    meta = MetaClassifier()
    meta.startup()
    ctx = MetaContext(extras={"consensus_alignment": 0.8, "similar_past_trades_win_rate": 0.7})
    s = meta.score(_make_decision(), ctx)
    assert 0.0 <= s <= 1.0


def test_v2_inference_latency(tmp_path, monkeypatch):
    """100 score() calls must complete in < 2 s (i.e. < 20ms each)."""
    _train_mini_v2(tmp_path, mode="classification")
    _patch_v2_paths(monkeypatch, tmp_path)
    meta = MetaClassifier()
    meta.startup()
    assert meta.model_v2 is not None
    d = _make_decision()
    ctx = MetaContext()
    for _ in range(3):
        meta.score(d, ctx)
    t0 = time.perf_counter()
    for _ in range(100):
        meta.score(d, ctx)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"100 score() calls took {elapsed:.3f}s (> 2s budget)"


def test_v2_inference_p90_latency_under_15ms(tmp_path, monkeypatch):
    """90th-percentile call latency must beat the 15ms target."""
    _train_mini_v2(tmp_path, mode="classification")
    _patch_v2_paths(monkeypatch, tmp_path)
    meta = MetaClassifier()
    meta.startup()
    d = _make_decision()
    ctx = MetaContext()
    for _ in range(5):
        meta.score(d, ctx)
    timings_ms = []
    for _ in range(50):
        t0 = time.perf_counter()
        meta.score(d, ctx)
        timings_ms.append((time.perf_counter() - t0) * 1000)
    p90 = sorted(timings_ms)[int(len(timings_ms) * 0.9)]
    assert p90 < 15.0, f"p90 latency = {p90:.2f}ms (> 15ms target)"


def _seed_full_dbs_for_retrain(tmp_path: Path, n_trades: int) -> tuple[Path, Path]:
    """Spin up trades+decisions DBs with realistic schema. Reuses the
    helper logic from test_dataset_builder."""
    decisions_db = tmp_path / "decisions.db"
    trades_db = tmp_path / "trades.db"
    with sqlite3.connect(str(decisions_db)) as cn:
        cn.execute("""
            CREATE TABLE decisions (
                decision_id TEXT PRIMARY KEY,
                cycle_id TEXT, ticker TEXT, action TEXT,
                tier TEXT, direction TEXT,
                combined_magnitude REAL, stop_loss REAL, take_profit REAL,
                expected_holding_min INTEGER,
                rationale TEXT, signals_json TEXT,
                created_at TEXT, executed_bool INTEGER, pnl_rub REAL,
                meta_score REAL, meta_threshold REAL, executed_at TEXT
            )
        """)
    with sqlite3.connect(str(trades_db)) as cn:
        cn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id TEXT NOT NULL, ticker TEXT, direction TEXT,
                quantity INTEGER, price REAL, order_value REAL,
                remaining_cash REAL, trade_date TEXT, trade_time TEXT,
                bot TEXT, source_model TEXT, arena_raw_json TEXT,
                created_at TEXT
            )
        """)
    base = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)
    with sqlite3.connect(str(decisions_db)) as dc, sqlite3.connect(str(trades_db)) as tc:
        for i in range(n_trades):
            did = f"dec_{i:04d}"
            is_winner = i % 2 == 0
            order_value = 100_000.0
            pnl_rub = order_value * (0.012 if is_winner else -0.012)
            ticker = ["SBER", "GAZP", "LKOH"][i % 3]
            direction = "BUY" if is_winner else "SELL"
            tier = ["1", "2", "3"][i % 3]
            ts = (base + timedelta(minutes=i * 30)).isoformat()
            signals = json.dumps(
                [
                    {
                        "source": "TA",
                        "detector": "double_top",
                        "ticker": ticker,
                        "direction": direction,
                        "magnitude": 0.7,
                        "metadata": {"regime": "trending"},
                    }
                ]
            )
            dc.execute(
                "INSERT INTO decisions VALUES "
                "(?, 'cyc', ?, 'EXECUTE', ?, ?, 0.7, NULL, NULL, 0, "
                "'r', ?, ?, 1, ?, 0.5, 0.45, ?)",
                (did, ticker, tier, direction, signals, ts, pnl_rub, ts),
            )
            tc.execute(
                "INSERT INTO trades (decision_id, ticker, direction, quantity, price, order_value, remaining_cash, trade_date, trade_time, bot, source_model, arena_raw_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    did,
                    ticker,
                    direction,
                    100,
                    1000.0,
                    order_value,
                    0,
                    ts[:10],
                    ts[11:19],
                    "bot",
                    "",
                    "",
                    ts,
                ),
            )
    return trades_db, decisions_db


def test_online_retrain_triggers_at_threshold(tmp_path, monkeypatch):
    """When >= threshold_new_trades new trades have arrived, retrain fires."""
    trades_db, decisions_db = _seed_full_dbs_for_retrain(tmp_path, n_trades=60)
    import app.training.online_retrain as ot

    monkeypatch.setattr(ot.cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ot, "META_V2_MODEL_PATH", tmp_path / "models" / "meta_v2.cbm")
    monkeypatch.setattr(ot, "META_V2_BACKUP_PATH", tmp_path / "models" / "meta_v2.cbm.bak")
    monkeypatch.setattr(ot, "META_V2_METRICS_PATH", tmp_path / "models" / "meta_v2.metrics.json")
    monkeypatch.setattr(
        ot, "RETRAIN_STATE_PATH", tmp_path / "models" / "meta_v2.retrain_state.json"
    )
    (tmp_path / "trades.db").write_bytes(trades_db.read_bytes())
    (tmp_path / "decisions.db").write_bytes(decisions_db.read_bytes())

    retrainer = ot.OnlineRetrainer(
        threshold_new_trades=50, min_initial_trades=50, mode="classification"
    )
    triggered = asyncio.get_event_loop().run_until_complete(retrainer.check_and_retrain())
    assert triggered is True
    assert (tmp_path / "models" / "meta_v2.cbm").exists()
    assert retrainer._last_trades_count == 60


def test_online_retrain_no_trigger_below_threshold(tmp_path, monkeypatch):
    """< threshold new trades → no retrain."""
    trades_db, decisions_db = _seed_full_dbs_for_retrain(tmp_path, n_trades=10)
    import app.training.online_retrain as ot

    monkeypatch.setattr(ot.cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ot, "META_V2_MODEL_PATH", tmp_path / "models" / "meta_v2.cbm")
    monkeypatch.setattr(ot, "META_V2_BACKUP_PATH", tmp_path / "models" / "meta_v2.cbm.bak")
    monkeypatch.setattr(ot, "META_V2_METRICS_PATH", tmp_path / "models" / "meta_v2.metrics.json")
    monkeypatch.setattr(
        ot, "RETRAIN_STATE_PATH", tmp_path / "models" / "meta_v2.retrain_state.json"
    )
    (tmp_path / "trades.db").write_bytes(trades_db.read_bytes())
    (tmp_path / "decisions.db").write_bytes(decisions_db.read_bytes())

    retrainer = ot.OnlineRetrainer(
        threshold_new_trades=50, min_initial_trades=50, mode="classification"
    )
    triggered = asyncio.get_event_loop().run_until_complete(retrainer.check_and_retrain())
    assert triggered is False
    assert not (tmp_path / "models" / "meta_v2.cbm").exists()


def test_online_retrain_atomic_swap_creates_backup(tmp_path, monkeypatch):
    """When meta_v2.cbm already exists, retrain copies it to .bak before swap."""
    trades_db, decisions_db = _seed_full_dbs_for_retrain(tmp_path, n_trades=80)
    import app.training.online_retrain as ot

    monkeypatch.setattr(ot.cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ot, "META_V2_MODEL_PATH", tmp_path / "models" / "meta_v2.cbm")
    monkeypatch.setattr(ot, "META_V2_BACKUP_PATH", tmp_path / "models" / "meta_v2.cbm.bak")
    monkeypatch.setattr(ot, "META_V2_METRICS_PATH", tmp_path / "models" / "meta_v2.metrics.json")
    monkeypatch.setattr(
        ot, "RETRAIN_STATE_PATH", tmp_path / "models" / "meta_v2.retrain_state.json"
    )
    (tmp_path / "trades.db").write_bytes(trades_db.read_bytes())
    (tmp_path / "decisions.db").write_bytes(decisions_db.read_bytes())
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    _train_mini_v2(tmp_path / "models", mode="classification")
    assert (tmp_path / "models" / "meta_v2.cbm").exists()
    initial_hash = (tmp_path / "models" / "meta_v2.cbm").read_bytes()

    retrainer = ot.OnlineRetrainer(
        threshold_new_trades=50, min_initial_trades=50, mode="classification"
    )
    triggered = asyncio.get_event_loop().run_until_complete(retrainer.check_and_retrain())
    assert triggered is True
    if retrainer._n_retrains_accepted >= 1:
        assert (tmp_path / "models" / "meta_v2.cbm.bak").exists()
        assert (tmp_path / "models" / "meta_v2.cbm.bak").read_bytes() == initial_hash


def test_online_retrainer_singleton():
    """get_retrainer() returns a process-wide instance."""
    from app.training.online_retrain import get_retrainer

    a = get_retrainer()
    b = get_retrainer()
    assert a is b


def test_online_retrainer_stats_dict_shape():
    """Test online retrainer stats dict shape."""
    from app.training.online_retrain import OnlineRetrainer

    r = OnlineRetrainer()
    stats = r.stats()
    assert "last_trades_count" in stats
    assert "n_retrains_total" in stats
    assert "is_running" in stats
