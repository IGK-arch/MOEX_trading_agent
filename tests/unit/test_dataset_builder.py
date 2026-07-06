"""
tests/unit/test_dataset_builder.py — meta_v2 DatasetBuilder.

Validates:
  - DataFrame columns match FEATURE_COLUMNS_V2 + meta cols
  - compute_labels adds outcome_bin, outcome_4class, sample_weight
  - 4-class bucketing logic (big_win / small_win / small_loss / big_loss)
  - sample_weight scales with |pnl_pct| and is capped
  - compute_class_balance flags imbalance > 2:1
  - save() round-trips through parquet OR pickle
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from app.training.dataset_builder import (
    LABEL_COLS,
    OUTCOME_4CLASS_ORDER,
    DatasetBuilder,
)
from app.training.feature_extractor import FEATURE_COLUMNS_V2, FeatureExtractor


def _seed_dbs(tmp_path: Path, n_trades: int = 60) -> tuple[Path, Path]:
    """Build a minimal trades.db + decisions.db with `n_trades` executed
    decisions, half winners with +1% pnl and half losers with -1% pnl."""
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
            pnl_rub = order_value * (0.01 if is_winner else -0.01)
            ticker = ["SBER", "GAZP", "LKOH"][i % 3]
            direction = "BUY" if i % 2 == 0 else "SELL"
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
                        "metadata": {"ofi": 0.15, "vpin": 0.1, "regime": "trending"},
                    }
                ]
            )
            dc.execute(
                """
                INSERT INTO decisions VALUES (
                    ?, 'cyc', ?, 'EXECUTE', ?, ?, 0.7, NULL, NULL, 0,
                    'r', ?, ?, 1, ?, 0.55, 0.45, ?
                )
            """,
                (did, ticker, tier, direction, signals, ts, pnl_rub, ts),
            )
            tc.execute(
                """
                INSERT INTO trades (
                    decision_id, ticker, direction, quantity, price,
                    order_value, remaining_cash, trade_date, trade_time,
                    bot, source_model, arena_raw_json, created_at
                ) VALUES (?, ?, ?, 100, 1000.0, ?, 0, ?, ?, 'bot', '', '', ?)
            """,
                (did, ticker, direction, order_value, ts[:10], ts[11:19], ts),
            )
    return trades_db, decisions_db


def test_collect_all_returns_correct_columns(tmp_path):
    """Test collect all returns correct columns."""
    trades_db, decisions_db = _seed_dbs(tmp_path, n_trades=30)
    fe = FeatureExtractor()
    b = DatasetBuilder(trades_db, decisions_db, fe)
    df = b.collect_all(days_back=365)
    for col in FEATURE_COLUMNS_V2:
        assert col in df.columns, f"missing feature col: {col}"
    for col in [
        "pnl_pct",
        "pnl_rub",
        "ticker",
        "direction",
        "ts_entry",
        "ts_exit",
        "exit_reason",
        "holding_min",
    ]:
        assert col in df.columns
    assert len(df) == 30
    assert df["ts_entry"].is_monotonic_increasing


def test_collect_all_with_empty_dbs(tmp_path):
    """Non-existent dbs return an empty DataFrame, not a crash."""
    fe = FeatureExtractor()
    b = DatasetBuilder(
        trades_db=tmp_path / "missing_trades.db",
        decisions_db=tmp_path / "missing_decisions.db",
        fe=fe,
    )
    df = b.collect_all(days_back=365)
    assert df.empty


def test_compute_labels_adds_label_columns(tmp_path):
    """Test compute labels adds label columns."""
    trades_db, decisions_db = _seed_dbs(tmp_path, n_trades=20)
    fe = FeatureExtractor()
    b = DatasetBuilder(trades_db, decisions_db, fe)
    df = b.collect_all(days_back=365)
    df = b.compute_labels(df)
    for col in LABEL_COLS:
        assert col in df.columns, f"missing label: {col}"
    assert df["outcome_bin"].sum() == 10
    assert (df["sample_weight"] >= 0.1).all()


def test_outcome_4class_bucketing(tmp_path):
    """Verify the 4 buckets behave as documented."""
    df = pd.DataFrame(
        {
            "pnl_pct": [0.02, 0.005, -0.005, -0.02, 0.0, float("nan")],
        }
    )
    fe = FeatureExtractor()
    b = DatasetBuilder(Path("nope_t"), Path("nope_d"), fe)
    df = b.compute_labels(df)
    assert df.loc[0, "outcome_4class"] == "big_win"
    assert df.loc[1, "outcome_4class"] == "small_win"
    assert df.loc[2, "outcome_4class"] == "small_loss"
    assert df.loc[3, "outcome_4class"] == "big_loss"
    assert df.loc[4, "outcome_4class"] == "small_loss"
    assert df.loc[5, "outcome_4class"] == "small_loss"


def test_sample_weight_scales_with_pnl():
    """Test sample weight scales with pnl."""
    df = pd.DataFrame(
        {
            "pnl_pct": [0.001, 0.005, 0.02, 0.10],
        }
    )
    fe = FeatureExtractor()
    b = DatasetBuilder(Path("a"), Path("b"), fe)
    df = b.compute_labels(df)
    assert df.loc[0, "sample_weight"] == 0.1
    assert df.loc[1, "sample_weight"] == 0.5
    assert df.loc[2, "sample_weight"] == 2.0
    assert df.loc[3, "sample_weight"] == 5.0


def test_compute_class_balance_imbalance_flag(tmp_path):
    """Test compute class balance imbalance flag."""
    trades_db, decisions_db = _seed_dbs(tmp_path, n_trades=20)
    fe = FeatureExtractor()
    b = DatasetBuilder(trades_db, decisions_db, fe)
    df = b.collect_all(days_back=365)
    df = b.compute_labels(df)
    cb = b.compute_class_balance(df)
    for cls in OUTCOME_4CLASS_ORDER:
        assert cls in cb["counts"]
    assert cb["total"] == 20
    assert 0.4 <= cb["pos_rate"] <= 0.6


def test_compute_class_balance_handles_empty():
    """Test compute class balance handles empty."""
    fe = FeatureExtractor()
    b = DatasetBuilder(Path("a"), Path("b"), fe)
    cb = b.compute_class_balance(pd.DataFrame())
    assert cb["total"] == 0
    assert cb["is_imbalanced"] is False


def test_save_and_load_roundtrip(tmp_path):
    """Test save and load roundtrip."""
    trades_db, decisions_db = _seed_dbs(tmp_path, n_trades=10)
    fe = FeatureExtractor()
    b = DatasetBuilder(trades_db, decisions_db, fe)
    df = b.collect_all(days_back=365)
    df = b.compute_labels(df)
    out = tmp_path / "dataset.parquet"
    saved_path = b.save(df, out)
    assert saved_path.exists()
    df_loaded = b.load(saved_path)
    assert len(df_loaded) == len(df)
    assert set(df_loaded.columns) == set(df.columns)


def test_skips_unexecuted_decisions(tmp_path):
    """Decisions with action != EXECUTE or executed_bool=0 should not appear."""
    trades_db = tmp_path / "t.db"
    decisions_db = tmp_path / "d.db"
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
        ts = datetime.now(tz=UTC).isoformat()
        cn.execute(
            "INSERT INTO decisions VALUES (?, 'c','SBER','VETO','1','BUY',0.7,NULL,NULL,0,'','[]',?,0,NULL,0.5,0.45,NULL)",
            ("d1", ts),
        )
        cn.execute(
            "INSERT INTO decisions VALUES (?, 'c','SBER','EXECUTE','1','BUY',0.7,NULL,NULL,0,'','[]',?,1,500.0,0.5,0.45,?)",
            ("d2", ts, ts),
        )
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
        ts = datetime.now(tz=UTC).isoformat()
        cn.execute(
            "INSERT INTO trades (decision_id, ticker, direction, quantity, price, order_value, remaining_cash, trade_date, trade_time, bot, source_model, arena_raw_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "d2",
                "SBER",
                "BUY",
                100,
                1000,
                100_000,
                0,
                "2026-05-26",
                "10:00:00",
                "bot",
                "",
                "",
                ts,
            ),
        )
    fe = FeatureExtractor()
    b = DatasetBuilder(trades_db, decisions_db, fe)
    df = b.collect_all(days_back=365)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "SBER"


def test_open_position_kept_with_nan_pnl(tmp_path):
    """A decision without pnl_rub (still-open trade) is kept but with NaN pnl_pct."""
    trades_db = tmp_path / "t.db"
    decisions_db = tmp_path / "d.db"
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
        ts = datetime.now(tz=UTC).isoformat()
        cn.execute(
            "INSERT INTO decisions VALUES (?, 'c','SBER','EXECUTE','1','BUY',0.7,NULL,NULL,0,'','[]',?,1,NULL,0.5,0.45,?)",
            ("dopen", ts, ts),
        )
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
        ts = datetime.now(tz=UTC).isoformat()
        cn.execute(
            "INSERT INTO trades (decision_id, ticker, direction, quantity, price, order_value, remaining_cash, trade_date, trade_time, bot, source_model, arena_raw_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "dopen",
                "SBER",
                "BUY",
                100,
                1000,
                100_000,
                0,
                "2026-05-26",
                "10:00:00",
                "bot",
                "",
                "",
                ts,
            ),
        )
    fe = FeatureExtractor()
    b = DatasetBuilder(trades_db, decisions_db, fe)
    df = b.collect_all(days_back=365)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["pnl_pct"])
    assert b.stats.n_dropped_open == 1
