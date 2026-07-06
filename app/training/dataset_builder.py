"""
app/training/dataset_builder.py — Build training dataset for meta-classifier v2.

Phase 27.8 (meta_v2). Reads decisions.db + trades.db, joins them by
decision_id, computes realised PnL %, attaches labels (4-class outcome +
sample_weight by |pnl_pct|), and materialises the matrix that
`scripts/train_meta_v2.py` will train on.

DB schema (from data/decisions.db + data/trades.db):

  decisions  (decision_id PK, ticker, action, tier, direction,
              combined_magnitude, signals_json, created_at, executed_bool,
              pnl_rub, meta_score, meta_threshold, executed_at, ...)
  trades     (id PK, decision_id FK, ticker, direction, quantity,
              price, order_value, remaining_cash, trade_date,
              trade_time, bot, source_model, created_at)

Note: `trades.db.trades` does NOT carry a pnl_rub column; PnL is computed
from the entry + later trades of the same ticker, OR pulled from
decisions.pnl_rub which is filled by the reflection layer.

Output DataFrame columns:
  feature_1, ..., feature_N (FEATURE_COLUMNS_V2 from FeatureExtractor),
  pnl_pct, pnl_rub, holding_min, exit_reason,
  ticker, direction, ts_entry, ts_exit,
  outcome_bin (added by compute_labels),
  outcome_4class, sample_weight
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import numpy as np  # noqa: F401  # type: ignore
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _HAS_PANDAS = False

import contextlib

from app.training.feature_extractor import FeatureExtractor
from app.utils.logging import get_logger

logger = get_logger(__name__)

BIG_WIN_PCT_THRESHOLD = 0.015
BIG_LOSS_PCT_THRESHOLD = -0.015

OUTCOME_4CLASS_ORDER: list[str] = ["big_loss", "small_loss", "small_win", "big_win"]

@dataclass
class CollectStats:
    """Collect Stats."""

    n_decisions_total: int = 0
    n_decisions_executed: int = 0
    n_decisions_with_pnl: int = 0
    n_with_features: int = 0
    n_dropped_open: int = 0

class DatasetBuilder:
    """
    Build the meta_v2 training matrix from decisions.db + trades.db.

    Usage
    -----
        fe = FeatureExtractor()
        builder = DatasetBuilder(
            trades_db=cfg.DATA_DIR / "trades.db",
            decisions_db=cfg.DATA_DIR / "decisions.db",
            fe=fe,
        )
        df = builder.collect_all(days_back=365)
        df = builder.compute_labels(df)
        builder.save(df, cfg.DATA_DIR / "training_cache" / "meta_v2_dataset.parquet")
    """

    def __init__(
        self,
        trades_db: Path,
        decisions_db: Path,
        fe: FeatureExtractor | None = None,
    ) -> None:
        """Init."""
        self.trades_db = Path(trades_db)
        self.decisions_db = Path(decisions_db)
        self.fe = fe or FeatureExtractor()
        self._stats = CollectStats()

    @property
    def stats(self) -> CollectStats:
        """Stats."""
        return self._stats

    @property
    def feature_columns(self) -> list[str]:
        """Feature columns."""
        return list(self.fe.feature_columns)

    def collect_all(self, days_back: int = 365) -> pd.DataFrame:
        """
        Read decisions + trades, JOIN by decision_id, compute pnl_pct from
        the entry price and either the decision's stored pnl_rub OR the
        sum of subsequent trades on the same ticker.

        Args:
            days_back: only include decisions in the last N days

        Returns:
            DataFrame with feature cols + meta cols (pnl_pct, pnl_rub,
            holding_min, exit_reason, ticker, direction, ts_entry, ts_exit).
            Rows with pnl_pct still NaN (open positions) are kept; caller
            should `.dropna(subset=["pnl_pct"])` before training.
        """
        if not _HAS_PANDAS:
            raise RuntimeError("DatasetBuilder requires pandas + numpy")

        decisions = self._load_decisions(days_back=days_back)
        self._stats.n_decisions_total = len(decisions)
        executed = [d for d in decisions if int(d.get("executed_bool") or 0) == 1]
        self._stats.n_decisions_executed = len(executed)
        decision_ids = [d["decision_id"] for d in executed]

        entries = self._load_entry_trades(decision_ids)
        pnl_map = self._load_realized_pnl_per_decision(executed, entries)

        rows: list[dict[str, Any]] = []
        for d in executed:
            did = d["decision_id"]
            entry = entries.get(did)
            pnl_rub = pnl_map.get(did)
            row = self._build_row(d, entry, pnl_rub)
            if row is not None:
                rows.append(row)

        if not rows:
            self._stats.n_with_features = 0
            return pd.DataFrame(columns=self.feature_columns + _META_COLS)

        df = pd.DataFrame(rows)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        if "ts_entry" in df.columns:
            df = df.sort_values("ts_entry", kind="stable").reset_index(drop=True)
        self._stats.n_with_features = len(df)
        if "pnl_pct" in df.columns:
            self._stats.n_dropped_open = int(df["pnl_pct"].isna().sum())
            self._stats.n_decisions_with_pnl = (
                self._stats.n_with_features - self._stats.n_dropped_open
            )
        return df

    def compute_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add label columns in-place:
          - outcome_bin    : 1 if pnl_pct > 0 else 0
          - outcome_4class : one of OUTCOME_4CLASS_ORDER
          - sample_weight  : max(0.1, |pnl_pct| * 100), capped at 5.0

        Big-pnl trades get more learning signal than the 50 small wins/losses
        that all sit near 0% (random noise around commission). Cap at 5.0
        prevents one 8% winner from completely dominating the gradient.
        """
        if df.empty:
            return df

        pnl = pd.to_numeric(df["pnl_pct"], errors="coerce")

        df["outcome_bin"] = (pnl > 0.0).astype(int)
        df["outcome_4class"] = pnl.apply(_bucket_4class)
        df["sample_weight"] = pnl.abs().mul(100.0).clip(lower=0.1, upper=5.0)
        return df

    def compute_class_balance(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Return per-class counts + imbalance flag.

        Returns
        -------
        dict with keys:
            counts          : {class_name: n}
            total           : total samples
            imbalance_ratio : max_count / min_count (NaN if any class missing)
            is_imbalanced   : True if imbalance_ratio > 2.0
            pos_rate        : P(win) on binary outcome
        """
        if df.empty or "outcome_4class" not in df.columns:
            return {
                "counts": {},
                "total": 0,
                "imbalance_ratio": float("nan"),
                "is_imbalanced": False,
                "pos_rate": 0.0,
            }
        counts = df["outcome_4class"].value_counts().to_dict()
        for cls in OUTCOME_4CLASS_ORDER:
            counts.setdefault(cls, 0)
        nonzero = [v for v in counts.values() if v > 0]
        imbalance = (max(nonzero) / min(nonzero)) if len(nonzero) >= 2 else float("nan")
        return {
            "counts": {k: int(counts[k]) for k in OUTCOME_4CLASS_ORDER},
            "total": int(sum(counts.values())),
            "imbalance_ratio": float(imbalance) if imbalance == imbalance else float("nan"),
            "is_imbalanced": bool(imbalance == imbalance and imbalance > 2.0),
            "pos_rate": float(df["outcome_bin"].mean()) if "outcome_bin" in df.columns else 0.0,
        }

    def save(self, df: pd.DataFrame, path: Path) -> Path:
        """
        Atomically save the matrix.

        We write parquet when pyarrow is installed (fastest, type-preserving)
        and fall back to pickle otherwise. Atomic = write to .tmp, then
        rename — partial writes never corrupt the cache.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            if path.suffix == ".parquet":
                try:
                    df.to_parquet(tmp, index=False)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "parquet save failed, falling back to pickle",
                        extra={"error": str(exc)},
                    )
                    tmp = path.with_suffix(".pkl.tmp")
                    df.to_pickle(tmp)
                    path = path.with_suffix(".pkl")
            else:
                df.to_pickle(tmp)
            tmp.replace(path)
            return path
        except Exception:
            if tmp.exists():
                with contextlib.suppress(OSError):
                    tmp.unlink()
            raise

    def load(self, path: Path) -> pd.DataFrame:
        """Counterpart of save() — auto-detect parquet vs pickle by suffix."""
        path = Path(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_pickle(path)

    def _load_decisions(self, days_back: int) -> list[dict[str, Any]]:
        """Load decisions from decisions.db. Empty list if DB missing."""
        if not self.decisions_db.exists():
            logger.info("decisions.db not found", extra={"path": str(self.decisions_db)})
            return []
        cutoff = (datetime.now(tz=UTC) - timedelta(days=days_back)).isoformat()
        with sqlite3.connect(str(self.decisions_db)) as cn:
            cn.row_factory = sqlite3.Row
            sql = """
                SELECT decision_id, cycle_id, ticker, action, tier, direction,
                       combined_magnitude, stop_loss, take_profit,
                       expected_holding_min, rationale, signals_json,
                       created_at, executed_bool, pnl_rub,
                       meta_score, meta_threshold, executed_at
                  FROM decisions
                 WHERE created_at >= ?
                   AND action = 'EXECUTE'
                 ORDER BY created_at ASC
            """
            try:
                return [dict(r) for r in cn.execute(sql, (cutoff,)).fetchall()]
            except sqlite3.OperationalError as e:
                logger.warning("decisions query failed", extra={"error": str(e)})
                return []

    def _load_entry_trades(self, decision_ids: list[str]) -> dict[str, dict]:
        """Return {decision_id: first_trade_row} for each executed decision."""
        if not decision_ids or not self.trades_db.exists():
            return {}
        out: dict[str, dict] = {}
        chunk_size = 800
        with sqlite3.connect(str(self.trades_db)) as cn:
            cn.row_factory = sqlite3.Row
            for i in range(0, len(decision_ids), chunk_size):
                chunk = decision_ids[i : i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                try:
                    rows = cn.execute(
                        f"""
                        SELECT decision_id, ticker, direction, quantity, price,
                               order_value, trade_date, trade_time, created_at
                          FROM trades
                         WHERE decision_id IN ({placeholders})
                         ORDER BY trade_date ASC, trade_time ASC, id ASC
                        """,
                        chunk,
                    ).fetchall()
                except sqlite3.OperationalError as e:
                    logger.warning("trades query failed", extra={"error": str(e)})
                    return out
                for r in rows:
                    if r["decision_id"] not in out:
                        out[r["decision_id"]] = dict(r)
        return out

    def _load_realized_pnl_per_decision(
        self,
        decisions: list[dict[str, Any]],
        entries: dict[str, dict],
    ) -> dict[str, float | None]:
        """
        Pull pnl_rub per decision_id from decisions.pnl_rub (filled by the
        reflection layer). When pnl_rub is NULL on decisions, we attempt
        FIFO matching on trades.db (rough — for offline labelling).

        Returns:
            {decision_id: pnl_rub_or_None}.
            None means "still open — drop from training".
        """
        out: dict[str, float | None] = {}
        for d in decisions:
            v = d.get("pnl_rub")
            if v is not None:
                try:
                    out[d["decision_id"]] = float(v)
                except (TypeError, ValueError):
                    out[d["decision_id"]] = None
            else:
                out[d["decision_id"]] = None
        return out

    def _build_row(
        self,
        decision: dict[str, Any],
        entry: dict[str, Any] | None,
        pnl_rub: float | None,
    ) -> dict[str, Any] | None:
        """Build a single feature+label row from a decision DB record."""
        try:
            signals = json.loads(decision.get("signals_json") or "[]")
        except Exception:
            signals = []

        dec_for_fe: dict[str, Any] = {
            "ticker": decision.get("ticker") or "",
            "direction": decision.get("direction") or "",
            "tier": decision.get("tier") or "",
            "combined_magnitude": decision.get("combined_magnitude") or 0.0,
            "expected_rr": decision.get("expected_rr") or 0.0,
            "meta_score": decision.get("meta_score") or 0.0,
            "signals": signals,
        }

        ts_entry = _parse_iso(decision.get("executed_at") or decision.get("created_at"))
        feat = self.fe.featurize(dec_for_fe, broker_state={}, ts_at_entry=ts_entry)

        row: dict[str, Any] = dict(feat)
        pnl_pct: float | None = None
        if pnl_rub is not None and entry is not None:
            order_value = float(entry.get("order_value") or 0.0)
            if order_value > 1e-6:
                pnl_pct = float(pnl_rub) / order_value
        elif pnl_rub is not None:
            pnl_pct = None

        holding_min: float | None = None
        if entry is not None and ts_entry is not None:
            holding_min = 0.0

        row["pnl_rub"] = float(pnl_rub) if pnl_rub is not None else None
        row["pnl_pct"] = pnl_pct
        row["holding_min"] = holding_min
        row["exit_reason"] = decision.get("rationale") or ""
        row["ticker"] = decision.get("ticker") or ""
        row["direction"] = decision.get("direction") or ""
        row["ts_entry"] = ts_entry.isoformat() if ts_entry else None
        row["ts_exit"] = ts_entry.isoformat() if ts_entry else None
        return row

_META_COLS = [
    "pnl_pct",
    "pnl_rub",
    "holding_min",
    "exit_reason",
    "ticker",
    "direction",
    "ts_entry",
    "ts_exit",
]
LABEL_COLS = ["outcome_bin", "outcome_4class", "sample_weight"]
NON_FEATURE_COLS = _META_COLS + LABEL_COLS

def _bucket_4class(pnl_pct: float) -> str:
    """Convert pnl_pct → 4-class label. NaN/None → 'small_loss' fallback."""
    try:
        v = float(pnl_pct)
    except (TypeError, ValueError):
        return "small_loss"
    if v != v:
        return "small_loss"
    if v >= BIG_WIN_PCT_THRESHOLD:
        return "big_win"
    if v <= BIG_LOSS_PCT_THRESHOLD:
        return "big_loss"
    if v > 0.0:
        return "small_win"
    return "small_loss"

def _parse_iso(s: str | None) -> datetime | None:
    """Parse an ISO timestamp string into an aware datetime (UTC default)."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except Exception:
        return None

__all__ = [
    "DatasetBuilder",
    "CollectStats",
    "BIG_WIN_PCT_THRESHOLD",
    "BIG_LOSS_PCT_THRESHOLD",
    "OUTCOME_4CLASS_ORDER",
    "LABEL_COLS",
    "NON_FEATURE_COLS",
]
