"""Атомарное сохранение состояния для холодного рестарта."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1

@dataclass
class RecoverySnapshot:
    """In-memory snapshot of recovery state."""

    schema_version: int = SCHEMA_VERSION
    last_save_ts_utc: float = 0.0

    circuit_state: dict[str, Any] = field(default_factory=dict)

    hmm_regime: str = "unknown"

    last_decision_ids: list[str] = field(default_factory=list)

    meta_score_history: list[float] = field(default_factory=list)

    daily_turnover_rub: float = 0.0
    n_trades_today: int = 0

    open_positions: list[dict[str, Any]] = field(default_factory=list)

    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize snapshot to JSON string.

        Returns:
            str: JSON-encoded snapshot
        """
        return json.dumps(asdict(self), default=str, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> RecoverySnapshot:
        """Deserialize JSON string into snapshot.

        Args:
            raw: JSON text
        Returns:
            RecoverySnapshot: parsed instance
        """
        obj = json.loads(raw)
        return cls(
            schema_version=int(obj.get("schema_version", SCHEMA_VERSION)),
            last_save_ts_utc=float(obj.get("last_save_ts_utc", 0.0)),
            circuit_state=obj.get("circuit_state", {}) or {},
            hmm_regime=str(obj.get("hmm_regime", "unknown")),
            last_decision_ids=list(obj.get("last_decision_ids", []) or []),
            meta_score_history=[float(x) for x in (obj.get("meta_score_history") or [])],
            daily_turnover_rub=float(obj.get("daily_turnover_rub", 0.0)),
            n_trades_today=int(obj.get("n_trades_today", 0)),
            open_positions=list(obj.get("open_positions", []) or []),
            extras=obj.get("extras", {}) or {},
        )

class RecoveryStateManager:
    """Owns the single recovery_state.json file."""

    def __init__(self, path: Path | None = None) -> None:
        """Init."""
        self.path = Path(path or cfg.RECOVERY_STATE_PATH)
        self._lock = asyncio.Lock()

    def load(self) -> RecoverySnapshot | None:
        """Read snapshot from disk.

        Returns:
            RecoverySnapshot | None: snapshot or None if missing/corrupt
        """
        if not self.path.exists():
            return None
        try:
            raw = self.path.read_text(encoding="utf-8")
            snap = RecoverySnapshot.from_json(raw)
            age_sec = max(0.0, time.time() - snap.last_save_ts_utc)
            if age_sec > cfg.RECOVERY_STALE_THRESHOLD_SEC:
                logger.critical(
                    "Recovery snapshot STALE — broker reconcile mandatory",
                    extra={
                        "age_sec": round(age_sec, 1),
                        "stale_threshold_sec": cfg.RECOVERY_STALE_THRESHOLD_SEC,
                        "n_open_positions_hint": len(snap.open_positions),
                    },
                )
            logger.info(
                "Recovery snapshot loaded",
                extra={
                    "age_sec": round(age_sec, 1),
                    "hmm_regime": snap.hmm_regime,
                    "n_last_decisions": len(snap.last_decision_ids),
                    "n_open_positions_hint": len(snap.open_positions),
                    "daily_turnover_rub": snap.daily_turnover_rub,
                    "n_trades_today": snap.n_trades_today,
                },
            )
            return snap
        except Exception as exc:
            logger.error(
                "Failed to load recovery snapshot",
                extra={"path": str(self.path), "error": str(exc)},
            )
            return None

    async def save_atomic(self, snap: RecoverySnapshot) -> None:
        """Write snapshot atomically (tmp + fsync + rename).

        Args:
            snap: snapshot to persist
        """
        snap.last_save_ts_utc = time.time()
        snap.schema_version = SCHEMA_VERSION
        raw = snap.to_json()
        async with self._lock:
            await asyncio.to_thread(self._write_atomic_blocking, raw)

    def _write_atomic_blocking(self, raw: str) -> None:
        """Blocking atomic write helper.

        Args:
            raw: JSON content to write
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cutoff = time.time() - 300.0
            for candidate in self.path.parent.glob(".recovery_state_*.tmp"):
                try:
                    if candidate.stat().st_mtime < cutoff:
                        candidate.unlink()
                except OSError:
                    continue
        except OSError:
            pass

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=".recovery_state_",
                suffix=".tmp",
                delete=False,
            ) as tf:
                tf.write(raw)
                tf.flush()
                os.fsync(tf.fileno())
                tmp_path = tf.name
            os.replace(tmp_path, str(self.path))
        except Exception as exc:
            logger.error("Recovery atomic write failed", extra={"error": str(exc)})

    @staticmethod
    def build_snapshot(
        *,
        circuit_state_dict: dict[str, Any] | None = None,
        hmm_regime: str = "unknown",
        last_decision_ids: list[str] | None = None,
        meta_score_history: list[float] | None = None,
        daily_turnover_rub: float = 0.0,
        n_trades_today: int = 0,
        open_positions: list[dict[str, Any]] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> RecoverySnapshot:
        """Build a RecoverySnapshot from kwargs.

        Args:
            circuit_state_dict: circuit breaker state dict
            hmm_regime: HMM regime label
            last_decision_ids: recent decision ids
            meta_score_history: meta scores history
            daily_turnover_rub: today's turnover
            n_trades_today: trade count
            open_positions: position hints
            extras: extra fields
        Returns:
            RecoverySnapshot: built snapshot
        """
        return RecoverySnapshot(
            circuit_state=circuit_state_dict or {},
            hmm_regime=hmm_regime,
            last_decision_ids=(last_decision_ids or [])[-100:],
            meta_score_history=(meta_score_history or [])[-200:],
            daily_turnover_rub=float(daily_turnover_rub),
            n_trades_today=int(n_trades_today),
            open_positions=list(open_positions or []),
            extras=extras or {},
        )

_manager: RecoveryStateManager | None = None

def get_recovery_manager() -> RecoveryStateManager:
    """Return process-wide RecoveryStateManager singleton.

    Returns:
        RecoveryStateManager: shared instance
    """
    global _manager
    if _manager is None:
        _manager = RecoveryStateManager()
    return _manager

__all__ = [
    "RecoveryStateManager",
    "RecoverySnapshot",
    "get_recovery_manager",
    "SCHEMA_VERSION",
]
