"""Evening retrain pipeline — runs once per day after MOEX close.

Schedule (MSK):
    19:00 — daily evening pipeline (after MOEX main close 18:50):
        A. snapshot trades + decisions
        B. refit meta-classifier on today's executed decisions
        C. recompute PER_TICKER_DIRECTION_BIAS_OBS overlay (7-day rolling WR)
        D. refresh NOISY_PATTERNS blacklist (WR<35% over 7 days)
        E. re-fit HMM on latest IMOEX daily candles
        F. run reflection (LLM lessons → RAG store)
        G. write summary JSON for ops dashboard

Sunday 03:00 — full CatBoost-primary retrain
        H. walk-forward 180-day × 20-ticker × 5m
        I. checkpoint to data/models/catboost_primary.cbm

The pipeline is *idempotent* — running twice in a day overwrites the
artifact files without harm. Each step is wrapped in try/except so a
failure in one step doesn't abort the others.

Phase 30 (v0.19.6).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACTS_DIR = cfg.DATA_DIR / "training_cache"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

EVENING_SUMMARY_PATH = ARTIFACTS_DIR / "evening_pipeline_summary.json"
DIRECTION_BIAS_OBS_PATH = ARTIFACTS_DIR / "direction_bias_observed.json"
NOISY_PATTERNS_PATH = ARTIFACTS_DIR / "noisy_patterns.json"
WEEKLY_FULL_RETRAIN_PATH = ARTIFACTS_DIR / "weekly_full_retrain_summary.json"

DEFAULT_LOOKBACK_DAYS_ROLLING = 7
NOISY_PATTERN_MIN_TRADES = 8
NOISY_PATTERN_MAX_WR = 0.35

class EveningPipeline:
    """Orchestrates the evening retrain steps. Stateless (singleton)."""

    def __init__(self) -> None:
        """Init."""
        self._last_run_ts_utc: float = 0.0
        self._last_summary: dict[str, Any] = {}

    async def run(self) -> dict[str, Any]:
        """Execute daily evening retrain (steps A-G).

        Returns:
            dict[str, Any]: summary of each step's outcome.
        """
        started_iso = datetime.now(tz=UTC).isoformat()
        t0 = time.time()
        logger.info("EveningPipeline START", extra={"ts_utc": started_iso})
        summary: dict[str, Any] = {
            "ts_utc": started_iso,
            "steps": {},
            "success": True,
        }

        snap = await self._run_step("snapshot", self._snapshot_today)
        summary["steps"]["snapshot"] = snap

        meta = await self._run_step("meta_retrain", self._retrain_meta)
        summary["steps"]["meta_retrain"] = meta

        bias = await self._run_step("direction_bias", self._update_direction_bias)
        summary["steps"]["direction_bias"] = bias

        noise = await self._run_step("noisy_patterns", self._update_noisy_patterns)
        summary["steps"]["noisy_patterns"] = noise

        hmm = await self._run_step("hmm_refit", self._refit_hmm)
        summary["steps"]["hmm_refit"] = hmm

        refl = await self._run_step("reflection", self._run_reflection)
        summary["steps"]["reflection"] = refl

        elapsed = round(time.time() - t0, 2)
        summary["elapsed_sec"] = elapsed
        summary["success"] = all((s or {}).get("ok") for s in summary["steps"].values())
        try:
            with open(EVENING_SUMMARY_PATH, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover
            logger.error("Evening summary dump failed", extra={"error": str(exc)})

        self._last_run_ts_utc = time.time()
        self._last_summary = summary
        logger.info(
            "EveningPipeline DONE",
            extra={
                "elapsed_sec": elapsed,
                "success": summary["success"],
                "steps_ok": sum(1 for s in summary["steps"].values() if (s or {}).get("ok")),
                "n_steps": len(summary["steps"]),
            },
        )
        return summary

    async def run_weekly_full(self) -> dict[str, Any]:
        """Sunday 03:00 — full CatBoost-primary retrain (step H/I).

        Delegated to scripts/train_catboost.py; if missing or fails we
        log + carry on so the next attempt isn't blocked.
        """
        started_iso = datetime.now(tz=UTC).isoformat()
        t0 = time.time()
        summary: dict[str, Any] = {"ts_utc": started_iso, "ok": False}
        try:
            import subprocess

            script = Path(__file__).resolve().parent.parent.parent / "scripts" / "train_catboost.py"
            if script.exists():
                cmd = ["python3", str(script), "--days", "180"]
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=1800.0,
                    )
                    summary["ok"] = proc.returncode == 0
                    summary["returncode"] = proc.returncode
                    summary["stdout_tail"] = (stdout or b"").decode("utf-8", "ignore")[-500:]
                    summary["stderr_tail"] = (stderr or b"").decode("utf-8", "ignore")[-500:]
                except TimeoutError:
                    proc.kill()
                    summary["ok"] = False
                    summary["error"] = "timeout 1800s"
            else:
                summary["ok"] = False
                summary["error"] = f"script not found: {script}"
        except Exception as exc:
            summary["ok"] = False
            summary["error"] = str(exc)

        summary["elapsed_sec"] = round(time.time() - t0, 2)
        try:
            with open(WEEKLY_FULL_RETRAIN_PATH, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=2)
        except Exception:  # pragma: no cover
            pass
        return summary

    async def _run_step(
        self,
        name: str,
        fn: callable,
    ) -> dict[str, Any]:
        """Wrap a step with timing and error capture."""
        t0 = time.time()
        try:
            result = await fn()
            ok = bool((result or {}).get("ok", True))
        except Exception as exc:
            logger.exception(
                "EveningPipeline step failed",
                extra={"step": name, "error": str(exc)},
            )
            return {"ok": False, "error": str(exc), "elapsed_sec": round(time.time() - t0, 2)}
        elapsed = round(time.time() - t0, 2)
        out = dict(result or {})
        out.setdefault("ok", ok)
        out["elapsed_sec"] = elapsed
        return out

    async def _snapshot_today(self) -> dict[str, Any]:
        """Step A — count decisions + trades for today."""
        today_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        decisions = self._sql_count(
            cfg.DATA_DIR / "decisions.db",
            "SELECT COUNT(*) FROM decisions WHERE created_at LIKE ?",
            (f"{today_str}%",),
        )
        executed = self._sql_count(
            cfg.DATA_DIR / "decisions.db",
            "SELECT COUNT(*) FROM decisions WHERE created_at LIKE ? AND executed_bool = 1",
            (f"{today_str}%",),
        )
        trades = self._sql_count(
            cfg.DATA_DIR / "trades.db",
            "SELECT COUNT(*) FROM trades WHERE trade_date = ?",
            (today_str,),
        )
        return {
            "ok": True,
            "date": today_str,
            "decisions_total": decisions,
            "decisions_executed": executed,
            "trades_count": trades,
        }

    async def _retrain_meta(self) -> dict[str, Any]:
        """Step B — invoke scripts/train_meta.py.

        Fallback: log "skipped" if script missing. Honors min-samples
        gate inside the script itself.
        """
        import subprocess

        script = Path(__file__).resolve().parent.parent.parent / "scripts" / "train_meta.py"
        if not script.exists():
            return {"ok": False, "skipped": True, "reason": "scripts/train_meta.py missing"}
        try:
            cmd = ["python3", str(script), "--days", "30"]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=600.0,
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout_tail": (stdout or b"").decode("utf-8", "ignore")[-300:],
                "stderr_tail": (stderr or b"").decode("utf-8", "ignore")[-300:],
            }
        except TimeoutError:
            return {"ok": False, "error": "meta retrain timeout"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def _update_direction_bias(self) -> dict[str, Any]:
        """Step C — compute 7-day rolling WR per ticker from trades.db.

        We don't mutate PER_TICKER_DIRECTION_BIAS in config (that
        would conflict with the static research baseline). We write
        an observational overlay to direction_bias_observed.json that
        the risk_manager can read at startup to blend.
        """
        cutoff_date = (
            datetime.now(tz=UTC) - timedelta(days=DEFAULT_LOOKBACK_DAYS_ROLLING)
        ).strftime("%Y-%m-%d")
        rows = self._sql_query(
            cfg.DATA_DIR / "trades.db",
            "SELECT ticker, direction, quantity, price, "
            "       (julianday('now') - julianday(created_at)) AS age_days "
            "  FROM trades WHERE trade_date >= ?",
            (cutoff_date,),
        )
        wr_per: dict[str, dict[str, float]] = {}
        from collections import deque

        fifo: dict[str, deque] = {}
        wins: dict[str, int] = defaultdict(int)
        losses: dict[str, int] = defaultdict(int)
        for r in rows:
            ticker = str(r[0] or "").upper()
            direction = str(r[1] or "").upper()
            qty = int(r[2] or 0)
            price = float(r[3] or 0.0)
            if not ticker or qty <= 0:
                continue
            if direction == "BUY":
                fifo.setdefault(ticker, deque()).append([qty, price])
            elif direction == "SELL":
                lots = fifo.get(ticker)
                remaining = qty
                while lots and remaining > 0:
                    lot_qty, lot_price = lots[0]
                    matched = min(remaining, lot_qty)
                    pnl = (price - lot_price) * matched
                    if pnl > 0:
                        wins[ticker] += 1
                    else:
                        losses[ticker] += 1
                    remaining -= matched
                    if matched >= lot_qty:
                        lots.popleft()
                    else:
                        lots[0][0] = lot_qty - matched
                        break
        for ticker in set(list(wins) + list(losses)):
            w = wins[ticker]
            n = w + losses[ticker]
            wr_per[ticker] = {
                "wr_7d": round(w / max(1, n), 4),
                "n_round_trips": n,
            }
        try:
            with open(DIRECTION_BIAS_OBS_PATH, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "ts_utc": datetime.now(tz=UTC).isoformat(),
                        "lookback_days": DEFAULT_LOOKBACK_DAYS_ROLLING,
                        "per_ticker": wr_per,
                    },
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:  # pragma: no cover
            pass
        return {"ok": True, "tickers": len(wr_per), "rows_scanned": len(rows)}

    async def _update_noisy_patterns(self) -> dict[str, Any]:
        """Step D — patterns with WR<35% over last 7d → blacklist file.

        Reads decisions.signals_json for emitted patterns; cross-checks
        with executed trade pnl in trades.db. We write a side-file the
        ta_trader can consult at startup (additive to DETECTOR_BLACKLIST).
        """
        cutoff = (datetime.now(tz=UTC) - timedelta(days=DEFAULT_LOOKBACK_DAYS_ROLLING)).isoformat()
        rows = self._sql_query(
            cfg.DATA_DIR / "decisions.db",
            "SELECT signals_json, executed_bool, pnl_rub, ticker "
            "  FROM decisions WHERE created_at >= ? AND action = 'EXECUTE'",
            (cutoff,),
        )
        agg: dict[str, dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0, "n": 0})
        for r in rows:
            try:
                signals = json.loads(r[0] or "[]")
            except Exception:
                signals = []
            pnl = float(r[2] or 0.0)
            if pnl == 0.0:
                continue
            won = pnl > 0
            for s in signals or []:
                pattern = str(s.get("pattern") or "").lower()
                if not pattern:
                    continue
                row = agg[pattern]
                row["n"] += 1
                if won:
                    row["wins"] += 1
                else:
                    row["losses"] += 1
        noisy: list[dict[str, Any]] = []
        for pattern, row in agg.items():
            if row["n"] < NOISY_PATTERN_MIN_TRADES:
                continue
            wr = row["wins"] / row["n"]
            if wr < NOISY_PATTERN_MAX_WR:
                noisy.append(
                    {
                        "pattern": pattern,
                        "wr_7d": round(wr, 4),
                        "n_trades": row["n"],
                    }
                )
        try:
            with open(NOISY_PATTERNS_PATH, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "ts_utc": datetime.now(tz=UTC).isoformat(),
                        "lookback_days": DEFAULT_LOOKBACK_DAYS_ROLLING,
                        "min_trades": NOISY_PATTERN_MIN_TRADES,
                        "max_wr": NOISY_PATTERN_MAX_WR,
                        "patterns": sorted(noisy, key=lambda p: p["wr_7d"]),
                    },
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:  # pragma: no cover
            pass
        return {"ok": True, "patterns_blacklisted": len(noisy), "candidates_scanned": len(agg)}

    async def _refit_hmm(self) -> dict[str, Any]:
        """Step E — refit HMM on the latest IMOEX daily candles."""
        import subprocess

        script = Path(__file__).resolve().parent.parent.parent / "scripts" / "train_hmm.py"
        if not script.exists():
            return {"ok": False, "skipped": True, "reason": "scripts/train_hmm.py missing"}
        try:
            cmd = ["python3", str(script), "--days", "120"]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=300.0,
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout_tail": (stdout or b"").decode("utf-8", "ignore")[-300:],
            }
        except TimeoutError:
            return {"ok": False, "error": "hmm fit timeout"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def _run_reflection(self) -> dict[str, Any]:
        """Step F — invoke ReflectionEngine.run_today (LLM lessons)."""
        try:
            from app.memory.reflection import get_reflection_engine

            engine = get_reflection_engine()
            result = await engine.run_today()
            if result is None:
                return {"ok": True, "note": "no decisions to reflect or LLM disabled"}
            lessons = result.get("lessons", [])
            return {"ok": True, "n_lessons": len(lessons)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @staticmethod
    def _sql_count(db_path: Path, sql: str, params: tuple) -> int:
        """Synchronous COUNT helper (small queries on local SQLite)."""
        try:
            if not db_path.exists():
                return 0
            with sqlite3.connect(str(db_path)) as cn:
                cur = cn.execute(sql, params)
                row = cur.fetchone()
                return int(row[0]) if row and row[0] is not None else 0
        except Exception:
            return 0

    @staticmethod
    def _sql_query(db_path: Path, sql: str, params: tuple) -> list[tuple]:
        """Synchronous fetchall helper (small queries on local SQLite)."""
        try:
            if not db_path.exists():
                return []
            with sqlite3.connect(str(db_path)) as cn:
                cur = cn.execute(sql, params)
                return list(cur.fetchall())
        except Exception:
            return []

_pipeline: EveningPipeline | None = None

def get_evening_pipeline() -> EveningPipeline:
    """Return process-wide EveningPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = EveningPipeline()
    return _pipeline

__all__ = [
    "EveningPipeline",
    "get_evening_pipeline",
    "EVENING_SUMMARY_PATH",
    "DIRECTION_BIAS_OBS_PATH",
    "NOISY_PATTERNS_PATH",
    "WEEKLY_FULL_RETRAIN_PATH",
]
