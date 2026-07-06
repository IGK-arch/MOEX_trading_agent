"""
app/dashboard/metrics_writer.py — Write live runtime metrics for the
GitLab platform Monitor (which can `cat` files from /data).

Two outputs:

  1. **`/data/metrics_live.jsonl`** — one JSON record every
     `cfg.METRICS_LIVE_INTERVAL_SEC` (default 30). Easy to grep / tail.
     Fields: ts_utc, equity_rub, daily_pnl_pct, max_dd_pct, current_dd_pct,
     winning_streak, losing_streak, n_trades_today, n_open_positions,
     hmm_regime, bus_queue_size, polza_budget_used_rub.

  2. **`/data/metrics_summary.json`** — single-row latest snapshot, easier
     to consume from a CI job (`python -c "import json; print(json.load(open(...)))"`).

In addition the snapshot is mirrored as a structured **INFO log line**
`metrics_snapshot` so the hackathon-provided Grafana (Loki) can grep
JSON fields directly from `/data/logs/*.jsonl` without our own panels.

Independent of Streamlit — runs in the trader process so it's always alive,
even when the dashboard fails or :8501 is not exposed.

The snapshot includes Phase 26C4 funnel/health fields:
  * ``win_rate_last_20_pct`` — sliding WR over the last 20 closed SELL trades
  * ``signal_to_decision_rate_pct`` — submitted / raw across the bot lifetime
  * ``decision_rejection_breakdown`` — reason → count histogram
  * ``avg_catboost_confidence`` — rolling mean of last 50 catboost scores
  * ``avg_meta_score`` — rolling mean of last 50 meta_scores
  * ``cash_utilization_pct`` — (deposit - cash) / deposit
  * ``open_pos_by_sector`` — sector → open-position count
  * ``recovery_mode_age_sec`` — age of recovery_state.json
  * ``last_model_retrain_hours`` — hours since catboost_ta.cbm mtime
  * ``signal_attrition`` — funnel snapshot (raw → submitted)
  * ``regime_name`` — NORMAL / CAUTIOUS / DEFENSIVE / CRISIS
  * ``n_positions_along_bias`` / ``n_positions_counter_bias`` — vs PER_TICKER_DIRECTION_BIAS
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

_RECENT_CATBOOST: deque[float] = deque(maxlen=50)
_RECENT_META: deque[float] = deque(maxlen=50)

def _sliding_win_rate(n: int = 20) -> float:
    """Return win-rate (%) of the most recent ``n`` closed (SELL) trades.

    FIFO-matched PnL — entry/exit pair for each ticker. Mirrors the
    accounting in ``app/memory/reflection.py`` but synchronous-only
    (sqlite3, not aiosqlite) so the metrics writer never blocks the
    event loop on async I/O.

    Args:
        n: rolling window size (default 20).
    Returns:
        float: WR as percentage, 0.0 when fewer than ``n`` closed trades.
    """
    db_path = cfg.DATA_DIR / "trades.db"
    if not db_path.exists():
        return 0.0
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0)
        try:
            rows = conn.execute(
                "SELECT ticker, direction, quantity, price FROM trades ORDER BY id"
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return 0.0

    fifo: dict[str, deque[list[float]]] = {}
    pnls: list[float] = []
    for ticker, direction, qty, price in rows:
        ticker = (ticker or "").upper()
        direction = (direction or "").upper()
        qty = int(qty or 0)
        price = float(price or 0.0)
        if qty <= 0 or price <= 0:
            continue
        if direction == "BUY":
            fifo.setdefault(ticker, deque()).append([float(qty), price])
        elif direction == "SELL":
            remaining = float(qty)
            lots = fifo.get(ticker)
            pnl = 0.0
            if lots:
                while remaining > 0 and lots:
                    lot_qty, lot_price = lots[0]
                    matched = min(remaining, lot_qty)
                    pnl += (price - lot_price) * matched
                    remaining -= matched
                    if matched >= lot_qty:
                        lots.popleft()
                    else:
                        lots[0][0] = lot_qty - matched
                        break
            pnls.append(pnl)
    if not pnls:
        return 0.0
    window = pnls[-n:]
    if not window:
        return 0.0
    wins = sum(1 for p in window if p > 0)
    return round(100.0 * wins / len(window), 2)

def _refresh_confidence_buffers(limit: int = 50) -> None:
    """Refresh rolling catboost/meta confidence buffers from decisions.db.

    Pulls the latest ``limit`` decisions, parses ``signals_json`` for
    ``catboost_score`` (max across the bundled signals), and ``rationale``
    for ``meta_score=…``. Populated in-place; cheap enough to run every
    metrics tick (~30s) since the table is tiny.
    """
    db_path = cfg.DATA_DIR / "decisions.db"
    if not db_path.exists():
        return
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0)
        try:
            rows = conn.execute(
                "SELECT signals_json, rationale FROM decisions ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return

    _RECENT_CATBOOST.clear()
    _RECENT_META.clear()
    for signals_json, rationale in rows:
        try:
            sigs = json.loads(signals_json or "[]") or []
            best = 0.0
            for s in sigs:
                md = (s or {}).get("metadata") or {}
                cb = md.get("catboost_score")
                if isinstance(cb, (int, float)) and cb > best:
                    best = float(cb)
            if best > 0:
                _RECENT_CATBOOST.append(best)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        try:
            rat = str(rationale or "")
            for key in ("meta_score=", "meta="):
                idx = rat.find(key)
                if idx >= 0:
                    tail = rat[idx + len(key) :]
                    num = ""
                    for ch in tail:
                        if ch.isdigit() or ch == ".":
                            num += ch
                        else:
                            break
                    if num:
                        _RECENT_META.append(float(num))
                        break
        except (TypeError, ValueError):
            pass

def _avg(buf: deque[float]) -> float:
    """Mean of a deque (0.0 if empty), rounded to 4 places."""
    if not buf:
        return 0.0
    return round(sum(buf) / len(buf), 4)

def build_metrics_snapshot() -> dict[str, Any]:
    """
    Pull the current live metrics from singletons.

    All getters are wrapped in try/except so a missing singleton (e.g.
    during early startup or in tests) just produces a default 0.0.
    """
    now_iso = datetime.now(tz=UTC).isoformat()
    snap: dict[str, Any] = {"ts_utc": now_iso}

    try:
        from app.risk.circuit_breakers import get_circuit_breaker

        cb = get_circuit_breaker()
        snap["equity_rub"] = float(getattr(cb.state, "current_equity_rub", 0.0))
        snap["peak_equity_rub"] = float(getattr(cb.state, "peak_equity_rub", 0.0))
        snap["daily_pnl_rub"] = float(getattr(cb.state, "daily_pnl_rub", 0.0))
        snap["daily_pnl_pct"] = float(getattr(cb.state, "daily_pnl_pct", 0.0))
        snap["current_dd_pct"] = float(getattr(cb.state, "current_drawdown_pct", 0.0))
        snap["max_dd_pct"] = float(getattr(cb.state, "max_drawdown_pct", 0.0))
        snap["winning_streak"] = int(getattr(cb.state, "winning_streak", 0))
        snap["losing_streak"] = int(getattr(cb.state, "losing_streak", 0))
        snap["n_trades_today"] = int(getattr(cb.state, "n_trades_today", 0))
        snap["circuit_blocked"] = bool(cb.state.is_blocked)
    except Exception:
        snap["equity_rub"] = 0.0

    try:
        from app.risk.position_book import get_position_book

        book = get_position_book()
        snap["n_open_positions"] = int(getattr(book, "n_open_positions", 0))
        snap["cash_balance_rub"] = float(getattr(book, "cash_balance", 0.0))
    except Exception:
        snap["n_open_positions"] = 0

    try:
        from app.agents.hmm_regime import get_hmm_detector

        snap["hmm_regime"] = getattr(get_hmm_detector(), "_current_label", "unknown")
    except Exception:
        snap["hmm_regime"] = "unknown"

    try:
        from app.news.ingestion_bus import get_bus

        snap["bus_queue_size"] = int(get_bus().stats().get("queue_size", 0))
        snap["bus_priority_queue_size"] = int(get_bus().stats().get("priority_queue_size", 0))
    except Exception:
        snap["bus_queue_size"] = 0

    try:
        from app.llm.polza_client import get_polza_client

        polza = get_polza_client()
        snap["polza_budget_used_rub"] = float(getattr(polza, "total_spent_rub", 0.0))
        snap["polza_budget_remaining_rub"] = max(
            0.0, cfg.POLZA_BUDGET_TOTAL_RUB - snap["polza_budget_used_rub"]
        )
    except Exception:
        snap["polza_budget_used_rub"] = 0.0

    snap["meta_min_proba"] = float(cfg.META_MIN_PROBA)
    snap["run_mode"] = cfg.RUN_MODE
    snap["live_sizing"] = cfg.LIVE_SIZING

    snap["win_rate_last_20_pct"] = _sliding_win_rate(20)

    with contextlib.suppress(Exception):
        _refresh_confidence_buffers(50)
    snap["avg_catboost_confidence"] = _avg(_RECENT_CATBOOST)
    snap["avg_meta_score"] = _avg(_RECENT_META)

    n_along = 0
    n_counter = 0
    pos_by_sector: dict[str, int] = {}
    cash_utilization_pct = 0.0
    try:
        from app.risk.position_book import get_position_book

        book = get_position_book()
        deposit = float(getattr(book, "deposit_total", 0.0)) or 0.0
        cash = float(getattr(book, "cash_balance", 0.0)) or 0.0
        if deposit > 0:
            cash_utilization_pct = round(100.0 * (deposit - cash) / deposit, 2)
        positions = getattr(book, "_positions", {}) or {}
        for ticker, pos in positions.items():
            sector = getattr(pos, "sector", "other") or "other"
            pos_by_sector[sector] = pos_by_sector.get(sector, 0) + 1
            bias = cfg.get_ticker_bias(ticker)
            qty = int(getattr(pos, "quantity", 0))
            if bias == "BUY" and qty > 0 or bias == "SELL" and qty < 0:
                n_along += 1
            elif bias == "BUY" and qty < 0 or bias == "SELL" and qty > 0:
                n_counter += 1
    except Exception:
        pass
    snap["cash_utilization_pct"] = cash_utilization_pct
    snap["open_pos_by_sector"] = pos_by_sector
    snap["n_positions_along_bias"] = n_along
    snap["n_positions_counter_bias"] = n_counter

    try:
        rec_path = Path(cfg.RECOVERY_STATE_PATH)
        if rec_path.exists():
            snap["recovery_mode_age_sec"] = int(max(0.0, time.time() - rec_path.stat().st_mtime))
        else:
            snap["recovery_mode_age_sec"] = -1
    except Exception:
        snap["recovery_mode_age_sec"] = -1

    try:
        model_path = cfg.MODELS_DIR / "catboost_ta.cbm"
        if model_path.exists():
            snap["last_model_retrain_hours"] = round(
                max(0.0, time.time() - model_path.stat().st_mtime) / 3600.0,
                2,
            )
        else:
            snap["last_model_retrain_hours"] = -1.0
    except Exception:
        snap["last_model_retrain_hours"] = -1.0

    try:
        from app.risk.adaptive_regime import compute_risk_regime
        from app.risk.circuit_breakers import get_circuit_breaker as _gcb
        from app.risk.position_book import get_position_book as _gpb

        _cb = _gcb()
        _book = _gpb()
        deposit = max(1.0, float(getattr(_book, "deposit_total", 1.0)))
        daily_pnl_pct = float(getattr(_cb.state, "daily_pnl_rub", 0.0)) / deposit
        regime = compute_risk_regime(
            current_drawdown_from_peak_pct=float(getattr(_cb.state, "current_drawdown_pct", 0.0)),
            losing_streak=int(getattr(_cb.state, "losing_streak", 0)),
            daily_pnl_pct=daily_pnl_pct,
            seconds_until_close=None,
        )
        snap["regime_name"] = regime.name
        snap["regime_size_mult"] = regime.size_multiplier
    except Exception:
        snap["regime_name"] = "UNKNOWN"

    try:
        from app.dispatcher.dispatcher import get_active_dispatcher

        d = get_active_dispatcher()
        if d is not None:
            stats = d.get_attrition_stats()
            stages = stats.get("stages", {}) or {}
            snap["signal_attrition"] = {
                "raw": int(stages.get("raw", 0)),
                "allowed": int(stages.get("allowed", 0)),
                "tier": int(stages.get("tier_passed", 0)),
                "meta": int(stages.get("meta_passed", 0)),
                "risk": int(stages.get("risk_passed", 0)),
                "submitted": int(stages.get("submitted", 0)),
            }
            raw = max(1, int(stages.get("raw", 0)))
            submitted = int(stages.get("submitted", 0))
            snap["signal_to_decision_rate_pct"] = round(100.0 * submitted / raw, 2)
            snap["decision_rejection_breakdown"] = {
                str(k): int(v) for k, v in (stats.get("rejection_breakdown") or {}).items()
            }
        else:
            snap["signal_attrition"] = {
                "raw": 0,
                "allowed": 0,
                "tier": 0,
                "meta": 0,
                "risk": 0,
                "submitted": 0,
            }
            snap["signal_to_decision_rate_pct"] = 0.0
            snap["decision_rejection_breakdown"] = {}
    except Exception:
        snap["signal_attrition"] = {
            "raw": 0,
            "allowed": 0,
            "tier": 0,
            "meta": 0,
            "risk": 0,
            "submitted": 0,
        }
        snap["signal_to_decision_rate_pct"] = 0.0
        snap["decision_rejection_breakdown"] = {}

    return snap

async def write_once(jsonl_path: Path, summary_path: Path) -> None:
    """Append one record to the JSONL + overwrite summary.json."""
    snap = build_metrics_snapshot()
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(snap, default=str, ensure_ascii=False)

    try:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning(
            "metrics_live append failed", extra={"path": str(jsonl_path), "error": str(exc)}
        )

    tmp = summary_path.with_suffix(".tmp")
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(line, encoding="utf-8")
        tmp.replace(summary_path)
    except Exception as exc:
        logger.warning(
            "metrics_summary write failed", extra={"path": str(summary_path), "error": str(exc)}
        )

    try:
        logger.info("metrics_snapshot", extra={"metrics": snap})
    except Exception as exc:
        logger.warning("metrics_snapshot emit failed", extra={"error": str(exc)})

async def metrics_writer_loop(
    jsonl_path: Path | None = None,
    summary_path: Path | None = None,
    interval_sec: float | None = None,
) -> None:
    """Background task — call from main.py via asyncio.create_task."""
    jsonl_path = jsonl_path or (cfg.DATA_DIR / "metrics_live.jsonl")
    summary_path = summary_path or (cfg.DATA_DIR / "metrics_summary.json")
    interval = float(interval_sec or cfg.METRICS_LIVE_INTERVAL_SEC)

    logger.info(
        "metrics_writer_loop started", extra={"interval_sec": interval, "jsonl": str(jsonl_path)}
    )
    while True:
        try:
            await write_once(jsonl_path, summary_path)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("metrics_writer iteration failed", extra={"error": str(exc)})
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise

__all__ = ["build_metrics_snapshot", "metrics_writer_loop", "write_once"]
