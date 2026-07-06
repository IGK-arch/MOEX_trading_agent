"""Guarded parameter updates driven by daily reflection.

Reflection is useful only if it can change tomorrow's behaviour, but direct
LLM-to-config writes are too risky for trading. This module is deliberately
small and conservative: the LLM may propose changes, while code applies only
whitelisted parameters, clamps every step, and blocks loosening after weak days.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ParamSpec:
    """Allowed runtime parameter adjustment."""

    min_value: float
    max_value: float
    max_step: float
    loosen_direction: int


PARAM_SPECS: dict[str, ParamSpec] = {
    "META_MIN_PROBA": ParamSpec(
        min_value=float(getattr(cfg, "META_MIN_PROBA_FLOOR", 0.45)),
        max_value=float(getattr(cfg, "META_MIN_PROBA_CEILING", 0.70)),
        max_step=0.02,
        loosen_direction=-1,
    ),
    "PAIR_Z_ENTRY_THRESHOLD": ParamSpec(
        min_value=1.20,
        max_value=2.50,
        max_step=0.10,
        loosen_direction=-1,
    ),
    "MEAN_REV_BB_STD": ParamSpec(
        min_value=1.50,
        max_value=2.80,
        max_step=0.10,
        loosen_direction=-1,
    ),
}


def apply_reflexive_adjustments(
    payload: dict[str, Any],
    *,
    date_str: str,
    day_stats: dict[str, Any],
) -> dict[str, Any]:
    """Validate and apply reflection-proposed parameter adjustments.

    Args:
        payload: parsed reflection LLM JSON.
        date_str: trading date YYYY-MM-DD.
        day_stats: objective metrics from today's decisions/trades.
    Returns:
        summary dict with applied/skipped adjustments.
    """
    if not bool(getattr(cfg, "REFLEXIVE_CONTROL_ENABLED", True)):
        return {"ok": True, "enabled": False, "applied": [], "skipped": []}

    proposals = payload.get("parameter_adjustments", [])
    if not isinstance(proposals, list):
        proposals = []

    min_trades = int(getattr(cfg, "REFLEXIVE_MIN_TRADES", 3))
    n_trades = int(day_stats.get("n_trades", 0) or 0)
    weak_day = _is_weak_day(day_stats)

    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    if n_trades < min_trades:
        reason = f"not enough trades for parameter control: {n_trades} < {min_trades}"
        skipped.extend(_skip_all(proposals, reason))
        summary = _summary(date_str, day_stats, applied, skipped)
        _write_runtime_override_summary(summary)
        return summary

    for proposal in proposals[:5]:
        decision = _validate_proposal(proposal, weak_day=weak_day)
        if not decision["ok"]:
            skipped.append(decision)
            continue

        param = decision["parameter"]
        new_value = float(decision["new_value"])
        old_value = float(decision["old_value"])
        _apply_runtime_value(param, new_value)
        applied.append(
            {
                "parameter": param,
                "old_value": old_value,
                "new_value": new_value,
                "delta": round(new_value - old_value, 6),
                "confidence": decision["confidence"],
                "reason": decision["reason"],
            }
        )

    summary = _summary(date_str, day_stats, applied, skipped)
    _write_runtime_override_summary(summary)
    if applied:
        logger.warning("Reflexive parameter adjustments applied", extra=summary)
    else:
        logger.info("Reflection produced no applied parameter adjustments", extra=summary)
    return summary


def apply_saved_reflexive_overrides() -> dict[str, Any]:
    """Restore fresh reflection-driven params after process restart.

    The saved block is intentionally short-lived: reflection controls tomorrow's
    behaviour, not a permanent config migration.
    """
    path = cfg.DATA_DIR / "runtime_overrides.json"
    if not path.exists():
        return {"ok": True, "applied": [], "reason": "no runtime overrides"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "applied": [], "reason": f"invalid overrides json: {exc}"}

    control = data.get("reflexive_control")
    if not isinstance(control, dict):
        return {"ok": True, "applied": [], "reason": "no reflexive control block"}

    expires_at = str(control.get("expires_at", "") or "")
    if expires_at:
        try:
            expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=UTC)
            if expires <= datetime.now(tz=UTC):
                return {"ok": True, "applied": [], "reason": "reflexive control expired"}
        except Exception:
            return {"ok": False, "applied": [], "reason": "invalid reflexive expiry"}

    applied: list[dict[str, Any]] = []
    for item in control.get("applied", []):
        if not isinstance(item, dict):
            continue
        param = str(item.get("parameter", "")).upper().strip()
        if param not in PARAM_SPECS:
            continue
        try:
            new_value = float(item["new_value"])
        except (KeyError, TypeError, ValueError):
            continue
        spec = PARAM_SPECS[param]
        bounded_value = max(spec.min_value, min(spec.max_value, new_value))
        _apply_runtime_value(param, bounded_value)
        applied.append({"parameter": param, "new_value": bounded_value})

    if applied:
        logger.warning("Saved reflexive parameter overrides restored", extra={"applied": applied})
    return {"ok": True, "applied": applied}


def _validate_proposal(proposal: Any, *, weak_day: bool) -> dict[str, Any]:
    if not isinstance(proposal, dict):
        return {"ok": False, "reason": "proposal is not an object", "proposal": str(proposal)[:120]}

    param = str(proposal.get("parameter", "")).upper().strip()
    spec = PARAM_SPECS.get(param)
    if spec is None:
        return {"ok": False, "parameter": param, "reason": "parameter not whitelisted"}

    confidence = float(proposal.get("confidence", 0.0) or 0.0)
    min_conf = float(getattr(cfg, "REFLEXIVE_MIN_CONFIDENCE", 0.65))
    if confidence < min_conf:
        return {"ok": False, "parameter": param, "confidence": confidence, "reason": "low confidence"}

    reason = str(proposal.get("reason", "") or proposal.get("evidence", "")).strip()
    if len(reason) < 10:
        return {"ok": False, "parameter": param, "confidence": confidence, "reason": "missing evidence"}

    direction = str(proposal.get("direction", "")).lower().strip()
    raw_delta = abs(float(proposal.get("delta", 0.0) or 0.0))
    if raw_delta <= 0:
        return {"ok": False, "parameter": param, "confidence": confidence, "reason": "non-positive delta"}
    signed = -raw_delta if direction in {"decrease", "down", "loosen", "lower"} else raw_delta
    if direction in {"increase", "up", "tighten", "raise"}:
        signed = raw_delta
    elif direction not in {"decrease", "down", "loosen", "lower"}:
        return {"ok": False, "parameter": param, "confidence": confidence, "reason": "unknown direction"}

    if weak_day and _is_loosening(signed, spec):
        return {
            "ok": False,
            "parameter": param,
            "confidence": confidence,
            "reason": "blocked loosening after weak day",
        }

    old_value = _current_value(param)
    step = max(-spec.max_step, min(spec.max_step, signed))
    new_value = max(spec.min_value, min(spec.max_value, old_value + step))
    if abs(new_value - old_value) < 1e-9:
        return {"ok": False, "parameter": param, "confidence": confidence, "reason": "clamped to current"}

    return {
        "ok": True,
        "parameter": param,
        "old_value": old_value,
        "new_value": round(new_value, 6),
        "confidence": confidence,
        "reason": reason[:240],
    }


def _current_value(param: str) -> float:
    if param == "MEAN_REV_BB_STD":
        return float(getattr(cfg, "MEAN_REV_BB_STD", getattr(cfg, "BB_STD", 2.0)))
    return float(getattr(cfg, param))


def _apply_runtime_value(param: str, value: float) -> None:
    if param == "META_MIN_PROBA":
        cfg.META_MIN_PROBA = value
        return
    if param == "PAIR_Z_ENTRY_THRESHOLD":
        cfg.PAIR_Z_ENTRY_THRESHOLD = value
        try:
            from app.agents.pair_trader import get_pair_trader

            get_pair_trader().z_entry = value
        except Exception as exc:
            logger.debug("pair z_entry runtime apply skipped", extra={"error": str(exc)})
        return
    if param == "MEAN_REV_BB_STD":
        cfg.MEAN_REV_BB_STD = value
        cfg.BB_STD = value
        try:
            from app.agents.mean_reversion import get_mean_reversion

            get_mean_reversion().bb_std = value
        except Exception as exc:
            logger.debug("mean reversion bb_std runtime apply skipped", extra={"error": str(exc)})


def _is_loosening(delta: float, spec: ParamSpec) -> bool:
    if delta == 0:
        return False
    return (delta > 0 and spec.loosen_direction > 0) or (delta < 0 and spec.loosen_direction < 0)


def _is_weak_day(day_stats: dict[str, Any]) -> bool:
    n_trades = int(day_stats.get("n_trades", 0) or 0)
    total_pnl = float(day_stats.get("total_pnl_rub", 0.0) or 0.0)
    win_rate = float(day_stats.get("win_rate", 0.0) or 0.0)
    if n_trades <= 0:
        return True
    return total_pnl < 0 or win_rate < 0.45


def _skip_all(proposals: list[Any], reason: str) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    for proposal in proposals[:5]:
        param = proposal.get("parameter", "?") if isinstance(proposal, dict) else "?"
        skipped.append({"ok": False, "parameter": str(param), "reason": reason})
    return skipped


def _summary(
    date_str: str,
    day_stats: dict[str, Any],
    applied: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "ok": True,
        "enabled": True,
        "date": date_str,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "expires_at": (datetime.now(tz=UTC) + timedelta(days=1)).isoformat(),
        "day_stats": day_stats,
        "applied": applied,
        "skipped": skipped,
    }


def _write_runtime_override_summary(summary: dict[str, Any]) -> None:
    path = cfg.DATA_DIR / "runtime_overrides.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["reflexive_control"] = summary
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
