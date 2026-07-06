"""Phase 27.5 — TA pattern noise blacklist.

A two-layer veto on top of `cfg.DETECTOR_BLACKLIST`:

1. STATIC_NOISE_PATTERNS — patterns whose 90d×20-ticker backtest produced
   Profit Factor < 1.0 with at least 5 trades (statistically money-losing).
   See data/training_cache/detector_rankings_after.json (column "kill_list").

2. dynamic noise overrides written by scripts/noise_review.py daily at
   03:30 МСК. The file lives at data/runtime_overrides.json and adds
   patterns that have shown rolling 7-day WR < NOISE_WR_THRESHOLD with at
   least NOISE_MIN_TRADES samples (config-tunable).

A pattern is considered noisy when it appears in either layer. The TA
trader and rank_detectors call `is_noisy()` to soft-mute (magnitude × 0.3)
or skip emission entirely.
"""

from __future__ import annotations

import json
import logging
import threading
import time

import app.config as cfg

logger = logging.getLogger(__name__)

STATIC_NOISE_PATTERNS: frozenset[str] = frozenset(
    {
        "smc_bos_bear",
        "smc_bos_bull",
        "smc_choch_bear",
        "smc_choch_bull",
        "smc_fvg_bear",
        "smc_fvg_bull",
        "smc_order_block_bear",
        "smc_order_block_bull",
        "smc_sweep_high",
        "smc_sweep_low",
        "diamond_bottom",
        "pivot_reversal_long",
        "compression_breakout_up",
        "megaphone_bottom",
        "falling_wedge",
        "rounding_bottom",
        "bull_flag",
        "bull_pennant",
        "inv_head_shoulders",
        "vcp",
    }
)

_RUNTIME_OVERRIDES_PATH = cfg.DATA_DIR / "runtime_overrides.json"
_dynamic_cache: dict[str, list[str]] = {"patterns": []}
_dynamic_cache_ts: float = 0.0
_DYN_REFRESH_SEC: float = 60.0
_dynamic_lock = threading.Lock()

def _load_dynamic_overrides() -> list[str]:
    """Read the dynamic blacklist from runtime_overrides.json with TTL caching.

    Returns:
        list[str]: pattern names marked noisy by scripts/noise_review.py
    """
    global _dynamic_cache_ts
    now = time.monotonic()
    with _dynamic_lock:
        if now - _dynamic_cache_ts < _DYN_REFRESH_SEC:
            return list(_dynamic_cache.get("patterns") or [])
        _dynamic_cache_ts = now
        if not _RUNTIME_OVERRIDES_PATH.exists():
            _dynamic_cache["patterns"] = []
            return []
        try:
            data = json.loads(_RUNTIME_OVERRIDES_PATH.read_text())
        except Exception as exc:
            logger.warning(
                "noise_blacklist: runtime overrides parse failed",
                extra={"path": str(_RUNTIME_OVERRIDES_PATH), "error": str(exc)},
            )
            _dynamic_cache["patterns"] = []
            return []
        patterns = list(data.get("noise_patterns") or [])
        _dynamic_cache["patterns"] = patterns
        return patterns

def is_noisy(detector: str) -> bool:
    """Return True when `detector` is on either the static or dynamic list.

    Args:
        detector: pattern / detector name (e.g. "rounding_bottom")
    Returns:
        bool: True when the detector should be muted
    """
    if not detector:
        return False
    name = detector.lower()
    if name in STATIC_NOISE_PATTERNS:
        return True
    return name in {p.lower() for p in _load_dynamic_overrides()}

def magnitude_penalty(detector: str) -> float:
    """Return magnitude multiplier (1.0 = pass-through, <1.0 = soft mute).

    Args:
        detector: pattern / detector name
    Returns:
        float: multiplier for unified_signal.magnitude. We apply a soft
        0.3× rather than 0 so confluence with a non-noisy partner can still
        revive the signal.
    """
    return 0.3 if is_noisy(detector) else 1.0

def update_dynamic_overrides(noise_patterns: list[str]) -> None:
    """Persist a new dynamic blacklist and bust the cache.

    Args:
        noise_patterns: patterns identified as noisy by adaptive review.
    """
    global _dynamic_cache_ts
    payload: dict[str, object] = {}
    if _RUNTIME_OVERRIDES_PATH.exists():
        try:
            payload = json.loads(_RUNTIME_OVERRIDES_PATH.read_text())
        except Exception:
            payload = {}
    payload["noise_patterns"] = list(noise_patterns)
    payload["updated_at_utc"] = (
        __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    )
    _RUNTIME_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RUNTIME_OVERRIDES_PATH.write_text(json.dumps(payload, indent=2))
    with _dynamic_lock:
        _dynamic_cache["patterns"] = list(noise_patterns)
        _dynamic_cache_ts = time.monotonic()
    logger.info(
        "noise_blacklist: dynamic overrides updated",
        extra={"n_patterns": len(noise_patterns)},
    )
