"""Broker availability monitor — drives SAFE MODE (close-only).

v1.0.10 (Cycle 10) — после ночного инцидента 27 May 2026 когда v1.0.8 ушёл
в CRISIS regime из-за сломанного URL variant у `/api/positions`, нам нужна
жёсткая защита: если broker недоступен дольше N секунд — переходить в
"close-only" режим, чтобы случайно не открыть entry в неизвестное состояние.

Один singleton на процесс. `on_positions_success()` обновляет
last-success timestamp; `on_positions_fail()` смотрит на возраст последнего
успеха и поднимает флаг `_safe_mode = True` если > 5 минут.

Public API:
    monitor = get_broker_health_monitor()
    monitor.on_positions_success()
    monitor.on_positions_fail()
    if monitor.is_safe_mode(): ...
    monitor.snapshot() → diagnostic dict

Feature flag: cfg.BROKER_SAFE_MODE_ENABLED (default ON). Когда выключен,
`is_safe_mode()` всегда возвращает False; вход в SAFE MODE никогда не
триггерится — для удобства полного отключения, если возникают ложные
срабатывания.
"""

from __future__ import annotations

import time

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

class BrokerHealthMonitor:
    """Track ArenaGo positions-endpoint reachability and gate entries.

    State machine:
        HEALTHY ──fail (age > threshold)──▶ SAFE_MODE
        SAFE_MODE ──success──▶ HEALTHY (after `_recovery_cooldown_sec`)

    The recovery cooldown prevents flapping when one good poll briefly
    interleaves a long outage. Default = 60s, matching the spec.
    """

    def __init__(
        self,
        *,
        safe_mode_threshold_sec: float = 300.0,
        recovery_cooldown_sec: float = 60.0,
        alert_interval_sec: float = 60.0,
    ) -> None:
        """Init.

        Args:
            safe_mode_threshold_sec: enter SAFE MODE if positions endpoint
                has been failing for at least this many seconds.
            recovery_cooldown_sec: after first success, exit SAFE MODE only
                if a follow-up grace period has elapsed (anti-flap).
            alert_interval_sec: throttle for CRITICAL "still unreachable"
                log lines while inside SAFE MODE.
        """
        self._last_success_ts = time.monotonic()
        self._first_recovery_ts: float | None = None
        self._safe_mode = False
        self._safe_mode_entered_at: float | None = None
        self._last_alert_ts: float = 0.0
        self._fail_count_in_safe_mode = 0

        self.safe_mode_threshold_sec = float(safe_mode_threshold_sec)
        self.recovery_cooldown_sec = float(recovery_cooldown_sec)
        self.alert_interval_sec = float(alert_interval_sec)

    def on_positions_success(self) -> None:
        """Record a successful positions fetch and try to exit SAFE MODE."""
        now = time.monotonic()
        self._last_success_ts = now
        if self._safe_mode:
            if self._first_recovery_ts is None:
                self._first_recovery_ts = now
                logger.warning(
                    "BrokerHealthMonitor: first positions success after outage — "
                    "entering recovery cooldown",
                    extra={
                        "cooldown_sec": self.recovery_cooldown_sec,
                        "outage_sec": round(
                            now - (self._safe_mode_entered_at or now), 1
                        ),
                    },
                )
                return
            if now - self._first_recovery_ts >= self.recovery_cooldown_sec:
                logger.info(
                    "BROKER HEALTH RESTORED — exiting SAFE MODE",
                    extra={
                        "outage_total_sec": round(
                            now - (self._safe_mode_entered_at or now), 1
                        ),
                        "fail_count_during_outage": self._fail_count_in_safe_mode,
                    },
                )
                self._safe_mode = False
                self._safe_mode_entered_at = None
                self._first_recovery_ts = None
                self._fail_count_in_safe_mode = 0

    def on_positions_fail(self) -> None:
        """Record a failed positions fetch; promote to SAFE MODE if stale."""
        now = time.monotonic()
        self._first_recovery_ts = None
        age = now - self._last_success_ts
        if self._safe_mode:
            self._fail_count_in_safe_mode += 1
            self._maybe_emit_alert(now, age)
            return
        if age > self.safe_mode_threshold_sec:
            self._safe_mode = True
            self._safe_mode_entered_at = now
            self._fail_count_in_safe_mode = 1
            self._last_alert_ts = now
            logger.critical(
                "BROKER UNREACHABLE > %.0f sec — ENTERING SAFE MODE (close-only)"
                % self.safe_mode_threshold_sec,
                extra={
                    "age_since_last_success_sec": round(age, 1),
                    "threshold_sec": self.safe_mode_threshold_sec,
                },
            )

    def _maybe_emit_alert(self, now: float, age_sec: float) -> None:
        """Throttle CRITICAL re-alerts while inside SAFE MODE.

        Args:
            now: current monotonic timestamp
            age_sec: seconds since the last successful positions fetch
        """
        if now - self._last_alert_ts < self.alert_interval_sec:
            return
        self._last_alert_ts = now
        logger.critical(
            "BROKER STILL UNREACHABLE — SAFE MODE active (close-only)",
            extra={
                "age_since_last_success_sec": round(age_sec, 1),
                "fail_count": self._fail_count_in_safe_mode,
                "alert_interval_sec": self.alert_interval_sec,
            },
        )

    def is_safe_mode(self) -> bool:
        """Return True if entries should be blocked (broker unreachable).

        Feature-flagged via ``cfg.BROKER_SAFE_MODE_ENABLED``. When the flag
        is False this always returns False so the rest of the trading
        pipeline behaves exactly as before.

        Returns:
            bool: gating decision for new entries.
        """
        if not getattr(cfg, "BROKER_SAFE_MODE_ENABLED", True):
            return False
        return self._safe_mode

    def seconds_since_last_success(self) -> float:
        """Return seconds since the last positive positions fetch.

        Returns:
            float: age of last success (>=0)
        """
        return max(0.0, time.monotonic() - self._last_success_ts)

    def snapshot(self) -> dict[str, float | bool | int]:
        """Return a serialisable snapshot for metrics/health dashboards.

        Returns:
            dict[str, float | bool | int]: state at call time
        """
        now = time.monotonic()
        return {
            "safe_mode": self._safe_mode,
            "enabled": bool(getattr(cfg, "BROKER_SAFE_MODE_ENABLED", True)),
            "age_since_last_success_sec": round(now - self._last_success_ts, 1),
            "threshold_sec": self.safe_mode_threshold_sec,
            "recovery_cooldown_sec": self.recovery_cooldown_sec,
            "fail_count_in_safe_mode": self._fail_count_in_safe_mode,
            "in_recovery_cooldown": self._first_recovery_ts is not None,
        }

_monitor: BrokerHealthMonitor | None = None

def get_broker_health_monitor() -> BrokerHealthMonitor:
    """Return process-wide BrokerHealthMonitor singleton.

    Returns:
        BrokerHealthMonitor: shared instance
    """
    global _monitor
    if _monitor is None:
        _monitor = BrokerHealthMonitor()
    return _monitor

def _reset_for_tests() -> None:
    """Drop the singleton between unit tests."""
    global _monitor
    _monitor = None

__all__ = [
    "BrokerHealthMonitor",
    "get_broker_health_monitor",
    "_reset_for_tests",
]
