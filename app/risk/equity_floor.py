"""Hard equity floor — emergency "all-stop" if equity drops below threshold.

v1.0.10 (Cycle 10) — за ночь 27 May 2026 капитал упал с 179882 ₽ до 155602 ₽
(-13.5%) пока бот сидел в неправильном URL. Если такое случится снова,
у нас должна быть жёсткая нижняя планка: если equity < 75% от стартового
капитала (= 750 000 ₽ при 1 млн стартового) — full close + permanent halt
до того как пользователь явно снимет блок.

v1.1.0 (Cycle 11) — добавлен **trailing-peak** режим: floor пересчитывается
от максимума equity, виданного с момента старта (peak × floor_pct), а не от
стартового капитала. Это lock-in profit: выросли с 1M до 1.5M → floor
поднимается с 750k до 1.125M, защищая прибыль.

В отличие от ``CircuitBreaker.DAILY_LOSS_HALT_PCT`` (2% / день) и
``MAX_DRAWDOWN_HALT_PCT`` (10% / 24h), это **долгосрочный** floor:
пережил 25% drawdown от пика → halt.

Public API:
    floor = get_equity_floor()
    ok, reason = floor.check(current_equity_rub)
    # peak обновляется автоматически при каждом check()

Feature flags:
    ``cfg.EQUITY_HARD_FLOOR_ENABLED``   — default ON
    ``cfg.EQUITY_TRAILING_PEAK_ENABLED`` — default ON (v1.1.0 new)

Persistence: peak_equity сохраняется в ``data/recovery_state.json`` через
``RecoveryStateManager.extras`` — переживает рестарт пода.
"""

from __future__ import annotations

import time

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_STARTING_CAPITAL_RUB: float = 1_000_000.0

class EquityFloor:
    """Single-shot equity floor guard with optional trailing-peak mode.

    Once ``check()`` returns ``(False, _)`` the breach is latched
    (``_breached = True``) — subsequent calls keep returning False until
    ``override_release()`` is called. This matches the "permanent halt
    до user override" requirement: a brief intraday spike back above the
    floor should NOT re-enable trading.

    Trailing-peak mode (v1.1.0): when ``trailing_peak_enabled=True`` the
    threshold is ``peak_equity_seen × floor_pct`` instead of
    ``starting_capital × floor_pct``. The peak monotonically grows as
    equity grows; once locked in, it does NOT decrease — this gives a
    profit-lock effect.
    """

    def __init__(
        self,
        *,
        starting_capital_rub: float | None = None,
        floor_pct: float | None = None,
        trailing_peak_enabled: bool | None = None,
    ) -> None:
        """Init.

        Args:
            starting_capital_rub: deposit baseline; defaults to cfg value
                or 1 000 000 ₽ when unset.
            floor_pct: fraction of starting capital (or trailing peak)
                that is the floor (e.g. 0.75 → 25% drawdown). Defaults
                to ``cfg.EQUITY_HARD_FLOOR_PCT``.
            trailing_peak_enabled: if True (default — see
                ``cfg.EQUITY_TRAILING_PEAK_ENABLED``) the floor pivots
                off ``max(current_equity, starting_capital)`` instead of
                only starting_capital.
        """
        self.starting_capital_rub = float(
            starting_capital_rub
            if starting_capital_rub is not None
            else getattr(cfg, "EQUITY_FLOOR_STARTING_CAPITAL_RUB", _DEFAULT_STARTING_CAPITAL_RUB)
        )
        self.floor_pct = float(
            floor_pct if floor_pct is not None else getattr(cfg, "EQUITY_HARD_FLOOR_PCT", 0.75)
        )
        self.trailing_peak_enabled = bool(
            trailing_peak_enabled
            if trailing_peak_enabled is not None
            else getattr(cfg, "EQUITY_TRAILING_PEAK_ENABLED", True)
        )
        self._peak_equity: float = self.starting_capital_rub
        self._breached = False
        self._breach_ts: float | None = None
        self._breach_equity: float | None = None

    @property
    def peak_equity_rub(self) -> float:
        """Return peak equity seen so far (≥ starting_capital).

        Returns:
            float: peak in RUB
        """
        return self._peak_equity

    @property
    def threshold_rub(self) -> float:
        """Return absolute floor in RUB.

        When trailing_peak is enabled: ``peak_equity × floor_pct``.
        Else: ``starting_capital × floor_pct``.

        Returns:
            float: current threshold
        """
        base = self._peak_equity if self.trailing_peak_enabled else self.starting_capital_rub
        return base * self.floor_pct

    @property
    def breached(self) -> bool:
        """Return True once the floor has been broken (latched).

        Returns:
            bool: latched breach state
        """
        return self._breached

    def update_peak(self, current_equity_rub: float) -> None:
        """Pump the trailing peak if equity made a new high.

        Safe to call independently of ``check()`` — useful when the
        caller wants to record a high-water mark without evaluating the
        floor (e.g. during recovery boot).

        Args:
            current_equity_rub: latest equity
        """
        import math

        try:
            equity = float(current_equity_rub)
        except (TypeError, ValueError):
            return
        if not math.isfinite(equity) or equity < 0:
            return
        if equity > self._peak_equity:
            self._peak_equity = equity

    def load_peak(self, peak_equity_rub: float) -> None:
        """Restore peak from persisted snapshot.

        Floor never accepts a peak lower than starting_capital — that's
        the absolute lower bound. Used by ``RecoveryStateManager`` on
        cold restart.

        Args:
            peak_equity_rub: peak from recovery_state.json
        """
        try:
            p = float(peak_equity_rub)
        except (TypeError, ValueError):
            return
        import math

        if not math.isfinite(p) or p <= 0:
            return
        self._peak_equity = max(p, self.starting_capital_rub)

    def check(
        self,
        current_equity_rub: float,
        peak_equity_rub: float | None = None,
    ) -> tuple[bool, str]:
        """Evaluate current equity against the floor.

        Trailing-peak mode: the peak is auto-updated on every call.
        Caller MAY pass an externally-tracked ``peak_equity_rub`` (e.g.
        from a checkpointed state) — that value seeds the in-memory peak
        before the update.

        Args:
            current_equity_rub: latest equity (cash + market value of
                open positions).
            peak_equity_rub: optional externally-tracked peak. When
                provided, the instance peak is bumped to
                ``max(self._peak_equity, peak_equity_rub)`` *before* the
                check. Backward compatibility: callers that don't track
                peak themselves pass ``None`` (the default) and the
                instance manages it internally.

        Returns:
            tuple[bool, str]: ``(ok, reason)``. ``ok=True`` means trading
                may continue; ``ok=False`` means the floor is hit
                (or was previously hit and not yet released).
        """
        if not getattr(cfg, "EQUITY_HARD_FLOOR_ENABLED", True):
            return True, "disabled"
        try:
            equity = float(current_equity_rub)
        except (TypeError, ValueError):
            return True, "non-numeric-equity"
        import math

        if not math.isfinite(equity) or equity < 0:
            return True, f"invalid-equity={equity}"

        if peak_equity_rub is not None:
            try:
                ext_peak = float(peak_equity_rub)
                if math.isfinite(ext_peak) and ext_peak > self._peak_equity:
                    self._peak_equity = ext_peak
            except (TypeError, ValueError):
                pass
        if self.trailing_peak_enabled and equity > self._peak_equity:
            self._peak_equity = equity

        if self._breached:
            return False, (
                f"EQUITY_HARD_FLOOR_LATCHED: breach @ {self._breach_equity:.0f} "
                f"< {self.threshold_rub:.0f} — awaiting user override"
            )

        if equity < self.threshold_rub:
            self._breached = True
            self._breach_ts = time.time()
            self._breach_equity = equity
            logger.critical(
                "EQUITY HARD FLOOR BREACHED — halting new entries, "
                "flattening book required",
                extra={
                    "current_equity_rub": round(equity, 2),
                    "threshold_rub": round(self.threshold_rub, 2),
                    "peak_equity_rub": round(self._peak_equity, 2),
                    "starting_capital_rub": self.starting_capital_rub,
                    "floor_pct": self.floor_pct,
                    "trailing_peak_enabled": self.trailing_peak_enabled,
                    "drawdown_from_peak_pct": round(
                        1.0 - (equity / self._peak_equity), 4
                    ),
                    "drawdown_from_start_pct": round(
                        1.0 - (equity / self.starting_capital_rub), 4
                    ),
                },
            )
            mode = "peak" if self.trailing_peak_enabled else "start"
            return False, (
                f"EQUITY_HARD_FLOOR[{mode}]: {equity:.0f} < {self.threshold_rub:.0f} "
                f"(={self.floor_pct:.0%} of "
                f"{(self._peak_equity if self.trailing_peak_enabled else self.starting_capital_rub):.0f})"
            )

        return True, "ok"

    def override_release(self, *, by: str = "user") -> None:
        """Manually clear the breach latch.

        Args:
            by: short tag for who released the latch (logged for audit).
        """
        if not self._breached:
            return
        logger.warning(
            "EquityFloor breach manually released — trading may resume",
            extra={
                "released_by": by,
                "was_breached_at": self._breach_ts,
                "breach_equity": self._breach_equity,
                "peak_equity_rub": self._peak_equity,
            },
        )
        self._breached = False
        self._breach_ts = None
        self._breach_equity = None

    def reset_for_stage(self, *, label: str = "stage2") -> None:
        """Полный reset: peak → starting_capital + unlatch breach.

        Используется однократно при старте новой фазы хакатона
        (например, Этап 2 когда ArenaGo обнуляет капитал до 1 000 000 ₽).
        После этого peak пересоберётся с нуля по новым trades.

        Args:
            label: метка фазы (Этап 2 / manual reset / ...). Идёт в логи.
        """
        logger.critical(
            "EquityFloor RESET_FOR_STAGE — peak and latch cleared",
            extra={
                "label": label,
                "old_peak_equity_rub": round(self._peak_equity, 2),
                "new_peak_equity_rub": round(self.starting_capital_rub, 2),
                "was_breached": self._breached,
            },
        )
        self._peak_equity = self.starting_capital_rub
        self._breached = False
        self._breach_ts = None
        self._breach_equity = None

    def snapshot(self) -> dict[str, float | bool | str | None]:
        """Return a serialisable snapshot for dashboards / persistence.

        ``peak_equity_rub`` and ``trailing_peak_enabled`` are written to
        ``recovery_state.json`` so the peak survives pod restarts.

        Returns:
            dict[str, float | bool | str | None]: state at call time
        """
        return {
            "enabled": bool(getattr(cfg, "EQUITY_HARD_FLOOR_ENABLED", True)),
            "starting_capital_rub": self.starting_capital_rub,
            "floor_pct": self.floor_pct,
            "trailing_peak_enabled": self.trailing_peak_enabled,
            "peak_equity_rub": self._peak_equity,
            "threshold_rub": self.threshold_rub,
            "breached": self._breached,
            "breach_ts": self._breach_ts,
            "breach_equity": self._breach_equity,
        }

def check_equity_floor(
    current_equity_rub: float,
    starting_capital_rub: float = _DEFAULT_STARTING_CAPITAL_RUB,
    peak_equity_rub: float | None = None,
) -> tuple[bool, str]:
    """Functional helper for one-off checks (matches Cycle 10/11 spec).

    Routes through the process-wide singleton so latch semantics are
    preserved across call sites. The ``starting_capital_rub`` argument
    is only honoured on first construction — afterwards the singleton's
    baseline wins. ``peak_equity_rub`` is forwarded to ``check()`` every
    call (used by trailing-peak mode).

    Args:
        current_equity_rub: latest equity in RUB
        starting_capital_rub: baseline deposit used for the threshold
            on first init. Ignored on subsequent calls. (back-compat)
        peak_equity_rub: optional externally-tracked peak. ``None``
            keeps backward-compatible behaviour where the singleton
            manages peak internally.

    Returns:
        tuple[bool, str]: ``(ok, reason)``
    """
    global _equity_floor
    if _equity_floor is None:
        _equity_floor = EquityFloor(starting_capital_rub=starting_capital_rub)
    return _equity_floor.check(current_equity_rub, peak_equity_rub=peak_equity_rub)

_equity_floor: EquityFloor | None = None

def get_equity_floor() -> EquityFloor:
    """Return process-wide EquityFloor singleton.

    Returns:
        EquityFloor: shared instance
    """
    global _equity_floor
    if _equity_floor is None:
        _equity_floor = EquityFloor()
    return _equity_floor

def _reset_for_tests() -> None:
    """Drop the singleton between unit tests."""
    global _equity_floor
    _equity_floor = None

__all__ = [
    "EquityFloor",
    "check_equity_floor",
    "get_equity_floor",
    "_reset_for_tests",
]
