"""Персистентное состояние circuit breakers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import aiosqlite  # type: ignore

    _HAS_AIOSQLITE = True
except ImportError:
    _HAS_AIOSQLITE = False

CB_DB_PATH = cfg.DATA_DIR / "trades.db"

@dataclass
class CircuitState:
    """In-memory snapshot of circuit breaker state."""

    daily_pnl_rub: float = 0.0
    peak_equity_rub: float = 1_000_000.0
    max_drawdown_pct: float = 0.0
    losing_streak: int = 0
    winning_streak: int = 0
    blocked_until_iso: str | None = None
    block_reason: str = ""

    current_equity_rub: float = 1_000_000.0
    current_drawdown_pct: float = 0.0
    daily_pnl_pct: float = 0.0
    n_trades_today: int = 0

    @property
    def is_blocked(self) -> bool:
        """Return True if breaker block window is still active.

        Returns:
            bool: blocked state
        """
        if not self.blocked_until_iso:
            return False
        try:
            blocked_until = datetime.fromisoformat(self.blocked_until_iso)
            return datetime.now(tz=UTC) < blocked_until
        except Exception:
            return False

    @property
    def sizing_multiplier(self) -> float:
        """Apply streak-based sizing adjustment.

        Returns:
            float: multiplier for position sizing
        """
        if self.losing_streak >= 3:
            return 0.5
        if self.winning_streak >= 5:
            return 1.5
        return 1.0

    @property
    def drawdown_kelly_multiplier(self) -> float:
        """Shrink position size as drawdown grows.

        Returns:
            float: Kelly-adjusted multiplier
        """
        import app.config as _cfg

        activation = 0.02
        if self.current_drawdown_pct < activation:
            return 1.0
        max_dd = max(0.01, abs(_cfg.CIRCUIT_MAX_DD_PCT))
        floor = 0.1
        ratio = self.current_drawdown_pct / max_dd
        return max(floor, 1.0 - ratio)

class CircuitBreaker:
    """Persistent circuit breaker state machine."""

    DAILY_LOSS_HALT_PCT = abs(getattr(cfg, "CIRCUIT_DAILY_LOSS_PCT", -0.02))
    MAX_DRAWDOWN_HALT_PCT = abs(getattr(cfg, "CIRCUIT_MAX_DD_PCT", -0.10))

    def __init__(self) -> None:
        """Init."""
        self.state = CircuitState()
        self._lock = asyncio.Lock()
        self._loaded = False

    async def load(self) -> None:
        """Load state from trades.db.circuit_breaker_state."""
        if not _HAS_AIOSQLITE:
            return
        try:
            async with (
                aiosqlite.connect(CB_DB_PATH) as db,
                db.execute(
                    "SELECT daily_pnl_rub, peak_equity_rub, max_drawdown_pct, "
                    "losing_streak, winning_streak, blocked_until, block_reason "
                    "FROM circuit_breaker_state WHERE id = 1"
                ) as cur,
            ):
                row = await cur.fetchone()
                if row:
                    self.state = CircuitState(
                        daily_pnl_rub=row[0] or 0.0,
                        peak_equity_rub=row[1] or 1_000_000.0,
                        max_drawdown_pct=row[2] or 0.0,
                        losing_streak=row[3] or 0,
                        winning_streak=row[4] or 0,
                        blocked_until_iso=row[5],
                        block_reason=row[6] or "",
                    )
            self._loaded = True
            logger.info(
                "CircuitBreaker state loaded",
                extra={
                    "state": {
                        "daily_pnl_rub": self.state.daily_pnl_rub,
                        "losing_streak": self.state.losing_streak,
                        "winning_streak": self.state.winning_streak,
                        "is_blocked": self.state.is_blocked,
                    }
                },
            )
        except Exception as exc:
            logger.error("CircuitBreaker load failed", extra={"error": str(exc)})

    async def _persist(self) -> None:
        """Write current state back to trades.db."""
        if not _HAS_AIOSQLITE:
            return
        now_iso = datetime.now(tz=UTC).isoformat()
        try:
            async with aiosqlite.connect(CB_DB_PATH) as db:
                await db.execute(
                    "UPDATE circuit_breaker_state "
                    "SET daily_pnl_rub=?, peak_equity_rub=?, max_drawdown_pct=?, "
                    "    losing_streak=?, winning_streak=?, blocked_until=?, "
                    "    block_reason=?, updated_at=? "
                    "WHERE id = 1",
                    (
                        self.state.daily_pnl_rub,
                        self.state.peak_equity_rub,
                        self.state.max_drawdown_pct,
                        self.state.losing_streak,
                        self.state.winning_streak,
                        self.state.blocked_until_iso,
                        self.state.block_reason,
                        now_iso,
                    ),
                )
                await db.commit()
        except Exception as exc:
            logger.error("CircuitBreaker persist failed", extra={"error": str(exc)})

    async def on_trade_closed(self, pnl_rub: float, current_equity_rub: float) -> None:
        """Update streaks and daily P&L when a trade closes.

        Args:
            pnl_rub: trade PnL in RUB
            current_equity_rub: equity after the close
        """
        import math

        if pnl_rub is None or (isinstance(pnl_rub, float) and math.isnan(pnl_rub)):
            logger.warning(
                "CircuitBreaker: skipping NaN/None pnl_rub",
                extra={"pnl_rub": str(pnl_rub)},
            )
            return
        if current_equity_rub is None or (
            isinstance(current_equity_rub, float) and math.isnan(current_equity_rub)
        ):
            logger.warning(
                "CircuitBreaker: skipping NaN/None equity",
                extra={"equity": str(current_equity_rub)},
            )
            return

        async with self._lock:
            self.state.daily_pnl_rub += pnl_rub
            self.state.n_trades_today += 1

            self.state.current_equity_rub = current_equity_rub
            if current_equity_rub > self.state.peak_equity_rub:
                self.state.peak_equity_rub = current_equity_rub
            dd_pct = (
                ((self.state.peak_equity_rub - current_equity_rub) / self.state.peak_equity_rub)
                if self.state.peak_equity_rub > 0
                else 0.0
            )
            self.state.current_drawdown_pct = max(0.0, dd_pct)
            self.state.max_drawdown_pct = max(self.state.max_drawdown_pct, dd_pct)
            self.state.daily_pnl_pct = (
                self.state.daily_pnl_rub / current_equity_rub if current_equity_rub > 0 else 0.0
            )

            if pnl_rub > 0:
                self.state.winning_streak += 1
                self.state.losing_streak = 0
            elif pnl_rub < 0:
                self.state.losing_streak += 1
                self.state.winning_streak = 0

            await self._check_halts(current_equity_rub)
            await self._persist()

        logger.info(
            "Trade closed",
            extra={
                "pnl_rub": pnl_rub,
                "daily_pnl_rub": self.state.daily_pnl_rub,
                "losing_streak": self.state.losing_streak,
                "winning_streak": self.state.winning_streak,
                "max_dd_pct": round(self.state.max_drawdown_pct * 100, 2),
                "is_blocked": self.state.is_blocked,
            },
        )

    async def _check_halts(self, current_equity_rub: float) -> None:
        """Trigger halts based on daily P&L or max DD.

        Args:
            current_equity_rub: equity used for thresholds
        """

        daily_pnl_pct = (
            (self.state.daily_pnl_rub / current_equity_rub) if current_equity_rub > 0 else 0.0
        )
        if daily_pnl_pct <= -self.DAILY_LOSS_HALT_PCT:
            tomorrow = datetime.now(tz=UTC).replace(hour=23, minute=59, second=0, microsecond=0)
            self.state.blocked_until_iso = tomorrow.isoformat()
            self.state.block_reason = f"daily_loss_halt (pnl_pct={daily_pnl_pct:.3f})"
            logger.critical(
                "CIRCUIT BREAKER: daily loss halt",
                extra={
                    "daily_pnl_pct": daily_pnl_pct,
                    "blocked_until": self.state.blocked_until_iso,
                },
            )

        if self.state.max_drawdown_pct >= self.MAX_DRAWDOWN_HALT_PCT:
            from datetime import timedelta

            unblock = datetime.now(tz=UTC) + timedelta(hours=24)
            self.state.blocked_until_iso = unblock.isoformat()
            self.state.block_reason = f"max_drawdown_halt (dd={self.state.max_drawdown_pct:.3f})"
            logger.critical(
                "CIRCUIT BREAKER: max drawdown halt",
                extra={
                    "max_dd": self.state.max_drawdown_pct,
                    "blocked_until": self.state.blocked_until_iso,
                },
            )

    async def reset_daily(self) -> None:
        """Reset daily counters at midnight MSK."""
        async with self._lock:
            self.state.daily_pnl_rub = 0.0
            self.state.daily_pnl_pct = 0.0
            self.state.n_trades_today = 0

            if self.state.block_reason.startswith("daily_loss"):
                self.state.blocked_until_iso = None
                self.state.block_reason = ""
            await self._persist()
        logger.info("CircuitBreaker daily reset")

    async def on_equity_update(self, current_equity_rub: float) -> None:
        """Update equity without closing a trade.

        Args:
            current_equity_rub: current equity in RUB
        """
        async with self._lock:
            self.state.current_equity_rub = current_equity_rub
            if current_equity_rub > self.state.peak_equity_rub:
                self.state.peak_equity_rub = current_equity_rub
            dd = (
                ((self.state.peak_equity_rub - current_equity_rub) / self.state.peak_equity_rub)
                if self.state.peak_equity_rub > 0
                else 0.0
            )
            self.state.current_drawdown_pct = max(0.0, dd)
            self.state.max_drawdown_pct = max(self.state.max_drawdown_pct, dd)
            self.state.daily_pnl_pct = (
                self.state.daily_pnl_rub / current_equity_rub if current_equity_rub > 0 else 0.0
            )

    def should_block_new_trades(self) -> tuple[bool, str]:
        """Return whether new trades should be blocked.

        Returns:
            tuple[bool, str]: (is_blocked, reason)
        """
        if self.state.is_blocked:
            return True, self.state.block_reason
        return False, ""

_cb: CircuitBreaker | None = None

def get_circuit_breaker() -> CircuitBreaker:
    """Return process-wide CircuitBreaker singleton.

    Returns:
        CircuitBreaker: shared instance
    """
    global _cb
    if _cb is None:
        _cb = CircuitBreaker()
    return _cb
