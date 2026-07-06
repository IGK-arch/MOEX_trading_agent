"""Мониторинг ежедневного оборота."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import aiosqlite  # type: ignore  # noqa: F401

    _HAS_AIOSQLITE = True
except ImportError:
    _HAS_AIOSQLITE = False

from app.utils.db_pool import get_conn

TRADES_DB = cfg.DATA_DIR / "trades.db"

DAILY_TARGET_RUB = cfg.TURNOVER_TARGET_DAILY_RUB
WEEKLY_TARGET_RUB = cfg.TURNOVER_WARNING_DAY_7_RUB
BIWEEKLY_TARGET_RUB = cfg.TURNOVER_TARGET_14D_RUB

class TurnoverTracker:
    """Tracks turnover and triggers threshold adjustments."""

    def __init__(self) -> None:
        """Init."""
        self._last_check_iso = ""
        self._daily_actual_rub: float = 0.0
        self._weekly_actual_rub: float = 0.0
        self._adjusted_thresholds = False

        self._original_meta_min_proba = cfg.META_MIN_PROBA

        self._recent_outcomes: list[int] = []
        self._max_recent = 30

        self._hackathon_start_iso: str = "2026-05-28"
        self._cum_turnover_rub: float = 0.0
        self._trades_today: int = 0
        self._intraday_boost_active: bool = False

    @property
    def today_volume(self) -> float:
        """Return current day's turnover.

        Returns:
            float: turnover in RUB
        """
        return self._daily_actual_rub

    async def run_check(self) -> dict[str, Any]:
        """Check daily and weekly turnover; adjust thresholds if behind.

        Returns:
            dict[str, Any]: turnover snapshot
        """
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        week_start = (datetime.now(tz=UTC) - timedelta(days=7)).strftime("%Y-%m-%d")

        self._daily_actual_rub = await self._sum_volume_since(today)
        self._weekly_actual_rub = await self._sum_volume_since(week_start)
        self._cum_turnover_rub = await self._sum_volume_since(self._hackathon_start_iso)

        daily_pct = self._daily_actual_rub / DAILY_TARGET_RUB if DAILY_TARGET_RUB > 0 else 0
        weekly_pct = self._weekly_actual_rub / WEEKLY_TARGET_RUB if WEEKLY_TARGET_RUB > 0 else 0

        await self._write_turnover_log(today)

        should_loosen = weekly_pct < 0.5
        if should_loosen and not self._adjusted_thresholds:
            await self._loosen_thresholds()
            self._adjusted_thresholds = True
        elif not should_loosen and self._adjusted_thresholds:
            await self._restore_thresholds()
            self._adjusted_thresholds = False

        day_index = self._compute_day_index()
        new_level = self._evaluate_escalation_ladder(
            day_index=day_index,
            cum_turnover_rub=self._cum_turnover_rub,
        )

        result = {
            "date": today,
            "daily_volume_rub": self._daily_actual_rub,
            "daily_target_rub": DAILY_TARGET_RUB,
            "daily_pct": daily_pct,
            "weekly_volume_rub": self._weekly_actual_rub,
            "weekly_target_rub": WEEKLY_TARGET_RUB,
            "weekly_pct": weekly_pct,
            "cumulative_turnover_rub": self._cum_turnover_rub,
            "hackathon_day_index": day_index,
            "turnover_escalation_level": new_level,
            "thresholds_loosened": self._adjusted_thresholds,
        }
        logger.info("Turnover check", extra=result)
        return result

    def _compute_day_index(self) -> int:
        """Compute days since hackathon start.

        Returns:
            int: 0-based day index
        """
        try:
            start = datetime.strptime(self._hackathon_start_iso, "%Y-%m-%d").replace(tzinfo=UTC)
            delta = datetime.now(tz=UTC) - start
            return max(0, delta.days)
        except Exception:
            return 0

    def _evaluate_escalation_ladder(
        self,
        *,
        day_index: int,
        cum_turnover_rub: float,
    ) -> int:
        """Decide and apply escalation level.

        Args:
            day_index: hackathon day index
            cum_turnover_rub: cumulative turnover
        Returns:
            int: new escalation level
        """
        current = int(getattr(cfg, "TURNOVER_ESCALATION_LEVEL", 0))
        proposed = 0
        if day_index >= 10 and cum_turnover_rub < cfg.TURNOVER_ESCALATION_DAY_10_THRESHOLD_RUB:
            proposed = 2
        elif day_index >= 7 and cum_turnover_rub < cfg.TURNOVER_ESCALATION_DAY_7_THRESHOLD_RUB:
            proposed = 1
        new_level = max(current, proposed)
        if new_level != current:
            cfg.apply_turnover_escalation(new_level)
            logger.warning(
                "Turnover escalation level changed",
                extra={
                    "old_level": current,
                    "new_level": new_level,
                    "day_index": day_index,
                    "cum_turnover_rub": cum_turnover_rub,
                    "size_mult": cfg.get_size_mult_for_escalation(),
                    "active_tickers": [
                        t for t, p in cfg.PER_TICKER_POLICY.items() if p != "DISABLED"
                    ],
                },
            )
        return new_level

    async def _sum_volume_since(self, date_iso: str) -> float:
        """Sum order_value from trades since date.

        Args:
            date_iso: YYYY-MM-DD lower bound
        Returns:
            float: total RUB volume
        """
        if not _HAS_AIOSQLITE:
            return 0.0
        try:
            db = await get_conn(TRADES_DB)
            async with db.execute(
                "SELECT COALESCE(SUM(order_value), 0) FROM trades WHERE trade_date >= ?",
                (date_iso,),
            ) as cur:
                row = await cur.fetchone()
                return float(row[0]) if row else 0.0
        except Exception as exc:
            logger.warning("turnover sum failed", extra={"error": str(exc)})
            return 0.0

    async def _write_turnover_log(self, date_iso: str) -> None:
        """Append a row to turnover_log table.

        Args:
            date_iso: YYYY-MM-DD date
        """
        if not _HAS_AIOSQLITE:
            return
        try:
            now_iso = datetime.now(tz=UTC).isoformat()
            db = await get_conn(TRADES_DB)
            await db.execute(
                """
                INSERT INTO turnover_log
                (date, daily_volume_rub, cumulative_volume_rub, trade_count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (date_iso, self._daily_actual_rub, self._weekly_actual_rub, 0, now_iso),
            )
            await db.commit()
        except Exception as exc:
            logger.warning("turnover log write failed", extra={"error": str(exc)})

    async def _loosen_thresholds(self) -> None:
        """Dynamically loosen entry thresholds and meta floor."""
        try:
            from app.agents.mean_reversion import get_mean_reversion
            from app.agents.pair_trader import get_pair_trader

            pt = get_pair_trader()
            pt.z_entry = max(1.5, pt.z_entry - 0.3)
            mr = get_mean_reversion()
            mr.bb_std = max(1.5, mr.bb_std - 0.3)

            floor = getattr(cfg, "META_MIN_PROBA_FLOOR", 0.45)
            new_meta = max(floor, cfg.META_MIN_PROBA - 0.05)
            cfg.META_MIN_PROBA = new_meta
            logger.warning(
                "Turnover behind target — thresholds loosened",
                extra={
                    "pair_z_entry": pt.z_entry,
                    "mr_bb_std": mr.bb_std,
                    "meta_min_proba": new_meta,
                },
            )
        except Exception as exc:
            logger.error("loosen_thresholds failed", extra={"error": str(exc)})

    async def _restore_thresholds(self) -> None:
        """Restore default thresholds."""
        try:
            from app.agents.mean_reversion import get_mean_reversion
            from app.agents.pair_trader import get_pair_trader

            pt = get_pair_trader()
            pt.z_entry = (
                cfg.PAIR_Z_ENTRY_THRESHOLD if hasattr(cfg, "PAIR_Z_ENTRY_THRESHOLD") else 2.0
            )
            mr = get_mean_reversion()
            mr.bb_std = cfg.BB_STD
            cfg.META_MIN_PROBA = self._original_meta_min_proba
            logger.info(
                "Turnover recovered — thresholds restored",
                extra={"meta_min_proba": cfg.META_MIN_PROBA},
            )
        except Exception as exc:
            logger.error("restore_thresholds failed", extra={"error": str(exc)})

    def on_trade_outcome(self, pnl_rub: float) -> None:
        """Record a closed trade's outcome.

        Args:
            pnl_rub: trade PnL in RUB
        """
        outcome = 1 if pnl_rub > 0 else 0
        self._recent_outcomes.append(outcome)
        if len(self._recent_outcomes) > self._max_recent:
            self._recent_outcomes.pop(0)

    def on_trade_opened(self) -> None:
        """Increment intraday trade counter."""
        self._trades_today += 1

    def reset_daily_counters(self) -> None:
        """Reset daily counters at session start."""
        self._trades_today = 0
        if self._intraday_boost_active:
            cfg.META_MIN_PROBA = self._original_meta_min_proba
            self._intraday_boost_active = False
            logger.info("Intraday boost reset at session start")

    def apply_intraday_meta_boost(self, now_msk_hour: int | None = None) -> float:
        """Cut META_MIN_PROBA when too few trades by cutoff hour.

        Args:
            now_msk_hour: current MSK hour or None
        Returns:
            float: current META_MIN_PROBA after boost
        """
        if self._intraday_boost_active:
            return cfg.META_MIN_PROBA
        if now_msk_hour is None:
            now_msk_hour = (datetime.now(tz=UTC) + timedelta(hours=3)).hour
        cutoff = int(getattr(cfg, "TURNOVER_INTRADAY_BOOST_CUTOFF_HOUR_MSK", 14))
        min_trades = int(getattr(cfg, "TURNOVER_INTRADAY_BOOST_MIN_TRADES", 5))
        if now_msk_hour < cutoff:
            return cfg.META_MIN_PROBA
        if self._trades_today >= min_trades:
            return cfg.META_MIN_PROBA
        delta = float(getattr(cfg, "TURNOVER_INTRADAY_BOOST_DELTA", 0.05))
        abs_floor = float(getattr(cfg, "TURNOVER_INTRADAY_BOOST_ABS_FLOOR", 0.25))
        new_val = max(abs_floor, cfg.META_MIN_PROBA - delta)
        if new_val < cfg.META_MIN_PROBA:
            logger.warning(
                "Intraday META_MIN_PROBA boost applied",
                extra={
                    "old": cfg.META_MIN_PROBA,
                    "new": new_val,
                    "trades_today": self._trades_today,
                    "hour_msk": now_msk_hour,
                    "min_trades_required": min_trades,
                },
            )
            cfg.META_MIN_PROBA = new_val
            self._intraday_boost_active = True
        return cfg.META_MIN_PROBA

    def adaptive_meta_step(self) -> float:
        """Adjust META_MIN_PROBA based on recent win-rate.

        Returns:
            float: updated META_MIN_PROBA
        """
        if len(self._recent_outcomes) < 10:
            return cfg.META_MIN_PROBA
        win_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)
        delta = 0.0
        if win_rate > 0.80:
            delta = +0.01
        elif win_rate < 0.55:
            delta = -0.01
        if delta == 0.0:
            return cfg.META_MIN_PROBA
        floor = getattr(cfg, "META_MIN_PROBA_FLOOR", 0.45)
        ceiling = getattr(cfg, "META_MIN_PROBA_CEILING", 0.70)
        new_val = max(floor, min(ceiling, cfg.META_MIN_PROBA + delta))
        if new_val != cfg.META_MIN_PROBA:
            logger.info(
                "Adaptive meta threshold",
                extra={
                    "old": cfg.META_MIN_PROBA,
                    "new": new_val,
                    "recent_win_rate": round(win_rate, 3),
                    "n_recent": len(self._recent_outcomes),
                },
            )
            cfg.META_MIN_PROBA = new_val
        return new_val

_tracker: TurnoverTracker | None = None

def get_turnover_tracker() -> TurnoverTracker:
    """Return process-wide TurnoverTracker singleton.

    Returns:
        TurnoverTracker: shared instance
    """
    global _tracker
    if _tracker is None:
        _tracker = TurnoverTracker()
    return _tracker
