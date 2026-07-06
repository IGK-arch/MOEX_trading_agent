"""APScheduler-обёртка для cron-задач."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

MSK_TZ = "Europe/Moscow"

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore
    from apscheduler.triggers.cron import CronTrigger  # noqa: F401  # type: ignore

    _HAS_APSCHEDULER = True
except ImportError:
    _HAS_APSCHEDULER = False
    logger.warning("APScheduler not installed; scheduled tasks will not run")

class MoexScheduler:
    """Thin wrapper around APScheduler AsyncIOScheduler (MSK timezone)."""

    def __init__(self) -> None:
        """Init."""
        if _HAS_APSCHEDULER:
            self._scheduler = AsyncIOScheduler(timezone=MSK_TZ)
        else:
            self._scheduler = None
        self._jobs: list[dict[str, Any]] = []

    def daily(self, time_msk: str) -> Callable:
        """Register an async function to run daily at time_msk (HH:MM)."""
        hour, minute = (int(x) for x in time_msk.split(":"))

        def decorator(func: Callable) -> Callable:
            """Decorator."""
            self._jobs.append(
                {
                    "func": func,
                    "trigger": "cron",
                    "hour": hour,
                    "minute": minute,
                    "id": f"daily_{func.__name__}_{hour:02d}{minute:02d}",
                }
            )
            return func

        return decorator

    def every(self, seconds: int) -> Callable:
        """Register an async function to run every N seconds."""

        def decorator(func: Callable) -> Callable:
            """Decorator."""
            self._jobs.append(
                {
                    "func": func,
                    "trigger": "interval",
                    "seconds": seconds,
                    "id": f"interval_{func.__name__}_{seconds}s",
                }
            )
            return func

        return decorator

    def weekly(self, day_of_week: str, time_msk: str) -> Callable:
        """Register an async function to run once a week at time_msk.

        Args:
            day_of_week: APS day-of-week string ('mon','tue',...,'sun').
            time_msk: HH:MM (MSK timezone).
        """
        hour, minute = (int(x) for x in time_msk.split(":"))

        def decorator(func: Callable) -> Callable:
            """Decorator."""
            self._jobs.append(
                {
                    "func": func,
                    "trigger": "cron",
                    "day_of_week": day_of_week,
                    "hour": hour,
                    "minute": minute,
                    "id": f"weekly_{func.__name__}_{day_of_week}_{hour:02d}{minute:02d}",
                }
            )
            return func

        return decorator

    def start(self) -> None:
        """Register all jobs and start the scheduler."""
        if not _HAS_APSCHEDULER or self._scheduler is None:
            logger.warning("Scheduler unavailable — skipping scheduled tasks")
            return

        for job in self._jobs:
            func = job.pop("func")
            job_id = job.pop("id", None)
            trigger = job.pop("trigger")

            if trigger == "cron":
                self._scheduler.add_job(
                    func,
                    "cron",
                    id=job_id,
                    timezone=MSK_TZ,
                    replace_existing=True,
                    **job,
                )
            elif trigger == "interval":
                self._scheduler.add_job(
                    func,
                    "interval",
                    id=job_id,
                    replace_existing=True,
                    **job,
                )

            logger.info(
                "Scheduled job registered",
                extra={"job_id": job_id, "trigger": trigger},
            )

        self._scheduler.start()
        logger.info("Scheduler started", extra={"timezone": MSK_TZ})

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=wait)
            logger.info("Scheduler stopped")

scheduler = MoexScheduler()
