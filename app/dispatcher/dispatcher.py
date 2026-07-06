"""Главный event loop диспетчера."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import app.config as cfg
from app.agents.base import BaseAdapter
from app.agents.meta_classifier import MetaContext
from app.dispatcher.aggregator import SignalAggregator
from app.dispatcher.entry_guard import get_entry_guard
from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    Direction,
    RiskCheckResult,
    UnifiedSignal,
)
from app.dispatcher.tier_classifier import apply_tier
from app.execution.order_manager import get_daily_trade_guard, get_order_manager
from app.risk.circuit_breakers import get_circuit_breaker
from app.risk.position_book import get_position_book
from app.risk.risk_manager import get_risk_manager
from app.utils.logging import get_logger, new_trace_id, set_trace_id
from app.utils.sessions import is_trading_open

logger = get_logger(__name__)

_MSK = timezone(timedelta(hours=3))

class Dispatcher:
    """Main coordinator. Runs the 30-second decision cycle."""

    def __init__(
        self,
        adapters: list[BaseAdapter],
        aggregator: SignalAggregator | None = None,
        cycle_seconds: float = 30.0,
        poll_timeout_seconds: float = 0.5,
    ) -> None:
        """Init."""
        self.adapters = adapters
        self.aggregator = aggregator or SignalAggregator()
        self.cycle_seconds = cycle_seconds
        self.poll_timeout_seconds = poll_timeout_seconds
        self.risk = get_risk_manager()
        self.orders = get_order_manager()
        self.entry_guard = get_entry_guard()
        self.daily_guard = get_daily_trade_guard()

        self._cycle_count = 0
        self._signals_gathered_total = 0
        self._decisions_executed_total = 0
        self._stop_event = asyncio.Event()

        self._consecutive_empty_polls = 0
        self._empty_polls_threshold = 3

        self.priority_event = asyncio.Event()

        self._attrition_counters: dict[str, int] = {
            "raw": 0,
            "allowed": 0,
            "tier_passed": 0,
            "meta_passed": 0,
            "risk_passed": 0,
            "submitted": 0,
        }
        self._rejection_breakdown: dict[str, int] = defaultdict(int)
        self._last_cycle_attrition: dict[str, int] = {}

    async def run(self) -> None:
        """Main loop; block until stop()."""
        logger.info(
            "Запуск цикла диспетчера",
            extra={
                "cycle_seconds": self.cycle_seconds,
                "adapters": [a.name for a in self.adapters],
            },
        )
        while not self._stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                await self._run_one_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Dispatcher cycle crashed", extra={"error": str(exc)})

            elapsed = time.monotonic() - cycle_start
            sleep_for = max(0.0, self.cycle_seconds - elapsed)
            if elapsed > self.cycle_seconds * 1.1:
                logger.warning(
                    "Dispatcher cycle exceeded budget",
                    extra={"elapsed_sec": round(elapsed, 2), "budget_sec": self.cycle_seconds},
                )

            stop_wait = asyncio.create_task(self._stop_event.wait())
            prio_wait = asyncio.create_task(self.priority_event.wait())
            try:
                done, pending = await asyncio.wait(
                    {stop_wait, prio_wait},
                    timeout=sleep_for,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for t in (stop_wait, prio_wait):
                    if not t.done():
                        t.cancel()
            if self.priority_event.is_set():
                self.priority_event.clear()
                logger.info(
                    "Диспетчер пробуждён приоритетным событием", extra={"elapsed_sec": round(elapsed, 2)}
                )

    def stop(self) -> None:
        """Signal the loop to stop."""
        self._stop_event.set()

    async def _run_one_cycle(self) -> None:
        """Run one dispatcher cycle: poll adapters, aggregate, execute."""
        cycle_bucket = int(time.time() // max(1, int(self.cycle_seconds)))
        cycle_id = f"c{cycle_bucket:x}"
        trace_id = new_trace_id()
        set_trace_id(trace_id)
        self._cycle_count += 1

        cycle_attrition: dict[str, int] = {
            "raw": 0,
            "allowed": 0,
            "tier_passed": 0,
            "meta_passed": 0,
            "risk_passed": 0,
            "submitted": 0,
        }
        cycle_rejections: dict[str, int] = defaultdict(int)

        if not is_trading_open():
            logger.debug(
                "Market closed — skipping cycle", extra={"cycle_id": cycle_id, "trace_id": trace_id}
            )
            return

        gather_start = time.monotonic()
        results = await asyncio.gather(
            *(a.safe_poll(timeout=self.poll_timeout_seconds) for a in self.adapters),
            return_exceptions=False,
        )
        gather_ms = round((time.monotonic() - gather_start) * 1000)

        all_empty = all(not s for s in results)
        if all_empty:
            self._consecutive_empty_polls += 1
            if self._consecutive_empty_polls == self._empty_polls_threshold:
                logger.critical(
                    "Диспетчер глух: все адаптеры пусты N циклов подряд",
                    extra={
                        "cycle_id": cycle_id,
                        "consecutive_empty_cycles": self._consecutive_empty_polls,
                        "adapters": [a.name for a in self.adapters],
                    },
                )
        else:
            self._consecutive_empty_polls = 0

        all_signals: list[UnifiedSignal] = []
        for sigs in results:
            if sigs:
                all_signals.extend(sigs)

        raw_count = len(all_signals)
        cycle_attrition["raw"] = raw_count
        all_signals = [s for s in all_signals if cfg.is_signal_allowed(s.ticker, s.detector)]
        cycle_attrition["allowed"] = len(all_signals)
        dropped = raw_count - len(all_signals)
        if dropped > 0:
            logger.info(
                "Сигналы отфильтрованы по PER_TICKER_POLICY",
                extra={
                    "raw": raw_count,
                    "allowed": len(all_signals),
                    "dropped": dropped,
                    "cycle_id": cycle_id,
                    "trace_id": trace_id,
                },
            )

        self._signals_gathered_total += len(all_signals)

        if not all_signals:
            logger.debug(
                "Dispatcher cycle: no signals",
                extra={"cycle_id": cycle_id, "gather_ms": gather_ms, "trace_id": trace_id},
            )
            return

        per_ticker: dict[str, list[UnifiedSignal]] = defaultdict(list)
        for s in all_signals:
            per_ticker[s.ticker].append(s)

        decisions: list[Decision] = []

        base_ctx = self._build_base_meta_context()

        supercandles_by_ticker = await self._fetch_supercandles_batch(list(per_ticker.keys()))

        meta_contexts = {
            ticker: self._ticker_meta_context(base_ctx, ticker, sigs)
            for ticker, sigs in per_ticker.items()
        }
        aggregate_start = time.monotonic()
        try:
            decisions = await self.aggregator.aggregate_batch(
                cycle_id=cycle_id,
                per_ticker_signals=per_ticker,
                meta_contexts=meta_contexts,
                supercandles_by_ticker=supercandles_by_ticker,
            )
        except Exception as exc:
            logger.error(
                "Aggregation batch failed",
                extra={"error": str(exc), "trace_id": trace_id},
            )
            decisions = []
        aggregate_ms = round((time.monotonic() - aggregate_start) * 1000)

        for decision in decisions:
            try:
                apply_tier(decision)
            except Exception as exc:
                logger.error(
                    "apply_tier failed",
                    extra={"ticker": decision.ticker, "error": str(exc), "trace_id": trace_id},
                )

        for decision in decisions:
            if decision.action == DecisionAction.EXECUTE:
                cycle_attrition["tier_passed"] += 1
                threshold = (
                    decision.meta_threshold
                    if decision.meta_threshold is not None
                    else cfg.META_MIN_PROBA
                )
                if decision.meta_score is None or decision.meta_score >= threshold:
                    cycle_attrition["meta_passed"] += 1
            elif decision.action == DecisionAction.NO_TRADE:
                reason = (decision.rationale or "no_trade").split(":")[0].strip().lower()
                cycle_rejections[reason or "no_trade"] += 1

        executed = 0
        rejected = 0
        no_trade = 0
        veto = 0
        risk_total_ms = 0
        submit_total_ms = 0

        for decision in decisions:
            if decision.action == DecisionAction.NO_TRADE:
                no_trade += 1
                continue
            if decision.action == DecisionAction.VETO:
                veto += 1
                cycle_rejections["veto"] += 1
                await self.orders._upsert_decision(decision)
                logger.info(
                    "Решение VETO",
                    extra={
                        "decision_id": decision.decision_id,
                        "ticker": decision.ticker,
                        "rationale": decision.rationale,
                        "trace_id": trace_id,
                    },
                )
                continue

            logger.info(
                "Решение EXECUTE",
                extra={
                    "decision_id": decision.decision_id,
                    "ticker": decision.ticker,
                    "direction": getattr(decision.direction, "value", str(decision.direction)),
                    "trace_id": trace_id,
                },
            )

            existing_for_guard = self.orders.book.get_position(decision.ticker)
            is_exit_for_guard = bool(
                existing_for_guard is not None
                and decision.direction in (Direction.BUY, Direction.SELL)
                and (
                    (existing_for_guard.quantity > 0 and decision.direction == Direction.SELL)
                    or (existing_for_guard.quantity < 0 and decision.direction == Direction.BUY)
                )
            )

            if not is_exit_for_guard:
                try:
                    from app.risk.broker_health_monitor import get_broker_health_monitor

                    if get_broker_health_monitor().is_safe_mode():
                        rejected += 1
                        cycle_rejections["safe_mode"] += 1
                        decision.action = DecisionAction.NO_TRADE
                        decision.rationale = (
                            "REJECTED_SAFE_MODE: broker unreachable > 5 min "
                            "| " + (decision.rationale or "")
                        )
                        await self.orders._upsert_decision(decision)
                        logger.warning(
                            "SAFE MODE — rejecting entry",
                            extra={
                                "decision_id": decision.decision_id,
                                "ticker": decision.ticker,
                                "trace_id": trace_id,
                            },
                        )
                        continue
                except Exception:  # pragma: no cover — never crash dispatcher
                    pass

            if not self.daily_guard.can_submit(is_exit=is_exit_for_guard):
                rejected += 1
                cycle_rejections["daily_trade_soft_cap"] += 1
                snap = self.daily_guard.snapshot()
                decision.action = DecisionAction.NO_TRADE
                decision.rationale = (
                    f"REJECTED_DAILY_SOFT_CAP ({snap['count']}/{snap['limit']}, "
                    f"soft={snap['soft_cap']}, exits-only) | " + (decision.rationale or "")
                )
                await self.orders._upsert_decision(decision)
                logger.warning(
                    "Daily trade soft-cap hit — blocking entry",
                    extra={
                        "decision_id": decision.decision_id,
                        "ticker": decision.ticker,
                        "is_exit": is_exit_for_guard,
                        "snapshot": snap,
                        "trace_id": trace_id,
                    },
                )
                continue

            risk_start = time.monotonic()
            try:
                risk_result = await self.risk.evaluate(decision)
            except Exception as exc:
                risk_total_ms += round((time.monotonic() - risk_start) * 1000)
                logger.error(
                    "Risk eval failed",
                    extra={
                        "decision_id": decision.decision_id,
                        "ticker": decision.ticker,
                        "trace_id": trace_id,
                        "error": str(exc),
                    },
                )
                continue
            risk_total_ms += round((time.monotonic() - risk_start) * 1000)

            decision.risk_check = risk_result.result
            if risk_result.result != RiskCheckResult.PASSED:
                rejected += 1
                cycle_rejections[f"risk_{risk_result.result.value.lower()}"] += 1
                decision.action = DecisionAction.NO_TRADE
                decision.rationale = (
                    f"REJECTED ({risk_result.result.value}): "
                    + risk_result.reason
                    + " | "
                    + decision.rationale
                )
                await self.orders._upsert_decision(decision)
                logger.warning(
                    "Risk отклонил решение",
                    extra={
                        "ticker": decision.ticker,
                        "result": risk_result.result.value,
                        "reason": risk_result.reason,
                        "decision_id": decision.decision_id,
                        "trace_id": trace_id,
                    },
                )
                continue

            cycle_attrition["risk_passed"] += 1
            decision.trade_request = risk_result.trade_request

            try:
                guard_ok, guard_reason = await self.entry_guard.confirm_entry(
                    decision,
                    supercandles_by_ticker.get(decision.ticker),
                )
            except Exception as exc:
                logger.warning(
                    "EntryGuard raised — failing open",
                    extra={
                        "decision_id": decision.decision_id,
                        "ticker": decision.ticker,
                        "error": str(exc),
                        "trace_id": trace_id,
                    },
                )
                guard_ok, guard_reason = True, "guard_error"

            if not guard_ok:
                rejected += 1
                cycle_rejections["late_stage_check"] += 1
                decision.action = DecisionAction.NO_TRADE
                decision.rationale = (
                    f"REJECTED_LATE_STAGE_CHECK ({guard_reason}) | " + decision.rationale
                )
                await self.orders._upsert_decision(decision)
                logger.info(
                    "EntryGuard skipped submit",
                    extra={
                        "decision_id": decision.decision_id,
                        "ticker": decision.ticker,
                        "reason": guard_reason,
                        "trace_id": trace_id,
                    },
                )
                continue

            submit_start = time.monotonic()
            try:
                await self.orders.submit(decision)
                executed += 1
                cycle_attrition["submitted"] += 1
                self._decisions_executed_total += 1
            except Exception as exc:
                logger.error(
                    "Submit failed",
                    extra={
                        "decision_id": decision.decision_id,
                        "ticker": decision.ticker,
                        "trace_id": trace_id,
                        "error": str(exc),
                    },
                )
            submit_total_ms += round((time.monotonic() - submit_start) * 1000)

        actionable = executed + rejected + veto
        summary_level = logger.info if actionable > 0 else logger.debug
        summary_level(
            "Сводка цикла диспетчера",
            extra={
                "cycle_id": cycle_id,
                "trace_id": trace_id,
                "signals": len(all_signals),
                "decisions": len(decisions),
                "executed": executed,
                "rejected": rejected,
                "no_trade": no_trade,
                "veto": veto,
                "gather_ms": gather_ms,
            },
        )

        for stage, n in cycle_attrition.items():
            self._attrition_counters[stage] += n
        for reason, n in cycle_rejections.items():
            self._rejection_breakdown[reason] += n
        self._last_cycle_attrition = dict(cycle_attrition)

        if cycle_attrition["raw"] > 0:
            logger.info(
                "Attrition сигналов цикла",
                extra={
                    "cycle_id": cycle_id,
                    "trace_id": trace_id,
                    "attrition": dict(cycle_attrition),
                    "rejections": dict(cycle_rejections),
                    "latencies_ms": {
                        "gather": gather_ms,
                        "aggregate": aggregate_ms,
                        "risk_eval": risk_total_ms,
                        "submit": submit_total_ms,
                    },
                },
            )

    @property
    def stats(self) -> dict[str, Any]:
        """Return cumulative dispatcher metrics.

        Returns:
            dict[str, Any]: stats snapshot
        """
        return {
            "cycles": self._cycle_count,
            "signals_gathered": self._signals_gathered_total,
            "decisions_executed": self._decisions_executed_total,
            "adapters": {a.name: a.stats for a in self.adapters},
            "orders": self.orders.stats,
        }

    def get_attrition_stats(self) -> dict[str, Any]:
        """Return cumulative signal-funnel attrition counters.

        Consumed by `app/dashboard/metrics_writer.py` so Grafana sees a
        running tally of how many signals survive each stage across the
        whole bot lifetime (raw → allowed → tier → meta → risk → submitted).

        Returns:
            dict[str, Any]: keys ``stages``, ``rejection_breakdown``,
                ``last_cycle``.
        """
        return {
            "stages": dict(self._attrition_counters),
            "rejection_breakdown": dict(self._rejection_breakdown),
            "last_cycle": dict(self._last_cycle_attrition),
        }

    def _build_base_meta_context(self) -> MetaContext:
        """Build cycle-level (portfolio + timing) meta-context.

        Returns:
            MetaContext: base context with portfolio/timing fields
        """
        cb = get_circuit_breaker()
        book = get_position_book()

        current_dd_pct = float(getattr(cb.state, "current_drawdown_pct", 0.0))
        daily_pnl_pct = float(getattr(cb.state, "daily_pnl_pct", 0.0))
        winning_streak = int(getattr(cb.state, "winning_streak", 0))
        losing_streak = int(getattr(cb.state, "losing_streak", 0))
        n_trades_today = int(getattr(cb.state, "n_trades_today", self._decisions_executed_total))
        n_open_positions = int(getattr(book, "n_open_positions", 0))

        now_msk = datetime.now(tz=_MSK)
        hour_of_day = now_msk.hour

        close_msk = now_msk.replace(hour=23, minute=50, second=0, microsecond=0)
        if close_msk < now_msk:
            close_msk += timedelta(days=1)
        minutes_to_close = int((close_msk - now_msk).total_seconds() / 60)

        return MetaContext(
            current_dd_pct=current_dd_pct,
            daily_pnl_pct=daily_pnl_pct,
            n_open_positions=n_open_positions,
            n_trades_today=n_trades_today,
            winning_streak=winning_streak,
            losing_streak=losing_streak,
            hour_of_day=hour_of_day,
            minutes_to_close=minutes_to_close,
        )

    async def _fetch_supercandles_batch(self, tickers: list[str]) -> dict[str, Any]:
        """Best-effort SuperCandles fetch for the cycle's tickers.

        Args:
            tickers: tickers to fetch
        Returns:
            dict[str, Any]: ticker → DataFrame
        """
        if not tickers:
            return {}
        try:
            from app.data.supercandles import get_supercandles
        except ImportError:
            return {}

        sem = asyncio.Semaphore(4)

        async def _one(t: str) -> tuple[str, Any]:
            """One."""
            async with sem:
                try:
                    df = await asyncio.wait_for(get_supercandles(t), timeout=1.5)
                    return (t, df)
                except (TimeoutError, Exception):
                    return (t, None)

        results = await asyncio.gather(*[_one(t) for t in tickers], return_exceptions=True)
        out: dict[str, Any] = {}
        for r in results:
            if isinstance(r, tuple) and r[1] is not None:
                out[r[0]] = r[1]
        return out

    @staticmethod
    def _ticker_meta_context(
        base: MetaContext,
        ticker: str,
        sigs: list[UnifiedSignal],
    ) -> MetaContext:
        """Build per-ticker MetaContext from common base.

        Args:
            base: cycle-level base context
            ticker: instrument code
            sigs: signals for this ticker
        Returns:
            MetaContext: per-ticker context
        """
        atr_pct = 0.0
        ofi = base.ofi
        kyles = base.kyles_lambda
        vpin = base.vpin
        vol_z = base.vol_z
        spread = base.spread_bbo_bps
        regime = base.regime

        for s in sigs:
            md = s.metadata or {}
            if s.atr > 0 and s.price > 0:
                atr_pct = max(atr_pct, s.atr / s.price * 100.0)

            ofi = md.get("ofi", ofi) if isinstance(md.get("ofi"), (int, float)) else ofi
            kyles = (
                md.get("kyles_lambda", kyles)
                if isinstance(md.get("kyles_lambda"), (int, float))
                else kyles
            )
            vpin = md.get("vpin", vpin) if isinstance(md.get("vpin"), (int, float)) else vpin
            vol_z = md.get("vol_z", vol_z) if isinstance(md.get("vol_z"), (int, float)) else vol_z
            spread = (
                md.get("spread_bbo_bps", spread)
                if isinstance(md.get("spread_bbo_bps"), (int, float))
                else spread
            )
            if isinstance(md.get("regime"), str):
                regime = md["regime"]

        return MetaContext(
            ofi=float(ofi),
            kyles_lambda=float(kyles),
            vpin=float(vpin),
            vol_z=float(vol_z),
            spread_bbo_bps=float(spread),
            atr_pct=float(atr_pct),
            regime=regime,
            current_dd_pct=base.current_dd_pct,
            daily_pnl_pct=base.daily_pnl_pct,
            n_open_positions=base.n_open_positions,
            n_trades_today=base.n_trades_today,
            winning_streak=base.winning_streak,
            losing_streak=base.losing_streak,
            hour_of_day=base.hour_of_day,
            minutes_to_close=base.minutes_to_close,
        )

_active_dispatcher: Dispatcher | None = None

def set_active_dispatcher(d: Dispatcher | None) -> None:
    """Register the running Dispatcher so `get_active_dispatcher()` can find it.

    Called once from `main.py` after the Dispatcher is constructed. The
    metrics_writer uses the singleton to pull attrition stats; tests that
    spin up their own Dispatcher don't need to register.

    Args:
        d: dispatcher instance or None to clear.
    """
    global _active_dispatcher
    _active_dispatcher = d

def get_active_dispatcher() -> Dispatcher | None:
    """Return the registered live Dispatcher (or None if not yet started).

    Returns:
        Dispatcher | None: live dispatcher or None.
    """
    return _active_dispatcher
