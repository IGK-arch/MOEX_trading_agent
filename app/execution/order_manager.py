"""Идемпотентная отправка ордеров."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import UTC, date, datetime

import app.config as cfg
from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    Direction,
)
from app.execution.arenago_client import SubmitResult, get_arenago_client
from app.execution.broker_reconciler import is_duplicate_today
from app.risk.position_book import get_position_book
from app.utils.logging import get_logger, get_trace_id

logger = get_logger(__name__)

class DailyTradeLimitGuard:
    """Process-local soft guard on ArenaGo's 1000-trade/day cap.

    v1.0.7 — short-circuit before risk eval/broker hop when we're already at
    90 % of the daily quota. The broker-side hard limit (and slowdown / entry
    halt) still live in :mod:`app.execution.arenago_client`; this guard is a
    *cheap* pre-check that saves a full evaluation when we know the entry
    can't go through.

    Behaviour:
      * Auto-resets at МСК midnight (ArenaGo's daily quota window).
      * ``can_submit_entry()`` returns False once we hit ``soft_pct`` of the
        cap → only exits/closes are allowed past this gate.
      * ``can_submit_exit()`` returns False only at the *hard* cap so a force-
        close still goes through up to the last allowed trade.
      * Increment is driven by the OrderManager on broker success.

    The dispatcher calls ``can_submit_entry()`` before risk eval; exits skip
    the soft gate. Both gates remain redundant with the broker-side counter
    in ArenaGoClient.
    """

    __slots__ = ("limit", "soft_pct", "_count", "_reset_date_msk")

    def __init__(
        self,
        limit: int | None = None,
        soft_pct: float = 0.9,
    ) -> None:
        """Init.

        Args:
            limit: hard cap (defaults to ``cfg.ARENAGO_DAILY_TRADE_LIMIT``).
            soft_pct: fraction of ``limit`` at which entries are halted
                (0.9 → 900/1000). Exits are still permitted up to the hard cap.
        """
        self.limit = int(limit if limit is not None else cfg.ARENAGO_DAILY_TRADE_LIMIT)
        self.soft_pct = float(soft_pct)
        self._count = 0
        self._reset_date_msk = self._today_msk()

    @staticmethod
    def _today_msk() -> date:
        """MSK calendar date — ArenaGo's daily counter rolls at МСК midnight."""
        from app.utils.session_profile import MSK_OFFSET

        return datetime.now(tz=MSK_OFFSET).date()

    def _maybe_reset(self) -> None:
        """Reset the local counter at МСК midnight."""
        today = self._today_msk()
        if today != self._reset_date_msk:
            logger.info(
                "DailyTradeLimitGuard reset at MSK midnight",
                extra={
                    "prev_count": self._count,
                    "prev_date_msk": self._reset_date_msk.isoformat(),
                    "new_date_msk": today.isoformat(),
                    "limit": self.limit,
                },
            )
            self._count = 0
            self._reset_date_msk = today

    @property
    def soft_cap(self) -> int:
        """Entry-halt threshold derived from ``limit * soft_pct``."""
        return max(0, int(self.limit * self.soft_pct))

    def can_submit(self, *, is_exit: bool = False) -> bool:
        """Return True if a new submit may proceed.

        Args:
            is_exit: True if the candidate decision is a closing leg —
                exits are only blocked at the hard cap.
        Returns:
            bool: ``True`` if the submit may proceed, ``False`` otherwise.
        """
        self._maybe_reset()
        if is_exit:
            return self._count < self.limit
        return self._count < self.soft_cap

    def can_submit_entry(self) -> bool:
        """Convenience alias used by the Dispatcher pre-risk gate."""
        return self.can_submit(is_exit=False)

    def remaining(self) -> int:
        """Trades remaining before the hard cap is hit."""
        self._maybe_reset()
        return max(0, self.limit - self._count)

    def on_submit_success(self, *, is_exit: bool = False) -> None:
        """Record a successful broker submit.

        Args:
            is_exit: True if the trade was an exit/close (kept for telemetry).
        """
        self._maybe_reset()
        self._count += 1
        if self._count == self.soft_cap:
            logger.critical(
                "DailyTradeLimitGuard soft cap reached — entries halted",
                extra={
                    "count": self._count,
                    "soft_cap": self.soft_cap,
                    "limit": self.limit,
                    "is_exit_trigger": is_exit,
                },
            )

    def snapshot(self) -> dict[str, int | str]:
        """Return a small dict for dashboards / logs.

        Returns:
            dict[str, int | str]: snapshot.
        """
        self._maybe_reset()
        return {
            "count": self._count,
            "limit": self.limit,
            "soft_cap": self.soft_cap,
            "remaining": self.remaining(),
            "reset_date_msk": self._reset_date_msk.isoformat(),
        }

_daily_guard: DailyTradeLimitGuard | None = None

def get_daily_trade_guard() -> DailyTradeLimitGuard:
    """Return process-wide DailyTradeLimitGuard singleton.

    Returns:
        DailyTradeLimitGuard: shared instance.
    """
    global _daily_guard
    if _daily_guard is None:
        _daily_guard = DailyTradeLimitGuard()
    return _daily_guard

try:
    import aiosqlite  # type: ignore  # noqa: F401

    _HAS_AIOSQLITE = True
except ImportError:
    _HAS_AIOSQLITE = False

from app.utils.db_pool import get_conn

DECISIONS_DB = cfg.DATA_DIR / "decisions.db"
TRADES_DB = cfg.DATA_DIR / "trades.db"

class OrderManager:
    """Idempotent order submission with persistence."""

    def __init__(self) -> None:
        """Init."""
        self.arenago = get_arenago_client()
        self.book = get_position_book()
        self.daily_guard = get_daily_trade_guard()
        self._submit_count = 0
        self._success_count = 0
        self._reject_count = 0
        self.recent_decision_ids: deque[str] = deque(maxlen=100)

    async def submit(self, decision: Decision) -> SubmitResult | None:
        """Submit decision's TradeRequest to ArenaGo (idempotent).

        Args:
            decision: decision to submit
        Returns:
            SubmitResult | None: broker result or None
        """
        if decision.action != DecisionAction.EXECUTE:
            return None
        if decision.trade_request is None:
            logger.error(
                "OrderManager: no trade_request on EXECUTE decision",
                extra={"decision_id": decision.decision_id},
            )
            return None

        await self._upsert_decision(decision)
        self.recent_decision_ids.append(decision.decision_id)

        tr = decision.trade_request

        if await is_duplicate_today(
            ticker=tr.ticker,
            direction=tr.direction.value,
            window_sec=300,
            db_path=TRADES_DB,
            exclude_decision_id=decision.decision_id,
        ):
            self._reject_count += 1
            logger.warning(
                "OrderManager: dropping duplicate same-direction order",
                extra={
                    "decision_id": decision.decision_id,
                    "ticker": tr.ticker,
                    "direction": tr.direction.value,
                    "trace_id": get_trace_id(),
                    "reason": "duplicate_window",
                },
            )
            return None

        self._submit_count += 1
        existing = None
        try:
            existing = self.book.get_position(tr.ticker)
        except Exception:
            existing = None
        is_exit = bool(
            existing is not None
            and (
                (existing.quantity > 0 and tr.direction == Direction.SELL)
                or (existing.quantity < 0 and tr.direction == Direction.BUY)
            )
        )
        submit_kwargs: dict = {
            "direction": tr.direction.value,
            "ticker": tr.ticker,
            "quantity": tr.quantity,
            "decision_id": decision.decision_id,
            "bot": tr.bot,
        }
        try:
            import inspect

            if "is_exit" in inspect.signature(self.arenago.submit_order).parameters:
                submit_kwargs["is_exit"] = is_exit
        except (TypeError, ValueError):
            pass
        try:
            result = await self.arenago.submit_order(**submit_kwargs)
        except Exception as exc:
            logger.error(
                "OrderManager: submit raised",
                extra={
                    "decision_id": decision.decision_id,
                    "ticker": tr.ticker,
                    "error": str(exc),
                    "trace_id": get_trace_id(),
                },
            )
            self._reject_count += 1
            return None

        if result.success:
            self._success_count += 1
            self.daily_guard.on_submit_success(is_exit=is_exit)
            dominant_source = getattr(decision, "dominant_source", None)
            mark_with_source = getattr(self.book, "mark_entry_with_source", None)
            if dominant_source and callable(mark_with_source):
                mark_with_source(tr.ticker, dominant_source)
            else:
                self.book.mark_entry(tr.ticker)
            await self._write_trade(decision, result)
            logger.info(
                "Order executed",
                extra={
                    "decision_id": decision.decision_id,
                    "ticker": tr.ticker,
                    "direction": tr.direction.value,
                    "quantity": result.quantity,
                    "price": result.price,
                    "order_value": result.order_value,
                    "remaining_cash": result.remaining_cash,
                    "trace_id": get_trace_id(),
                },
            )
        else:
            self._reject_count += 1
            logger.error(
                "Order rejected",
                extra={
                    "decision_id": decision.decision_id,
                    "ticker": tr.ticker,
                    "message": result.message,
                    "arena_error": result.arena_error.value if result.arena_error else "?",
                    "trace_id": get_trace_id(),
                },
            )

        return result

    async def submit_pair_atomically(
        self,
        leg_a: Decision,
        leg_b: Decision,
    ) -> tuple[SubmitResult | None, SubmitResult | None]:
        """Submit two pair legs as atomically as possible.

        Args:
            leg_a: first leg
            leg_b: second leg
        Returns:
            tuple[SubmitResult | None, SubmitResult | None]: results
        """

        result_a, result_b = await asyncio.gather(
            self.submit(leg_a),
            self.submit(leg_b),
            return_exceptions=False,
        )

        a_ok = result_a is not None and result_a.success
        b_ok = result_b is not None and result_b.success
        if a_ok and not b_ok:
            logger.warning(
                "Pair leg B failed, closing leg A",
                extra={"pair": leg_a.ticker + "/" + leg_b.ticker},
            )
            await self._emergency_close(leg_a)
        elif b_ok and not a_ok:
            logger.warning(
                "Pair leg A failed, closing leg B",
                extra={"pair": leg_a.ticker + "/" + leg_b.ticker},
            )
            await self._emergency_close(leg_b)

        return result_a, result_b

    async def _emergency_close(self, decision: Decision) -> None:
        """Close a position by submitting opposite direction.

        Args:
            decision: the leg to flatten
        """
        if decision.trade_request is None:
            return
        tr = decision.trade_request
        close_direction = "SELL" if tr.direction == Direction.BUY else "BUY"
        try:
            close_kwargs: dict = {
                "direction": close_direction,
                "ticker": tr.ticker,
                "quantity": tr.quantity,
                "decision_id": f"close_{decision.decision_id}",
                "bot": tr.bot,
            }
            try:
                import inspect

                if "is_exit" in inspect.signature(self.arenago.submit_order).parameters:
                    close_kwargs["is_exit"] = True
            except (TypeError, ValueError):
                pass
            await self.arenago.submit_order(**close_kwargs)
            logger.info(
                "Emergency close submitted",
                extra={"ticker": tr.ticker, "decision_id": decision.decision_id},
            )
        except Exception as exc:
            logger.critical(
                "Emergency close failed", extra={"ticker": tr.ticker, "error": str(exc)}
            )

    async def _upsert_decision(self, decision: Decision) -> None:
        """Insert decision into decisions.db (idempotent on decision_id).

        Args:
            decision: decision to persist
        """
        if not _HAS_AIOSQLITE:
            return
        now_iso = datetime.now(tz=UTC).isoformat()
        try:
            tr_json = (
                json.dumps(decision.trade_request.model_dump(mode="json"))
                if decision.trade_request
                else None
            )
            signals_json = json.dumps([s.model_dump(mode="json") for s in decision.signals])
            db = await get_conn(DECISIONS_DB)
            await db.execute(
                """
                INSERT OR IGNORE INTO decisions
                (decision_id, cycle_id, ticker, action, tier, direction,
                 combined_magnitude, risk_check, stop_loss, take_profit,
                 expected_holding_min, rationale, signals_json, trade_request_json,
                 created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    decision.cycle_id,
                    decision.ticker,
                    decision.action.value,
                    decision.tier.value,
                    decision.direction.value,
                    decision.combined_magnitude,
                    decision.risk_check.value,
                    decision.stop_loss,
                    decision.take_profit,
                    decision.expected_holding_min,
                    decision.rationale,
                    signals_json,
                    tr_json,
                    now_iso,
                ),
            )
            await db.commit()
        except Exception as exc:
            logger.error(
                "Decision DB write failed",
                extra={"decision_id": decision.decision_id, "error": str(exc)},
            )

    async def _write_trade(self, decision: Decision, result: SubmitResult) -> None:
        """Insert executed trade into trades.db.

        Args:
            decision: source decision
            result: broker result
        """
        if not _HAS_AIOSQLITE:
            return
        now = datetime.now(tz=UTC)
        tr = decision.trade_request
        if tr is None:
            return
        try:
            db = await get_conn(TRADES_DB)
            await db.execute(
                """
                INSERT INTO trades
                (decision_id, ticker, direction, quantity, price, order_value,
                 remaining_cash, trade_date, trade_time, bot, source_model,
                 arena_raw_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    tr.ticker,
                    tr.direction.value,
                    result.quantity,
                    result.price,
                    result.order_value,
                    result.remaining_cash,
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    tr.bot,
                    ",".join({s.source.value for s in decision.signals}),
                    json.dumps(result.to_dict()),
                    now.isoformat(),
                ),
            )
            await db.commit()
        except Exception as exc:
            logger.error(
                "Trade DB write failed",
                extra={"decision_id": decision.decision_id, "error": str(exc)},
            )

    async def reconcile_pending_decisions(self, lookback_hours: int = 4) -> dict[str, int]:
        """Cross-check pending decisions vs broker positions.

        Args:
            lookback_hours: how far back to scan decisions.db
        Returns:
            dict[str, int]: {scanned, repaired, orphans_found}
        """
        result = {"scanned": 0, "repaired": 0, "orphans_found": 0}
        if not _HAS_AIOSQLITE:
            return result
        try:
            broker_positions = {
                str(p.get("secid", "")).upper(): int(p.get("position", 0))
                for p in await self.arenago.get_positions()
                if int(p.get("position", 0)) != 0
            }
        except Exception as exc:
            logger.warning("reconcile: broker get_positions failed", extra={"error": str(exc)})
            broker_positions = {}

        if not broker_positions:
            try:
                local = await self.book._derive_positions_from_trades()
                broker_positions = {t: int(q) for t, (q, _vwap) in local.items() if q > 0}
            except Exception:
                pass

        from datetime import timedelta

        since_iso = (datetime.now(tz=UTC) - timedelta(hours=lookback_hours)).isoformat()
        try:
            db = await get_conn(DECISIONS_DB)
            async with db.execute(
                """
                SELECT decision_id, ticker, direction, created_at
                FROM decisions
                WHERE action = 'EXECUTE'
                  AND executed_bool = 0
                  AND created_at >= ?
                ORDER BY created_at DESC
                """,
                (since_iso,),
            ) as cur:
                rows = await cur.fetchall()
            result["scanned"] = len(rows)
            for did, ticker, direction, created_at in rows:
                t = str(ticker or "").upper()
                broker_qty = broker_positions.get(t, 0)
                if broker_qty == 0:
                    continue
                result["orphans_found"] += 1
                logger.critical(
                    "Orphan order reconciled — broker has position, decisions.db was pending",
                    extra={
                        "decision_id": did,
                        "ticker": t,
                        "direction": direction,
                        "broker_qty": broker_qty,
                        "created_at": created_at,
                    },
                )
                now_iso = datetime.now(tz=UTC).isoformat()
                try:
                    await db.execute(
                        "UPDATE decisions SET executed_bool=1, "
                        "executed_at=COALESCE(executed_at, ?), "
                        "arena_response_json=COALESCE(arena_response_json, ?) "
                        "WHERE decision_id=?",
                        (
                            now_iso,
                            json.dumps(
                                {
                                    "success": True,
                                    "message": "RECONCILED_ON_STARTUP",
                                    "quantity": broker_qty,
                                    "decision_id": did,
                                }
                            ),
                            did,
                        ),
                    )
                    result["repaired"] += 1
                except Exception as exc:
                    logger.error(
                        "reconcile UPDATE failed", extra={"decision_id": did, "error": str(exc)}
                    )
            await db.commit()
        except Exception as exc:
            logger.error("reconcile_pending_decisions failed", extra={"error": str(exc)})
        return result

    @property
    def stats(self) -> dict[str, int]:
        """Return submit/success/reject counters.

        Returns:
            dict[str, int]: cumulative counters
        """
        return {
            "submit_count": self._submit_count,
            "success_count": self._success_count,
            "reject_count": self._reject_count,
        }

_order_manager: OrderManager | None = None

def get_order_manager() -> OrderManager:
    """Return process-wide OrderManager singleton.

    Returns:
        OrderManager: shared instance
    """
    global _order_manager
    if _order_manager is None:
        _order_manager = OrderManager()
    return _order_manager
