"""Broker reconciliation — make local state agree with ArenaGo."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover
    import aiosqlite  # noqa: F401  # type: ignore

    _HAS_AIOSQLITE = True
except ImportError:  # pragma: no cover
    _HAS_AIOSQLITE = False

TRADES_DB = cfg.DATA_DIR / "trades.db"
DECISIONS_DB = cfg.DATA_DIR / "decisions.db"

DEFAULT_INTERVAL_SEC: float = 300.0

CASH_DIVERGENCE_TOL_RUB: float = 0.01

_SYNTH_SL_ATR: float = 1.75
_SYNTH_RR: float = 2.0
_SYNTH_DEFAULT_ATR_PCT: float = 0.015

@dataclass
class BrokerPosition:
    """Normalised view of one broker-side position."""

    ticker: str
    quantity: int
    avg_price: float
    bot: str = ""

    @classmethod
    def from_arenago(cls, raw: dict[str, Any], default_bot: str = "") -> BrokerPosition:
        """Build BrokerPosition from raw ArenaGo dict.

        Args:
            raw: raw broker dict
            default_bot: fallback bot name
        Returns:
            BrokerPosition: parsed instance
        """
        return cls(
            ticker=str(raw.get("secid", "")).upper(),
            quantity=int(raw.get("position", 0)),
            avg_price=float(raw.get("average_price", 0) or 0),
            bot=str(raw.get("bot", default_bot)),
        )

@dataclass
class ReconcileReport:
    """Summary of what the reconciliation cycle did."""

    fetched_positions: int = 0
    fetched_trades: int = 0
    synthetic_added: list[str] = field(default_factory=list)
    marked_closed: list[str] = field(default_factory=list)
    cash_divergence_rub: float = 0.0
    has_mismatch: bool = False
    broker_cash: float = 0.0
    local_cash: float = 0.0
    diff_summary: str = ""

    def log_warning_if_needed(self) -> None:
        """Emit a single WARNING line when there is divergence."""
        if not self.has_mismatch:
            return
        logger.warning(
            "Broker reconciliation found divergence",
            extra={
                "synthetic_added": self.synthetic_added,
                "marked_closed": self.marked_closed,
                "cash_divergence_rub": round(self.cash_divergence_rub, 4),
                "broker_cash": self.broker_cash,
                "local_cash": self.local_cash,
                "diff_summary": self.diff_summary,
            },
        )

class BrokerReconciler:
    """Pull truth from ArenaGo and merge it into local state."""

    def __init__(
        self,
        arenago: Any = None,
        position_book: Any = None,
        interval_sec: float = DEFAULT_INTERVAL_SEC,
    ) -> None:
        """Init."""
        self._arenago = arenago
        self._book = position_book
        self.interval_sec = interval_sec
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed_marker_set: set[str] = set()

    async def reconcile_once(self) -> ReconcileReport:
        """Run a single reconciliation pass.

        Returns:
            ReconcileReport: summary of what was changed
        """
        report = ReconcileReport()

        broker_positions, positions_confirmed = await self._fetch_broker_positions()
        broker_trades = await self._fetch_broker_trades()
        report.fetched_positions = len(broker_positions)
        report.fetched_trades = len(broker_trades)

        local_positions = self._current_local_positions()

        for ticker, bp in broker_positions.items():
            if ticker in local_positions:
                continue
            if bp.quantity == 0:
                continue
            await self._create_synthetic_local(bp)
            report.synthetic_added.append(ticker)

        if positions_confirmed:
            for ticker, lp in local_positions.items():
                if ticker in broker_positions:
                    continue
                await self._mark_position_closed(ticker, local_pos=lp)
                report.marked_closed.append(ticker)
        else:
            if local_positions:
                logger.warning(
                    "Reconciler: broker positions UNCONFIRMED (transient) — "
                    "preserving local state, will retry next cycle",
                    extra={
                        "local_position_tickers": sorted(local_positions.keys()),
                    },
                )

        report.local_cash = self._current_local_cash()
        try:
            report.broker_cash = await self._fetch_broker_cash()
        except Exception as exc:  # pragma: no cover
            logger.debug("Reconciler: broker cash fetch failed", extra={"error": str(exc)})
            report.broker_cash = report.local_cash

        report.cash_divergence_rub = abs(report.broker_cash - report.local_cash)
        if (
            report.cash_divergence_rub > CASH_DIVERGENCE_TOL_RUB
            or report.synthetic_added
            or report.marked_closed
        ):
            report.has_mismatch = True
            report.diff_summary = (
                f"+{len(report.synthetic_added)} synth / "
                f"-{len(report.marked_closed)} closed / "
                f"Δcash={report.cash_divergence_rub:.2f} RUB"
            )

        return report

    async def start_periodic(self) -> None:
        """Spawn the periodic background reconciliation task."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop(), name="broker_reconciler")
        logger.info(
            "BrokerReconciler periodic loop started",
            extra={"interval_sec": self.interval_sec},
        )

    async def stop(self) -> None:
        """Stop the periodic loop."""
        self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        logger.info("BrokerReconciler stopped")

    async def _loop(self) -> None:
        """Internal periodic loop body."""
        while not self._stop_event.is_set():
            try:
                report = await self.reconcile_once()
                report.log_warning_if_needed()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "BrokerReconciler cycle failed",
                    extra={"error": str(exc)},
                )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_sec)

    async def _fetch_broker_positions(self) -> tuple[dict[str, BrokerPosition], bool]:
        """Return ({ticker: BrokerPosition}, confirmed) from ArenaGo.

        Returns:
            tuple[dict[str, BrokerPosition], bool]: (positions, confirmed)
        """
        arenago = self._resolve_arenago()
        if arenago is None:
            return {}, False

        confirmed = True
        raw: list[Any]
        try:
            if hasattr(arenago, "get_positions_safe"):
                raw, confirmed = await arenago.get_positions_safe()
            else:
                raw = await arenago.get_positions()
        except Exception as exc:
            logger.warning(
                "Reconciler: get_positions failed (treated as transient)", extra={"error": str(exc)}
            )
            return {}, False

        out: dict[str, BrokerPosition] = {}
        bot_name = getattr(arenago, "_bot_name", "") or cfg.ARENAGO_BOT_NAME
        for r in raw or []:
            try:
                bp = BrokerPosition.from_arenago(r, default_bot=bot_name)
            except Exception:
                continue
            if not bp.ticker:
                continue
            if bp.quantity == 0:
                continue
            out[bp.ticker] = bp
        return out, confirmed

    async def _fetch_broker_trades(self) -> list[dict[str, Any]]:
        """Return today's broker trades.

        Returns:
            list[dict[str, Any]]: trades or empty list
        """
        arenago = self._resolve_arenago()
        if arenago is None:
            return []
        try:
            raw = await arenago.get_trades()
        except Exception as exc:
            logger.debug("Reconciler: get_trades failed", extra={"error": str(exc)})
            return []
        return list(raw or [])

    async def _fetch_broker_cash(self) -> float:
        """Return broker cash balance.

        Returns:
            float: cash in RUB
        """
        arenago = self._resolve_arenago()
        if arenago is None:
            return 0.0
        try:
            return float(await arenago.get_cash_balance())
        except Exception:
            return 0.0

    def _current_local_positions(self) -> dict[str, Any]:
        """Return PositionBook cache snapshot.

        Returns:
            dict[str, Any]: ticker → Position
        """
        book = self._resolve_book()
        if book is None:
            return {}
        try:
            return dict(book.positions)
        except Exception:
            return {}

    def _current_local_cash(self) -> float:
        """Return local cash balance.

        Returns:
            float: cash in RUB
        """
        book = self._resolve_book()
        if book is None:
            return 0.0
        try:
            return float(book.cash_balance)
        except Exception:
            return 0.0

    async def _create_synthetic_local(self, bp: BrokerPosition) -> None:
        """Write synthetic decisions+trades rows for broker-only position.

        Args:
            bp: broker position to synthesise
        """
        if not _HAS_AIOSQLITE:
            return

        synthetic_id = f"synth_{bp.ticker}_{int(time.time())}"
        direction = "BUY" if bp.quantity > 0 else "SELL"
        abs_qty = abs(bp.quantity)
        now = datetime.now(tz=UTC)
        now_iso = now.isoformat()

        atr_proxy = max(bp.avg_price * _SYNTH_DEFAULT_ATR_PCT, 0.01)
        if direction == "BUY":
            stop_loss = bp.avg_price - _SYNTH_SL_ATR * atr_proxy
            take_profit = bp.avg_price + _SYNTH_RR * _SYNTH_SL_ATR * atr_proxy
        else:
            stop_loss = bp.avg_price + _SYNTH_SL_ATR * atr_proxy
            take_profit = bp.avg_price - _SYNTH_RR * _SYNTH_SL_ATR * atr_proxy

        try:
            from app.utils.db_pool import get_conn

            db_dec = await get_conn(DECISIONS_DB)
            await db_dec.execute(
                """
                INSERT OR IGNORE INTO decisions
                (decision_id, cycle_id, ticker, action, tier, direction,
                 combined_magnitude, risk_check, stop_loss, take_profit,
                 expected_holding_min, rationale, signals_json,
                 trade_request_json, executed_bool, arena_response_json,
                 reflection_status, created_at, executed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                """,
                (
                    synthetic_id,
                    "reconcile",
                    bp.ticker,
                    "EXECUTE",
                    "NONE",
                    direction,
                    0.0,
                    "PASSED",
                    float(stop_loss),
                    float(take_profit),
                    0,
                    "synthetic — reconciled from broker get_positions",
                    json.dumps(
                        [
                            {
                                "source": "RECONCILER",
                                "pattern": "synthetic",
                                "atr": atr_proxy,
                                "entry_level": bp.avg_price,
                            }
                        ]
                    ),
                    None,
                    json.dumps(
                        {
                            "success": True,
                            "message": "synthetic-reconcile",
                            "quantity": abs_qty,
                            "price": bp.avg_price,
                            "order_value": abs_qty * bp.avg_price,
                        }
                    ),
                    "SYNTHETIC",
                    now_iso,
                    now_iso,
                ),
            )
            await db_dec.commit()

            db_tr = await get_conn(TRADES_DB)
            await db_tr.execute(
                """
                INSERT INTO trades
                (decision_id, ticker, direction, quantity, price,
                 order_value, remaining_cash, trade_date, trade_time,
                 bot, source_model, arena_raw_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    synthetic_id,
                    bp.ticker,
                    direction,
                    abs_qty,
                    bp.avg_price,
                    abs_qty * bp.avg_price,
                    self._current_local_cash(),
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    bp.bot or cfg.ARENAGO_BOT_NAME,
                    "reconciler",
                    json.dumps({"synthetic": True, "source": "broker_reconcile"}),
                    now_iso,
                ),
            )
            await db_tr.commit()

            logger.info(
                "Reconciler: created synthetic local entry for broker-only position",
                extra={
                    "ticker": bp.ticker,
                    "quantity": abs_qty,
                    "avg_price": bp.avg_price,
                    "direction": direction,
                    "synthetic_id": synthetic_id,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                },
            )
        except Exception as exc:
            logger.error(
                "Reconciler: failed to create synthetic entry",
                extra={"ticker": bp.ticker, "error": str(exc)},
            )

    async def _mark_position_closed(self, ticker: str, local_pos: Any) -> None:
        """Insert a flattening trade row to net FIFO to zero.

        Args:
            ticker: ticker to mark closed
            local_pos: local Position snapshot
        """
        if not _HAS_AIOSQLITE:
            return
        if ticker in self._closed_marker_set:
            return
        self._closed_marker_set.add(ticker)

        qty = int(getattr(local_pos, "quantity", 0))
        if qty == 0:
            return
        flatten_direction = "SELL" if qty > 0 else "BUY"
        abs_qty = abs(qty)
        avg_price = float(getattr(local_pos, "avg_price", 0.0))
        now = datetime.now(tz=UTC)
        now_iso = now.isoformat()
        marker_id = f"reconcile_close_{ticker}_{int(time.time())}"

        try:
            from app.utils.db_pool import get_conn

            db = await get_conn(TRADES_DB)
            await db.execute(
                """
                INSERT INTO trades
                (decision_id, ticker, direction, quantity, price,
                 order_value, remaining_cash, trade_date, trade_time,
                 bot, source_model, arena_raw_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    marker_id,
                    ticker,
                    flatten_direction,
                    abs_qty,
                    avg_price,
                    abs_qty * avg_price,
                    self._current_local_cash(),
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    getattr(local_pos, "bot", "") or cfg.ARENAGO_BOT_NAME,
                    "reconciler",
                    json.dumps({"synthetic_close": True, "reason": "broker_no_longer_holds"}),
                    now_iso,
                ),
            )
            await db.commit()

            logger.warning(
                "Reconciler: marked local-only position as externally closed",
                extra={
                    "ticker": ticker,
                    "quantity": abs_qty,
                    "direction_flatten": flatten_direction,
                    "marker_id": marker_id,
                },
            )
        except Exception as exc:
            logger.error(
                "Reconciler: failed to mark position closed",
                extra={"ticker": ticker, "error": str(exc)},
            )

    def _resolve_arenago(self) -> Any:
        """Resolve ArenaGo singleton lazily.

        Returns:
            Any: ArenaGoClient or None
        """
        if self._arenago is not None:
            return self._arenago
        try:
            from app.execution.arenago_client import get_arenago_client

            self._arenago = get_arenago_client()
        except Exception:  # pragma: no cover
            return None
        return self._arenago

    def _resolve_book(self) -> Any:
        """Resolve PositionBook singleton lazily.

        Returns:
            Any: PositionBook or None
        """
        if self._book is not None:
            return self._book
        try:
            from app.risk.position_book import get_position_book

            self._book = get_position_book()
        except Exception:  # pragma: no cover
            return None
        return self._book

async def is_duplicate_today(
    ticker: str,
    direction: str,
    window_sec: int = 300,
    *,
    db_path: Any = None,
    exclude_decision_id: str | None = None,
) -> bool:
    """Return True if a recent same-direction trade already exists.

    Args:
        ticker: instrument code
        direction: BUY or SELL
        window_sec: lookback window
        db_path: trades.db path override
        exclude_decision_id: decision id to skip
    Returns:
        bool: True if duplicate found
    """
    if not _HAS_AIOSQLITE:
        return False
    path = db_path if db_path is not None else TRADES_DB
    cutoff = datetime.now(tz=UTC) - timedelta(seconds=window_sec)
    cutoff_date = cutoff.strftime("%Y-%m-%d")
    cutoff_dt_iso = cutoff.isoformat()
    try:
        from app.utils.db_pool import get_conn

        db = await get_conn(path)
        query = """
            SELECT COUNT(*) FROM trades
            WHERE ticker = ?
              AND direction = ?
              AND trade_date >= ?
              AND created_at >= ?
        """
        params: list[Any] = [
            ticker.upper(),
            direction.upper(),
            cutoff_date,
            cutoff_dt_iso,
        ]
        if exclude_decision_id is not None:
            query += " AND decision_id <> ?"
            params.append(exclude_decision_id)
        async with db.execute(query, params) as cur:
            row = await cur.fetchone()
        return bool(row and row[0] and int(row[0]) > 0)
    except Exception as exc:
        logger.debug("is_duplicate_today probe failed", extra={"error": str(exc)})
        return False

_reconciler: BrokerReconciler | None = None

def get_broker_reconciler() -> BrokerReconciler:
    """Return process-wide BrokerReconciler singleton.

    Returns:
        BrokerReconciler: shared instance
    """
    global _reconciler
    if _reconciler is None:
        _reconciler = BrokerReconciler()
    return _reconciler

def _reset_for_tests() -> None:
    """Drop the singleton between unit tests."""
    global _reconciler
    _reconciler = None

__all__ = [
    "BrokerPosition",
    "BrokerReconciler",
    "ReconcileReport",
    "get_broker_reconciler",
    "is_duplicate_today",
    "_reset_for_tests",
]
