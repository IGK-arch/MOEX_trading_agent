"""Монитор стоп-лоссов и тейк-профитов."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import app.config as cfg
from app.risk.partial_exit import (
    plan_partial_exit,
    tp1_hit,
    tp2_hit,
)
from app.risk.sl_tp_rules import family_of, rule_for
from app.risk.trailing_stop import (
    compute_r_trailing_stop,
    compute_trailing_stop,
    effective_stop,
    ladder_for_family,
    should_time_stop,
)
from app.utils.logging import get_logger
from app.utils.sessions import is_trading_open

logger = get_logger(__name__)

try:
    import aiosqlite  # type: ignore

    _HAS_AIOSQLITE = True
except ImportError:
    _HAS_AIOSQLITE = False

DECISIONS_DB = cfg.DATA_DIR / "decisions.db"

@dataclass
class StopLossLevel:
    """SL/TP levels associated with an open position."""

    ticker: str
    direction: str
    decision_id: str
    stop_loss: float | None
    take_profit: float | None
    take_profit_1: float | None = None
    take_profit_2: float | None = None
    atr: float | None = None
    entry_price: float | None = None
    pattern: str | None = None
    family: str | None = None
    entry_at_utc: str | None = None

class StopLossMonitor:
    """Background task: poll open positions, close on SL/TP hit."""

    def __init__(self, check_interval_sec: float | None = None) -> None:
        """Init."""
        self.check_interval_sec = float(
            check_interval_sec
            if check_interval_sec is not None
            else getattr(cfg, "SL_MONITOR_INTERVAL_SEC", 10.0)
        )
        self._stop_event = asyncio.Event()
        self._closed_ids: set[str] = set()
        self._task: asyncio.Task | None = None
        self._trailing_stops: dict[str, float] = {}
        self._tp1_closed: set[str] = set()
        self._hard_stop_pct = float(getattr(cfg, "HARD_STOP_LOSS_PCT", 0.015))
        self._hard_time_stop_hours = float(getattr(cfg, "HARD_TIME_STOP_HOURS", 24.0))
        self._hard_tp_pct = float(getattr(cfg, "HARD_TAKE_PROFIT_PCT", 0.04))
        self._hard_tp_aged_pct = float(getattr(cfg, "HARD_TAKE_PROFIT_AGED_PCT", 0.015))
        self._hard_tp_aged_hours = float(getattr(cfg, "HARD_TP_AGED_HOURS", 4.0))

    async def start(self) -> None:
        """Start monitor task and re-arm SL/TP for open positions."""
        try:
            from app.risk.position_book import get_position_book

            book = get_position_book()
            tickers = list(book.positions.keys())
            if tickers:
                levels = await self._load_sl_tp_for_open(tickers)
                logger.info(
                    "StopLossMonitor: восстановление SL/TP на старте",
                    extra={
                        "n_open_positions": len(tickers),
                        "n_sl_tp_loaded": len(levels),
                        "tickers": list(levels.keys()),
                    },
                )
                missing = [t for t in tickers if t not in levels]
                if missing:
                    logger.critical(
                        "Open positions without SL/TP after reattach — "
                        "monitor cannot protect these",
                        extra={"tickers": missing},
                    )
        except Exception as exc:
            logger.warning("StopLossMonitor reattach probe failed", extra={"error": str(exc)})

        self._task = asyncio.create_task(self._run(), name="stop_loss_monitor")
        logger.info("StopLossMonitor запущен", extra={"interval_sec": self.check_interval_sec})

    async def stop(self) -> None:
        """Stop the monitor task."""
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        logger.info("StopLossMonitor остановлен")

    async def _run(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            try:
                if is_trading_open():
                    await self._check_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("StopLossMonitor cycle failed", extra={"error": str(exc)})

            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.check_interval_sec,
                )

    def _get_adaptive_hard_stops(self) -> tuple[float | None, float | None]:
        """Compute adaptive hard SL/TP thresholds.

        Returns:
            tuple[float | None, float | None]: (hard_sl_pct, hard_tp_pct)
        """
        try:
            from app.risk.adaptive_regime import compute_risk_regime
            from app.risk.circuit_breakers import get_circuit_breaker

            cb = get_circuit_breaker()
            cb_state = cb.state
            deposit = float(getattr(cb_state, "peak_equity_rub", 0.0) or 1_000_000.0)
            daily_pnl_pct = float(cb_state.daily_pnl_rub) / max(1.0, deposit)
            regime = compute_risk_regime(
                current_drawdown_from_peak_pct=float(cb_state.current_drawdown_pct),
                losing_streak=int(cb_state.losing_streak),
                daily_pnl_pct=daily_pnl_pct,
                seconds_until_close=None,
            )
            return regime.hard_sl_pct, regime.hard_tp_pct
        except Exception as exc:
            logger.debug(
                "Failed to compute adaptive hard stops; falling back to None",
                extra={"error": str(exc)},
            )
            return None, None

    async def _check_once(self) -> None:
        """Fetch open positions, compare current price vs SL/TP, close on hit."""
        from app.data.iss_client import get_iss_client
        from app.execution.arenago_client import get_arenago_client
        from app.risk.position_book import get_position_book

        book = get_position_book()
        if not book.positions:
            return

        await self._hard_stop_sweep(book)

        levels = await self._load_sl_tp_for_open(list(book.positions.keys()))
        if not levels:
            return

        iss = get_iss_client()
        arenago = get_arenago_client()

        try:
            from datetime import timedelta

            till = datetime.now(tz=UTC)
            from_dt = till - timedelta(minutes=5)
            candles = await iss.get_candles_multi(
                list(levels.keys()),
                interval=1,
                from_dt=from_dt,
                till_dt=till,
            )
        except Exception as exc:
            logger.warning("SL monitor: ISS fetch failed", extra={"error": str(exc)})
            return

        for ticker, level in levels.items():
            df = candles.get(ticker)
            if df is None or len(df) == 0:
                continue
            try:
                last_price = float(df.iloc[-1].get("close", 0))
            except Exception:
                continue
            if last_price <= 0:
                continue

            position = book.positions.get(ticker)
            if position is None or position.quantity == 0:
                continue

            entry = level.entry_price or position.avg_price
            family = level.family or family_of(level.pattern or "")
            ladder = ladder_for_family(family)
            new_trail = compute_trailing_stop(
                direction=level.direction,
                entry_price=entry,
                current_price=last_price,
                atr=level.atr or 0.0,
                prev_trailing=self._trailing_stops.get(ticker),
                ladder=ladder,
            )
            if new_trail is not None:
                self._trailing_stops[ticker] = new_trail

            if (
                getattr(cfg, "TRAILING_STOP_ENABLED", True)
                and level.stop_loss is not None
                and entry > 0
            ):
                r_trail = compute_r_trailing_stop(
                    direction=level.direction,
                    entry_price=entry,
                    current_price=last_price,
                    original_sl=float(level.stop_loss),
                    r_to_be=float(getattr(cfg, "TRAILING_STOP_R_TO_BE", 1.0)),
                    r_to_r1=float(getattr(cfg, "TRAILING_STOP_R_TO_R1", 2.0)),
                )
                if r_trail is not None:
                    combined = effective_stop(
                        level.direction, self._trailing_stops.get(ticker), r_trail
                    )
                    if combined is not None:
                        self._trailing_stops[ticker] = combined

            current_trail = self._trailing_stops.get(ticker)

            eff_sl = effective_stop(level.direction, level.stop_loss, current_trail)

            plan = plan_partial_exit(
                quantity=position.quantity if ticker not in self._tp1_closed else position.quantity,
                take_profit=level.take_profit,
                take_profit_1=level.take_profit_1,
                take_profit_2=level.take_profit_2,
                entry_price=entry,
            )

            bars_held = 0
            if level.entry_at_utc:
                try:
                    entry_dt = datetime.fromisoformat(level.entry_at_utc.replace("Z", "+00:00"))
                    delta = datetime.now(tz=UTC) - entry_dt
                    bars_held = max(0, int(delta.total_seconds() // 3600))
                except Exception:
                    bars_held = 0
            rule = rule_for(level.pattern or "")
            time_stop_triggered = should_time_stop(
                direction=level.direction,
                entry_price=entry,
                current_price=last_price,
                bars_held=bars_held,
                max_bars=rule.time_stop_bars,
            )

            action: str | None = None
            close_qty = 0
            close_price = 0.0
            close_suffix = ""

            adaptive_hard_sl_pct, adaptive_hard_tp_pct = self._get_adaptive_hard_stops()
            if adaptive_hard_sl_pct is not None and entry > 0:
                if level.direction == "BUY":
                    hard_sl_price = entry * (1.0 - adaptive_hard_sl_pct)
                    if last_price <= hard_sl_price:
                        action = "HARD_SL"
                        close_qty = position.quantity
                        close_price = last_price
                        close_suffix = "_hardsl"
                else:
                    hard_sl_price = entry * (1.0 + adaptive_hard_sl_pct)
                    if last_price >= hard_sl_price:
                        action = "HARD_SL"
                        close_qty = position.quantity
                        close_price = last_price
                        close_suffix = "_hardsl"

            if action is None and adaptive_hard_tp_pct is not None and entry > 0:
                if level.direction == "BUY":
                    hard_tp_price = entry * (1.0 + adaptive_hard_tp_pct)
                    if last_price >= hard_tp_price:
                        action = "HARD_TP"
                        close_qty = position.quantity
                        close_price = last_price
                        close_suffix = "_hardtp"
                else:
                    hard_tp_price = entry * (1.0 - adaptive_hard_tp_pct)
                    if last_price <= hard_tp_price:
                        action = "HARD_TP"
                        close_qty = position.quantity
                        close_price = last_price
                        close_suffix = "_hardtp"

            if action is not None:
                pass
            elif level.direction == "BUY":
                if tp2_hit("BUY", last_price, plan.tp2_price):
                    action = "TP2"
                    close_qty = position.quantity
                    close_price = plan.tp2_price or 0.0
                    close_suffix = ""
                elif (
                    plan.has_partial
                    and ticker not in self._tp1_closed
                    and tp1_hit("BUY", last_price, plan.tp1_price)
                ):
                    action = "TP1"
                    close_qty = min(plan.qty_tp1, position.quantity)
                    close_price = plan.tp1_price or 0.0
                    close_suffix = "_tp1"
                elif eff_sl is not None and last_price <= eff_sl:
                    action = "SL"
                    close_qty = position.quantity
                    close_price = eff_sl
                    close_suffix = ""
                elif time_stop_triggered:
                    action = "TIME_STOP"
                    close_qty = position.quantity
                    close_price = last_price
                    close_suffix = "_ts"
            else:
                if tp2_hit("SELL", last_price, plan.tp2_price):
                    action = "TP2"
                    close_qty = position.quantity
                    close_price = plan.tp2_price or 0.0
                    close_suffix = ""
                elif (
                    plan.has_partial
                    and ticker not in self._tp1_closed
                    and tp1_hit("SELL", last_price, plan.tp1_price)
                ):
                    action = "TP1"
                    close_qty = min(plan.qty_tp1, position.quantity)
                    close_price = plan.tp1_price or 0.0
                    close_suffix = "_tp1"
                elif eff_sl is not None and last_price >= eff_sl:
                    action = "SL"
                    close_qty = position.quantity
                    close_price = eff_sl
                    close_suffix = ""
                elif time_stop_triggered:
                    action = "TIME_STOP"
                    close_qty = position.quantity
                    close_price = last_price
                    close_suffix = "_ts"

            if action is None or close_qty <= 0:
                continue

            close_id = f"sl_{level.decision_id}{close_suffix}"
            if close_id in self._closed_ids:
                continue
            self._closed_ids.add(close_id)

            reason = f"{action}@{close_price:.2f}"
            if (
                action == "SL"
                and current_trail is not None
                and (
                    (level.direction == "BUY" and current_trail > (level.stop_loss or -1e18))
                    or (level.direction == "SELL" and current_trail < (level.stop_loss or 1e18))
                )
            ):
                reason = f"TRAIL@{current_trail:.2f}"

            opposite = "SELL" if level.direction == "BUY" else "BUY"
            logger.critical(
                "Сработал SL/TP — закрываем позицию",
                extra={
                    "ticker": ticker,
                    "open_direction": level.direction,
                    "close_direction": opposite,
                    "quantity": close_qty,
                    "last_price": last_price,
                    "reason": reason,
                    "action": action,
                    "trailing_stop": current_trail,
                    "decision_id": level.decision_id,
                },
            )
            try:
                sl_kwargs: dict = {
                    "direction": opposite,
                    "ticker": ticker,
                    "quantity": close_qty,
                    "decision_id": close_id,
                }
                try:
                    import inspect

                    if "is_exit" in inspect.signature(arenago.submit_order).parameters:
                        sl_kwargs["is_exit"] = True
                except (TypeError, ValueError):
                    pass
                await arenago.submit_order(**sl_kwargs)
                if action == "TP1":
                    self._tp1_closed.add(ticker)
                elif action in ("SL", "TP2", "TIME_STOP"):
                    self._tp1_closed.discard(ticker)
                    self._trailing_stops.pop(ticker, None)
                try:
                    from app.risk.position_book import get_position_book

                    book_for_ts = get_position_book()
                    if hasattr(book_for_ts, "_last_entry_ts"):
                        book_for_ts._last_entry_ts[ticker.upper()] = time.time()
                except Exception:
                    pass
            except Exception as exc:
                self._closed_ids.discard(close_id)
                logger.error("SL закрытие позиции не удалось", extra={"ticker": ticker, "error": str(exc)})

    async def _hard_stop_sweep(self, book) -> None:
        """Безусловная защита: закрыть позицию при loss > HARD_STOP_LOSS_PCT
        или возрасте > HARD_TIME_STOP_HOURS, независимо от наличия SL в DB.

        Это страховка против ситуации «позиция висит в минусе часами, бот
        не реагирует» — основной StopLossMonitor работает только когда
        SL/TP записаны в decisions.db, что не всегда так (старые позиции,
        reconciler-synth, слишком мягкий adaptive_regime).
        """
        if self._hard_stop_pct <= 0 and self._hard_time_stop_hours <= 0:
            return
        from app.data.iss_client import get_iss_client
        from app.execution.arenago_client import get_arenago_client

        tickers = list(book.positions.keys())
        if not tickers:
            return
        iss = get_iss_client()
        try:
            from datetime import timedelta

            till = datetime.now(tz=UTC)
            from_dt = till - timedelta(minutes=5)
            candles = await iss.get_candles_multi(
                tickers, interval=1, from_dt=from_dt, till_dt=till
            )
        except Exception as exc:
            logger.debug("hard_stop_sweep: ISS fetch failed", extra={"error": str(exc)})
            return

        arenago = get_arenago_client()
        now_utc = datetime.now(tz=UTC)

        for ticker in tickers:
            pos = book.positions.get(ticker)
            if pos is None or pos.quantity == 0:
                continue
            df = candles.get(ticker)
            if df is None or len(df) == 0:
                continue
            try:
                last_price = float(df.iloc[-1].get("close", 0))
            except Exception:
                continue
            if last_price <= 0:
                continue

            entry = float(pos.avg_price or 0)
            if entry <= 0:
                continue
            qty_signed = int(pos.quantity)
            direction = "BUY" if qty_signed > 0 else "SELL"
            if direction == "BUY":
                pnl_pct = (last_price - entry) / entry
            else:
                pnl_pct = (entry - last_price) / entry

            triggered: str | None = None
            if self._hard_stop_pct > 0 and pnl_pct <= -self._hard_stop_pct:
                triggered = "HARD_STOP_LOSS"

            opened_at = getattr(pos, "opened_at_utc", None) or getattr(pos, "opened_at", None)
            age_hours = 0.0
            if opened_at:
                try:
                    if isinstance(opened_at, str):
                        opened_dt = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
                    else:
                        opened_dt = opened_at
                    if opened_dt.tzinfo is None:
                        opened_dt = opened_dt.replace(tzinfo=UTC)
                    age_hours = (now_utc - opened_dt).total_seconds() / 3600.0
                except Exception:
                    age_hours = 0.0
            if triggered is None and self._hard_tp_pct > 0:
                is_aged = (
                    self._hard_tp_aged_hours > 0
                    and age_hours >= self._hard_tp_aged_hours
                )
                tp_threshold = (
                    self._hard_tp_aged_pct if is_aged and self._hard_tp_aged_pct > 0
                    else self._hard_tp_pct
                )
                if pnl_pct >= tp_threshold:
                    triggered = "HARD_TAKE_PROFIT_AGED" if is_aged else "HARD_TAKE_PROFIT"
            if (
                triggered is None
                and self._hard_time_stop_hours > 0
                and age_hours >= self._hard_time_stop_hours
                and pnl_pct < 0.005
            ):
                triggered = "HARD_TIME_STOP"

            if triggered is None:
                continue

            close_id = f"hard_{ticker.lower()}_{int(now_utc.timestamp())}"
            if close_id in self._closed_ids:
                continue
            self._closed_ids.add(close_id)

            opposite = "SELL" if direction == "BUY" else "BUY"
            logger.critical(
                "HARD_STOP сработал — закрываем убыточную позицию",
                extra={
                    "trigger": triggered,
                    "ticker": ticker,
                    "open_direction": direction,
                    "close_direction": opposite,
                    "quantity": abs(qty_signed),
                    "entry_price": round(entry, 4),
                    "last_price": round(last_price, 4),
                    "pnl_pct": round(pnl_pct * 100, 3),
                    "age_hours": round(age_hours, 2),
                },
            )
            try:
                sl_kwargs: dict = {
                    "direction": opposite,
                    "ticker": ticker,
                    "quantity": abs(qty_signed),
                    "decision_id": close_id,
                }
                try:
                    import inspect

                    if "is_exit" in inspect.signature(arenago.submit_order).parameters:
                        sl_kwargs["is_exit"] = True
                except (TypeError, ValueError):
                    pass
                await arenago.submit_order(**sl_kwargs)
                self._trailing_stops.pop(ticker, None)
                self._tp1_closed.discard(ticker)
                try:
                    from app.risk.position_book import get_position_book

                    book_for_ts = get_position_book()
                    if hasattr(book_for_ts, "_last_entry_ts"):
                        book_for_ts._last_entry_ts[ticker.upper()] = time.time()
                except Exception:
                    pass
            except Exception as exc:
                self._closed_ids.discard(close_id)
                logger.error(
                    "hard_stop отправка ордера не удалась",
                    extra={"ticker": ticker, "error": str(exc)},
                )

    async def _load_sl_tp_for_open(self, tickers: list[str]) -> dict[str, StopLossLevel]:
        """Load most-recent EXECUTED SL/TP rows for given tickers.

        Args:
            tickers: tickers to load
        Returns:
            dict[str, StopLossLevel]: ticker → SL/TP record
        """
        if not _HAS_AIOSQLITE or not tickers:
            return {}
        result: dict[str, StopLossLevel] = {}
        placeholders = ",".join("?" * len(tickers))
        try:
            async with aiosqlite.connect(DECISIONS_DB) as db:
                cols_cur = await db.execute("PRAGMA table_info(decisions)")
                col_rows = await cols_cur.fetchall()
                await cols_cur.close()
                col_names = {str(r[1]) for r in col_rows}
                has_tp12 = "take_profit_1" in col_names and "take_profit_2" in col_names
                tp_select = (
                    "stop_loss, take_profit, take_profit_1, take_profit_2"
                    if has_tp12
                    else "stop_loss, take_profit, NULL, NULL"
                )
                async with db.execute(
                    f"""
                    SELECT ticker, direction, decision_id,
                           {tp_select}, signals_json, executed_at,
                           MAX(created_at) AS latest
                    FROM decisions
                    WHERE ticker IN ({placeholders})
                      AND action = 'EXECUTE'
                      AND (stop_loss IS NOT NULL OR take_profit IS NOT NULL)
                    GROUP BY ticker
                    """,
                    [t.upper() for t in tickers],
                ) as cur:
                    rows = await cur.fetchall()
            for r in rows:
                ticker = str(r[0]).upper()
                if ticker not in [t.upper() for t in tickers]:
                    continue
                atr_val = None
                entry_val = None
                pattern_val: str | None = None
                signals_json = r[7] if len(r) > 7 else None
                if signals_json:
                    try:
                        sigs = json.loads(signals_json)
                        for s in sigs:
                            if atr_val is None and float(s.get("atr") or 0) > 0:
                                atr_val = float(s["atr"])
                            if entry_val is None and float(s.get("entry_level") or 0) > 0:
                                entry_val = float(s["entry_level"])
                            if pattern_val is None:
                                pat = s.get("pattern") or s.get("signal_type") or s.get("source")
                                if pat:
                                    pattern_val = str(pat)
                            if atr_val and entry_val and pattern_val:
                                break
                    except (ValueError, TypeError, json.JSONDecodeError):
                        pass
                executed_at = r[8] if len(r) > 8 else None
                result[ticker] = StopLossLevel(
                    ticker=ticker,
                    direction=str(r[1] or "BUY").upper(),
                    decision_id=str(r[2] or ""),
                    stop_loss=float(r[3]) if r[3] is not None else None,
                    take_profit=float(r[4]) if r[4] is not None else None,
                    take_profit_1=float(r[5]) if r[5] is not None else None,
                    take_profit_2=float(r[6]) if r[6] is not None else None,
                    atr=atr_val,
                    entry_price=entry_val,
                    pattern=pattern_val,
                    family=family_of(pattern_val or ""),
                    entry_at_utc=str(executed_at) if executed_at else None,
                )
        except Exception as exc:
            logger.error("SL/TP DB load failed", extra={"error": str(exc)})
        return result

_monitor: StopLossMonitor | None = None

def get_stop_loss_monitor() -> StopLossMonitor:
    """Return process-wide StopLossMonitor singleton.

    Returns:
        StopLossMonitor: shared instance
    """
    global _monitor
    if _monitor is None:
        _monitor = StopLossMonitor()
    return _monitor
