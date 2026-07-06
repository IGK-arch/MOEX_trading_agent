"""Вечерняя рефлексия через polza.ai."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import app.config as cfg
from app.llm.polza_client import get_polza_client
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import aiosqlite  # type: ignore

    _HAS_AIOSQLITE = True
except ImportError:
    _HAS_AIOSQLITE = False

DECISIONS_DB = cfg.DATA_DIR / "decisions.db"
TRADES_DB = cfg.DATA_DIR / "trades.db"

REFLECTION_PROMPT = """Ты — старший трейдер MOEX, делающий вечерний разбор сегодняшних сделок.

ДАННЫЕ ЗА ДЕНЬ:
{trades_summary}

КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ:
- Всего решений: {n_decisions}
- Исполненных сделок: {n_trades}
- Общий PnL: {total_pnl_rub:.0f} ₽ ({pnl_pct:+.2%})
- Win rate: {win_rate:.1%}
- Текущий капитал: {current_equity_rub:.0f} ₽

ЗАДАЧА:
Выдели 5-7 КОНКРЕТНЫХ уроков на завтра. Не общие фразы — точные паттерны:
- Какие тикеры/режимы/сигналы дали профит и почему?
- Где были ошибки и что их связывало?
- Какие фильтры/правила нужно усилить/ослабить?

Верни JSON:
{{
  "lessons": [
    {{"importance": float ∈ [0.0, 1.0], "category": "wins|losses|process|risk", "lesson": "..."}},
    ...
  ],
  "tomorrow_focus": "Одна фраза — на что обратить главное внимание завтра"
}}

Also return optional guarded parameter recommendations:
{{
  "parameter_adjustments": [
    {{
      "parameter": "META_MIN_PROBA|PAIR_Z_ENTRY_THRESHOLD|MEAN_REV_BB_STD",
      "direction": "increase|decrease",
      "delta": 0.01,
      "confidence": 0.0,
      "reason": "specific evidence from today's trades"
    }}
  ]
}}

Rules for parameter_adjustments:
- At most 3 items.
- Use only the whitelisted parameters above.
- Increase thresholds after losses/noisy trades.
- Decrease thresholds only after profitable high-confidence days with missed opportunity.
- If evidence is weak, return an empty list.
"""

class ReflectionEngine:
    """Evening deferred reflection — runs once per day."""

    def __init__(self) -> None:
        """Init."""
        self.polza = get_polza_client()

    async def run_today(self) -> dict[str, Any] | None:
        """Run reflection on today's PENDING decisions."""
        if cfg.DISABLE_LLM:
            logger.info("reflection: DISABLE_LLM=1, skipping LLM reflection")
            return None
        if not self.polza._started:
            await self.polza.startup()

        today_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        decisions = await self._fetch_pending_decisions(today_str)
        trades = await self._fetch_today_trades(today_str)

        if not decisions:
            logger.info("Reflection: no PENDING decisions today")
            return None

        n_decisions = len(decisions)
        n_trades = len(trades)
        total_pnl = sum(t.get("pnl_rub", 0) or 0 for t in trades)
        wins = sum(1 for t in trades if (t.get("pnl_rub") or 0) > 0)
        win_rate = wins / max(1, n_trades)
        day_stats = {
            "n_decisions": n_decisions,
            "n_trades": n_trades,
            "total_pnl_rub": round(float(total_pnl), 2),
            "win_rate": round(float(win_rate), 4),
        }

        summary = self._format_summary(decisions, trades)

        start_dep = float(getattr(cfg, "STARTING_DEPOSIT_RUB", 1_000_000.0))
        prompt = REFLECTION_PROMPT.format(
            trades_summary=summary,
            n_decisions=n_decisions,
            n_trades=n_trades,
            total_pnl_rub=total_pnl,
            pnl_pct=total_pnl / start_dep if start_dep > 0 else 0.0,
            win_rate=win_rate,
            current_equity_rub=start_dep + total_pnl,
        )

        try:
            response = await self.polza.chat_json(
                messages=[
                    {"role": "system", "content": "Ты опытный трейдер MOEX. Отвечай только JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=cfg.POLZA_MODEL_REASONING,
                purpose="evening_reflection",
                max_tokens=4000,
            )
        except Exception as exc:
            logger.error("Reflection LLM call failed", extra={"error": str(exc)})
            return None

        if not response:
            logger.warning("Reflection: invalid LLM response")
            return None

        payload = response.get("parsed") if isinstance(response.get("parsed"), dict) else response
        if "lessons" not in payload:
            logger.warning("Reflection: invalid LLM response")
            return None

        lessons = payload.get("lessons", [])

        await self._save_lessons(lessons, today_str)

        try:
            from app.memory.reflexive_overrides import apply_reflexive_adjustments

            payload["reflexive_control"] = apply_reflexive_adjustments(
                payload,
                date_str=today_str,
                day_stats=day_stats,
            )
        except Exception as exc:
            logger.error("Reflexive control failed", extra={"error": str(exc)})

        await self._mark_reflected(today_str)

        logger.info(
            "Reflection complete",
            extra={
                "lessons_count": len(lessons),
                "tomorrow_focus": payload.get("tomorrow_focus", "")[:80],
                "pnl_rub": total_pnl,
                "win_rate": win_rate,
            },
        )

        return payload

    async def _fetch_pending_decisions(self, date_str: str) -> list[dict]:
        """Fetch pending decisions."""
        if not _HAS_AIOSQLITE:
            return []
        results = []
        try:
            async with (
                aiosqlite.connect(DECISIONS_DB) as db,
                db.execute(
                    "SELECT decision_id, ticker, action, tier, direction, "
                    "combined_magnitude, rationale, created_at "
                    "FROM decisions WHERE created_at LIKE ? AND reflection_status='PENDING'",
                    (f"{date_str}%",),
                ) as cur,
            ):
                rows = await cur.fetchall()
                for r in rows:
                    results.append(
                        {
                            "decision_id": r[0],
                            "ticker": r[1],
                            "action": r[2],
                            "tier": r[3],
                            "direction": r[4],
                            "combined_magnitude": r[5],
                            "rationale": r[6],
                            "created_at": r[7],
                        }
                    )
        except Exception as exc:
            logger.error("Fetch decisions failed", extra={"error": str(exc)})
        return results

    async def _fetch_today_trades(self, date_str: str) -> list[dict]:
        """Phase 19 (v0.0.16) — compute real FIFO PnL per closing trade.

        Algorithm: maintain a per-ticker FIFO queue of BUY lots
        (qty, price). Each SELL pops lots and computes:
            pnl = sum( (sell_price - lot_price) * matched_qty ) - commission

        Commission heuristic: 0.05% per side (10 bps roundtrip) — matches
        typical MOEX retail rate; ArenaGo sandbox doesn't charge but live
        does, so we account for it in the reflection / streak metrics.
        """
        if not _HAS_AIOSQLITE:
            return []
        results: list[dict] = []
        commission_pct = cfg.ARENAGO_COMMISSION_PCT
        try:
            async with (
                aiosqlite.connect(TRADES_DB) as db,
                db.execute(
                    "SELECT decision_id, ticker, direction, quantity, price, order_value "
                    "FROM trades WHERE trade_date = ? ORDER BY trade_time",
                    (date_str,),
                ) as cur,
            ):
                rows = await cur.fetchall()

            from collections import deque

            fifo: dict[str, deque] = {}

            for r in rows:
                ticker = str(r[1]).upper()
                direction = str(r[2]).upper()
                qty = int(r[3] or 0)
                price = float(r[4] or 0)
                order_value = float(r[5] or 0)
                pnl_rub = 0.0

                if direction == "BUY":
                    fifo.setdefault(ticker, deque()).append([qty, price])
                    pnl_rub = -order_value * commission_pct
                elif direction == "SELL":
                    remaining = qty
                    lots = fifo.get(ticker)
                    if lots:
                        while remaining > 0 and lots:
                            lot_qty, lot_price = lots[0]
                            matched = min(remaining, lot_qty)
                            pnl_rub += (price - lot_price) * matched
                            remaining -= matched
                            if matched >= lot_qty:
                                lots.popleft()
                            else:
                                lots[0][0] = lot_qty - matched
                                break
                    pnl_rub -= order_value * commission_pct

                results.append(
                    {
                        "decision_id": r[0],
                        "ticker": ticker,
                        "direction": direction,
                        "quantity": qty,
                        "price": price,
                        "order_value": order_value,
                        "pnl_rub": round(pnl_rub, 2),
                    }
                )

            try:
                from app.risk.circuit_breakers import get_circuit_breaker
                from app.risk.position_book import get_position_book

                cb = get_circuit_breaker()
                book = get_position_book()
                equity = float(book.cash_balance) + float(book.total_market_value())
                for t in results:
                    if t["direction"] == "SELL" and abs(t["pnl_rub"]) > 1.0:
                        await cb.on_trade_closed(t["pnl_rub"], equity)
            except Exception as exc:
                logger.warning("CB on_trade_closed failed in reflection", extra={"error": str(exc)})

        except Exception as exc:
            logger.error("Fetch trades failed", extra={"error": str(exc)})
        return results

    @staticmethod
    def _format_summary(decisions: list[dict], trades: list[dict]) -> str:
        """Format summary."""
        lines = ["TICKER | ACTION | TIER | DIR  | MAG  | RATIONALE"]
        lines.append("-" * 100)
        for d in decisions[:30]:
            lines.append(
                f"{d['ticker']:6s} | {d['action']:9s} | {d['tier']:4s} | "
                f"{d['direction']:4s} | {(d['combined_magnitude'] or 0):.2f} | "
                f"{(d['rationale'] or '')[:60]}"
            )
        lines.append("")
        lines.append(f"Trades executed: {len(trades)}")
        return "\n".join(lines)

    async def _save_lessons(self, lessons: list[dict], date_str: str) -> None:
        """Persist lessons to a JSON file for now (ChromaDB integration later)."""
        if not lessons:
            return
        lessons_dir = cfg.DATA_DIR / "lessons"
        lessons_dir.mkdir(parents=True, exist_ok=True)
        path = lessons_dir / f"{date_str}.json"
        try:
            with open(path, "w") as f:
                json.dump({"date": date_str, "lessons": lessons}, f, ensure_ascii=False, indent=2)
            logger.info("Lessons saved", extra={"path": str(path), "count": len(lessons)})
        except Exception as exc:
            logger.error("Lessons save failed", extra={"error": str(exc)})

    async def _mark_reflected(self, date_str: str) -> None:
        """Mark reflected."""
        if not _HAS_AIOSQLITE:
            return
        try:
            async with aiosqlite.connect(DECISIONS_DB) as db:
                await db.execute(
                    "UPDATE decisions SET reflection_status='REFLECTED' "
                    "WHERE created_at LIKE ? AND reflection_status='PENDING'",
                    (f"{date_str}%",),
                )
                await db.commit()
        except Exception as exc:
            logger.error("Mark reflected failed", extra={"error": str(exc)})

_reflection: ReflectionEngine | None = None

def get_reflection_engine() -> ReflectionEngine:
    """Get reflection engine."""
    global _reflection
    if _reflection is None:
        _reflection = ReflectionEngine()
    return _reflection
