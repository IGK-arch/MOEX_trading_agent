"""Daily trade analyzer — detailed per-trade post-mortem (Phase 27.5).

The legacy ``ReflectionEngine`` writes a flat list of LLM "lessons" once
per evening and nobody reads it the next morning. This script produces a
much richer artefact:

* per-trade metrics (entry/exit/PnL/holding/alpha vs IMOEX/expected vs
  actual RR/winning_pattern_id/HMM regime at entry/...)
* aggregation tables (per ticker / per source_strategy / per detector /
  per tier / per regime / per time-of-day bucket / per direction)
* a top-5 winners + top-5 losers ranking
* an optional LLM-generated narrative summary (polza-flash)
* JSON for downstream consumers (``data/training_cache/daily_analysis_<date>.json``)
* Markdown for human consumption (``data/morning_plans/post_mortem_<date>.md``)
* optional indexing of every trade into the RAG store via
  :class:`app.memory.trade_outcome_rag.TradeOutcomeIndexer` (``--write-rag``)
  so the morning consensus + reactive comparator can recall historical
  outcomes for similar setups.

CLI::

    python3 scripts/daily_trade_analyzer.py                          # yesterday
    python3 scripts/daily_trade_analyzer.py --date 2026-05-26
    python3 scripts/daily_trade_analyzer.py --date 2026-05-26 --write-rag
    python3 scripts/daily_trade_analyzer.py --output-json /tmp/x.json --output-md /tmp/x.md

The script is purposely best-effort: empty trades.db → graceful empty
report (no exception, no missing files). The morning planner can rely on
the JSON existing even on a day with zero trades.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.config as cfg  # noqa: E402
from app.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

DECISIONS_DB = cfg.DATA_DIR / "decisions.db"
TRADES_DB = cfg.DATA_DIR / "trades.db"
POST_MORTEM_DIR = cfg.DATA_DIR / "morning_plans"
TRAINING_CACHE_DIR = cfg.DATA_DIR / "training_cache"

COMMISSION_PCT_PER_SIDE = cfg.ARENAGO_COMMISSION_PCT

def _load_trades(db_path: Path, date_str: str) -> list[dict[str, Any]]:
    """Return all trades for ``date_str`` (YYYY-MM-DD) ordered by trade_time.

    Empty list on missing DB / missing date — never raises.
    """
    if not db_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(
                    """
                    SELECT id, decision_id, ticker, direction, quantity, price,
                           order_value, remaining_cash, trade_date, trade_time,
                           bot, source_model, arena_raw_json, created_at
                    FROM trades
                    WHERE trade_date = ?
                    ORDER BY trade_time, id
                    """,
                    (date_str,),
                )
            except sqlite3.OperationalError as exc:
                logger.warning(
                    "daily_trade_analyzer: trades schema mismatch",
                    extra={"error": str(exc)},
                )
                return []
            for r in cur.fetchall():
                rows.append({k: r[k] for k in r})
    except sqlite3.DatabaseError as exc:
        logger.warning(
            "daily_trade_analyzer: failed to read trades.db",
            extra={"error": str(exc), "path": str(db_path)},
        )
        return []
    return rows

def _load_decisions(db_path: Path, date_str: str) -> dict[str, dict[str, Any]]:
    """Map ``decision_id`` → decision row for ``date_str``.

    Empty dict on missing DB — never raises.
    """
    if not db_path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(
                    """
                    SELECT decision_id, cycle_id, ticker, action, tier,
                           direction, combined_magnitude, risk_check,
                           stop_loss, take_profit, expected_holding_min,
                           rationale, signals_json, executed_bool,
                           created_at, executed_at
                    FROM decisions
                    WHERE created_at LIKE ?
                    """,
                    (f"{date_str}%",),
                )
            except sqlite3.OperationalError as exc:
                logger.warning(
                    "daily_trade_analyzer: decisions schema mismatch",
                    extra={"error": str(exc)},
                )
                return {}
            for r in cur.fetchall():
                out[str(r["decision_id"])] = {k: r[k] for k in r}
    except sqlite3.DatabaseError as exc:
        logger.warning(
            "daily_trade_analyzer: failed to read decisions.db",
            extra={"error": str(exc), "path": str(db_path)},
        )
        return {}
    return out

def _parse_signals(signals_json: Any) -> list[dict[str, Any]]:
    """Defensively parse the signals_json blob from decisions.db."""
    if not signals_json:
        return []
    if isinstance(signals_json, list):
        return [s for s in signals_json if isinstance(s, dict)]
    try:
        parsed = json.loads(signals_json)
        if isinstance(parsed, list):
            return [s for s in parsed if isinstance(s, dict)]
        return []
    except (TypeError, ValueError):
        return []

def _parse_arena_json(blob: Any) -> dict[str, Any]:
    """Parse arena json."""
    if not blob:
        return {}
    if isinstance(blob, dict):
        return blob
    try:
        v = json.loads(blob)
        return v if isinstance(v, dict) else {}
    except (TypeError, ValueError):
        return {}

def _fetch_imoex_curve(date_str: str) -> dict[str, float] | None:
    """Return a ``HH:MM`` → close mapping for IMOEX on ``date_str``.

    Falls back to ``None`` if moexalgo isn't installed or the day has no
    data. The script tolerates a missing curve — alpha_vs_imoex is set
    to ``None`` for affected trades.
    """
    try:
        from moexalgo import Ticker  # type: ignore
    except ImportError:
        logger.debug("daily_trade_analyzer: moexalgo not installed, skipping IMOEX")
        return None
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None
    try:
        candles = Ticker("IMOEX").candles(start=d, end=d, period="5min")
    except Exception as exc:
        logger.warning(
            "daily_trade_analyzer: IMOEX fetch failed",
            extra={"date": date_str, "error": str(exc)},
        )
        return None
    if candles is None:
        return None
    try:
        rows = list(candles)
    except TypeError:
        rows = candles
    out: dict[str, float] = {}
    for c in rows:
        try:
            begin = getattr(c, "begin", None) or (c.get("begin") if isinstance(c, dict) else None)
            close = getattr(c, "close", None) or (c.get("close") if isinstance(c, dict) else None)
            if begin is None or close is None:
                continue
            begin_dt = datetime.fromisoformat(begin) if isinstance(begin, str) else begin
            hhmm = begin_dt.strftime("%H:%M")
            out[hhmm] = float(close)
        except Exception:
            continue
    return out or None

def _nearest_imoex(curve: dict[str, float], hhmm: str) -> float | None:
    """Return the closest IMOEX value to ``hhmm`` (HH:MM)."""
    if not curve:
        return None
    if hhmm in curve:
        return curve[hhmm]
    try:
        target = datetime.strptime(hhmm, "%H:%M")
    except ValueError:
        return None
    best_key: str | None = None
    best_delta: float = 1e18
    for k in curve:
        try:
            kt = datetime.strptime(k, "%H:%M")
        except ValueError:
            continue
        delta = abs((kt - target).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best_key = k
    return curve[best_key] if best_key else None

def _coerce_time(date_str: str, t: Any) -> datetime | None:
    """Coerce time."""
    if t is None:
        return None
    s = str(t).strip()
    if not s:
        return None
    for fmt in ("%H:%M:%S", "%H:%M:%S.%f", "%H:%M"):
        try:
            dt = datetime.strptime(s, fmt)
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            return datetime.combine(d, dt.time(), tzinfo=UTC)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None

def _time_of_day_bucket(dt: datetime | None) -> str:
    """Time of day bucket."""
    if dt is None:
        return "unknown"
    h = dt.hour
    if h < 12:
        return "10-12"
    if h < 15:
        return "12-15"
    if h < 18:
        return "15-18"
    return "18-23"

def _classify_exit(
    *,
    holding_min: float | None,
    pnl_pct: float | None,
    stop_loss: float | None,
    take_profit: float | None,
    exit_price: float | None,
    direction: str,
) -> str:
    """Best-effort exit classification from price + decision SL/TP.

    Heuristics:
    * If we have explicit SL/TP and exit price is within 0.2% of either,
      label accordingly.
    * Else if pnl_pct < 0 and holding ≥ expected → time_stop.
    * Else default to ``manual``.
    """
    if exit_price is not None and stop_loss is not None and stop_loss > 0:
        if abs(exit_price - stop_loss) / max(1e-9, stop_loss) < 0.002:
            return "stop_loss"
    if exit_price is not None and take_profit is not None and take_profit > 0:
        if abs(exit_price - take_profit) / max(1e-9, take_profit) < 0.002:
            return "take_profit"
    if holding_min is not None and holding_min >= 60 and (pnl_pct or 0) <= 0:
        return "time_stop"
    return "manual"

def _build_round_trips(
    trades: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    date_str: str,
    imoex_curve: dict[str, float] | None,
) -> list[dict[str, Any]]:
    """FIFO-match BUYs vs SELLs per ticker and emit one record per closed
    leg.

    The exit_ts is the SELL leg's time; entry_ts is the matched lot's
    BUY time. Open positions at end of day are reported as ``open=True``
    with ``pnl_pct=None``.
    """
    fifo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    out: list[dict[str, Any]] = []

    for t in trades:
        ticker = str(t.get("ticker") or "").upper()
        direction = str(t.get("direction") or "").upper()
        qty = int(t.get("quantity") or 0)
        price = float(t.get("price") or 0.0)
        order_value = float(t.get("order_value") or 0.0)
        trade_time = t.get("trade_time")
        entry_dt = _coerce_time(date_str, trade_time)
        decision = decisions.get(str(t.get("decision_id") or "")) or {}
        sigs = _parse_signals(decision.get("signals_json"))
        first_sig = sigs[0] if sigs else {}
        meta = first_sig.get("metadata") if isinstance(first_sig.get("metadata"), dict) else {}

        if direction == "BUY":
            fifo[ticker].append(
                {
                    "qty": qty,
                    "price": price,
                    "trade_id": t.get("id"),
                    "decision_id": t.get("decision_id"),
                    "entry_ts": entry_dt,
                    "entry_time_str": str(trade_time or ""),
                    "order_value": order_value,
                    "decision": decision,
                    "first_sig": first_sig,
                    "meta": meta,
                }
            )
            continue

        if direction == "SELL":
            remaining = qty
            lots = fifo.get(ticker, [])
            while remaining > 0 and lots:
                lot = lots[0]
                matched = min(remaining, int(lot["qty"]))
                entry_price = float(lot["price"])
                pnl_gross = (price - entry_price) * matched
                commission = (
                    entry_price * matched * COMMISSION_PCT_PER_SIDE
                    + price * matched * COMMISSION_PCT_PER_SIDE
                )
                pnl_rub = pnl_gross - commission
                notional = entry_price * matched
                pnl_pct = (pnl_rub / notional * 100.0) if notional > 0 else 0.0
                holding_min: float | None = None
                if isinstance(lot["entry_ts"], datetime) and isinstance(entry_dt, datetime):
                    holding_min = (entry_dt - lot["entry_ts"]).total_seconds() / 60.0
                alpha_pct: float | None = None
                if (
                    imoex_curve
                    and isinstance(lot["entry_ts"], datetime)
                    and isinstance(entry_dt, datetime)
                ):
                    in_imoex = _nearest_imoex(imoex_curve, lot["entry_ts"].strftime("%H:%M"))
                    out_imoex = _nearest_imoex(imoex_curve, entry_dt.strftime("%H:%M"))
                    if in_imoex and out_imoex and in_imoex > 0:
                        imoex_ret = (out_imoex - in_imoex) / in_imoex * 100.0
                        alpha_pct = pnl_pct - imoex_ret
                dec = lot["decision"]
                first_sig = lot["first_sig"]
                meta = lot["meta"]
                source_strategy = str(first_sig.get("source") or "").upper() or "UNKNOWN"
                detector = str(first_sig.get("detector") or "n/a")
                tier = str(dec.get("tier") or "NONE")
                magnitude = first_sig.get("magnitude") or dec.get("combined_magnitude") or 0.0
                hmm_regime = (
                    meta.get("hmm_regime")
                    or meta.get("regime")
                    or meta.get("hmm_state")
                    or "unknown"
                )
                expected_rr = float(first_sig.get("expected_rr") or 0.0)
                stop = dec.get("stop_loss")
                tp = dec.get("take_profit")
                risk_pct: float | None = None
                if stop and entry_price > 0:
                    risk_pct = abs(entry_price - float(stop)) / entry_price * 100.0
                actual_rr: float | None = None
                if risk_pct and risk_pct > 1e-6:
                    actual_rr = pnl_pct / risk_pct
                exit_reason = _classify_exit(
                    holding_min=holding_min,
                    pnl_pct=pnl_pct,
                    stop_loss=float(stop) if stop else None,
                    take_profit=float(tp) if tp else None,
                    exit_price=price,
                    direction="LONG",
                )
                rec: dict[str, Any] = {
                    "trade_id": f"{lot['trade_id']}->{t.get('id')}",
                    "ticker": ticker,
                    "direction": "BUY",
                    "entry_ts": (
                        lot["entry_ts"].isoformat()
                        if isinstance(lot["entry_ts"], datetime)
                        else None
                    ),
                    "exit_ts": (entry_dt.isoformat() if isinstance(entry_dt, datetime) else None),
                    "entry_price": entry_price,
                    "exit_price": price,
                    "qty": matched,
                    "notional_rub": notional,
                    "pnl_rub": round(pnl_rub, 2),
                    "pnl_pct": round(pnl_pct, 4),
                    "holding_min": round(holding_min, 2) if holding_min is not None else None,
                    "exit_reason": exit_reason,
                    "source_strategy": source_strategy,
                    "detector": detector,
                    "tier": tier,
                    "meta_score": float(first_sig.get("raw_confidence") or 0.0),
                    "hmm_regime_at_entry": hmm_regime,
                    "magnitude": float(magnitude or 0.0),
                    "expected_rr": expected_rr,
                    "actual_rr": round(actual_rr, 3) if actual_rr is not None else None,
                    "risk_pct": round(risk_pct, 4) if risk_pct is not None else None,
                    "alpha_vs_imoex_pct": round(alpha_pct, 4) if alpha_pct is not None else None,
                    "time_of_day": _time_of_day_bucket(lot["entry_ts"]),
                    "rationale": str(dec.get("rationale") or "")[:400],
                    "would_have_been_better": _retrospective_better(
                        pnl_pct=pnl_pct,
                        entry_price=entry_price,
                        exit_price=price,
                    ),
                    "open": False,
                }
                out.append(rec)
                remaining -= matched
                if matched >= int(lot["qty"]):
                    lots.pop(0)
                else:
                    lot["qty"] = int(lot["qty"]) - matched

    for ticker, lots in fifo.items():
        for lot in lots:
            decision = lot["decision"]
            first_sig = lot["first_sig"]
            meta = lot["meta"]
            rec = {
                "trade_id": f"{lot['trade_id']}->OPEN",
                "ticker": ticker,
                "direction": "BUY",
                "entry_ts": (
                    lot["entry_ts"].isoformat() if isinstance(lot["entry_ts"], datetime) else None
                ),
                "exit_ts": None,
                "entry_price": float(lot["price"]),
                "exit_price": None,
                "qty": int(lot["qty"]),
                "notional_rub": float(lot["price"]) * int(lot["qty"]),
                "pnl_rub": None,
                "pnl_pct": None,
                "holding_min": None,
                "exit_reason": "open",
                "source_strategy": str(first_sig.get("source") or "").upper() or "UNKNOWN",
                "detector": str(first_sig.get("detector") or "n/a"),
                "tier": str(decision.get("tier") or "NONE"),
                "meta_score": float(first_sig.get("raw_confidence") or 0.0),
                "hmm_regime_at_entry": (meta.get("hmm_regime") or meta.get("regime") or "unknown"),
                "magnitude": float(
                    first_sig.get("magnitude") or decision.get("combined_magnitude") or 0.0
                ),
                "expected_rr": float(first_sig.get("expected_rr") or 0.0),
                "actual_rr": None,
                "risk_pct": None,
                "alpha_vs_imoex_pct": None,
                "time_of_day": _time_of_day_bucket(lot["entry_ts"]),
                "rationale": str(decision.get("rationale") or "")[:400],
                "would_have_been_better": None,
                "open": True,
            }
            out.append(rec)
    return out

def _retrospective_better(*, pnl_pct: float, entry_price: float, exit_price: float) -> str | None:
    """Simple sanity hint for negative PnL — "would 5min later have helped?"

    We can't access intra-bar data here, so produce a qualitative hint
    only: distinguishes "premature stop" vs "late entry" by sign of the
    price move.
    """
    if pnl_pct >= 0:
        return None
    move = (exit_price - entry_price) / max(1e-9, entry_price)
    if move < -0.002:
        return "entry was too high; consider tighter entry filter"
    return "stop fired prematurely; consider wider SL or different exit logic"

def _aggregate_by(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    """Group by ``rows[key]`` and produce WR / PnL / count / avg_holding."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("open"):
            continue
        if r.get("pnl_rub") is None:
            continue
        groups[str(r.get(key) or "unknown")].append(r)
    out: dict[str, dict[str, Any]] = {}
    for k, items in groups.items():
        wins = sum(1 for r in items if (r.get("pnl_rub") or 0) > 0)
        losses = sum(1 for r in items if (r.get("pnl_rub") or 0) < 0)
        n = len(items)
        total_pnl = sum(float(r.get("pnl_rub") or 0.0) for r in items)
        avg_holding = (
            statistics.mean(
                [
                    float(r.get("holding_min") or 0.0)
                    for r in items
                    if r.get("holding_min") is not None
                ]
            )
            if any(r.get("holding_min") is not None for r in items)
            else 0.0
        )
        alphas = [
            float(r["alpha_vs_imoex_pct"]) for r in items if r.get("alpha_vs_imoex_pct") is not None
        ]
        avg_alpha = statistics.mean(alphas) if alphas else None
        out[k] = {
            "n_trades": n,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / n) if n else 0.0,
            "total_pnl_rub": round(total_pnl, 2),
            "avg_holding_min": round(avg_holding, 2),
            "avg_alpha_vs_imoex_pct": round(avg_alpha, 4) if avg_alpha is not None else None,
        }
    return out

def _top_n(
    rows: list[dict[str, Any]],
    n: int,
    reverse: bool = True,
) -> list[dict[str, Any]]:
    """Top n."""
    closed = [r for r in rows if not r.get("open") and r.get("pnl_rub") is not None]
    closed.sort(key=lambda r: float(r["pnl_rub"]), reverse=reverse)
    return closed[:n]

def _summary_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary block."""
    closed = [r for r in rows if not r.get("open") and r.get("pnl_rub") is not None]
    n = len(closed)
    if n == 0:
        return {
            "n_round_trips": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl_rub": 0.0,
            "avg_pnl_pct": 0.0,
            "best_trade_rub": 0.0,
            "worst_trade_rub": 0.0,
            "avg_holding_min": 0.0,
            "avg_alpha_vs_imoex_pct": None,
        }
    wins = sum(1 for r in closed if r["pnl_rub"] > 0)
    losses = sum(1 for r in closed if r["pnl_rub"] < 0)
    total_pnl = sum(float(r["pnl_rub"]) for r in closed)
    pnl_pcts = [float(r["pnl_pct"]) for r in closed if r.get("pnl_pct") is not None]
    holds = [float(r["holding_min"]) for r in closed if r.get("holding_min") is not None]
    alphas = [
        float(r["alpha_vs_imoex_pct"]) for r in closed if r.get("alpha_vs_imoex_pct") is not None
    ]
    return {
        "n_round_trips": n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / n, 4) if n else 0.0,
        "total_pnl_rub": round(total_pnl, 2),
        "avg_pnl_pct": round(statistics.mean(pnl_pcts), 4) if pnl_pcts else 0.0,
        "best_trade_rub": round(max(r["pnl_rub"] for r in closed), 2),
        "worst_trade_rub": round(min(r["pnl_rub"] for r in closed), 2),
        "avg_holding_min": round(statistics.mean(holds), 2) if holds else 0.0,
        "avg_alpha_vs_imoex_pct": round(statistics.mean(alphas), 4) if alphas else None,
    }

async def _llm_narrative(analysis: dict[str, Any]) -> str:
    """Ask polza-flash for a 3-5 bullet pattern summary. Falls back to ''
    if the LLM is disabled or returns an error.
    """
    if cfg.DISABLE_LLM:
        return ""
    try:
        from app.llm.polza_client import get_polza_client  # noqa: WPS433

        client = get_polza_client()
        if not client._started:
            await client.startup()
        stats = {
            "summary": analysis.get("summary"),
            "by_ticker": analysis.get("by_ticker"),
            "by_detector": analysis.get("by_detector"),
            "by_strategy": analysis.get("by_strategy"),
            "by_tier": analysis.get("by_tier"),
            "by_regime": analysis.get("by_regime"),
            "by_time_of_day": analysis.get("by_time_of_day"),
            "top_winners": analysis.get("top_winners"),
            "top_losers": analysis.get("top_losers"),
        }
        prompt = (
            "Ниже статистика торгового дня MOEX-бота. Найди 3–5 закономерностей "
            "(что работало, что нет) и предложи конкретные правки к завтрашней сессии. "
            "Не общие фразы, а адресно: тикер / детектор / режим / время дня.\n\n"
            f"СТАТИСТИКА:\n{json.dumps(stats, ensure_ascii=False, indent=2)}\n\n"
            'Верни JSON: {"narrative": "<text>", "recommendations": ["..."]}'
        )
        resp = await client.chat_json(
            messages=[
                {"role": "system", "content": "Ты — старший трейдер MOEX. Отвечай только JSON."},
                {"role": "user", "content": prompt},
            ],
            model=cfg.POLZA_MODEL_REACTIVE,
            max_tokens=900,
            purpose="daily_trade_analyzer_narrative",
        )
        parsed = resp.get("parsed") if isinstance(resp, dict) else None
        if not isinstance(parsed, dict):
            return ""
        narrative = str(parsed.get("narrative") or "")
        recs = parsed.get("recommendations") or []
        if recs and isinstance(recs, list):
            narrative += "\n\nRecommendations:\n" + "\n".join(
                f"  - {str(r)[:240]}" for r in recs[:8]
            )
        return narrative.strip()
    except Exception as exc:
        logger.warning(
            "daily_trade_analyzer: LLM narrative failed",
            extra={"error": str(exc)},
        )
        return ""

def _render_markdown(analysis: dict[str, Any]) -> str:
    """Render markdown."""
    date = analysis["date"]
    summary = analysis["summary"]
    lines: list[str] = []
    lines.append(f"# Post-mortem: {date}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Round-trips: **{summary['n_round_trips']}**  "
        f"(wins {summary['wins']} / losses {summary['losses']})"
    )
    lines.append(f"- Win-rate: **{summary['win_rate'] * 100:.1f}%**")
    lines.append(f"- Total PnL: **{summary['total_pnl_rub']:+.2f} ₽**")
    lines.append(f"- Avg PnL: {summary['avg_pnl_pct']:+.3f}%")
    lines.append(
        f"- Best / Worst: {summary['best_trade_rub']:+.2f} ₽ / {summary['worst_trade_rub']:+.2f} ₽"
    )
    lines.append(f"- Avg holding: {summary['avg_holding_min']:.1f} min")
    if summary.get("avg_alpha_vs_imoex_pct") is not None:
        lines.append(f"- Avg alpha vs IMOEX: {summary['avg_alpha_vs_imoex_pct']:+.3f}%")
    lines.append("")
    lines.append(f"- Open positions at EOD: {analysis.get('n_open', 0)}")
    lines.append(f"- Decisions logged: {analysis.get('n_decisions', 0)}")
    lines.append("")

    def _table(title: str, agg: dict[str, dict[str, Any]]) -> None:
        """Table."""
        if not agg:
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| key | n | WR | PnL ₽ | avg hold min | α vs IMOEX |")
        lines.append("|---|---|---|---|---|---|")
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1]["total_pnl_rub"]):
            alpha = v.get("avg_alpha_vs_imoex_pct")
            alpha_s = f"{alpha:+.3f}%" if alpha is not None else "—"
            lines.append(
                f"| {k} | {v['n_trades']} | "
                f"{v['win_rate'] * 100:.1f}% | "
                f"{v['total_pnl_rub']:+.2f} | "
                f"{v['avg_holding_min']:.1f} | {alpha_s} |"
            )
        lines.append("")

    _table("By ticker", analysis.get("by_ticker", {}))
    _table("By source strategy", analysis.get("by_strategy", {}))
    _table("By detector", analysis.get("by_detector", {}))
    _table("By tier", analysis.get("by_tier", {}))
    _table("By HMM regime", analysis.get("by_regime", {}))
    _table("By time-of-day", analysis.get("by_time_of_day", {}))
    _table("By direction", analysis.get("by_direction", {}))

    def _topn_md(title: str, rows: list[dict[str, Any]]) -> None:
        """Topn md."""
        if not rows:
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            "| trade_id | ticker | detector | tier | regime | PnL ₽ | PnL % | hold min | α |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            alpha = r.get("alpha_vs_imoex_pct")
            alpha_s = f"{alpha:+.2f}%" if alpha is not None else "—"
            lines.append(
                f"| {r['trade_id']} | {r['ticker']} | {r['detector']} | "
                f"{r['tier']} | {r['hmm_regime_at_entry']} | "
                f"{r['pnl_rub']:+.2f} | {(r.get('pnl_pct') or 0):+.2f}% | "
                f"{(r.get('holding_min') or 0):.1f} | {alpha_s} |"
            )
        lines.append("")

    _topn_md("Top-5 winners", analysis.get("top_winners", []))
    _topn_md("Top-5 losers", analysis.get("top_losers", []))

    narrative = analysis.get("llm_narrative") or ""
    if narrative:
        lines.append("## LLM narrative")
        lines.append("")
        lines.append(narrative)
        lines.append("")

    return "\n".join(lines) + "\n"

def build_analysis(
    date_str: str,
    *,
    decisions_db: Path = DECISIONS_DB,
    trades_db: Path = TRADES_DB,
    imoex_curve: dict[str, float] | None = None,
    skip_imoex: bool = False,
) -> dict[str, Any]:
    """Pure builder — no I/O beyond the two DB reads and optional IMOEX
    fetch. Returns the full analysis dict.
    """
    trades = _load_trades(trades_db, date_str)
    decisions = _load_decisions(decisions_db, date_str)
    if imoex_curve is None and not skip_imoex and trades:
        imoex_curve = _fetch_imoex_curve(date_str)
    round_trips = _build_round_trips(trades, decisions, date_str, imoex_curve)

    analysis: dict[str, Any] = {
        "date": date_str,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "n_trades_raw": len(trades),
        "n_decisions": len(decisions),
        "n_round_trips": sum(1 for r in round_trips if not r.get("open")),
        "n_open": sum(1 for r in round_trips if r.get("open")),
        "trades": round_trips,
        "summary": _summary_block(round_trips),
        "by_ticker": _aggregate_by(round_trips, "ticker"),
        "by_strategy": _aggregate_by(round_trips, "source_strategy"),
        "by_detector": _aggregate_by(round_trips, "detector"),
        "by_tier": _aggregate_by(round_trips, "tier"),
        "by_regime": _aggregate_by(round_trips, "hmm_regime_at_entry"),
        "by_time_of_day": _aggregate_by(round_trips, "time_of_day"),
        "by_direction": _aggregate_by(round_trips, "direction"),
        "top_winners": _top_n(round_trips, 5, reverse=True),
        "top_losers": _top_n(round_trips, 5, reverse=False),
    }
    return analysis

async def run(
    *,
    date_str: str,
    write_rag: bool,
    output_json: Path,
    output_md: Path,
    skip_llm: bool,
    skip_imoex: bool,
) -> dict[str, Any]:
    """Run."""
    analysis = build_analysis(date_str, skip_imoex=skip_imoex)

    if not skip_llm and analysis["n_round_trips"] > 0:
        narrative = await _llm_narrative(analysis)
        if narrative:
            analysis["llm_narrative"] = narrative

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, default=str))
    output_md.write_text(_render_markdown(analysis))
    logger.info(
        "daily_trade_analyzer: wrote outputs",
        extra={
            "date": date_str,
            "json": str(output_json),
            "md": str(output_md),
            "n_round_trips": analysis["n_round_trips"],
        },
    )

    if write_rag and analysis["n_round_trips"] > 0:
        try:
            from app.memory.rag_store import get_rag_store
            from app.memory.trade_outcome_rag import TradeOutcomeIndexer

            rag = get_rag_store()
            indexer = TradeOutcomeIndexer(rag)
            n = await indexer.index_daily_trades(date_str, analysis)
            analysis["rag_indexed"] = n
            logger.info(
                "daily_trade_analyzer: indexed trade outcomes into RAG",
                extra={"date": date_str, "indexed": n},
            )
        except Exception as exc:
            logger.warning(
                "daily_trade_analyzer: RAG indexing failed",
                extra={"error": str(exc)},
            )
    return analysis

def _print_console_topn(analysis: dict[str, Any]) -> None:
    """Print console topn."""
    summary = analysis["summary"]
    print(f"=== Daily analysis: {analysis['date']} ===")
    print(
        f"Round-trips: {summary['n_round_trips']}  "
        f"WR: {summary['win_rate'] * 100:.1f}%  "
        f"PnL: {summary['total_pnl_rub']:+.2f} ₽  "
        f"Open: {analysis.get('n_open', 0)}"
    )
    print()
    print("Top winners:")
    for r in analysis.get("top_winners", []):
        print(
            f"  {r['ticker']:<6s} {r['detector']:<20s} tier={r['tier']:<4s} "
            f"PnL={r['pnl_rub']:+.2f}₽ ({(r.get('pnl_pct') or 0):+.2f}%)"
        )
    print()
    print("Top losers:")
    for r in analysis.get("top_losers", []):
        print(
            f"  {r['ticker']:<6s} {r['detector']:<20s} tier={r['tier']:<4s} "
            f"PnL={r['pnl_rub']:+.2f}₽ ({(r.get('pnl_pct') or 0):+.2f}%)"
        )

def _default_date_str() -> str:
    """Default date str."""
    yesterday = datetime.now(tz=UTC).date() - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")

def main(argv: Iterable[str] | None = None) -> int:
    """Main."""
    parser = argparse.ArgumentParser(description="Daily trade post-mortem analyzer")
    parser.add_argument(
        "--date",
        default=_default_date_str(),
        help="YYYY-MM-DD (default: yesterday UTC)",
    )
    parser.add_argument(
        "--write-rag",
        action="store_true",
        help="Index each trade into the RAG store for next-day recall.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Override JSON output path (default: data/training_cache/daily_analysis_<date>.json)",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Override Markdown output path (default: data/morning_plans/post_mortem_<date>.md)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the LLM narrative call (Markdown still produced).",
    )
    parser.add_argument(
        "--skip-imoex",
        action="store_true",
        help="Skip IMOEX benchmark fetch (alpha_vs_imoex_pct will be None).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    date_str = args.date
    output_json = Path(args.output_json or (TRAINING_CACHE_DIR / f"daily_analysis_{date_str}.json"))
    output_md = Path(args.output_md or (POST_MORTEM_DIR / f"post_mortem_{date_str}.md"))
    try:
        analysis = asyncio.run(
            run(
                date_str=date_str,
                write_rag=args.write_rag,
                output_json=output_json,
                output_md=output_md,
                skip_llm=args.skip_llm,
                skip_imoex=args.skip_imoex,
            )
        )
    except KeyboardInterrupt:
        return 130
    _print_console_topn(analysis)
    return 0

if __name__ == "__main__":
    sys.exit(main())
