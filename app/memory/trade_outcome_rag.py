"""Trade-outcome indexer for the RAG store.

Phase 27.5 — vectorise post-mortem trade results into the same RAG store
that the morning consensus + reactive comparator already use, so that
``ConsensusComparator`` (and the morning planner) can recall *similar
historical setups* with their actual win/loss outcome when scoring a
fresh signal.

Why reuse the news RAG store
----------------------------
The existing :class:`app.memory.rag_store.RAGStore` already provides:

* persistent embedding (chromadb or hash+numpy fallback)
* cosine similarity search with ticker + recency filters
* ticker-tagged retrieval

We piggy-back on it by writing trade outcomes through the same
``add_news`` entry point with ``source="trade_post_mortem"`` and a
``[OUTCOME]`` prefix in the event id, so a downstream consumer can filter
``type == "trade_outcome"`` by sniffing the source / event_id prefix.
The "metadata" payload travels inside the indexed text itself plus the
``source`` field — search results round-trip those values.

Design choices
--------------
* Indexer is fully synchronous (in keeping with RAGStore's design — the
  embedding step blocks the GIL for ~10ms per record at the 384-d MiniLM
  layer, so async wrappers add overhead with no parallelism gain).
* ``index_daily_trades`` is idempotent on ``trade_id`` thanks to the
  RAGStore dedup-by-event-id contract.
* We never raise — partial indexing is preferable to a full daily
  pipeline failure.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from app.memory.rag_store import RAGStore
from app.utils.logging import get_logger

logger = get_logger(__name__)

TRADE_OUTCOME_EVENT_PREFIX = "trade_outcome::"
TRADE_OUTCOME_SOURCE = "trade_post_mortem"

def _coerce_ts(ts: Any) -> datetime:
    """Best-effort conversion of arbitrary timestamp inputs to UTC datetime."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=UTC)
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts), tz=UTC)
        except (OSError, ValueError, OverflowError):
            return datetime.now(tz=UTC)
    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            return datetime.now(tz=UTC)
        s = s.replace(" ", "T") if "T" not in s and ":" in s else s
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            return datetime.now(tz=UTC)
    return datetime.now(tz=UTC)

def _format_trade_text(trade: dict[str, Any]) -> str:
    """Compose the human-readable text we feed to the embedder.

    The phrasing intentionally mirrors the *queries* a comparator will
    issue ("Ticker SBER BUY in trending-режиме, detector double_top,
    tier A, magnitude 0.72") so cosine similarity is maximised on
    semantically meaningful tokens.
    """
    ticker = str(trade.get("ticker") or "").upper() or "UNK"
    direction = str(trade.get("direction") or "").upper() or "?"
    hmm_regime = str(trade.get("hmm_regime_at_entry") or "unknown")
    detector = str(trade.get("detector") or "n/a")
    tier = str(trade.get("tier") or "?")
    magnitude = trade.get("magnitude")
    try:
        magnitude_f = float(magnitude) if magnitude is not None else 0.0
    except (TypeError, ValueError):
        magnitude_f = 0.0
    pnl_pct = trade.get("pnl_pct")
    try:
        pnl_pct_f = float(pnl_pct) if pnl_pct is not None else 0.0
    except (TypeError, ValueError):
        pnl_pct_f = 0.0
    exit_reason = str(trade.get("exit_reason") or "manual")
    rationale = str(trade.get("rationale") or "")[:280]
    alpha = trade.get("alpha_vs_imoex_pct")
    try:
        alpha_f = float(alpha) if alpha is not None else 0.0
    except (TypeError, ValueError):
        alpha_f = 0.0
    source_strategy = str(trade.get("source_strategy") or "n/a")
    outcome = "win" if pnl_pct_f > 0 else ("loss" if pnl_pct_f < 0 else "flat")
    return (
        f"[OUTCOME] Ticker {ticker} {direction} в {hmm_regime}-режиме, "
        f"detector {detector}, tier {tier}, magnitude {magnitude_f:.2f}, "
        f"strategy {source_strategy}. {exit_reason}. PnL: {pnl_pct_f:.2f}% "
        f"({outcome}). Alpha vs IMOEX: {alpha_f:.2f}%. Rationale: {rationale}"
    )

class TradeOutcomeIndexer:
    """Persist per-trade post-mortems into the RAG store.

    Parameters
    ----------
    rag:
        Live :class:`RAGStore` instance — typically the process-wide
        singleton from ``get_rag_store()``.
    """

    def __init__(self, rag: RAGStore) -> None:
        """Init."""
        self.rag = rag
        self._indexed = 0
        self._skipped = 0
        self._failed = 0

    async def index_daily_trades(
        self,
        date: str,
        analysis: dict[str, Any],
    ) -> int:
        """Index every per-trade record from a daily analysis dict.

        Expected ``analysis`` shape (subset)::

            {
                "date": "2026-05-26",
                "trades": [
                    {
                        "trade_id": "...",
                        "ticker": "SBER",
                        "direction": "BUY",
                        "exit_ts": "2026-05-26T15:30:00+00:00",
                        "pnl_pct": 1.23,
                        "rationale": "...",
                        "detector": "double_top",
                        "tier": "A",
                        "hmm_regime_at_entry": "trending",
                        ...
                    },
                    ...
                ],
            }

        Returns the number of trades successfully indexed.
        """
        trades = analysis.get("trades") or []
        if not isinstance(trades, list) or not trades:
            logger.info(
                "TradeOutcomeIndexer: no trades to index",
                extra={"date": date},
            )
            return 0
        count = 0
        for tr in trades:
            if not isinstance(tr, dict):
                self._skipped += 1
                continue
            try:
                ok = self._index_one(date=date, trade=tr)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "TradeOutcomeIndexer: failed to index trade",
                    extra={"error": str(exc), "trade_id": tr.get("trade_id")},
                )
                self._failed += 1
                continue
            if ok:
                count += 1
                self._indexed += 1
            else:
                self._skipped += 1
        logger.info(
            "TradeOutcomeIndexer: indexed daily trades",
            extra={
                "date": date,
                "indexed": count,
                "skipped": self._skipped,
                "failed": self._failed,
                "rag_size": len(self.rag),
            },
        )
        return count

    def _index_one(self, *, date: str, trade: dict[str, Any]) -> bool:
        """Index a single trade. Returns True on success.

        Side-effects: writes one record into the RAG store with a
        ``trade_outcome::`` event-id prefix so callers can filter on
        ``startswith()`` or sniff ``source == 'trade_post_mortem'``.
        """
        trade_id = trade.get("trade_id")
        if not trade_id:
            return False
        ticker = str(trade.get("ticker") or "").upper()
        if not ticker:
            return False
        ts = _coerce_ts(trade.get("exit_ts") or trade.get("entry_ts"))
        text = _format_trade_text(trade)
        try:
            pnl_pct_f = float(trade.get("pnl_pct") or 0.0)
        except (TypeError, ValueError):
            pnl_pct_f = 0.0
        outcome = "win" if pnl_pct_f > 0 else ("loss" if pnl_pct_f < 0 else "flat")
        headline = (
            f"{ticker} {trade.get('direction', '?')} "
            f"{trade.get('detector', 'n/a')} → {outcome} ({pnl_pct_f:+.2f}%)"
        )[:200]
        event_id = f"{TRADE_OUTCOME_EVENT_PREFIX}{date}::{trade_id}"
        body_payload = _format_body_payload(date=date, outcome=outcome, trade=trade)
        try:
            self.rag.add_news(
                event_id=event_id,
                text=text,
                ts_utc=ts,
                tickers=[ticker],
                source=TRADE_OUTCOME_SOURCE,
                source_tier="F",
                headline=headline,
                body=body_payload,
            )
        except Exception as exc:
            logger.warning(
                "TradeOutcomeIndexer: rag.add_news failed",
                extra={"error": str(exc), "event_id": event_id},
            )
            return False
        return True

    def stats(self) -> dict[str, int]:
        """Stats."""
        return {
            "indexed": self._indexed,
            "skipped": self._skipped,
            "failed": self._failed,
        }

def _format_body_payload(*, date: str, outcome: str, trade: dict[str, Any]) -> str:
    """Render a compact key=value block we can stuff into ``body``.

    The RAG store stores the body verbatim; we keep it short so callers
    looking at search results can grep it for, e.g., ``outcome=win`` or
    ``hmm_regime=trending`` without doing JSON.
    """
    parts: list[str] = [
        "type=trade_outcome",
        f"date={date}",
        f"outcome={outcome}",
    ]
    for key in (
        "ticker",
        "direction",
        "detector",
        "tier",
        "source_strategy",
        "hmm_regime_at_entry",
        "magnitude",
        "expected_rr",
        "actual_rr",
        "pnl_pct",
        "pnl_rub",
        "alpha_vs_imoex_pct",
        "exit_reason",
        "holding_min",
    ):
        val = trade.get(key)
        if val is None:
            continue
        if isinstance(val, float):
            parts.append(f"{key}={val:.4f}")
        else:
            parts.append(f"{key}={val}")
    return "; ".join(parts)

def filter_trade_outcomes(
    results: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only records that look like trade outcomes.

    Filtering on either the event_id prefix OR the source label makes
    this resilient to legacy entries that may have been indexed with a
    slightly different convention.
    """
    out: list[dict[str, Any]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        eid = str(r.get("event_id") or "")
        src = str(r.get("source") or "")
        if eid.startswith(TRADE_OUTCOME_EVENT_PREFIX) or src == TRADE_OUTCOME_SOURCE:
            out.append(r)
    return out

def parse_outcome_from_body(body: str) -> dict[str, Any]:
    """Parse the ``k=v; k=v`` body we wrote in :func:`_format_body_payload`.

    Returns an empty dict on any failure — never raises.
    """
    out: dict[str, Any] = {}
    if not body or not isinstance(body, str):
        return out
    for chunk in body.split(";"):
        chunk = chunk.strip()
        if "=" not in chunk:
            continue
        k, _, v = chunk.partition("=")
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        if v in ("True", "False"):
            out[k] = v == "True"
            continue
        try:
            if "." in v or "e" in v.lower():
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v
    return out

__all__ = [
    "TradeOutcomeIndexer",
    "TRADE_OUTCOME_EVENT_PREFIX",
    "TRADE_OUTCOME_SOURCE",
    "filter_trade_outcomes",
    "parse_outcome_from_body",
]
