"""Phase 27.5 — daily adaptive noise review.

Reads the last NOISE_LOOKBACK_DAYS of trades from data/trades.db (and
optional decisions.db for live ledger), computes rolling per-detector win
rate, then writes any pattern whose WR < NOISE_WR_THRESHOLD with at least
NOISE_MIN_TRADES samples to data/runtime_overrides.json.

Run manually:
    python3 scripts/noise_review.py            # default 7d
    python3 scripts/noise_review.py --days 14  # override
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.config as cfg  # noqa: E402
from app.agents.ta_patterns.noise_blacklist import (  # noqa: E402
    STATIC_NOISE_PATTERNS,
    update_dynamic_overrides,
)

DECISIONS_DB = cfg.DATA_DIR / "decisions.db"

def _load_trades_with_pnl(days: int) -> list[tuple[str, float]]:
    """Pull (detector_name, pnl_pct) tuples from decisions.db within window.

    The decisions table stores `signals` as JSON and `pnl_rub`. We translate
    rub pnl into pct using the order notional captured by RiskManager. When
    pnl is unavailable we skip the row.
    """
    if not DECISIONS_DB.exists():
        return []
    cutoff = (datetime.now(tz=UTC) - timedelta(days=days)).isoformat()
    rows: list[tuple[str, float]] = []
    try:
        con = sqlite3.connect(str(DECISIONS_DB))
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        for r in cur.execute(
            """
            SELECT signals_json, pnl_rub, trade_request_json
            FROM decisions
            WHERE created_at >= ? AND pnl_rub IS NOT NULL
            """,
            (cutoff,),
        ):
            try:
                signals = json.loads(r["signals_json"] or "[]")
            except Exception:
                continue
            try:
                tr = json.loads(r["trade_request_json"] or "{}")
            except Exception:
                tr = {}
            pnl = float(r["pnl_rub"] or 0.0)
            qty = float(tr.get("quantity") or 0.0)
            price = float(tr.get("price_at_signal") or 0.0)
            notional = qty * price
            if notional <= 0:
                continue
            pct = pnl / notional
            for sig in signals:
                det = (sig.get("detector") or "").lower()
                if not det:
                    continue
                rows.append((det, pct))
        con.close()
    except Exception as exc:
        print(f"  [warn] decisions.db read failed: {exc}")
    return rows

def review(days: int) -> list[str]:
    """Identify noisy patterns based on the last `days` of decision pnl."""
    trades = _load_trades_with_pnl(days)
    if not trades:
        print(f"No graded decisions in last {days} days — keeping current overrides")
        return list(STATIC_NOISE_PATTERNS)
    by_det: dict[str, list[float]] = {}
    for det, pnl in trades:
        by_det.setdefault(det, []).append(pnl)
    noisy: list[str] = []
    print(f"\nDetector WR over last {days}d:")
    for det in sorted(by_det):
        pnls = by_det[det]
        n = len(pnls)
        if n < cfg.NOISE_MIN_TRADES:
            continue
        wr = sum(1 for p in pnls if p > 0) / n
        marker = ""
        if wr < cfg.NOISE_WR_THRESHOLD:
            noisy.append(det)
            marker = " ← NOISE"
        print(f"  {det:38s} N={n:4d}  WR={wr:.3f}{marker}")
    merged = sorted(set(STATIC_NOISE_PATTERNS) | set(noisy))
    return merged

def main() -> int:
    """Main."""
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=cfg.NOISE_LOOKBACK_DAYS)
    args = p.parse_args()
    patterns = review(args.days)
    update_dynamic_overrides(patterns)
    print(f"\nWrote {len(patterns)} patterns to {cfg.DATA_DIR / 'runtime_overrides.json'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
