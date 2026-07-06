"""
scripts/bootstrap_db.py — Create SQLite schemas on first run.
Run once before starting the main app, or call via main.py at startup.

Tables created:
  - decisions.db: decisions, budget_log
  - trades.db: trades, circuit_breaker_state
"""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC
from pathlib import Path

_DEFAULT = Path(__file__).parent.parent / "data"
DATA_DIR = Path(os.getenv("DATA_DIR", str(_DEFAULT)))
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _verify_or_rebuild(db_path: Path) -> None:
    """Chaos-engineering: scrub a corrupt SQLite file at boot.

    If `db_path` exists but `PRAGMA integrity_check` returns anything other
    than "ok", we unlink the file (plus its `-shm` / `-wal` siblings) so the
    create-table calls below build a fresh schema. The alternative — letting
    the bot crash-loop on every read — is worse than losing the trade log:
    we still have ArenaGo's server-side history if we need to reconcile.

    A successful integrity check is silent.
    """
    if not db_path.exists():
        return
    try:
        with sqlite3.connect(str(db_path)) as cn:
            cn.execute("PRAGMA busy_timeout = 2000")
            row = cn.execute("PRAGMA integrity_check").fetchone()
        ok = bool(row) and (row[0] == "ok")
    except sqlite3.DatabaseError:
        ok = False
    if ok:
        return
    print(f" ! integrity_check FAILED on {db_path.name} — rebuilding from scratch")
    for suffix in ("", "-shm", "-wal", "-journal"):
        sibling = db_path.with_name(db_path.name + suffix)
        try:
            if sibling.exists():
                sibling.unlink()
        except OSError as exc:
            print(f"   could not remove {sibling}: {exc}")

def _apply_pragmas(cur: sqlite3.Cursor) -> None:
    """Phase 12 / v0.9.0 — WAL + NORMAL sync + cache_size for concurrent writers.

    All MoexML SQLite DBs are touched by multiple writers (order_manager,
    circuit_breakers, polza_client, recovery loop, turnover_tracker, dashboard).

    - journal_mode=WAL: eliminates "database is locked" under load, ~3x faster
      than rollback journal (writers no longer fsync on every commit).
    - synchronous=NORMAL: safe under WAL, drops fsync from per-commit to
      per-checkpoint (~10x faster writes; durability guarantee = "no torn pages
      on power loss" which is what we want for a trading log).
    - cache_size=-10000: 10 MB page cache per connection. Default is 2 MB which
      forces re-read of hot pages (decisions index, circuit_breaker state) every
      query. Negative value = kibibytes (positive = pages).
    - busy_timeout=5000: 5s wait before SQLITE_BUSY surfaces — enough for any
      WAL checkpoint contention from the dashboard reader.
    """
    cur.execute("PRAGMA journal_mode = WAL")
    cur.execute("PRAGMA busy_timeout = 5000")
    cur.execute("PRAGMA synchronous = NORMAL")
    cur.execute("PRAGMA cache_size = -10000")
    cur.execute("PRAGMA temp_store = MEMORY")

def create_decisions_db() -> None:
    """Create decisions db."""
    db_path = DATA_DIR / "decisions.db"
    _verify_or_rebuild(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    _apply_pragmas(cur)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            decision_id     TEXT PRIMARY KEY,
            cycle_id        TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            action          TEXT NOT NULL,
            tier            TEXT NOT NULL DEFAULT 'NONE',
            direction       TEXT NOT NULL DEFAULT 'NEUTRAL',
            combined_magnitude REAL DEFAULT 0.0,
            risk_check      TEXT NOT NULL DEFAULT 'PASSED',
            stop_loss       REAL,
            take_profit     REAL,
            expected_holding_min INTEGER DEFAULT 0,
            rationale       TEXT DEFAULT '',
            signals_json    TEXT DEFAULT '[]',
            trade_request_json TEXT,
            git_commit      TEXT DEFAULT '',
            executed_bool   INTEGER DEFAULT 0,
            arena_response_json TEXT,
            pnl_rub         REAL,
            reflection_status TEXT DEFAULT 'PENDING',
            created_at      TEXT NOT NULL,
            executed_at     TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS budget_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT NOT NULL,
            model           TEXT NOT NULL,
            input_tokens    INTEGER DEFAULT 0,
            output_tokens   INTEGER DEFAULT 0,
            cost_rub        REAL DEFAULT 0.0,
            cumulative_rub  REAL DEFAULT 0.0,
            purpose         TEXT DEFAULT ''
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prompt_cache (
            cache_key       TEXT PRIMARY KEY,
            model           TEXT NOT NULL,
            response_json   TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            expires_at      TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS phase_status (
            phase           TEXT PRIMARY KEY,
            status          TEXT DEFAULT 'TODO',
            updated_at      TEXT NOT NULL,
            notes           TEXT DEFAULT ''
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_decisions_created_at
        ON decisions(created_at DESC)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_decisions_ticker_created
        ON decisions(ticker, created_at)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_budget_log_ts
        ON budget_log(ts)
    """)

    conn.commit()
    conn.close()
    print(f" decisions.db created at {db_path}")

def create_trades_db() -> None:
    """Create trades db."""
    db_path = DATA_DIR / "trades.db"
    _verify_or_rebuild(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    _apply_pragmas(cur)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id     TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            direction       TEXT NOT NULL,
            quantity        INTEGER NOT NULL,
            price           REAL NOT NULL,
            order_value     REAL NOT NULL,
            remaining_cash  REAL NOT NULL,
            trade_date      TEXT NOT NULL,
            trade_time      TEXT NOT NULL,
            bot             TEXT NOT NULL,
            source_model    TEXT DEFAULT '',
            arena_raw_json  TEXT DEFAULT '',
            created_at      TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS circuit_breaker_state (
            id              INTEGER PRIMARY KEY CHECK (id = 1),
            daily_pnl_rub   REAL DEFAULT 0.0,
            peak_equity_rub REAL DEFAULT 1000000.0,
            max_drawdown_pct REAL DEFAULT 0.0,
            losing_streak   INTEGER DEFAULT 0,
            winning_streak  INTEGER DEFAULT 0,
            blocked_until   TEXT,
            block_reason    TEXT DEFAULT '',
            updated_at      TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS turnover_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            daily_volume_rub REAL DEFAULT 0.0,
            cumulative_volume_rub REAL DEFAULT 0.0,
            trade_count     INTEGER DEFAULT 0,
            updated_at      TEXT NOT NULL
        )
    """)

    from datetime import datetime

    now_str = datetime.now(tz=UTC).isoformat()
    cur.execute(
        """
        INSERT OR IGNORE INTO circuit_breaker_state
        (id, daily_pnl_rub, peak_equity_rub, max_drawdown_pct,
         losing_streak, winning_streak, blocked_until, block_reason, updated_at)
        VALUES (1, 0.0, 1000000.0, 0.0, 0, 0, NULL, '', ?)
    """,
        (now_str,),
    )

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_date_time
        ON trades(trade_date, trade_time)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_ticker_date
        ON trades(ticker, trade_date)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_decision_id
        ON trades(decision_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_turnover_log_date
        ON turnover_log(date)
    """)

    conn.commit()
    conn.close()
    print(f" trades.db created at {db_path}")

def create_feeds_db() -> None:
    """Numeric data feeds cache (yfinance, FRED, CBR, EIA)."""
    db_path = DATA_DIR / "feeds.db"
    _verify_or_rebuild(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    _apply_pragmas(cur)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            symbol          TEXT NOT NULL,
            ts              TEXT NOT NULL,
            data_type       TEXT NOT NULL,   -- candle | commodity | fx | index
            open            REAL,
            high            REAL,
            low             REAL,
            close           REAL,
            volume          REAL,
            source          TEXT DEFAULT '',
            PRIMARY KEY (symbol, ts, data_type)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS macro_data (
            series_id       TEXT NOT NULL,
            date            TEXT NOT NULL,
            value           REAL,
            source          TEXT DEFAULT '',
            updated_at      TEXT NOT NULL,
            PRIMARY KEY (series_id, date)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_events (
            event_id         TEXT PRIMARY KEY,
            ts_utc           TEXT NOT NULL,
            source           TEXT NOT NULL,
            source_tier      TEXT NOT NULL,
            headline         TEXT NOT NULL,
            body             TEXT DEFAULT '',
            url              TEXT DEFAULT '',
            language         TEXT DEFAULT 'ru',
            tickers_json     TEXT DEFAULT '[]',
            matched_keywords_json TEXT DEFAULT '[]',
            event_type       TEXT DEFAULT 'other',
            is_material      INTEGER DEFAULT 0,
            is_sanctions     INTEGER DEFAULT 0,
            text_norm        TEXT DEFAULT '',
            raw_payload_json TEXT DEFAULT '{}',
            llm_direction    TEXT DEFAULT 'NEUTRAL',
            llm_magnitude    REAL DEFAULT 0.0,
            horizon_min      INTEGER DEFAULT 0,
            reason           TEXT DEFAULT '',
            created_at       TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_reactions (
            event_id         TEXT NOT NULL,
            ticker           TEXT NOT NULL,
            window_min       INTEGER NOT NULL,
            price_t0         REAL DEFAULT 0.0,
            price_tn         REAL DEFAULT 0.0,
            return_pct       REAL DEFAULT 0.0,
            abs_move_pct     REAL DEFAULT 0.0,
            created_at       TEXT NOT NULL,
            PRIMARY KEY (event_id, ticker, window_min)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_context_snapshots (
            event_id         TEXT NOT NULL,
            ticker           TEXT NOT NULL,
            price            REAL DEFAULT 0.0,
            atr              REAL DEFAULT 0.0,
            atr_pct          REAL DEFAULT 0.0,
            rsi              REAL DEFAULT 0.0,
            vol_z            REAL DEFAULT 0.0,
            ret_30m_pct      REAL DEFAULT 0.0,
            regime           TEXT DEFAULT 'unknown',
            regime_proba_json TEXT DEFAULT '{}',
            catboost_score   REAL DEFAULT 0.0,
            ta_pattern       TEXT DEFAULT '',
            ta_direction     TEXT DEFAULT '',
            ta_expected_rr   REAL DEFAULT 0.0,
            historical_bias  REAL DEFAULT 0.0,
            retrieval_cases  INTEGER DEFAULT 0,
            context_json     TEXT DEFAULT '{}',
            created_at       TEXT NOT NULL,
            PRIMARY KEY (event_id, ticker)
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_news_events_ts
        ON news_events(ts_utc DESC)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_news_events_source_ts
        ON news_events(source, ts_utc)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_market_data_symbol_ts
        ON market_data(symbol, ts DESC)
    """)

    conn.commit()
    conn.close()
    print(f" feeds.db created at {db_path}")

def main() -> None:
    """Main."""
    print("Bootstrapping MoexML databases...")
    create_decisions_db()
    create_trades_db()
    create_feeds_db()
    print(" All databases ready.")

if __name__ == "__main__":
    main()
