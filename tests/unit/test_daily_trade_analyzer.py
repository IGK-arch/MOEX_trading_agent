"""Unit tests for scripts/daily_trade_analyzer.py.

We exercise the pure ``build_analysis`` builder against a hand-built
SQLite fixture so we don't depend on the live trades.db.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "daily_trade_analyzer.py"


@pytest.fixture(scope="module")
def dta_module():
    """Load daily_trade_analyzer.py as a module (it lives in scripts/)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "daily_trade_analyzer",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["daily_trade_analyzer"] = mod
    spec.loader.exec_module(mod)
    return mod


def _seed_decisions(db_path: Path, rows: list[dict]) -> None:
    """Recreate decisions.db with the schema (mirrors prod) and insert
    ``rows``. Note that ``executed_at`` is part of the production schema,
    so the analyzer SELECTs it — we must declare it here too."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
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
            """
        )
        for r in rows:
            conn.execute(
                """INSERT INTO decisions
                (decision_id, cycle_id, ticker, action, tier, direction,
                 combined_magnitude, risk_check, stop_loss, take_profit,
                 expected_holding_min, rationale, signals_json,
                 executed_bool, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    r["decision_id"],
                    r["cycle_id"],
                    r["ticker"],
                    r["action"],
                    r.get("tier", "A"),
                    r.get("direction", "BUY"),
                    r.get("combined_magnitude", 0.7),
                    r.get("risk_check", "PASSED"),
                    r.get("stop_loss"),
                    r.get("take_profit"),
                    r.get("expected_holding_min", 60),
                    r.get("rationale", ""),
                    json.dumps(r.get("signals", [])),
                    1,
                    r["created_at"],
                ),
            )
        conn.commit()


def _seed_trades(db_path: Path, rows: list[dict]) -> None:
    """Seed trades."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
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
            """
        )
        for r in rows:
            conn.execute(
                """INSERT INTO trades
                (decision_id, ticker, direction, quantity, price,
                 order_value, remaining_cash, trade_date, trade_time,
                 bot, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    r["decision_id"],
                    r["ticker"],
                    r["direction"],
                    int(r["quantity"]),
                    float(r["price"]),
                    float(r["order_value"]),
                    float(r.get("remaining_cash", 1_000_000)),
                    r["trade_date"],
                    r["trade_time"],
                    r.get("bot", "test"),
                    r.get("created_at", datetime.now(tz=UTC).isoformat()),
                ),
            )
        conn.commit()


def _make_round_trip(
    *,
    ticker: str,
    entry_time: str,
    exit_time: str,
    entry_price: float,
    exit_price: float,
    qty: int,
    detector: str = "double_top",
    tier: str = "A",
    source_strategy: str = "TA",
    regime: str = "trending",
) -> tuple[list[dict], list[dict]]:
    """Return (decisions, trades) for a BUY+SELL pair forming one round-trip."""
    buy_id = f"d_buy_{uuid4().hex[:8]}"
    sell_id = f"d_sell_{uuid4().hex[:8]}"
    date_str = "2026-05-26"
    sig_buy = {
        "source": source_strategy,
        "detector": detector,
        "ticker": ticker,
        "direction": "BUY",
        "magnitude": 0.72,
        "raw_confidence": 0.81,
        "expected_rr": 2.5,
        "metadata": {"hmm_regime": regime},
    }
    decisions = [
        {
            "decision_id": buy_id,
            "cycle_id": "c1",
            "ticker": ticker,
            "action": "EXECUTE",
            "tier": tier,
            "direction": "BUY",
            "combined_magnitude": 0.72,
            "stop_loss": entry_price * 0.99,
            "take_profit": entry_price * 1.03,
            "rationale": f"{ticker} long on {detector}",
            "signals": [sig_buy],
            "created_at": f"{date_str}T{entry_time}+00:00",
        },
        {
            "decision_id": sell_id,
            "cycle_id": "c1",
            "ticker": ticker,
            "action": "EXECUTE",
            "tier": tier,
            "direction": "SELL",
            "combined_magnitude": 0.6,
            "rationale": f"{ticker} exit",
            "signals": [{"source": source_strategy, "detector": "exit", "direction": "SELL"}],
            "created_at": f"{date_str}T{exit_time}+00:00",
        },
    ]
    trades = [
        {
            "decision_id": buy_id,
            "ticker": ticker,
            "direction": "BUY",
            "quantity": qty,
            "price": entry_price,
            "order_value": entry_price * qty,
            "trade_date": date_str,
            "trade_time": entry_time,
        },
        {
            "decision_id": sell_id,
            "ticker": ticker,
            "direction": "SELL",
            "quantity": qty,
            "price": exit_price,
            "order_value": exit_price * qty,
            "trade_date": date_str,
            "trade_time": exit_time,
        },
    ]
    return decisions, trades


def test_empty_dbs_produce_empty_report(dta_module, tmp_path):
    """No trades, no decisions → zero round-trips, no exception."""
    trades_db = tmp_path / "trades.db"
    decisions_db = tmp_path / "decisions.db"
    _seed_trades(trades_db, [])
    _seed_decisions(decisions_db, [])
    analysis = dta_module.build_analysis(
        "2026-05-26",
        decisions_db=decisions_db,
        trades_db=trades_db,
        skip_imoex=True,
    )
    assert analysis["n_round_trips"] == 0
    assert analysis["trades"] == []
    assert analysis["summary"]["win_rate"] == 0.0


def test_five_round_trips_match_decisions(dta_module, tmp_path):
    """Five matched BUY/SELL pairs → five round-trip records."""
    trades_db = tmp_path / "trades.db"
    decisions_db = tmp_path / "decisions.db"
    all_decisions: list[dict] = []
    all_trades: list[dict] = []
    fixtures = [
        ("SBER", "10:30:00", "11:30:00", 300.0, 305.0, 100, "double_top", "A", "TA", "trending"),
        (
            "GAZP",
            "10:45:00",
            "12:30:00",
            150.0,
            148.0,
            200,
            "ofi_spike",
            "B",
            "ANOMALY",
            "mean_reverting",
        ),
        ("LKOH", "11:15:00", "13:00:00", 7000.0, 7150.0, 5, "harmonic_bat", "A", "TA", "trending"),
        ("ROSN", "12:00:00", "14:00:00", 450.0, 442.0, 50, "vpvr_breakout", "B", "TA", "trending"),
        ("MGNT", "13:45:00", "15:30:00", 5500.0, 5610.0, 8, "smc_choch", "A", "TA", "trending"),
    ]
    for fx in fixtures:
        (ticker, entry_time, exit_time, ep, xp, qty, det, tier, strat, regime) = fx
        decs, trs = _make_round_trip(
            ticker=ticker,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=ep,
            exit_price=xp,
            qty=qty,
            detector=det,
            tier=tier,
            source_strategy=strat,
            regime=regime,
        )
        all_decisions.extend(decs)
        all_trades.extend(trs)
    _seed_decisions(decisions_db, all_decisions)
    _seed_trades(trades_db, all_trades)

    analysis = dta_module.build_analysis(
        "2026-05-26",
        decisions_db=decisions_db,
        trades_db=trades_db,
        skip_imoex=True,
    )
    assert analysis["n_round_trips"] == 5
    assert len(analysis["trades"]) == 5
    sber = next(r for r in analysis["trades"] if r["ticker"] == "SBER")
    assert sber["pnl_rub"] > 0
    assert sber["entry_price"] == 300.0
    assert sber["exit_price"] == 305.0
    assert sber["qty"] == 100
    gazp = next(r for r in analysis["trades"] if r["ticker"] == "GAZP")
    assert gazp["pnl_rub"] < 0
    assert sber["detector"] == "double_top"
    assert sber["tier"] == "A"
    assert sber["source_strategy"] == "TA"
    assert sber["hmm_regime_at_entry"] == "trending"
    assert sber["holding_min"] == pytest.approx(60.0, abs=0.1)


def test_aggregation_per_ticker_is_correct(dta_module, tmp_path):
    """Two trades on SBER, one on GAZP → correct per-ticker counts."""
    trades_db = tmp_path / "trades.db"
    decisions_db = tmp_path / "decisions.db"
    all_decisions, all_trades = [], []
    for entry_time, exit_time, ep, xp in [
        ("10:00:00", "11:00:00", 300.0, 303.0),
        ("11:30:00", "12:30:00", 304.0, 301.0),
    ]:
        decs, trs = _make_round_trip(
            ticker="SBER",
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=ep,
            exit_price=xp,
            qty=100,
        )
        all_decisions.extend(decs)
        all_trades.extend(trs)
    decs, trs = _make_round_trip(
        ticker="GAZP",
        entry_time="13:00:00",
        exit_time="14:00:00",
        entry_price=150.0,
        exit_price=155.0,
        qty=200,
        detector="ofi_spike",
        source_strategy="ANOMALY",
    )
    all_decisions.extend(decs)
    all_trades.extend(trs)

    _seed_decisions(decisions_db, all_decisions)
    _seed_trades(trades_db, all_trades)

    analysis = dta_module.build_analysis(
        "2026-05-26",
        decisions_db=decisions_db,
        trades_db=trades_db,
        skip_imoex=True,
    )
    by_ticker = analysis["by_ticker"]
    assert "SBER" in by_ticker
    assert "GAZP" in by_ticker
    assert by_ticker["SBER"]["n_trades"] == 2
    assert by_ticker["GAZP"]["n_trades"] == 1
    assert by_ticker["SBER"]["win_rate"] == pytest.approx(0.5, abs=1e-6)
    assert by_ticker["GAZP"]["win_rate"] == pytest.approx(1.0)

    by_det = analysis["by_detector"]
    assert "double_top" in by_det
    assert "ofi_spike" in by_det
    assert by_det["double_top"]["n_trades"] == 2

    by_strat = analysis["by_strategy"]
    assert by_strat["TA"]["n_trades"] == 2
    assert by_strat["ANOMALY"]["n_trades"] == 1


def test_alpha_vs_imoex_computed_when_curve_provided(dta_module, tmp_path):
    """Provide a synthetic IMOEX curve and verify alpha is computed."""
    trades_db = tmp_path / "trades.db"
    decisions_db = tmp_path / "decisions.db"
    decs, trs = _make_round_trip(
        ticker="SBER",
        entry_time="10:00:00",
        exit_time="11:00:00",
        entry_price=100.0,
        exit_price=103.0,
        qty=100,
    )
    _seed_decisions(decisions_db, decs)
    _seed_trades(trades_db, trs)
    imoex_curve = {f"{h:02d}:00": 1000.0 for h in range(10, 16)}
    imoex_curve["10:00"] = 1000.0
    imoex_curve["11:00"] = 1010.0

    analysis = dta_module.build_analysis(
        "2026-05-26",
        decisions_db=decisions_db,
        trades_db=trades_db,
        imoex_curve=imoex_curve,
    )
    sber = analysis["trades"][0]
    assert sber["alpha_vs_imoex_pct"] is not None
    assert 1.5 < sber["alpha_vs_imoex_pct"] < 2.5


def test_open_position_when_no_matching_sell(dta_module, tmp_path):
    """A BUY without a SELL leaves an open position record."""
    trades_db = tmp_path / "trades.db"
    decisions_db = tmp_path / "decisions.db"
    decs, trs = _make_round_trip(
        ticker="SBER",
        entry_time="10:00:00",
        exit_time="11:00:00",
        entry_price=300.0,
        exit_price=305.0,
        qty=100,
    )
    _seed_decisions(decisions_db, decs)
    _seed_trades(trades_db, [trs[0]])
    analysis = dta_module.build_analysis(
        "2026-05-26",
        decisions_db=decisions_db,
        trades_db=trades_db,
        skip_imoex=True,
    )
    assert analysis["n_round_trips"] == 0
    assert analysis["n_open"] == 1
    open_rec = analysis["trades"][0]
    assert open_rec["open"] is True
    assert open_rec["pnl_rub"] is None


def test_json_schema_has_required_keys(dta_module, tmp_path):
    """Every per-trade record carries the contract keys."""
    trades_db = tmp_path / "trades.db"
    decisions_db = tmp_path / "decisions.db"
    decs, trs = _make_round_trip(
        ticker="SBER",
        entry_time="10:00:00",
        exit_time="11:00:00",
        entry_price=300.0,
        exit_price=305.0,
        qty=100,
    )
    _seed_decisions(decisions_db, decs)
    _seed_trades(trades_db, trs)
    analysis = dta_module.build_analysis(
        "2026-05-26",
        decisions_db=decisions_db,
        trades_db=trades_db,
        skip_imoex=True,
    )
    required_keys = {
        "trade_id",
        "ticker",
        "direction",
        "entry_ts",
        "exit_ts",
        "entry_price",
        "exit_price",
        "qty",
        "notional_rub",
        "pnl_rub",
        "pnl_pct",
        "holding_min",
        "exit_reason",
        "source_strategy",
        "detector",
        "tier",
        "hmm_regime_at_entry",
        "magnitude",
        "expected_rr",
        "actual_rr",
        "alpha_vs_imoex_pct",
        "rationale",
    }
    rec = analysis["trades"][0]
    missing = required_keys - set(rec.keys())
    assert not missing, f"missing required keys: {missing}"
    for k in (
        "date",
        "summary",
        "by_ticker",
        "by_detector",
        "by_strategy",
        "by_tier",
        "by_regime",
        "by_time_of_day",
        "top_winners",
        "top_losers",
    ):
        assert k in analysis


def test_top_winners_and_losers_ordered(dta_module, tmp_path):
    """Top-5 winners/losers sorted correctly."""
    trades_db = tmp_path / "trades.db"
    decisions_db = tmp_path / "decisions.db"
    all_decisions, all_trades = [], []
    fixtures = [
        ("AAA", "10:00:00", "11:00:00", 100.0, 110.0),
        ("BBB", "10:05:00", "11:00:00", 100.0, 105.0),
        ("CCC", "10:10:00", "11:00:00", 100.0, 102.0),
        ("DDD", "10:15:00", "11:00:00", 100.0, 95.0),
        ("EEE", "10:20:00", "11:00:00", 100.0, 92.0),
        ("FFF", "10:25:00", "11:00:00", 100.0, 90.0),
    ]
    for tk, et, xt, ep, xp in fixtures:
        decs, trs = _make_round_trip(
            ticker=tk,
            entry_time=et,
            exit_time=xt,
            entry_price=ep,
            exit_price=xp,
            qty=100,
        )
        all_decisions.extend(decs)
        all_trades.extend(trs)
    _seed_decisions(decisions_db, all_decisions)
    _seed_trades(trades_db, all_trades)

    analysis = dta_module.build_analysis(
        "2026-05-26",
        decisions_db=decisions_db,
        trades_db=trades_db,
        skip_imoex=True,
    )
    winners = analysis["top_winners"]
    losers = analysis["top_losers"]
    assert len(winners) <= 5
    assert len(losers) <= 5
    pnls_w = [r["pnl_rub"] for r in winners]
    assert pnls_w == sorted(pnls_w, reverse=True)
    pnls_l = [r["pnl_rub"] for r in losers]
    assert pnls_l == sorted(pnls_l)
    assert winners[0]["ticker"] == "AAA"
    assert losers[0]["ticker"] == "FFF"


def test_render_markdown_renders_summary(dta_module, tmp_path):
    """Markdown renderer doesn't crash on a minimal analysis dict."""
    analysis = {
        "date": "2026-05-26",
        "n_open": 1,
        "n_decisions": 3,
        "summary": {
            "n_round_trips": 2,
            "wins": 1,
            "losses": 1,
            "win_rate": 0.5,
            "total_pnl_rub": 100.0,
            "avg_pnl_pct": 0.5,
            "best_trade_rub": 200.0,
            "worst_trade_rub": -100.0,
            "avg_holding_min": 45.0,
            "avg_alpha_vs_imoex_pct": 0.3,
        },
        "by_ticker": {
            "SBER": {
                "n_trades": 2,
                "wins": 1,
                "losses": 1,
                "win_rate": 0.5,
                "total_pnl_rub": 100.0,
                "avg_holding_min": 45.0,
                "avg_alpha_vs_imoex_pct": 0.3,
            }
        },
        "by_detector": {},
        "by_strategy": {},
        "by_tier": {},
        "by_regime": {},
        "by_time_of_day": {},
        "by_direction": {},
        "top_winners": [],
        "top_losers": [],
        "llm_narrative": "some narrative",
    }
    md = dta_module._render_markdown(analysis)
    assert "Post-mortem: 2026-05-26" in md
    assert "Win-rate" in md
    assert "SBER" in md
    assert "LLM narrative" in md
    assert "some narrative" in md
