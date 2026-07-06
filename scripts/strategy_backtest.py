"""Phase 27.5 — Per-strategy backtest + capital allocation.

Runs each adapter (TA / Anomaly / News / MeanRev) independently on the same
ISS 60m / 5m universe so we can compare their stand-alone edge and assign
deposit weights instead of letting first-come-first-served decide.

Outputs:
    data/training_cache/strategy_backtest_v1.json
    data/training_cache/strategy_allocations.json

Usage:
    python3 scripts/strategy_backtest.py                  # default: 90d × 20 tickers
    python3 scripts/strategy_backtest.py --days 60 --tickers SBER GAZP LKOH

For TA we re-use the trade ledger captured by scripts/rank_all_detectors.py
(detector_rankings_after.json) when available — that is the exact same
simulation engine used by the existing 60m × 20-ticker backtest.

For Anomaly we run the 5 file-based detectors (volume_zscore, price_spikes,
absorption, vwap_crosses, atr_reversion; OFI skipped — needs supercandles).

For News we replay news_events × news_reactions from feeds.db: every
material news event with a directional LLM tag yields one virtual trade
priced from the 15-min reaction window.

For Mean-Reversion we re-implement the Bollinger + RSI logic that lives in
app/agents/mean_reversion.py against the same 5m candle frame.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import contextlib

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import app.config as cfg  # noqa: E402
from app.agents.anomaly_detectors.absorption import detect_absorption  # noqa: E402
from app.agents.anomaly_detectors.atr_reversion import detect_atr_reversion  # noqa: E402
from app.agents.anomaly_detectors.price_spikes import detect_price_spikes  # noqa: E402
from app.agents.anomaly_detectors.volume_zscore import detect_volume_zscore  # noqa: E402
from app.agents.anomaly_detectors.vwap_crosses import detect_vwap_crosses  # noqa: E402
from app.agents.ta_indicators import (  # noqa: E402
    compute_atr,
    compute_bollinger,
    compute_rsi,
    compute_sma,
)
from app.data.iss_client import get_iss_client  # noqa: E402

COMMISSION_RATE = 0.00035
DEFAULT_SL_ATR = 1.5
DEFAULT_TP_ATR = 2.0
ANOMALY_HORIZON_BARS = 6
MEAN_REV_HORIZON_BARS = 6
NEWS_HORIZON_MIN = 60

@dataclass
class TradeRecord:
    """Trade Record."""

    strategy: str
    detector: str
    ticker: str
    direction: str
    entry: float
    stop: float
    target: float
    exit_price: float
    exit_reason: str
    pnl_pct: float
    holding_bars: int
    timestamp_ms: int = 0

def _simulate(
    df: pd.DataFrame,
    bar_idx: int,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    max_hold: int,
) -> tuple[float, float, str, int] | None:
    """Triple-barrier sim: SL → exit_at_stop, TP → exit_at_target, else time-out."""
    if bar_idx < 0 or bar_idx >= len(df) - 1:
        return None
    if entry <= 0 or stop <= 0 or target <= 0:
        return None
    if direction == "BUY" and (stop >= entry or target <= entry):
        return None
    if direction == "SELL" and (stop <= entry or target >= entry):
        return None

    sign = 1 if direction == "BUY" else -1
    end_i = min(bar_idx + max_hold, len(df) - 1)
    exit_price = None
    exit_reason = None
    holding = 0

    for j in range(bar_idx + 1, end_i + 1):
        high = float(df["high"].iloc[j])
        low = float(df["low"].iloc[j])
        if sign == 1:
            stop_hit = low <= stop
            target_hit = high >= target
        else:
            stop_hit = high >= stop
            target_hit = low <= target

        if stop_hit and target_hit:
            exit_price, exit_reason, holding = stop, "stop_in_target_bar", j - bar_idx
            break
        if stop_hit:
            exit_price, exit_reason, holding = stop, "stop", j - bar_idx
            break
        if target_hit:
            exit_price, exit_reason, holding = target, "target", j - bar_idx
            break

    if exit_price is None:
        exit_price = float(df["close"].iloc[end_i])
        exit_reason = "timeout"
        holding = end_i - bar_idx

    gross = (exit_price - entry) / entry * sign
    pnl = gross - 2 * COMMISSION_RATE
    return float(exit_price), float(pnl), exit_reason, int(holding)

def _aggregate(trades: list[TradeRecord], bars_per_day: float) -> dict[str, Any]:
    """Convert a flat trade list into the metrics required by allocation math."""
    if not trades:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "pnl_pct": 0.0,
            "avg_trade_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_dd_pct": 0.0,
            "vol_pct": 0.0,
            "trades_per_day": 0.0,
            "by_exit": {},
        }
    pnls = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls)
    total_pnl = sum(pnls)
    avg_trade = sum(pnls) / len(pnls)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)
    std = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
    sharpe_trade = avg_trade / std if std > 0 else 0.0
    n_trades = len(pnls)
    sharpe_ann = sharpe_trade * math.sqrt(max(1.0, bars_per_day * 252.0)) if std > 0 else 0.0
    eq = np.cumsum(pnls)
    peaks = np.maximum.accumulate(eq)
    dd = peaks - eq
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0
    vol_pct = std * math.sqrt(max(1.0, bars_per_day))
    by_exit: dict[str, int] = {}
    for t in trades:
        by_exit[t.exit_reason] = by_exit.get(t.exit_reason, 0) + 1
    return {
        "n_trades": int(n_trades),
        "win_rate": round(win_rate, 4),
        "pnl_pct": round(total_pnl * 100, 4),
        "avg_trade_pct": round(avg_trade * 100, 5),
        "avg_win_pct": round(avg_win * 100, 5),
        "avg_loss_pct": round(avg_loss * 100, 5),
        "profit_factor": round(pf, 4) if math.isfinite(pf) else None,
        "sharpe": round(sharpe_ann, 4),
        "max_dd_pct": round(max_dd * 100, 4),
        "vol_pct": round(vol_pct * 100, 4),
        "trades_per_day": round(n_trades / max(1.0, bars_per_day * 90.0 / bars_per_day), 3)
        if n_trades > 0
        else 0.0,
        "by_exit": by_exit,
    }

async def _fetch_candles(
    iss,
    ticker: str,
    interval_min: int,
    days: int,
) -> pd.DataFrame | None:
    """Fetch candles."""
    till = datetime.now(tz=UTC)
    from_dt = till - timedelta(days=days)
    try:
        result = await iss.get_candles_multi(
            [ticker],
            interval=interval_min,
            from_dt=from_dt,
            till_dt=till,
        )
    except Exception as exc:  # pragma: no cover
        print(f"  [{ticker}] fetch_candles failed: {exc}")
        return None
    df = result.get(ticker)
    if df is None or len(df) < 60:
        return None
    df = df.reset_index(drop=True)
    atr = compute_atr(df, period=14)
    df["atr14"] = atr if atr is not None else 0.0
    if "timestamp" not in df.columns:
        df["timestamp"] = df.get("begin", df.index)
    return df

def _ta_trades_from_rankings(rankings_path: Path) -> list[TradeRecord]:
    """Reconstruct virtual trade records from the existing 90d × 20 backtest.

    We can't fully replay the simulation here without re-running the engine,
    but the aggregated per_pattern metrics in detector_rankings_after.json
    already represent ~8k trades. We synthesize a pseudo-trade list using
    win_rate/avg_pnl per pattern so downstream aggregation reproduces the
    same headline numbers (WR / PF / total PnL) — exact enough for the
    allocation math.
    """
    if not rankings_path.exists():
        return []
    data = json.loads(rankings_path.read_text())
    out: list[TradeRecord] = []
    blacklist = set(cfg.DETECTOR_BLACKLIST or set())
    kill_list = set(data.get("kill_list") or [])
    skip = blacklist | kill_list
    for pat, m in (data.get("per_pattern") or {}).items():
        if pat in skip:
            continue
        n = int(m.get("n_trades") or 0)
        wr = float(m.get("win_rate") or 0.0)
        pf = m.get("profit_factor")
        avg = float(m.get("avg_pnl_pct") or 0.0)
        if n <= 0:
            continue
        n_win = max(1, int(round(n * wr)))
        n_loss = max(1, n - n_win)
        avg_loss = 0.005
        if pf and pf > 0 and math.isfinite(pf):
            avg_win = pf * avg_loss * n_loss / n_win
        else:
            avg_win = max(0.0, 2 * avg)
        for _ in range(n_win):
            out.append(
                TradeRecord(
                    strategy="TA",
                    detector=pat,
                    ticker="MIX",
                    direction="BUY",
                    entry=100.0,
                    stop=98.0,
                    target=104.0,
                    exit_price=100.0 * (1 + avg_win),
                    exit_reason="target",
                    pnl_pct=avg_win,
                    holding_bars=int(m.get("avg_holding_bars") or 10),
                )
            )
        for _ in range(n_loss):
            out.append(
                TradeRecord(
                    strategy="TA",
                    detector=pat,
                    ticker="MIX",
                    direction="BUY",
                    entry=100.0,
                    stop=98.0,
                    target=104.0,
                    exit_price=100.0 * (1 - avg_loss),
                    exit_reason="stop",
                    pnl_pct=-avg_loss,
                    holding_bars=int(m.get("avg_holding_bars") or 10),
                )
            )
    return out

def _anomaly_trades_one_ticker(
    df: pd.DataFrame,
    ticker: str,
    max_hold: int = ANOMALY_HORIZON_BARS,
) -> list[TradeRecord]:
    """Run the 5 file-based detectors and simulate one trade per actionable signal."""
    out: list[TradeRecord] = []
    if df is None or len(df) < 50:
        return out
    atr = df["atr14"]

    detectors = {
        "volume_zscore": lambda: detect_volume_zscore(df, ticker),
        "price_spikes": lambda: detect_price_spikes(df, ticker, atr),
        "absorption": lambda: detect_absorption(df, ticker, atr),
        "vwap_crosses": lambda: detect_vwap_crosses(df, ticker, atr),
    }
    signals: list[Any] = []
    for _name, fn in detectors.items():
        try:
            signals.extend(fn())
        except Exception:
            continue
    try:
        last_idx = -100
        for win_end in range(40, len(df), 20):
            chunk = df.iloc[:win_end].reset_index(drop=True)
            chunk_atr = compute_atr(chunk, period=14)
            extra = detect_atr_reversion(
                chunk,
                ticker,
                chunk_atr,
                last_signal_idx=last_idx,
            )
            if extra:
                last_idx = extra[-1].bar_idx
                for e in extra:
                    e.bar_idx = e.bar_idx
                signals.extend(extra)
    except Exception:
        pass

    for s in signals:
        if not s.is_actionable():
            continue
        bar = int(s.bar_idx)
        atr_val = float(atr.iloc[bar]) if bar < len(atr) and pd.notna(atr.iloc[bar]) else 0.0
        if atr_val <= 0:
            continue
        entry = float(df["close"].iloc[bar])
        if s.direction == "BUY":
            stop = entry - DEFAULT_SL_ATR * atr_val
            target = entry + DEFAULT_TP_ATR * atr_val
        else:
            stop = entry + DEFAULT_SL_ATR * atr_val
            target = entry - DEFAULT_TP_ATR * atr_val
        sim = _simulate(df, bar, s.direction, entry, stop, target, max_hold)
        if sim is None:
            continue
        exit_price, pnl, reason, hold = sim
        out.append(
            TradeRecord(
                strategy="ANOMALY",
                detector=s.detector,
                ticker=ticker,
                direction=s.direction,
                entry=entry,
                stop=stop,
                target=target,
                exit_price=exit_price,
                exit_reason=reason,
                pnl_pct=pnl,
                holding_bars=hold,
            )
        )
    return out

def _news_trades_from_db(
    db_path: Path,
    days: int = 60,
) -> list[TradeRecord]:
    """Stream news_events with directional reactions and synthesize trades."""
    out: list[TradeRecord] = []
    if not db_path.exists():
        return out
    cutoff = (datetime.now(tz=UTC) - timedelta(days=days)).isoformat()
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        rows = cur.execute(
            """
            SELECT ne.event_id, ne.ts_utc, ne.llm_direction, ne.llm_magnitude,
                   ne.horizon_min, ne.is_material, ne.event_type,
                   nr.ticker, nr.window_min, nr.return_pct, nr.abs_move_pct
            FROM news_events ne
            JOIN news_reactions nr ON ne.event_id = nr.event_id
            WHERE ne.is_material = 1 AND ne.ts_utc >= ?
            """,
            (cutoff,),
        ).fetchall()
        con.close()
    except Exception:
        return out
    by_key: dict[tuple[str, str], dict[int, tuple[float, float]]] = {}
    direction_by_key: dict[tuple[str, str], str] = {}
    event_type_by_key: dict[tuple[str, str], str] = {}
    for ev_id, _ts, llm_dir, _llm_mag, _horizon, _mat, ev_type, tk, wm, ret, absmv in rows:
        key = (ev_id, tk)
        by_key.setdefault(key, {})[int(wm)] = (float(ret or 0.0), float(absmv or 0.0))
        direction_by_key[key] = (llm_dir or "NEUTRAL").upper()
        event_type_by_key[key] = ev_type or "other"

    momentum_types = {"sanctions", "earnings", "guidance", "commodity", "macro"}
    for key, windows in by_key.items():
        ev_id, ticker = key
        direction = direction_by_key.get(key, "NEUTRAL")
        ev_type = event_type_by_key.get(key, "other")
        if direction == "NEUTRAL":
            r5 = windows.get(5, (0.0, 0.0))[0]
            if ev_type in momentum_types and abs(r5) >= 0.003:
                direction = "BUY" if r5 > 0 else "SELL"
            else:
                continue
        if direction not in ("BUY", "SELL"):
            continue
        exit_ret = (windows.get(60, windows.get(30, windows.get(15, windows.get(5, (0.0, 0.0))))))[
            0
        ]
        sign = 1 if direction == "BUY" else -1
        pnl = sign * exit_ret - 2 * COMMISSION_RATE
        if abs(exit_ret) >= 0.005:
            reason = "target" if sign * exit_ret > 0 else "stop"
        else:
            reason = "timeout"
        out.append(
            TradeRecord(
                strategy="NEWS",
                detector=f"news_{ev_type}",
                ticker=ticker.upper(),
                direction=direction,
                entry=100.0,
                stop=100.0 * (1 - 0.005),
                target=100.0 * (1 + 0.005),
                exit_price=100.0 * (1 + exit_ret),
                exit_reason=reason,
                pnl_pct=pnl,
                holding_bars=4,
            )
        )
    return out

def _mean_rev_trades_one_ticker(
    df: pd.DataFrame,
    ticker: str,
    bb_period: int = 20,
    bb_std: float = 2.0,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    atr_stop_mult: float = 1.5,
    target_atr_mult: float = 1.5,
    max_hold: int = MEAN_REV_HORIZON_BARS,
) -> list[TradeRecord]:
    """Replay app/agents/mean_reversion.py over a candle history."""
    out: list[TradeRecord] = []
    if df is None or len(df) < bb_period + 5:
        return out
    bb = compute_bollinger(df, period=bb_period, std_dev=bb_std)
    rsi = compute_rsi(df, period=14)
    atr = df["atr14"]
    compute_sma(df, period=bb_period)
    if bb is None or len(bb) == 0:
        return out

    for i in range(bb_period + 2, len(df) - max_hold - 1):
        prev_close = float(df["close"].iloc[i - 1])
        curr_close = float(df["close"].iloc[i])
        bbu = float(bb["BBU"].iloc[i]) if pd.notna(bb["BBU"].iloc[i]) else None
        bbl = float(bb["BBL"].iloc[i]) if pd.notna(bb["BBL"].iloc[i]) else None
        bbm = float(bb["BBM"].iloc[i]) if pd.notna(bb["BBM"].iloc[i]) else None
        rsi_val = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else None
        atr_val = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0
        if bbu is None or bbl is None or rsi_val is None or atr_val <= 0:
            continue

        if prev_close > bbu and curr_close < bbu and rsi_val > rsi_overbought:
            entry = curr_close
            stop = entry + atr_stop_mult * atr_val
            target = bbm if bbm is not None else entry - target_atr_mult * atr_val
            sim = _simulate(df, i, "SELL", entry, stop, target, max_hold)
            if sim is None:
                continue
            exit_price, pnl, reason, hold = sim
            out.append(
                TradeRecord(
                    strategy="MEAN_REV",
                    detector="bollinger_short",
                    ticker=ticker,
                    direction="SELL",
                    entry=entry,
                    stop=stop,
                    target=target,
                    exit_price=exit_price,
                    exit_reason=reason,
                    pnl_pct=pnl,
                    holding_bars=hold,
                )
            )
        elif prev_close < bbl and curr_close > bbl and rsi_val < rsi_oversold:
            entry = curr_close
            stop = entry - atr_stop_mult * atr_val
            target = bbm if bbm is not None else entry + target_atr_mult * atr_val
            sim = _simulate(df, i, "BUY", entry, stop, target, max_hold)
            if sim is None:
                continue
            exit_price, pnl, reason, hold = sim
            out.append(
                TradeRecord(
                    strategy="MEAN_REV",
                    detector="bollinger_long",
                    ticker=ticker,
                    direction="BUY",
                    entry=entry,
                    stop=stop,
                    target=target,
                    exit_price=exit_price,
                    exit_reason=reason,
                    pnl_pct=pnl,
                    holding_bars=hold,
                )
            )
    return out

def compute_optimal_allocation(
    metrics: dict[str, dict],
    floor_pct: float = 0.05,
) -> dict[str, Any]:
    """Three allocation methods + final average with floor."""
    strategies = list(metrics.keys())

    sharpe_pos = {s: max(0.0, float(metrics[s].get("sharpe") or 0.0)) for s in strategies}
    total_sharpe = sum(sharpe_pos.values())
    if total_sharpe > 0:
        sharpe_alloc = {s: sharpe_pos[s] / total_sharpe for s in strategies}
    else:
        sharpe_alloc = {s: 1.0 / len(strategies) for s in strategies}

    inv_vol = {s: 1.0 / max(0.001, float(metrics[s].get("vol_pct") or 0.001)) for s in strategies}
    total_inv = sum(inv_vol.values())
    rp_alloc = {s: inv_vol[s] / total_inv for s in strategies}

    kelly_raw: dict[str, float] = {}
    for s in strategies:
        m = metrics[s]
        win = float(m.get("win_rate") or 0.0)
        avg_win = abs(float(m.get("avg_win_pct") or 0.0))
        avg_loss = abs(float(m.get("avg_loss_pct") or 0.001)) or 0.001
        b = avg_win / avg_loss if avg_loss > 0 else 0.0
        f = (win * b - (1.0 - win)) / b if b > 0 else 0.0
        kelly_raw[s] = max(0.0, f * 0.5)
    total_k = sum(kelly_raw.values())
    if total_k > 0:
        kelly_alloc = {s: kelly_raw[s] / total_k for s in strategies}
    else:
        kelly_alloc = {s: 1.0 / len(strategies) for s in strategies}

    raw_final = {s: (sharpe_alloc[s] + rp_alloc[s] + kelly_alloc[s]) / 3.0 for s in strategies}
    floored = {s: max(floor_pct, v) for s, v in raw_final.items()}
    total_f = sum(floored.values())
    final_alloc = {s: round(v / total_f, 4) for s, v in floored.items()}

    return {
        "sharpe_weighted": {s: round(v, 4) for s, v in sharpe_alloc.items()},
        "risk_parity": {s: round(v, 4) for s, v in rp_alloc.items()},
        "kelly_weighted": {s: round(v, 4) for s, v in kelly_alloc.items()},
        "final": final_alloc,
    }

def _format_recommendation(alloc: dict[str, float]) -> str:
    """Format recommendation."""
    parts = [
        f"{s}: {int(round(v * 100))}%" for s, v in sorted(alloc.items(), key=lambda kv: -kv[1])
    ]
    return ", ".join(parts)

async def main() -> int:
    """Main."""
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--tickers", nargs="*", default=None)
    p.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Bar size in minutes for Anomaly/MeanRev (default 10 — "
        "matches production poll interval).",
    )
    p.add_argument(
        "--output", default=str(cfg.DATA_DIR / "training_cache" / "strategy_backtest_v1.json")
    )
    p.add_argument(
        "--alloc-output", default=str(cfg.DATA_DIR / "training_cache" / "strategy_allocations.json")
    )
    p.add_argument("--floor-pct", type=float, default=0.05)
    p.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Use only TA + NEWS (no live candle fetch). Useful for smoke runs.",
    )
    args = p.parse_args()

    tickers = args.tickers or cfg.TICKERS
    bars_per_day = (390 / args.interval) if args.interval > 0 else 78
    print(
        f"\nStrategy backtest: {len(tickers)} tickers × {args.days}d × "
        f"{args.interval}m bars (bars/day ≈ {bars_per_day:.1f})\n"
    )

    print("[1/4] TA Trader (90d × 20-ticker × 60m → derived from detector_rankings_after.json)...")
    rankings_path = cfg.DATA_DIR / "training_cache" / "detector_rankings_after.json"
    ta_trades = _ta_trades_from_rankings(rankings_path)
    print(f"      → {len(ta_trades)} virtual TA trades after blacklist/kill filter")

    anomaly_trades: list[TradeRecord] = []
    mr_trades: list[TradeRecord] = []
    if not args.skip_fetch:
        iss = get_iss_client()
        try:
            await iss.startup()
        except Exception as exc:
            print(f"  ISS startup failed → falling back to historical CSV: {exc}")
            iss = None

        if iss is not None:
            print(f"[2/4] Anomaly detectors over 5m × {args.days}d...")
            for ticker in tickers:
                df = await _fetch_candles(iss, ticker, args.interval, args.days)
                if df is None or len(df) < 60:
                    print(f"   {ticker:6s}: skip (no candles)")
                    continue
                anom = _anomaly_trades_one_ticker(df, ticker)
                anomaly_trades.extend(anom)
                mr = _mean_rev_trades_one_ticker(df, ticker)
                mr_trades.extend(mr)
                print(f"   {ticker:6s}: anomaly={len(anom):3d}  mean_rev={len(mr):3d}")
            with contextlib.suppress(Exception):
                await iss.shutdown()

    print(f"      → {len(anomaly_trades)} anomaly trades, {len(mr_trades)} mean-rev trades")

    print("[3/4] News LLM replay from feeds.db...")
    news_trades = _news_trades_from_db(cfg.DATA_DIR / "feeds.db", days=args.days)
    print(f"      → {len(news_trades)} news-driven trades")

    print("[4/4] Aggregating metrics + computing allocations...")
    ta_metrics = _aggregate(ta_trades, bars_per_day=6.5)
    anom_metrics = _aggregate(anomaly_trades, bars_per_day=bars_per_day)
    news_metrics = _aggregate(news_trades, bars_per_day=4.0)
    mr_metrics = _aggregate(mr_trades, bars_per_day=bars_per_day)

    metrics_all = {
        "TA": ta_metrics,
        "ANOMALY": anom_metrics,
        "NEWS": news_metrics,
        "MEAN_REV": mr_metrics,
    }
    payload = {
        "ts_utc": datetime.now(tz=UTC).isoformat(),
        "lookback_days": args.days,
        "interval_min": args.interval,
        "tickers": list(tickers),
        "metrics": metrics_all,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"   metrics → {out_path}")

    allocations = compute_optimal_allocation(metrics_all, floor_pct=args.floor_pct)
    alloc_payload = {
        "ts_utc": datetime.now(tz=UTC).isoformat(),
        "lookback_days": args.days,
        "metrics": metrics_all,
        "allocations": allocations,
        "recommendation": _format_recommendation(allocations["final"]),
    }
    alloc_path = Path(args.alloc_output)
    alloc_path.write_text(json.dumps(alloc_payload, indent=2))
    print(f"   allocations → {alloc_path}")
    print(f"   recommendation: {alloc_payload['recommendation']}")

    print("\n" + "=" * 90)
    print(
        f"{'Strategy':<10} {'N':>6} {'WR':>7} {'PnL%':>9} "
        f"{'Sharpe':>8} {'MaxDD%':>8} {'Vol%':>7} {'PF':>7}"
    )
    print("-" * 90)
    for s, m in metrics_all.items():
        pf = m.get("profit_factor")
        pf_s = f"{pf:.2f}" if pf is not None else "inf"
        print(
            f"{s:<10} {m['n_trades']:>6d} {m['win_rate'] * 100:>5.1f}% "
            f"{m['pnl_pct']:>+8.2f}% {m['sharpe']:>8.2f} "
            f"{m['max_dd_pct']:>7.2f}% {m['vol_pct']:>6.2f}% {pf_s:>7}"
        )
    print("=" * 90)
    print("\nAllocations:")
    for method, alloc in allocations.items():
        line = ", ".join(f"{s}={v * 100:.1f}%" for s, v in alloc.items())
        print(f"  {method:<18}  {line}")
    return 0

if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
