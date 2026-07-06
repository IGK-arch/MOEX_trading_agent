"""
app/backtest/engine.py — Pure numpy/pandas backtester for pattern trades.

Designed around the same triple-barrier labels used in training, so backtest
↔ training ↔ live use the IDENTICAL exit rules. This is the only way to
get an honest estimate of live PnL.

Inputs:
  - df:        OHLCV DataFrame
  - patterns:  list of PatternSignal (any source — chart / harmonic / SMC)
  - sizing:    fraction of equity per trade (e.g. 0.02 = 2%)

Output: BacktestReport with Sharpe, MaxDD, WinRate, expectancy, turnover.

The simulator walks chronologically through `patterns`, opens each trade at
the pattern's `entry` bar, exits when one of the triple barriers fires, and
updates equity. Commissions / slippage configurable via kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.training.labeling import label_triple_barrier
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import numpy as np
    import pandas as pd

    _READY = True
except ImportError:
    _READY = False

@dataclass
class BacktestReport:
    """Summary of one backtest run."""

    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    n_timeouts: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    expectancy_pct: float = 0.0
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_holding_bars: float = 0.0
    turnover_rub: float = 0.0
    equity_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "n_timeouts": self.n_timeouts,
            "win_rate": round(self.win_rate, 3),
            "total_pnl_pct": round(self.total_pnl_pct, 3),
            "expectancy_pct": round(self.expectancy_pct, 4),
            "sharpe": round(self.sharpe, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 3),
            "avg_holding_bars": round(self.avg_holding_bars, 1),
            "turnover_rub": round(self.turnover_rub, 0),
        }

class BacktestEngine:
    """
    Replays a list of pattern signals against price data using the SAME
    triple-barrier exit logic as training & live. Returns a BacktestReport.
    """

    def __init__(
        self,
        starting_equity_rub: float = 1_000_000.0,
        sizing_pct: float = 0.02,
        commission_bps: float = 5.0,
        slippage_bps: float = 5.0,
        horizon_bars: int = 24,
        atr_mult_top: float = 2.0,
        atr_mult_bot: float = 1.0,
    ) -> None:
        """Init."""
        self.start_eq = starting_equity_rub
        self.sizing_pct = sizing_pct
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.horizon_bars = horizon_bars
        self.atr_mult_top = atr_mult_top
        self.atr_mult_bot = atr_mult_bot

    def run(
        self,
        df: pd.DataFrame,
        patterns: list,
    ) -> BacktestReport:
        """Run."""
        if not _READY or df is None or not patterns:
            return BacktestReport(equity_curve=[self.start_eq])

        df = df.reset_index(drop=True)

        patterns_sorted = sorted(patterns, key=lambda p: p.bar_idx)

        equity = self.start_eq
        equity_curve = [equity]
        n_wins = n_losses = n_timeouts = 0
        holding_bars: list[int] = []
        trade_returns: list[float] = []
        turnover = 0.0
        open_until_bar = -1

        for p in patterns_sorted:
            if p.bar_idx <= open_until_bar:
                continue

            label = label_triple_barrier(
                df,
                bar_idx=p.bar_idx,
                direction=p.direction,
                entry=p.entry,
                stop=p.stop,
                target=p.target,
                atr_at_entry=p.atr_at_entry,
                horizon_bars=self.horizon_bars,
                atr_mult_top=self.atr_mult_top,
                atr_mult_bot=self.atr_mult_bot,
            )
            if label.barrier_hit == "no_data":
                continue

            notional = equity * self.sizing_pct
            qty = max(1, int(notional / max(p.entry, 1e-6)))
            actual_notional = qty * p.entry
            turnover += actual_notional

            exit_price = label.exit_price
            if p.direction == "BUY":
                pnl_rub = (exit_price - p.entry) * qty
            else:
                pnl_rub = (p.entry - exit_price) * qty

            cost_bps = (self.commission_bps + self.slippage_bps) * 2
            cost_rub = actual_notional * cost_bps / 10_000
            pnl_rub -= cost_rub

            if label.barrier_hit == "top":
                n_wins += 1
            elif label.barrier_hit == "bottom":
                n_losses += 1
            else:
                n_timeouts += 1
                if pnl_rub > 0:
                    n_wins += 1
                else:
                    n_losses += 1

            equity += pnl_rub
            equity_curve.append(equity)
            trade_returns.append(pnl_rub / actual_notional if actual_notional > 0 else 0.0)
            holding_bars.append(label.holding_bars)
            open_until_bar = label.exit_bar_idx

        n_trades = len(trade_returns)
        if n_trades == 0:
            return BacktestReport(equity_curve=equity_curve)

        trade_returns_np = np.asarray(trade_returns)
        equity_np = np.asarray(equity_curve)
        peak = np.maximum.accumulate(equity_np)
        drawdowns = (peak - equity_np) / np.maximum(peak, 1e-9)
        max_dd = float(drawdowns.max())

        std = trade_returns_np.std()
        sharpe = float(trade_returns_np.mean() / std * np.sqrt(252)) if std > 0 else 0.0
        win_rate = n_wins / n_trades

        return BacktestReport(
            n_trades=n_trades,
            n_wins=n_wins,
            n_losses=n_losses,
            n_timeouts=n_timeouts,
            win_rate=win_rate,
            total_pnl_pct=(equity / self.start_eq - 1.0),
            expectancy_pct=float(trade_returns_np.mean()),
            sharpe=sharpe,
            max_drawdown_pct=max_dd,
            avg_holding_bars=float(np.mean(holding_bars)) if holding_bars else 0.0,
            turnover_rub=turnover,
            equity_curve=[float(v) for v in equity_curve],
        )

def simulate_pattern_trades(
    df: pd.DataFrame,
    patterns: list,
    **engine_kwargs,
) -> BacktestReport:
    """Convenience wrapper for one-shot backtest runs."""
    return BacktestEngine(**engine_kwargs).run(df, patterns)

__all__ = ["BacktestEngine", "BacktestReport", "simulate_pattern_trades"]
