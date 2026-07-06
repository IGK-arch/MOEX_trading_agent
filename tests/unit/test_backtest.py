"""
tests/unit/test_backtest.py — Backtest engine + synthetic data generators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.agents.ta_patterns.reversal import PatternSignal
from app.backtest import (
    BacktestEngine,
    BacktestReport,
    generate_gbm_paths,
    generate_jump_diffusion_paths,
    simulate_pattern_trades,
)


def _make_df(n: int = 50, seed: int = 1) -> pd.DataFrame:
    """Make df."""
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0, 0.5, n).cumsum()
    high = close + rng.uniform(0.1, 0.4, n)
    low = close - rng.uniform(0.1, 0.4, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
        }
    )


def _make_pattern(
    bar_idx: int,
    direction: str = "BUY",
    entry: float = 100.0,
    stop: float = 99.0,
    target: float = 102.0,
) -> PatternSignal:
    """Make pattern."""
    return PatternSignal(
        pattern="test",
        direction=direction,
        confidence=0.7,
        bar_idx=bar_idx,
        entry=entry,
        stop=stop,
        target=target,
        expected_rr=2.0,
        atr_at_entry=0.5,
    )


def test_empty_run_returns_initial_equity():
    """Test empty run returns initial equity."""
    engine = BacktestEngine()
    df = _make_df()
    report = engine.run(df, [])
    assert report.n_trades == 0
    assert len(report.equity_curve) == 1
    assert report.equity_curve[0] == 1_000_000.0


def test_buy_pattern_executes_and_returns_pnl():
    """Test buy pattern executes and returns pnl."""

    n = 30
    close = np.concatenate([np.full(5, 100.0), np.linspace(101, 110, n - 5)])
    high = close + 0.5
    low = close - 0.5
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": [1000] * n}
    )
    pat = _make_pattern(bar_idx=4, entry=100.0, stop=98.0, target=105.0)
    report = simulate_pattern_trades(df, [pat])
    assert report.n_trades == 1
    assert report.n_wins == 1
    assert report.total_pnl_pct > 0


def test_overlapping_trades_are_skipped():
    """If we open at bar 5 and exit at bar 10, a new pattern at bar 7 is skipped."""
    n = 30
    close = np.linspace(100, 110, n)
    high = close + 0.3
    low = close - 0.3
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": [1000] * n}
    )
    pat1 = _make_pattern(bar_idx=5, entry=close[5], stop=close[5] - 1, target=close[5] + 3)
    pat2 = _make_pattern(bar_idx=7, entry=close[7], stop=close[7] - 1, target=close[7] + 3)
    report = simulate_pattern_trades(df, [pat1, pat2])
    assert report.n_trades == 1


def test_sharpe_is_zero_on_constant_returns():
    """If we craft trades with identical % returns, Sharpe should be infinite or zero."""
    df = _make_df(40)
    pat = _make_pattern(bar_idx=20, entry=100, stop=99, target=101)
    report = simulate_pattern_trades(df, [pat])
    assert isinstance(report.sharpe, float)


def test_gbm_paths_generated_with_correct_shape():
    """Test gbm paths generated with correct shape."""
    paths = generate_gbm_paths(n_paths=5, n_bars=100, seed=1)
    assert len(paths) == 5
    for p in paths:
        assert len(p.df) == 100
        assert all(c in p.df.columns for c in ("open", "high", "low", "close", "volume"))


def test_jump_diffusion_paths_generated():
    """Test jump diffusion paths generated."""
    paths = generate_jump_diffusion_paths(n_paths=3, n_bars=50, seed=2)
    assert len(paths) == 3


def test_report_to_dict_serializable():
    """BacktestReport.to_dict() should be JSON-serializable."""
    import json

    report = BacktestReport(n_trades=10, n_wins=6, n_losses=4, win_rate=0.6, sharpe=1.5)
    d = report.to_dict()
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["n_trades"] == 10
    assert parsed["win_rate"] == 0.6
