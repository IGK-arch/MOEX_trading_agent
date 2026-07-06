"""
app/backtest — Lightweight backtesting engine for triple-barrier-labeled events.

NOT a Streamlit-grade visual backtester. Designed for two concrete uses:

  1. **Walk-forward eval** of the full pipeline (TA patterns + meta + risk)
     against historical CSV from MOEX ISS.
  2. **AlterGiga-style validation**: run the same strategy across 1000
     synthetic price paths to verify the model isn't overfit to noise.

Output is a small `BacktestReport` dataclass with the metrics we care about:
Sharpe, max DD, win rate, expectancy, turnover, n_trades.
"""

from app.backtest.engine import BacktestEngine, BacktestReport, simulate_pattern_trades
from app.backtest.synthetic import (
    SyntheticPath,
    generate_gbm_paths,
    generate_jump_diffusion_paths,
)

__all__ = [
    "BacktestEngine",
    "BacktestReport",
    "simulate_pattern_trades",
    "SyntheticPath",
    "generate_gbm_paths",
    "generate_jump_diffusion_paths",
]
