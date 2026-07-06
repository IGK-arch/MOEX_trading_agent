"""
app/backtest/synthetic.py — Generators of synthetic OHLCV paths.

For AlterGiga-style validation: run the strategy over many synthetic price
paths drawn from a model fit to real data. If real Sharpe is comparable
or worse than the synthetic mean → we may be picking up noise.

Two generators:
  1. **Geometric Brownian Motion (GBM)** — Black-Scholes style, no jumps.
  2. **Jump diffusion** — Merton (1976) — GBM with Poisson-distributed jumps.

Both produce H/L/O/C bars by drawing intraday volatility around the close
random walk. Sufficient for triple-barrier evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
    import pandas as pd

    _READY = True
except ImportError:
    _READY = False

@dataclass
class SyntheticPath:
    """Wrapper for a generated OHLCV DataFrame."""

    df: pd.DataFrame
    params: dict[str, Any]
    seed: int

def _build_ohlcv(close: np.ndarray, noise_pct: float = 0.002, seed: int = 0) -> pd.DataFrame:
    """Build OHLCV from a closing-price series. High/low are random noise around close."""
    rng = np.random.default_rng(seed)
    n = len(close)
    noise = rng.uniform(0, 1, n) * noise_pct
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
        }
    )

def generate_gbm_paths(
    n_paths: int = 100,
    n_bars: int = 240,
    mu_per_bar: float = 0.0001,
    sigma_per_bar: float = 0.01,
    s0: float = 100.0,
    seed: int = 42,
) -> list[SyntheticPath]:
    """
    Geometric Brownian Motion:
        d ln(S) ~ N(mu, sigma)

    Default values calibrate to ~3% daily volatility on 5-min bars
    (48 bars × 0.01² × √(48) ≈ 6.9%; rough match for MOEX blue-chips).
    """
    if not _READY:
        return []

    np.random.default_rng(seed)
    paths: list[SyntheticPath] = []
    for i in range(n_paths):
        path_seed = seed + i
        sub_rng = np.random.default_rng(path_seed)
        log_returns = sub_rng.normal(mu_per_bar, sigma_per_bar, n_bars)
        close = s0 * np.exp(np.cumsum(log_returns))
        df = _build_ohlcv(close, noise_pct=sigma_per_bar * 0.2, seed=path_seed)
        paths.append(
            SyntheticPath(
                df=df,
                params={"model": "gbm", "mu": mu_per_bar, "sigma": sigma_per_bar},
                seed=path_seed,
            )
        )
    return paths

def generate_jump_diffusion_paths(
    n_paths: int = 100,
    n_bars: int = 240,
    mu_per_bar: float = 0.0001,
    sigma_per_bar: float = 0.008,
    jump_rate: float = 0.005,
    jump_mean: float = 0.0,
    jump_std: float = 0.03,
    s0: float = 100.0,
    seed: int = 42,
) -> list[SyntheticPath]:
    """
    Merton jump diffusion: GBM + N(jump_mean, jump_std) jumps at Poisson
    intervals. Captures fat tails typical of MOEX news days.
    """
    if not _READY:
        return []

    paths: list[SyntheticPath] = []
    for i in range(n_paths):
        path_seed = seed + i
        rng = np.random.default_rng(path_seed)
        log_ret = rng.normal(mu_per_bar, sigma_per_bar, n_bars)

        jumps = rng.poisson(jump_rate, n_bars)
        jump_sizes = rng.normal(jump_mean, jump_std, n_bars)
        log_ret += jumps * jump_sizes
        close = s0 * np.exp(np.cumsum(log_ret))
        df = _build_ohlcv(close, noise_pct=sigma_per_bar * 0.2, seed=path_seed)
        paths.append(
            SyntheticPath(
                df=df,
                params={
                    "model": "jump_diffusion",
                    "mu": mu_per_bar,
                    "sigma": sigma_per_bar,
                    "jump_rate": jump_rate,
                    "jump_std": jump_std,
                },
                seed=path_seed,
            )
        )
    return paths

def fit_gbm_to_history(df_close: pd.DataFrame) -> tuple[float, float]:
    """
    Fit (mu, sigma) of GBM to a real closing-price series.

    Returns (mu_per_bar, sigma_per_bar) estimated from log-returns.
    """
    if not _READY or df_close is None or len(df_close) < 5:
        return 0.0001, 0.01
    close = df_close["close"].astype(float).to_numpy()
    log_ret = np.diff(np.log(close))
    return float(log_ret.mean()), float(log_ret.std())

__all__ = [
    "SyntheticPath",
    "generate_gbm_paths",
    "generate_jump_diffusion_paths",
    "fit_gbm_to_history",
]
