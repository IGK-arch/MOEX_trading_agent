"""
app/agents/microstructure/kyles_lambda.py — Kyle's lambda price-impact estimator.

Definition (Kyle 1985):
    Δp_i = λ · q_i + ε_i
    where q_i is the signed volume of bar i (positive = net buy, negative = net sell)
    and λ is the regression slope.

In practice we use signed dollar-volume:
    Δp_i = λ · (vol_b - vol_s)_i

A high lambda means each unit of imbalanced flow moves the price more — that's
characteristic of an informed market environment, which is dangerous for our
liquidity-taker style. We compare current lambda to its 30-day 95th percentile
in `MicrostructureGates`.

This implementation uses pure-numpy least squares over the last `window` bars.
"""

from __future__ import annotations

try:
    import numpy as np
    import pandas as pd

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

def compute_kyles_lambda(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    vol_b_col: str = "vol_b",
    vol_s_col: str = "vol_s",
    window: int = 20,
    min_obs: int = 5,
) -> float:
    """
    Estimate Kyle's lambda over the last `window` bars.

    Returns the regression slope of Δp on signed_volume. If we don't have
    `vol_b/vol_s`, falls back to using `volume` × sign(Δp) (cruder but signed).

    Returns 0.0 when not enough data or input columns missing.
    """
    if not _HAS_NUMPY or df is None or len(df) < min_obs:
        return 0.0

    sub = df.tail(window).copy()
    if close_col not in sub.columns or len(sub) < min_obs:
        return 0.0
    close = sub[close_col].astype(float).to_numpy()
    dp = np.diff(close)
    if len(dp) < min_obs - 1:
        return 0.0

    if vol_b_col in sub.columns and vol_s_col in sub.columns:
        signed = (
            sub[vol_b_col].fillna(0).astype(float).to_numpy()
            - sub[vol_s_col].fillna(0).astype(float).to_numpy()
        )[1:]
    elif "volume" in sub.columns:
        signed = sub["volume"].fillna(0).astype(float).to_numpy()[1:] * np.sign(dp)
    else:
        return 0.0

    if len(signed) != len(dp):
        return 0.0

    x = signed
    y = dp
    x_mean = x.mean()
    y_mean = y.mean()
    var_x = ((x - x_mean) ** 2).sum()
    if var_x <= 0:
        return 0.0
    cov_xy = ((x - x_mean) * (y - y_mean)).sum()
    lam = cov_xy / var_x
    return float(lam)

__all__ = ["compute_kyles_lambda"]
