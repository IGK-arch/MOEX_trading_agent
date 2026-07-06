"""
app/agents/microstructure/vpin.py — Volume-synchronized PIN (toxic-flow indicator).

Reference: Easley, López de Prado, O'Hara (2012),
"Flow Toxicity and Liquidity in a High-Frequency World".

Procedure
---------
1. Aggregate trades into N equal-volume "buckets" (we use bar-volume here as
   a practical approximation — exact tick-by-tick requires per-trade data).
2. For each bucket, compute:
       a_i = |buy_vol - sell_vol|_i  /  total_vol_i
3. VPIN = mean(a_i) over the most recent `lookback` buckets.

Range: [0, 1]. Empirically VPIN > 0.45 precedes flash crashes / informed
runs on liquid US equities; we use that as a defensive gate.

When sell/buy split isn't available (only `volume` and price), fall back
to using bulk-volume classification by sign of Δprice within the bucket.
"""

from __future__ import annotations

try:
    import numpy as np
    import pandas as pd

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

def compute_vpin(
    df: pd.DataFrame,
    *,
    n_buckets: int = 50,
    close_col: str = "close",
    vol_b_col: str = "vol_b",
    vol_s_col: str = "vol_s",
    volume_col: str = "volume",
) -> float:
    """
    Compute VPIN over the last `n_buckets` bars of a SuperCandles DataFrame.

    Returns value in [0, 1] (or 0.0 if not enough data).
    """
    if not _HAS_NUMPY or df is None or len(df) == 0:
        return 0.0

    sub = df.tail(n_buckets).copy()
    if len(sub) < 2:
        return 0.0

    has_explicit = vol_b_col in sub.columns and vol_s_col in sub.columns
    if has_explicit:
        vb = sub[vol_b_col].fillna(0.0).astype(float).to_numpy()
        vs = sub[vol_s_col].fillna(0.0).astype(float).to_numpy()
        total = vb + vs

        mask = total > 0
        if not mask.any():
            return 0.0
        a = np.where(mask, np.abs(vb - vs) / np.maximum(total, 1e-12), 0.0)
        return float(np.clip(a[mask].mean(), 0.0, 1.0))

    if close_col not in sub.columns or volume_col not in sub.columns:
        return 0.0
    close = sub[close_col].astype(float).to_numpy()
    vol = sub[volume_col].astype(float).to_numpy()
    dp = np.diff(close)

    vol_aligned = vol[1:]
    if len(vol_aligned) == 0:
        return 0.0
    sign = np.sign(dp)
    buy_v = np.where(sign > 0, vol_aligned, 0.0)
    sell_v = np.where(sign < 0, vol_aligned, 0.0)
    total = buy_v + sell_v
    mask = total > 0
    if not mask.any():
        return 0.0
    a = np.where(mask, np.abs(buy_v - sell_v) / np.maximum(total, 1e-12), 0.0)
    return float(np.clip(a[mask].mean(), 0.0, 1.0))

__all__ = ["compute_vpin"]
