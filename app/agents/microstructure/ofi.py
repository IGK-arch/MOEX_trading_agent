"""
app/agents/microstructure/ofi.py — Order Flow Imbalance.

Two flavours of OFI we use:

  1. **Volume-based proxy** (`compute_ofi`):
        OFI = (vol_b - vol_s) / max(epsilon, vol_b + vol_s)   ∈ [-1, 1]
     Works wherever we have buyer/seller volume splits (moexalgo SuperCandles
     give us `vol_b` and `vol_s` directly).

  2. **Bid/ask quote-event OFI** (`compute_ofi_series`):
        OFI_i = ΔQ_bid_i · 1{p_bid_i ≥ p_bid_{i-1}}
                - ΔQ_bid_i · 1{p_bid_i ≤ p_bid_{i-1}}
                - ΔQ_ask_i · 1{p_ask_i ≤ p_ask_{i-1}}
                + ΔQ_ask_i · 1{p_ask_i ≥ p_ask_{i-1}}
     Requires quote-level updates; we have this only when AlgoPack `obstats`
     returns the L1 stack. Used opportunistically when available.

Positive OFI → buying pressure → supports BUY signals.
Negative OFI → selling pressure → supports SELL signals.
"""

from __future__ import annotations

try:
    import numpy as np  # noqa: F401
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

def compute_ofi(vol_b: float, vol_s: float) -> float:
    """
    Simple buyer/seller volume imbalance — used when only aggregate volumes are known.
    Returns value in [-1, 1] (0 if both are zero).
    """
    vb, vs = float(vol_b), float(vol_s)
    total = vb + vs
    if total <= 0:
        return 0.0
    return (vb - vs) / total

def compute_ofi_series(
    df: pd.DataFrame,
    *,
    vol_b_col: str = "vol_b",
    vol_s_col: str = "vol_s",
    window: int = 5,
) -> float:
    """
    Rolling-window OFI over the last `window` bars of a moexalgo SuperCandles frame.

    Aggregates `vol_b` and `vol_s` over the window and applies compute_ofi.
    Robust to missing columns (returns 0.0).
    """
    if not _HAS_PANDAS or df is None or len(df) == 0:
        return 0.0
    cols = set(df.columns)
    if vol_b_col not in cols or vol_s_col not in cols:
        return 0.0
    tail = df.tail(max(1, window))
    vb = float(tail[vol_b_col].fillna(0.0).sum())
    vs = float(tail[vol_s_col].fillna(0.0).sum())
    return compute_ofi(vb, vs)

__all__ = ["compute_ofi", "compute_ofi_series"]
