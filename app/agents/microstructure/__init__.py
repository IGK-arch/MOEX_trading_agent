"""
app/agents/microstructure — Pure-numpy market microstructure metrics.

Three core estimators used as defensive gates and as meta-classifier features:

  - OFI (Order Flow Imbalance) — Cont, Cucuringu, Zhang (2014):
        OFI = (vol_b - vol_s) / (vol_b + vol_s)
    Predictive of short-horizon mid-price movement (3-min, accuracy 57-63%
    on liquid equities — see arxiv:2411.08382).

  - Kyle's lambda — measures permanent price impact per unit volume.
    High λ ⇒ informed flow dominates ⇒ blue-chip alpha decays fast.
    Used as a defensive gate: skip entries when λ > rolling-30d 95th percentile.

  - VPIN (Volume-synchronized PIN) — Easley, Lopez de Prado, O'Hara (2012):
    Aggregate trades into N equal-volume buckets, then average
    abs(buy_vol - sell_vol) / total_vol across buckets.
    VPIN > 0.45 historically precedes flash crashes / informed runs.

All functions take pandas DataFrames or numpy arrays and return floats.
No external dependencies beyond numpy / pandas.
"""

from app.agents.microstructure.kyles_lambda import compute_kyles_lambda
from app.agents.microstructure.ofi import compute_ofi, compute_ofi_series
from app.agents.microstructure.vpin import compute_vpin

__all__ = [
    "compute_ofi",
    "compute_ofi_series",
    "compute_kyles_lambda",
    "compute_vpin",
]
