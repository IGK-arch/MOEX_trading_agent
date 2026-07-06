"""
app/agents/microstructure_gates.py — Defensive gates BEFORE meta-classifier.

Three independent gates, applied in order. Each returns a `GateResult` that
the aggregator inspects:

  1. **VPIN gate** — block when volume-synchronized PIN > cfg.VPIN_BLOCK_THRESHOLD
     (default 0.45 per Easley/Lopez de Prado paper). VPIN > 0.45 historically
     precedes flash crashes & informed runs.

  2. **Kyle's lambda gate** — block when current lambda exceeds the 30-day
     95th percentile. Stored in a rolling cache per ticker (in-memory + SQLite).

  3. **OFI-direction gate** — when OFI strongly contradicts decision direction
     (|OFI| > cfg.OFI_OPPOSITION_THRESHOLD AND opposite sign), the gate
     downgrades to "weaken" (×0.7 magnitude) rather than block outright.
     This prefers conservative continuation over outright rejection.

Failure mode: when SuperCandles unavailable for a ticker, all gates default
to "pass" — we don't block on missing data. This is the safe default since
the existing TA / pair / news signals are already filtered.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import app.config as cfg
from app.agents.microstructure import (
    compute_kyles_lambda,
    compute_ofi_series,
    compute_vpin,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class GateResult:
    """Outcome of MicrostructureGates.check() for one (ticker, direction)."""

    blocked: bool = False
    weakened: bool = False
    reason: str = "ok"
    ofi: float = 0.0
    kyles_lambda: float = 0.0
    vpin: float = 0.0
    weakening_multiplier: float = 1.0

class MicrostructureGates:
    """
    Stateful gate engine — keeps per-ticker rolling history of Kyle's lambda
    so we can compute the 95th-percentile threshold without hitting moexalgo
    again on every cycle.

    History window is intentionally short (last `kyles_history_size` values)
    to adapt fast to changing regimes.
    """

    def __init__(self, kyles_history_size: int = 300) -> None:
        """Init."""
        self._lam_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=kyles_history_size)
        )

    async def check(
        self,
        ticker: str,
        direction: str,
        supercandles_df: Any | None = None,
    ) -> GateResult:
        """
        Apply all three gates. Returns the combined GateResult.

        `supercandles_df` is the moexalgo DataFrame for this ticker. When None
        the caller is signaling that data is unavailable — return pass-through.
        """
        if not cfg.MICROSTRUCTURE_GATES_ENABLED:
            return GateResult(reason="gates_disabled")
        if supercandles_df is None or len(supercandles_df) < 5:
            return GateResult(reason="no_data")

        vpin = compute_vpin(supercandles_df, n_buckets=cfg.VPIN_N_BUCKETS)

        lam = compute_kyles_lambda(
            supercandles_df,
            window=cfg.KYLES_LAMBDA_WINDOW,
        )
        self._lam_history[ticker].append(abs(lam))

        history = list(self._lam_history[ticker])
        lam_p95 = 0.0
        if len(history) >= 20:
            sorted_hist = sorted(history)
            idx = int(len(sorted_hist) * cfg.KYLES_LAMBDA_BLOCK_PCT)
            lam_p95 = sorted_hist[min(idx, len(sorted_hist) - 1)]

        ofi = compute_ofi_series(
            supercandles_df,
            window=cfg.OFI_WINDOW_BARS,
        )

        if vpin > cfg.VPIN_BLOCK_THRESHOLD:
            return GateResult(
                blocked=True,
                reason=f"vpin {vpin:.2f} > {cfg.VPIN_BLOCK_THRESHOLD:.2f}",
                ofi=ofi,
                kyles_lambda=lam,
                vpin=vpin,
            )
        if lam_p95 > 0 and abs(lam) > lam_p95:
            return GateResult(
                blocked=True,
                reason=f"kyles |λ| {abs(lam):.4g} > p95 {lam_p95:.4g}",
                ofi=ofi,
                kyles_lambda=lam,
                vpin=vpin,
            )

        dir_upper = direction.upper()
        if abs(ofi) > cfg.OFI_OPPOSITION_THRESHOLD:
            opposes = (dir_upper == "BUY" and ofi < -cfg.OFI_OPPOSITION_THRESHOLD) or (
                dir_upper == "SELL" and ofi > cfg.OFI_OPPOSITION_THRESHOLD
            )
            if opposes:
                return GateResult(
                    blocked=False,
                    weakened=True,
                    reason=f"ofi {ofi:.2f} opposes {dir_upper}",
                    ofi=ofi,
                    kyles_lambda=lam,
                    vpin=vpin,
                    weakening_multiplier=cfg.OFI_OPPOSITION_WEAKEN_MULT,
                )

        return GateResult(
            blocked=False,
            weakened=False,
            reason="ok",
            ofi=ofi,
            kyles_lambda=lam,
            vpin=vpin,
        )

_gates: MicrostructureGates | None = None

def get_microstructure_gates() -> MicrostructureGates:
    """Get microstructure gates."""
    global _gates
    if _gates is None:
        _gates = MicrostructureGates()
    return _gates

__all__ = ["MicrostructureGates", "GateResult", "get_microstructure_gates"]
