"""Phase 27.10 — Soft Sharpe-filter for low-quality TA detectors.

Acts as a SOFT companion to :mod:`app.agents.ta_patterns.noise_blacklist`
(which HARD-mutes a detector via ``magnitude × 0.3``). The Sharpe filter
applies a milder ``× 0.6`` haircut to detectors that survive the hard
veto but still show a low risk-adjusted edge — i.e. statistically
non-noise but marginal.

A detector is added to :data:`LOW_SHARPE_DETECTORS` when, on the latest
``data/training_cache/detector_rankings_after.json``:

* it accumulated **at least 30 trades** (statistically observable), AND
* its profit-factor is **below 1.2** (proxy for Sharpe < 0.5), AND
* it is **not** already on :data:`noise_blacklist.STATIC_NOISE_PATTERNS`
  (those are already hard-muted to ``× 0.3``).

The filter is composable with the noise blacklist: the multipliers stack
(``× 0.3 × 0.6 = 0.18`` if both fire). In practice noise-blacklisted
detectors are skipped (return ``1.0`` from this filter) so the two are
disjoint.
"""

from __future__ import annotations

from app.agents.ta_patterns.noise_blacklist import is_noisy

LOW_SHARPE_DETECTORS: frozenset[str] = frozenset(
    {
        "cdl_dojistar",
        "double_bottom",
        "cdl_shootingstar",
        "cdl_eveningdojistar",
    }
)

LOW_SHARPE_HAIRCUT: float = 0.6

def detector_magnitude_haircut(detector: str) -> float:
    """Return the magnitude multiplier for ``detector``.

    Pure function. Returns ``1.0`` (no change) when:
      * ``detector`` is empty,
      * the detector is already noise-blacklisted (avoid double penalty),
      * the detector is not on :data:`LOW_SHARPE_DETECTORS`.

    Returns :data:`LOW_SHARPE_HAIRCUT` otherwise.

    Args:
        detector: detector / pattern name (case-insensitive).
    Returns:
        float: multiplier in ``[0, 1]`` to scale ``magnitude``.
    """
    if not detector:
        return 1.0
    name = detector.lower()
    if is_noisy(name):
        return 1.0
    if name in LOW_SHARPE_DETECTORS:
        return LOW_SHARPE_HAIRCUT
    return 1.0

__all__ = [
    "LOW_SHARPE_DETECTORS",
    "LOW_SHARPE_HAIRCUT",
    "detector_magnitude_haircut",
]
