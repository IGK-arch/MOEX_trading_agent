"""
app/agents/anomaly_detectors/base.py — Base type for microstructure anomalies.

Each of the 6 detectors returns a list[AnomalySignal].
Detectors that only produce alerts (no actionable direction) emit direction=NEUTRAL
— these are still useful as `context` for News LLM and morning_plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class AnomalySignal:
    """
    A single microstructure anomaly detection.

    Fields:
        ticker:    Symbol the anomaly is on (e.g. "SBER")
        detector:  Name of the producing detector (e.g. "volume_zscore", "ofi_spike")
        direction: "BUY" | "SELL" | "NEUTRAL" — NEUTRAL = context only, no trade
        confidence: 0.0–1.0
        ts:        UTC timestamp of the anomaly bar
        price:     Price at detection (close)
        volume:    Volume at detection bar
        atr:       ATR at detection bar (for sizing context)
        bar_idx:   Index in df (helpful for backtest replay)
        metadata:  Detector-specific fields (e.g. z_score, ofi_value)
    """

    ticker: str
    detector: str
    direction: str
    confidence: float
    ts: Any
    price: float
    volume: float = 0.0
    atr: float = 0.0
    bar_idx: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post init."""
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.direction not in ("BUY", "SELL", "NEUTRAL"):
            self.direction = "NEUTRAL"

    def is_actionable(self) -> bool:
        """Is actionable."""
        return self.direction in ("BUY", "SELL")
