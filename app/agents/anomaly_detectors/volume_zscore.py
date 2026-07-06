"""
app/agents/anomaly_detectors/volume_zscore.py

Detects abnormal volume bars: Z-score vs the trailing rolling window.

Direction: NEUTRAL (context only — "something is happening").
The TATrader / NewsLLM combination uses this to gate signals (e.g. confirm a flag
breakout only if volume_z > 1.5).
"""

from __future__ import annotations

from app.utils.logging import get_logger

from .base import AnomalySignal

logger = get_logger(__name__)

try:
    import numpy as np  # noqa: F401  # type: ignore
    import pandas as pd  # type: ignore

    _READY = True
except ImportError:
    _READY = False

def detect_volume_zscore(
    df: pd.DataFrame,
    ticker: str,
    z_threshold: float = 3.0,
    window: int = 30,
) -> list[AnomalySignal]:
    """
    Returns NEUTRAL anomalies on any bar where volume Z-score exceeds threshold.

    Args:
        df:           5m OHLCV DataFrame
        ticker:       Symbol
        z_threshold:  e.g. 3.0 = 99.7% percentile
        window:       Rolling window length in bars (30 × 5m = 2.5h)
    """
    if not _READY or df is None or len(df) < window + 1 or "volume" not in df.columns:
        return []

    vol = df["volume"].astype(float)
    roll_mean = vol.rolling(window).mean()
    roll_std = vol.rolling(window).std().replace(0, 1)
    z = (vol - roll_mean) / roll_std

    signals: list[AnomalySignal] = []
    recent_start = max(0, len(df) - 5)
    for idx in range(recent_start, len(df)):
        z_val = z.iloc[idx]
        if pd.isna(z_val) or abs(z_val) < z_threshold:
            continue

        signals.append(
            AnomalySignal(
                ticker=ticker,
                detector="volume_zscore",
                direction="NEUTRAL",
                confidence=min(1.0, abs(float(z_val)) / 6.0),
                ts=df["begin"].iloc[idx] if "begin" in df.columns else None,
                price=float(df["close"].iloc[idx]),
                volume=float(vol.iloc[idx]),
                bar_idx=int(idx),
                metadata={"z_score": round(float(z_val), 2), "window": window},
            )
        )

    if signals:
        logger.debug("volume_zscore signals", extra={"ticker": ticker, "count": len(signals)})
    return signals
