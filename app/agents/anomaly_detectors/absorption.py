"""
app/agents/anomaly_detectors/absorption.py

Detects absorption — large volume that's absorbed by the opposite side without moving price:
  - Volume Z-score > 2.5 (vs 100-bar history)
  - Price change < 0.3×ATR_1m on that bar

Direction: opposite of the side that was absorbed.
  - If close < open (sellers were strong but price did not drop much) → BUY
    (buyers absorbed the selling, expect rebound)
  - If close > open (buyers strong but price barely rose) → SELL
"""

from __future__ import annotations

from app.utils.logging import get_logger

from .base import AnomalySignal

logger = get_logger(__name__)

try:
    import pandas as pd  # type: ignore

    _READY = True
except ImportError:
    _READY = False

def detect_absorption(
    df: pd.DataFrame,
    ticker: str,
    atr: pd.Series,
    vol_z_threshold: float = 2.5,
    max_move_atr: float = 0.3,
    window: int = 100,
) -> list[AnomalySignal]:
    """
    Detect absorption bars in recent data (last 5 bars).
    """
    if not _READY or df is None or len(df) < window or atr is None:
        return []
    if "volume" not in df.columns:
        return []

    vol = df["volume"].astype(float)
    vol_mean = vol.rolling(window).mean().replace(0, 1)
    vol_std = vol.rolling(window).std().replace(0, 1)
    vol_z = (vol - vol_mean) / vol_std

    signals: list[AnomalySignal] = []
    start = max(window, len(df) - 5)

    for idx in range(start, len(df)):
        atr_val = float(atr.iloc[idx]) if idx < len(atr) and pd.notna(atr.iloc[idx]) else 0.0
        if atr_val <= 0:
            continue

        z = float(vol_z.iloc[idx])
        if pd.isna(z) or z < vol_z_threshold:
            continue

        bar_move = abs(float(df["close"].iloc[idx]) - float(df["open"].iloc[idx]))
        if bar_move > max_move_atr * atr_val:
            continue

        o = float(df["open"].iloc[idx])
        c = float(df["close"].iloc[idx])

        if c < o:
            direction = "BUY"
        elif c > o:
            direction = "SELL"
        else:
            direction = "NEUTRAL"

        signals.append(
            AnomalySignal(
                ticker=ticker,
                detector="absorption",
                direction=direction,
                confidence=min(1.0, 0.55 + (z - vol_z_threshold) * 0.08),
                ts=df["begin"].iloc[idx] if "begin" in df.columns else None,
                price=c,
                volume=float(vol.iloc[idx]),
                atr=atr_val,
                bar_idx=int(idx),
                metadata={
                    "vol_z": round(z, 2),
                    "bar_move_atrs": round(bar_move / atr_val, 3),
                },
            )
        )

    if signals:
        logger.debug("absorption signals", extra={"ticker": ticker, "count": len(signals)})
    return signals
