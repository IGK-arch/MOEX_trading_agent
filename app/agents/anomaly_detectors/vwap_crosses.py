"""
app/agents/anomaly_detectors/vwap_crosses.py

Detects daily-VWAP crossings with 2-minute confirmation.
"""

from __future__ import annotations

from app.utils.logging import get_logger

from .base import AnomalySignal

logger = get_logger(__name__)

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    _READY = True
except ImportError:
    _READY = False

def _intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP from the start of the slice."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    cum_tpv = (typical * df["volume"]).cumsum()
    return cum_tpv / cum_vol

def detect_vwap_crosses(
    df: pd.DataFrame,
    ticker: str,
    atr: pd.Series,
    confirm_bars: int = 2,
    min_cross_atr: float = 0.1,
) -> list[AnomalySignal]:
    """
    Find VWAP cross-up / cross-down within recent bars.
    Confirmation: close stays on the new side for `confirm_bars` bars.

    NOTE: assumes df already represents one trading day (intraday slice).
    For multi-day df, only the last day's VWAP would be meaningful — caller
    should pass a single-day slice in production.
    """
    if not _READY or df is None or len(df) < 10 or "volume" not in df.columns:
        return []

    vwap = _intraday_vwap(df)
    close = df["close"]

    diff = close - vwap
    cross_up = (diff > 0) & (diff.shift(1) <= 0)
    cross_down = (diff < 0) & (diff.shift(1) >= 0)

    signals: list[AnomalySignal] = []
    n = len(df)
    start = max(5, n - 10 - confirm_bars)

    for idx in range(start, n - confirm_bars):
        atr_val = float(atr.iloc[idx]) if idx < len(atr) and pd.notna(atr.iloc[idx]) else 0.0
        if atr_val <= 0:
            continue

        if cross_up.iloc[idx]:
            ok = all(
                float(close.iloc[idx + k]) >= float(vwap.iloc[idx + k]) + min_cross_atr * atr_val
                for k in range(1, confirm_bars + 1)
            )
            if ok:
                signals.append(
                    AnomalySignal(
                        ticker=ticker,
                        detector="vwap_cross_up",
                        direction="BUY",
                        confidence=0.55,
                        ts=df["begin"].iloc[idx + confirm_bars] if "begin" in df.columns else None,
                        price=float(close.iloc[idx + confirm_bars]),
                        volume=float(df["volume"].iloc[idx + confirm_bars]),
                        atr=atr_val,
                        bar_idx=int(idx + confirm_bars),
                        metadata={
                            "vwap_at_cross": round(float(vwap.iloc[idx]), 3),
                            "confirm_bars": confirm_bars,
                        },
                    )
                )
        elif cross_down.iloc[idx]:
            ok = all(
                float(close.iloc[idx + k]) <= float(vwap.iloc[idx + k]) - min_cross_atr * atr_val
                for k in range(1, confirm_bars + 1)
            )
            if ok:
                signals.append(
                    AnomalySignal(
                        ticker=ticker,
                        detector="vwap_cross_down",
                        direction="SELL",
                        confidence=0.55,
                        ts=df["begin"].iloc[idx + confirm_bars] if "begin" in df.columns else None,
                        price=float(close.iloc[idx + confirm_bars]),
                        volume=float(df["volume"].iloc[idx + confirm_bars]),
                        atr=atr_val,
                        bar_idx=int(idx + confirm_bars),
                        metadata={
                            "vwap_at_cross": round(float(vwap.iloc[idx]), 3),
                            "confirm_bars": confirm_bars,
                        },
                    )
                )

    if signals:
        logger.debug("vwap_cross signals", extra={"ticker": ticker, "count": len(signals)})
    return signals
