"""
app/agents/anomaly_detectors/price_spikes.py

Detects false breakouts ("прострелы"):
  1. Bar with range > 2×ATR_1m AND volume > 3×average
  2. Within the next 2-3 bars, price retraces ≥ 50% of the bar's range
  3. → Direction = opposite of spike direction (BUY for failed breakdown, SELL for failed breakout)
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

def detect_price_spikes(
    df: pd.DataFrame,
    ticker: str,
    atr: pd.Series,
    range_atr_threshold: float = 2.0,
    vol_threshold_x: float = 3.0,
    retrace_threshold: float = 0.5,
    lookback_bars: int = 3,
) -> list[AnomalySignal]:
    """
    Detect false breakouts in recent bars.
    """
    if not _READY or df is None or len(df) < 30 or atr is None or len(atr) == 0:
        return []

    bar_range = df["high"] - df["low"]
    vol = df["volume"].astype(float) if "volume" in df.columns else None
    if vol is None:
        return []
    vol_avg = vol.rolling(20).mean().replace(0, 1)
    vol_ratio = vol / vol_avg

    signals: list[AnomalySignal] = []
    n = len(df)

    start = max(20, n - 10 - lookback_bars)
    end = max(start + 1, n - lookback_bars)
    for idx in range(start, end):
        atr_val = float(atr.iloc[idx]) if idx < len(atr) and pd.notna(atr.iloc[idx]) else 0.0
        if atr_val <= 0:
            continue

        rng = float(bar_range.iloc[idx])
        if rng < range_atr_threshold * atr_val:
            continue
        if float(vol_ratio.iloc[idx]) < vol_threshold_x:
            continue

        spike_high = float(df["high"].iloc[idx])
        spike_low = float(df["low"].iloc[idx])
        spike_close = float(df["close"].iloc[idx])
        spike_open = float(df["open"].iloc[idx])

        spike_up = spike_close > spike_open
        retrace_needed = retrace_threshold * rng

        for retr_idx in range(idx + 1, min(idx + 1 + lookback_bars, n)):
            r_close = float(df["close"].iloc[retr_idx])
            if spike_up:
                pulled_back = spike_high - r_close
                if pulled_back >= retrace_needed:
                    signals.append(
                        AnomalySignal(
                            ticker=ticker,
                            detector="price_spike_failed",
                            direction="SELL",
                            confidence=min(1.0, 0.55 + (rng / atr_val - 2.0) * 0.10),
                            ts=df["begin"].iloc[retr_idx] if "begin" in df.columns else None,
                            price=r_close,
                            volume=float(vol.iloc[idx]),
                            atr=atr_val,
                            bar_idx=retr_idx,
                            metadata={
                                "spike_bar_idx": idx,
                                "spike_range_atrs": round(rng / atr_val, 2),
                                "vol_ratio": round(float(vol_ratio.iloc[idx]), 2),
                                "retrace_pct": round(pulled_back / rng * 100, 1),
                            },
                        )
                    )
                    break
            else:
                pulled_back = r_close - spike_low
                if pulled_back >= retrace_needed:
                    signals.append(
                        AnomalySignal(
                            ticker=ticker,
                            detector="price_spike_failed",
                            direction="BUY",
                            confidence=min(1.0, 0.55 + (rng / atr_val - 2.0) * 0.10),
                            ts=df["begin"].iloc[retr_idx] if "begin" in df.columns else None,
                            price=r_close,
                            volume=float(vol.iloc[idx]),
                            atr=atr_val,
                            bar_idx=retr_idx,
                            metadata={
                                "spike_bar_idx": idx,
                                "spike_range_atrs": round(rng / atr_val, 2),
                                "vol_ratio": round(float(vol_ratio.iloc[idx]), 2),
                                "retrace_pct": round(pulled_back / rng * 100, 1),
                            },
                        )
                    )
                    break

    if signals:
        logger.debug("price_spike_failed signals", extra={"ticker": ticker, "count": len(signals)})
    return signals
