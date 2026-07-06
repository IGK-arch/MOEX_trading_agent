"""
app/agents/anomaly_detectors/ofi_spikes.py

Detects Order Flow Imbalance spikes.

Primary source: AlgoPack obstats (`imbalance_vol_bbo` field).
Fallback: ISS candle-based OFI approximation (built into AlgoPackClient._fallback_ofi):
    ofi_approx = (close - open) / (high - low) * volume

Cycle-5: when caller passes ``threshold_pct=None`` the function resolves the
threshold via :func:`app.config.ofi_threshold_for_ticker` (per-ticker override
when ``OFI_PER_TICKER_THRESHOLDS=True``, else the global ``ANOMALY_OFI_THRESHOLD``).
Backward compatibility: any explicit numeric ``threshold_pct`` keeps its previous
hard-coded behaviour.
"""

from __future__ import annotations

import app.config as cfg
from app.utils.logging import get_logger

from .base import AnomalySignal

logger = get_logger(__name__)

try:
    import pandas as pd  # type: ignore

    _READY = True
except ImportError:
    _READY = False

def detect_ofi_spikes(
    obstats_df: pd.DataFrame,
    ticker: str,
    threshold_pct: float | None = None,
    window_seconds: int = 30,
) -> list[AnomalySignal]:
    """
    Detect bars where order flow imbalance > threshold_pct (e.g. 60%).

    Args:
        obstats_df: DataFrame from AlgoPackClient.get_obstats() — has
                    `imbalance_vol_bbo` field (or `imbalance_vol`) in real-time,
                    or the fallback contains `imbalance_vol_bbo` (synthetic).
        ticker:     symbol
        threshold_pct: explicit override; if ``None`` (default), resolved via
            :func:`app.config.ofi_threshold_for_ticker` — per-ticker map when
            ``OFI_PER_TICKER_THRESHOLDS=True``, else the global
            ``ANOMALY_OFI_THRESHOLD`` (0.60).
    """
    if not _READY or obstats_df is None or len(obstats_df) == 0:
        return []

    if threshold_pct is None:
        threshold_pct = cfg.ofi_threshold_for_ticker(ticker)

    field_name: str | None = None
    for fn in ("imbalance_vol_bbo", "imbalance_vol", "ofi_approx"):
        if fn in obstats_df.columns:
            field_name = fn
            break
    if field_name is None:
        return []

    signals: list[AnomalySignal] = []
    n = len(obstats_df)
    start = max(0, n - 5)

    for idx in range(start, n):
        ofi = obstats_df[field_name].iloc[idx]
        if pd.isna(ofi):
            continue
        ofi_f = float(ofi)

        if abs(ofi_f) < threshold_pct:
            continue

        direction = "BUY" if ofi_f > 0 else "SELL"

        price = 0.0
        for pcol in ("vwap_b", "vwap_s", "vwap", "close", "price"):
            if pcol in obstats_df.columns:
                p = obstats_df[pcol].iloc[idx]
                if pd.notna(p):
                    price = float(p)
                    break

        ts = None
        for tcol in ("ts", "begin", "tradetime"):
            if tcol in obstats_df.columns:
                ts = obstats_df[tcol].iloc[idx]
                break

        signals.append(
            AnomalySignal(
                ticker=ticker,
                detector="ofi_spike",
                direction=direction,
                confidence=min(1.0, abs(ofi_f) * 1.2),
                ts=ts,
                price=price,
                volume=0.0,
                bar_idx=int(idx),
                metadata={
                    "ofi_value": round(ofi_f, 3),
                    "field": field_name,
                    "threshold": threshold_pct,
                },
            )
        )

    if signals:
        logger.debug("ofi_spike signals", extra={"ticker": ticker, "count": len(signals)})
    return signals
