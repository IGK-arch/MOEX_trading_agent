"""
app/agents/anomaly_detectors/atr_reversion.py

Detects ATR-distance reversions:
  1. Within a 10-bar window, find a move ≥ 2×ATR from window-open price.
  2. Then within next 5 bars, if a retrace of ≥ 40% of the move happens →
     emit signal in REVERSION direction (opposite to original move).
  3. Cooldown 45 minutes per ticker (≈ 9 bars on 5m).
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

def detect_atr_reversion(
    df: pd.DataFrame,
    ticker: str,
    atr: pd.Series,
    move_atr_threshold: float = 2.0,
    retrace_pct: float = 0.4,
    move_window: int = 10,
    retrace_window: int = 5,
    cooldown_bars: int = 9,
    last_signal_idx: int = -100,
) -> list[AnomalySignal]:
    """
    Find recent ATR-distance reversions.

    Args:
        last_signal_idx: bar index of the previous signal for this ticker
                         (caller provides to enforce cooldown across polls).
    """
    if not _READY or df is None or len(df) < move_window + retrace_window + 1:
        return []
    if atr is None or len(atr) == 0:
        return []

    signals: list[AnomalySignal] = []
    n = len(df)

    earliest = max(20, n - 20 - retrace_window)
    for win_start in range(earliest, n - retrace_window - 1):
        atr_val = (
            float(atr.iloc[win_start])
            if win_start < len(atr) and pd.notna(atr.iloc[win_start])
            else 0.0
        )
        if atr_val <= 0:
            continue

        open_price = float(df["open"].iloc[win_start])

        end_w = min(n - retrace_window - 1, win_start + move_window)
        if end_w <= win_start + 1:
            continue

        window = df.iloc[win_start : end_w + 1]
        max_up = float(window["high"].max()) - open_price
        max_down = open_price - float(window["low"].min())

        if max(max_up, max_down) < move_atr_threshold * atr_val:
            continue

        is_up_move = max_up > max_down
        excursion = max_up if is_up_move else max_down

        extreme_idx = int(window["high"].idxmax()) if is_up_move else int(window["low"].idxmin())

        if extreme_idx - last_signal_idx < cooldown_bars:
            continue

        retrace_target = retrace_pct * excursion
        end_r = min(n, extreme_idx + 1 + retrace_window)
        for ri in range(extreme_idx + 1, end_r):
            r_close = float(df["close"].iloc[ri])
            if is_up_move:
                pulled_back = float(df["high"].iloc[extreme_idx]) - r_close
                if pulled_back >= retrace_target:
                    direction = "SELL"
                    break
            else:
                pulled_back = r_close - float(df["low"].iloc[extreme_idx])
                if pulled_back >= retrace_target:
                    direction = "BUY"
                    break
        else:
            continue

        signals.append(
            AnomalySignal(
                ticker=ticker,
                detector="atr_reversion",
                direction=direction,
                confidence=min(0.85, 0.50 + (excursion / atr_val - 2.0) * 0.08),
                ts=df["begin"].iloc[ri] if "begin" in df.columns else None,
                price=float(df["close"].iloc[ri]),
                volume=float(df["volume"].iloc[ri]) if "volume" in df.columns else 0.0,
                atr=atr_val,
                bar_idx=int(ri),
                metadata={
                    "excursion_atrs": round(excursion / atr_val, 2),
                    "retrace_pct": round(pulled_back / excursion * 100, 1),
                    "extreme_bar_idx": extreme_idx,
                    "is_up_move": is_up_move,
                },
            )
        )

        break

    if signals:
        logger.debug("atr_reversion signals", extra={"ticker": ticker, "count": len(signals)})
    return signals
