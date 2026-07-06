"""Confluence filters — Phase 27.2 (v0.0.39).

Lightweight per-pattern gates that reject low-quality TA setups *before* they
reach CatBoost / meta-classifier. Designed for the 41% baseline win-rate;
literature + observed data both say:

  1. Patterns fired on below-average volume are noise. Cut bars below
     `multiplier × rolling20 mean`.
  2. Reversal patterns work in mean-reverting regimes, continuation in
     trending. In crisis (high vol) NOTHING is reliable.
  3. The first 15 min of the MOEX session (10:00-10:15 МСК) are auction
     overflow; the closing window (18:20-18:50 МСК) has thin liquidity
     and lots of position-squaring. Skip both buckets.
  4. ATR-percentile gate eliminates dead-flat and crisis-level volatility:
     pattern fires only when current ATR is in [low_p, high_p] over the
     trailing ~20 days (lookback=480 H1 bars).

Every filter is a *pure* function over (df, idx) / (family, regime) /
(timestamp). All return True when the signal should be kept and False to
veto. NaN-safe and bounds-safe — out-of-range indices return False (=veto).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

def passes_volume_check(
    df: pd.DataFrame,
    idx: int,
    multiplier: float = 1.3,
    lookback: int = 20,
) -> bool:
    """Return True if volume at `idx` is at least `multiplier × rolling20 mean`.

    Missing volume column → True (graceful: we don't penalise tickers
    without volume data). NaN volume → False.
    """
    if not _HAS_PANDAS or df is None or not hasattr(df, "columns"):
        return True
    if "volume" not in df.columns:
        return True
    n = len(df)
    if n == 0:
        return False
    if idx < 0:
        idx = n + idx
    if idx < 0 or idx >= n:
        return False
    if idx < lookback:
        return True
    try:
        vol_now = float(df["volume"].iloc[idx])
        if not np.isfinite(vol_now) or vol_now <= 0:
            return False
        window = df["volume"].iloc[max(0, idx - lookback) : idx]
        if len(window) == 0:
            return True
        mean = float(window.astype(float).mean())
        if not np.isfinite(mean) or mean <= 0:
            return True
        return vol_now >= multiplier * mean
    except Exception as exc:
        logger.debug("passes_volume_check error", extra={"error": str(exc)})
        return True

_REGIME_ALIGNMENT: dict[tuple[str, str], bool] = {
    ("reversal", "mean_reverting"): True,
    ("reversal", "trending"): False,
    ("reversal", "crisis"): False,
    ("continuation", "trending"): True,
    ("continuation", "mean_reverting"): False,
    ("continuation", "crisis"): False,
}

def passes_hmm_alignment(family: str, regime: str) -> bool:
    """Return True if the detector family is aligned with the current regime.

    Reversal patterns require mean-reverting regime; continuation patterns
    require trending.

    v0.19.6 (filter audit, Phase 30): crisis is no longer a blanket veto.
    Literature consensus (volatilitybox.com, quantifiedstrategies.com): even
    in volatility crises, continuation patterns still work — at smaller
    size. Sizing happens upstream via adaptive_regime.size_multiplier;
    here we only veto REVERSAL in crisis (fading a crash is widow-maker
    territory). Continuation + all other families pass through crisis.

    Unknown regime → True (don't crash trading because the model is missing).
    """
    if not isinstance(family, str) or not isinstance(regime, str):
        return True
    regime_norm = regime.strip().lower()
    family_norm = family.strip().lower()
    if regime_norm in ("", "unknown"):
        return True
    if regime_norm == "crisis":
        return family_norm != "reversal"
    if (family_norm, regime_norm) in _REGIME_ALIGNMENT:
        return _REGIME_ALIGNMENT[(family_norm, regime_norm)]
    return True

_MSK_OFFSET = timedelta(hours=3)

def _to_msk(ts_utc: Any) -> datetime | None:
    """Convert UTC datetime / pandas Timestamp / str to naive MSK datetime."""
    if ts_utc is None:
        return None
    if isinstance(ts_utc, (int, float)):
        try:
            ts_utc = datetime.fromtimestamp(float(ts_utc), tz=UTC)
        except (ValueError, OSError):
            return None
    if hasattr(ts_utc, "to_pydatetime"):
        try:
            ts_utc = ts_utc.to_pydatetime()
        except Exception:
            return None
    if isinstance(ts_utc, str):
        try:
            ts_utc = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(ts_utc, datetime):
        return None
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=UTC)
    return (ts_utc + _MSK_OFFSET).replace(tzinfo=None)

def passes_time_of_day(ts_utc: Any) -> bool:
    """Return True if the timestamp is OUTSIDE the no-trade MSK windows.

    Forbidden buckets (False):
      * any time < 10:00 MSK that is also not within the evening session
        (19:05-23:50 MSK is the second leg — also tradeable on MOEX)
      * 10:00-10:15 MSK  (auction-overflow noise)
      * 18:20-18:50 MSK  (close-window scramble)

    Missing/invalid timestamp → True (don't kill signals because the bar
    lost its index). Aggregator/Risk layer enforces stricter session windows
    upstream when needed.
    """
    msk = _to_msk(ts_utc)
    if msk is None:
        return True
    minute_of_day = msk.hour * 60 + msk.minute

    if 600 <= minute_of_day < 615:
        return False
    if 1100 <= minute_of_day <= 1130:
        return False
    morning_open = 600
    evening_open = 19 * 60 + 5
    evening_close = 23 * 60 + 50
    if minute_of_day < morning_open:
        return evening_open <= minute_of_day <= evening_close
    return True

def passes_atr_percentile(
    df: pd.DataFrame,
    idx: int,
    low_p: float = 30.0,
    high_p: float = 90.0,
    lookback: int = 480,
    atr_period: int = 14,
) -> bool:
    """Return True if the current ATR is within [low_p, high_p] percentile
    of the trailing `lookback` window.

    We compute ATR inline (no dependency on indicators cache) to keep this a
    pure helper. NaN / insufficient history → True (be permissive on warm-up).
    """
    if not _HAS_PANDAS or df is None:
        return True
    n = len(df)
    if n == 0:
        return False
    if idx < 0:
        idx = n + idx
    if idx < 0 or idx >= n:
        return False
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(df.columns)):
        return True
    if idx < atr_period + 1:
        return True

    try:
        start = max(0, idx - lookback)
        sub = df.iloc[start : idx + 1]
        if len(sub) < atr_period + 1:
            return True
        high = sub["high"].astype(float)
        low = sub["low"].astype(float)
        close = sub["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()
        if atr.empty:
            return True
        atr_now = float(atr.iloc[-1])
        if not np.isfinite(atr_now) or atr_now <= 0:
            return True
        atr_history = atr.dropna()
        if len(atr_history) < max(atr_period + 1, 30):
            return True
        low_thresh = float(np.percentile(atr_history.values, low_p))
        high_thresh = float(np.percentile(atr_history.values, high_p))
        return low_thresh <= atr_now <= high_thresh
    except Exception as exc:
        logger.debug("passes_atr_percentile error", extra={"error": str(exc)})
        return True
