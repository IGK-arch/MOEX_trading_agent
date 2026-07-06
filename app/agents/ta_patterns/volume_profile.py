"""Volume Profile (VPVR) detector — Phase 27 / v0.0.38.

Volume Profile Visible Range is a classical price-action edge that maps how
volume is distributed across price (not time) over a recent lookback window.
The output is a histogram of (price_low, price_high, total_volume) bins.

Two structural concepts drive the signal:

  - **HVN** (High-Volume Node): price area with concentrated trade
    activity. Acts as a magnet/fair-value zone. Approaches *from above*
    with declining momentum tend to be rejected (mean-revert back down),
    while pushes through them often re-test as support. We trade the
    rejection: approach from above + declining volume → SELL.

  - **LVN** (Low-Volume Node): price area trades skipped quickly with
    little resting volume — these zones break easily on momentum.
    Approach with a volume spike (z>2) → BUY breakout.

Why this is worth adding to TA Trader v0.0.37:
  - The 11 chart + 5 candle + 6 anomaly detectors all use *price geometry*
    (pivots / channels / wicks). None of them consume volume *distribution*.
  - Volume Profile is orthogonal to the other detectors — confluence with
    a momentum break or a chart pattern boosts confidence in the aggregator.
  - Computationally cheap: O(n_bars × n_bins), runs in <1ms on 30d × 10m.

Outputs `PatternSignal`-compatible objects via `detect_vpvr_signal()` so
the existing ta_trader → aggregator → risk_manager pipeline consumes them
without changes. Behind `cfg.VPVR_ENABLED` and (after backtest gating)
`cfg.VPVR_PRODUCTION_ENABLED`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    _READY = True
except ImportError:
    _READY = False

@dataclass
class VolumeBin:
    """One bin of the volume profile."""

    price_low: float
    price_high: float
    total_volume: float

    @property
    def mid(self) -> float:
        """Mid."""
        return 0.5 * (self.price_low + self.price_high)

    def contains(self, price: float) -> bool:
        """Contains."""
        return self.price_low <= price <= self.price_high

    def as_tuple(self) -> tuple[float, float, float]:
        """As tuple."""
        return (self.price_low, self.price_high, self.total_volume)

@dataclass
class VPVRSignal:
    """A VPVR-derived setup. Shape matches `ResearchPattern`/`DashaPattern`
    so the ta_trader wrapper can produce a `PatternSignal` from it directly.
    """

    pattern: str
    direction: str
    bar_idx: int
    entry: float
    stop: float
    target: float
    confidence: float
    atr_at_entry: float
    height: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expected_rr(self) -> float:
        """Expected rr."""
        risk = abs(self.entry - self.stop)
        reward = abs(self.target - self.entry)
        return reward / risk if risk > 1e-9 else 0.0

class VolumeProfile:
    """Compute Volume Profile Visible Range over an OHLCV DataFrame.

    Usage:
        vp = VolumeProfile()
        profile = vp.compute_vpvr(df, n_bins=50)
        hvns = vp.find_hvn(profile, top_k=5)
        lvns = vp.find_lvn(profile, threshold_pct=20)
        sig  = vp.detect_vpvr_signal(df, current_price, profile)
    """

    def __init__(self) -> None:
        """Init."""
        if not _READY:
            logger.warning("VolumeProfile: numpy/pandas not available")

    def compute_vpvr(
        self,
        df: pd.DataFrame,
        n_bins: int = 50,
    ) -> list[VolumeBin]:
        """Build a VPVR histogram from OHLCV bars.

        Each bar's volume is *spread evenly* across the price bins its
        (low, high) range covers. This is the standard approximation
        (the typical "uniform distribution across the bar's range") and
        is far more representative than just counting close-price volume.
        """
        if not _READY or df is None or len(df) == 0:
            return []
        if n_bins < 2:
            n_bins = 2

        try:
            highs = df["high"].astype(float).values
            lows = df["low"].astype(float).values
            volumes = df["volume"].astype(float).values
        except (KeyError, AttributeError):
            return []

        mask = (
            np.isfinite(highs)
            & np.isfinite(lows)
            & np.isfinite(volumes)
            & (volumes > 0)
            & (highs >= lows)
        )
        if not mask.any():
            return []
        highs = highs[mask]
        lows = lows[mask]
        volumes = volumes[mask]

        p_min = float(np.min(lows))
        p_max = float(np.max(highs))
        if p_max <= p_min:
            return [VolumeBin(p_min, p_max, float(np.sum(volumes)))]

        edges = np.linspace(p_min, p_max, n_bins + 1)
        bin_width = (p_max - p_min) / n_bins
        bin_volumes = np.zeros(n_bins, dtype=float)

        for h, l, v in zip(highs, lows, volumes, strict=False):
            if h == l:
                idx = int(min(n_bins - 1, max(0, (l - p_min) / bin_width)))
                bin_volumes[idx] += v
                continue
            lo_idx = int(max(0, np.floor((l - p_min) / bin_width)))
            hi_idx = int(min(n_bins - 1, np.floor((h - p_min) / bin_width)))
            if hi_idx <= lo_idx:
                bin_volumes[lo_idx] += v
                continue
            bar_range = h - l
            for b in range(lo_idx, hi_idx + 1):
                bin_lo = edges[b]
                bin_hi = edges[b + 1]
                overlap = max(0.0, min(bin_hi, h) - max(bin_lo, l))
                bin_volumes[b] += v * (overlap / bar_range)

        return [
            VolumeBin(float(edges[i]), float(edges[i + 1]), float(bin_volumes[i]))
            for i in range(n_bins)
        ]

    def find_hvn(
        self,
        profile: list[VolumeBin],
        top_k: int = 5,
    ) -> list[VolumeBin]:
        """Return the `top_k` bins by total_volume, sorted by descending volume.

        Only **local** maxima are eligible (a bin is local max if its volume
        >= both neighbours), which avoids ranking a fat continuous range as
        five separate HVNs.
        """
        if not profile or top_k <= 0:
            return []
        n = len(profile)
        if n == 1:
            return list(profile)

        local_max: list[VolumeBin] = []
        for i, b in enumerate(profile):
            left = profile[i - 1].total_volume if i > 0 else -1.0
            right = profile[i + 1].total_volume if i < n - 1 else -1.0
            if b.total_volume >= left and b.total_volume >= right and b.total_volume > 0:
                local_max.append(b)

        local_max.sort(key=lambda x: x.total_volume, reverse=True)
        return local_max[:top_k]

    def find_lvn(
        self,
        profile: list[VolumeBin],
        threshold_pct: float = 20.0,
    ) -> list[VolumeBin]:
        """Return bins whose volume is below `threshold_pct` of the max bin volume.

        Empty-volume bins are excluded — those usually fall outside the
        traded range (artifacts of `compute_vpvr` filtering).
        """
        if not profile:
            return []
        max_v = max(b.total_volume for b in profile)
        if max_v <= 0:
            return []
        threshold = max_v * threshold_pct / 100.0
        return [b for b in profile if 0 < b.total_volume < threshold]

    def detect_vpvr_signal(
        self,
        df: pd.DataFrame,
        current_price: float,
        profile: list[VolumeBin] | None = None,
        *,
        atr_col: str = "atr14",
        proximity_atrs: float = 0.6,
        vol_zscore_threshold: float = 2.0,
        momentum_lookback: int = 5,
    ) -> list[VPVRSignal]:
        """Build VPVR signal(s) at the *current* bar.

        Two setups:

          - **HVN rejection (SELL):** price is approaching an HVN *from
            above* (current_price > HVN price), close-to-close momentum
            is *declining* (recent slope <= 0), and the latest volume
            z-score >= `vol_zscore_threshold`. Interpretation: trader
            interest is flooding back to a fair-value zone and the move
            up is exhausted — short into the HVN with stop above the
            most recent swing high.

          - **LVN breakout (BUY):** price is moving *up* into an LVN
            with a fresh volume spike (z >= threshold) and a positive
            momentum slope. Stop below the most recent low; target the
            far side of the LVN (the next HVN above).
        """
        if not _READY or df is None or len(df) < momentum_lookback + 5:
            return []
        if current_price <= 0:
            return []

        if profile is None:
            profile = self.compute_vpvr(df, n_bins=50)
        if not profile:
            return []

        try:
            closes = df["close"].astype(float).values
            highs = df["high"].astype(float).values
            lows = df["low"].astype(float).values
            volumes = df["volume"].astype(float).values
        except (KeyError, AttributeError):
            return []

        if atr_col in df.columns:
            atr_series = df[atr_col].astype(float).values
            atr_now = float(atr_series[-1]) if np.isfinite(atr_series[-1]) else 0.0
        else:
            atr_now = self._fallback_atr(highs, lows, closes, period=14)
        if atr_now <= 0:
            return []

        vol_window = volumes[-30:] if len(volumes) >= 30 else volumes
        vol_window = vol_window[np.isfinite(vol_window) & (vol_window > 0)]
        if len(vol_window) < 5:
            return []
        vol_mean = float(np.mean(vol_window))
        vol_std = float(np.std(vol_window))
        if vol_std <= 0:
            return []
        last_vol = float(volumes[-1])
        vol_z = (last_vol - vol_mean) / vol_std

        recent = closes[-momentum_lookback:]
        slope = float(recent[-1] - recent[0])

        out: list[VPVRSignal] = []
        hvns = self.find_hvn(profile, top_k=5)
        lvns = self.find_lvn(profile, threshold_pct=20.0)

        below_hvns = [b for b in hvns if b.mid < current_price]
        if below_hvns and slope <= 0 and vol_z >= vol_zscore_threshold:
            hvn = max(below_hvns, key=lambda b: b.mid)
            dist = current_price - hvn.mid
            if 0 < dist <= proximity_atrs * atr_now:
                entry = current_price
                stop_lookback = max(10, momentum_lookback * 2)
                stop = float(np.max(highs[-stop_lookback:])) + 0.2 * atr_now
                target = hvn.mid
                if stop > entry and target < entry:
                    out.append(
                        VPVRSignal(
                            pattern="vpvr_hvn_rejection",
                            direction="SELL",
                            bar_idx=int(len(df) - 1),
                            entry=float(entry),
                            stop=float(stop),
                            target=float(target),
                            confidence=0.62,
                            atr_at_entry=float(atr_now),
                            height=float(stop - entry),
                            metadata={
                                "source": "vpvr",
                                "hvn_price": float(hvn.mid),
                                "hvn_volume": float(hvn.total_volume),
                                "vol_zscore": round(vol_z, 2),
                                "slope": round(slope, 4),
                                "dist_atrs": round(dist / atr_now, 2),
                            },
                        )
                    )

        above_lvns = [b for b in lvns if b.mid > current_price]
        if above_lvns and slope > 0 and vol_z >= vol_zscore_threshold:
            lvn = min(above_lvns, key=lambda b: b.mid)
            dist = lvn.mid - current_price
            if 0 < dist <= proximity_atrs * atr_now:
                entry = current_price
                stop_lookback = max(10, momentum_lookback * 2)
                stop = float(np.min(lows[-stop_lookback:])) - 0.2 * atr_now
                hvn_above = [b for b in hvns if b.mid > lvn.price_high]
                if hvn_above:
                    target = float(min(hvn_above, key=lambda b: b.mid).mid)
                else:
                    target = float(lvn.price_high + 2.0 * (entry - stop))
                if stop < entry and target > entry:
                    out.append(
                        VPVRSignal(
                            pattern="vpvr_lvn_breakout",
                            direction="BUY",
                            bar_idx=int(len(df) - 1),
                            entry=float(entry),
                            stop=float(stop),
                            target=float(target),
                            confidence=0.60,
                            atr_at_entry=float(atr_now),
                            height=float(entry - stop),
                            metadata={
                                "source": "vpvr",
                                "lvn_price": float(lvn.mid),
                                "lvn_volume": float(lvn.total_volume),
                                "vol_zscore": round(vol_z, 2),
                                "slope": round(slope, 4),
                                "dist_atrs": round(dist / atr_now, 2),
                            },
                        )
                    )

        return out

    @staticmethod
    def _fallback_atr(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> float:
        """Wilder-style ATR estimate over the last `period` bars. Used only
        when the caller didn't supply an `atr14` column."""
        n = len(closes)
        if n < period + 1:
            return 0.0
        trs = []
        for i in range(1, n):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            if np.isfinite(tr):
                trs.append(tr)
        if not trs:
            return 0.0
        window = trs[-period:]
        return float(np.mean(window)) if window else 0.0

_DEFAULT_VP = VolumeProfile()

def compute_vpvr(df: pd.DataFrame, n_bins: int = 50) -> list[VolumeBin]:
    """Module-level shortcut around `VolumeProfile.compute_vpvr`."""
    return _DEFAULT_VP.compute_vpvr(df, n_bins=n_bins)

def find_hvn(profile: list[VolumeBin], top_k: int = 5) -> list[VolumeBin]:
    """Module-level shortcut around `VolumeProfile.find_hvn`."""
    return _DEFAULT_VP.find_hvn(profile, top_k=top_k)

def find_lvn(profile: list[VolumeBin], threshold_pct: float = 20.0) -> list[VolumeBin]:
    """Module-level shortcut around `VolumeProfile.find_lvn`."""
    return _DEFAULT_VP.find_lvn(profile, threshold_pct=threshold_pct)

def detect_vpvr_signal(
    df: pd.DataFrame,
    current_price: float,
    profile: list[VolumeBin] | None = None,
    *,
    atr_col: str = "atr14",
) -> list[VPVRSignal]:
    """Module-level shortcut around `VolumeProfile.detect_vpvr_signal`."""
    return _DEFAULT_VP.detect_vpvr_signal(df, current_price, profile, atr_col=atr_col)

__all__ = [
    "VolumeBin",
    "VPVRSignal",
    "VolumeProfile",
    "compute_vpvr",
    "find_hvn",
    "find_lvn",
    "detect_vpvr_signal",
]
