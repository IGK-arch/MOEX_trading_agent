"""Unit tests for Volume Profile / VPVR detector (Phase 27 / v0.0.38)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.agents.ta_patterns.volume_profile import (
    VolumeBin,
    VolumeProfile,
    compute_vpvr,
    find_hvn,
    find_lvn,
)


def _make_ohlcv(
    closes: list[float],
    volumes: list[float] | None = None,
    spread: float = 0.5,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a closes list."""
    n = len(closes)
    if volumes is None:
        volumes = [1000.0] * n
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) + spread for o, c in zip(opens, closes, strict=False)]
    lows = [min(o, c) - spread for o, c in zip(opens, closes, strict=False)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def _add_atr14(df: pd.DataFrame) -> pd.DataFrame:
    """Add atr14."""
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def test_compute_vpvr_returns_n_bins_and_conserves_volume():
    """The profile must have exactly n_bins entries and the sum across
    all bins must equal the total OHLC volume input (within floating eps)."""
    closes = list(np.linspace(100.0, 120.0, 60))
    volumes = [1000.0] * 60
    df = _make_ohlcv(closes, volumes)

    vp = VolumeProfile()
    profile = vp.compute_vpvr(df, n_bins=20)

    assert len(profile) == 20
    assert all(isinstance(b, VolumeBin) for b in profile)
    total_input = float(np.sum(volumes))
    total_binned = sum(b.total_volume for b in profile)
    assert abs(total_binned - total_input) < 1e-6 * total_input
    for b1, b2 in zip(profile[:-1], profile[1:], strict=False):
        assert b1.price_high <= b2.price_low + 1e-9


def test_compute_vpvr_concentrates_volume_at_oscillation_price():
    """If price oscillates around a single value with all volume there,
    the bin containing that value must hold the most volume."""
    closes = [100.0] * 40 + [100.5] * 40
    df = _make_ohlcv(closes, spread=0.1)
    profile = compute_vpvr(df, n_bins=20)

    sorted_bins = sorted(profile, key=lambda b: b.total_volume, reverse=True)
    top3_mids = [b.mid for b in sorted_bins[:3]]
    assert any(99.5 <= m <= 101.0 for m in top3_mids)


def test_find_hvn_returns_top_k_local_maxima_sorted_desc():
    """HVNs are returned sorted by total_volume descending, capped at top_k,
    and every returned bin is a local maximum vs. its neighbours."""
    bins = [
        VolumeBin(0, 1, 10),
        VolumeBin(1, 2, 100),
        VolumeBin(2, 3, 30),
        VolumeBin(3, 4, 80),
        VolumeBin(4, 5, 20),
        VolumeBin(5, 6, 60),
        VolumeBin(6, 7, 5),
    ]
    hvns = find_hvn(bins, top_k=2)
    assert len(hvns) == 2
    assert hvns[0].total_volume == 100
    assert hvns[1].total_volume == 80
    for h in hvns:
        idx = bins.index(h)
        left = bins[idx - 1].total_volume if idx > 0 else -1
        right = bins[idx + 1].total_volume if idx < len(bins) - 1 else -1
        assert h.total_volume >= left
        assert h.total_volume >= right


def test_find_hvn_handles_empty_and_top_k_zero():
    """Test find hvn handles empty and top k zero."""
    assert find_hvn([], top_k=5) == []
    assert find_hvn([VolumeBin(0, 1, 10)], top_k=0) == []


def test_find_lvn_returns_below_threshold_only():
    """LVNs are bins whose volume is below `threshold_pct` of the max bin
    volume. Empty bins (vol=0) are excluded."""
    bins = [
        VolumeBin(0, 1, 100),
        VolumeBin(1, 2, 15),
        VolumeBin(2, 3, 0),
        VolumeBin(3, 4, 30),
        VolumeBin(4, 5, 5),
    ]
    lvns = find_lvn(bins, threshold_pct=20.0)
    vols = sorted(b.total_volume for b in lvns)
    assert vols == [5, 15]


def test_detect_vpvr_signal_emits_lvn_breakout_buy():
    """Construct a path with a clear HVN below (high resting volume) and
    a thin gap above (LVN), then push price up with a volume spike.
    Expect a BUY signal pointing to the next HVN."""
    rng = np.random.default_rng(7)
    p1_closes = [100 + rng.normal(0, 0.3) for _ in range(60)]
    p1_vols = [5000.0] * 60
    p2_closes = [100.3, 100.7, 101.1, 101.5, 101.9]
    p2_vols = [400.0, 400.0, 400.0, 400.0, 400.0]
    closes = p1_closes + p2_closes + [102.3]
    volumes = p1_vols + p2_vols + [50000.0]
    df = _add_atr14(_make_ohlcv(closes, volumes, spread=0.15))

    vp = VolumeProfile()
    profile = vp.compute_vpvr(df, n_bins=50)
    signals = vp.detect_vpvr_signal(
        df,
        current_price=float(df["close"].iloc[-1]),
        profile=profile,
        proximity_atrs=5.0,
        vol_zscore_threshold=2.0,
    )

    lvn_signals = [s for s in signals if s.pattern == "vpvr_lvn_breakout"]
    assert len(lvn_signals) >= 1
    s = lvn_signals[0]
    assert s.direction == "BUY"
    assert s.stop < s.entry < s.target
    assert s.expected_rr > 0
    assert s.metadata["source"] == "vpvr"
    assert s.metadata["vol_zscore"] >= 2.0


def test_detect_vpvr_signal_emits_hvn_rejection_sell():
    """Build an HVN below the current price, then a tail-off in price with a
    volume spike → expect SELL pointing back into the HVN."""
    rng = np.random.default_rng(3)
    p1_closes = [100 + rng.normal(0, 0.3) for _ in range(60)]
    p1_vols = [5000.0] * 60
    p2_closes = [100.5, 100.8, 101.0, 101.0, 100.95, 100.85, 100.7]
    p2_vols = [800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 50000.0]
    closes = p1_closes + p2_closes
    volumes = p1_vols + p2_vols
    df = _add_atr14(_make_ohlcv(closes, volumes, spread=0.15))

    vp = VolumeProfile()
    profile = vp.compute_vpvr(df, n_bins=50)
    signals = vp.detect_vpvr_signal(
        df,
        current_price=float(df["close"].iloc[-1]),
        profile=profile,
        proximity_atrs=5.0,
        vol_zscore_threshold=2.0,
    )

    hvn_signals = [s for s in signals if s.pattern == "vpvr_hvn_rejection"]
    assert len(hvn_signals) >= 1
    s = hvn_signals[0]
    assert s.direction == "SELL"
    assert s.target < s.entry < s.stop
    assert s.expected_rr > 0
    assert s.metadata["source"] == "vpvr"
    assert s.metadata["vol_zscore"] >= 2.0


def test_compute_vpvr_empty_df_returns_empty():
    """Test compute vpvr empty df returns empty."""
    assert compute_vpvr(pd.DataFrame(columns=["high", "low", "volume"])) == []


def test_compute_vpvr_flat_range_single_bin():
    """Degenerate case: all bars at one price → one bin with all volume."""
    closes = [100.0] * 10
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * 10,
        }
    )
    profile = compute_vpvr(df, n_bins=20)
    assert len(profile) == 1
    assert profile[0].total_volume == 10000.0


def test_detect_vpvr_signal_no_volume_spike_no_signal():
    """Without a volume spike, the detector must emit no signal — even
    if price is approaching an HVN."""
    rng = np.random.default_rng(1)
    closes = [100 + rng.normal(0, 0.3) for _ in range(60)] + [100.5, 100.8, 101.0]
    volumes = [5000.0] * 60 + [5100.0, 5050.0, 5020.0]
    df = _add_atr14(_make_ohlcv(closes, volumes, spread=0.15))

    vp = VolumeProfile()
    signals = vp.detect_vpvr_signal(
        df,
        current_price=float(df["close"].iloc[-1]),
        vol_zscore_threshold=2.0,
    )
    assert signals == []


def test_module_level_helpers_match_class_methods():
    """compute_vpvr / find_hvn / find_lvn module-level helpers must mirror
    the class-method behaviour exactly (no divergence)."""
    closes = list(np.linspace(100.0, 110.0, 30))
    df = _make_ohlcv(closes, [1000.0] * 30)
    vp = VolumeProfile()
    p1 = vp.compute_vpvr(df, n_bins=15)
    p2 = compute_vpvr(df, n_bins=15)
    assert [b.as_tuple() for b in p1] == [b.as_tuple() for b in p2]

    h1 = vp.find_hvn(p1, top_k=3)
    h2 = find_hvn(p2, top_k=3)
    assert [b.as_tuple() for b in h1] == [b.as_tuple() for b in h2]

    l1 = vp.find_lvn(p1, threshold_pct=25)
    l2 = find_lvn(p2, threshold_pct=25)
    assert [b.as_tuple() for b in l1] == [b.as_tuple() for b in l2]
