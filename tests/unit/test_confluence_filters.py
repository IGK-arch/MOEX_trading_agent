"""Unit tests for app.agents.ta_patterns.confluence_filters (Phase 27.2).

5 deterministic tests covering each of the four confluence gates:
  * volume_check (above / below threshold)
  * hmm_alignment (aligned / vetoed)
  * time_of_day (auction window, mid-session, close window)
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from app.agents.ta_patterns.confluence_filters import (
    passes_atr_percentile,
    passes_hmm_alignment,
    passes_time_of_day,
    passes_volume_check,
)


def _df_with_volume(volumes: list[float], n_pad: int = 30) -> pd.DataFrame:
    """Build a synthetic OHLCV frame whose volume column equals `volumes`
    (padded on the left with the first value so rolling means converge).
    """
    pad = [volumes[0]] * n_pad
    full = pad + list(volumes)
    idx = pd.date_range("2026-04-01", periods=len(full), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.linspace(100, 101, len(full)),
            "high": np.linspace(100.5, 101.5, len(full)),
            "low": np.linspace(99.5, 100.5, len(full)),
            "close": np.linspace(100, 101, len(full)),
            "volume": full,
        },
        index=idx,
    )


def test_volume_check_above_threshold_passes():
    """Bar with volume = 2× rolling20 mean → True."""
    base_vol = 1000.0
    df = _df_with_volume([base_vol] * 25 + [2 * base_vol])
    last_idx = len(df) - 1
    assert passes_volume_check(df, last_idx, multiplier=1.3) is True


def test_volume_check_below_threshold_fails():
    """Bar with volume = 0.5× rolling20 mean → False."""
    base_vol = 1000.0
    df = _df_with_volume([base_vol] * 25 + [0.5 * base_vol])
    last_idx = len(df) - 1
    assert passes_volume_check(df, last_idx, multiplier=1.3) is False


def test_hmm_alignment_reversal_mean_reverting_passes():
    """Reversal patterns are aligned with mean-reverting regime."""
    assert passes_hmm_alignment("reversal", "mean_reverting") is True


def test_hmm_alignment_continuation_crisis_v0196_passes():
    """v0.19.6: continuation now PASSES crisis (downsized via adaptive_regime).

    Only reversal is vetoed in crisis ("don't fade crashes"). Other
    families (continuation/research/dasha/smc/candle/harmonic) pass
    through so the bot can still earn at 0.25× sizing during turbulence.
    """
    assert passes_hmm_alignment("continuation", "crisis") is True
    assert passes_hmm_alignment("reversal", "crisis") is False


def test_time_of_day_auction_window_vetoes_and_midday_passes():
    """10:05 MSK is in the auction-overflow window (False).
    11:30 MSK is mid-session (True).
    18:30 MSK is in the close window (False).

    MSK = UTC + 3 → 07:05 UTC == 10:05 MSK.
    """
    auction_msk = datetime(2026, 5, 27, 7, 5, tzinfo=UTC)
    mid_msk = datetime(2026, 5, 27, 8, 30, tzinfo=UTC)
    close_msk = datetime(2026, 5, 27, 15, 30, tzinfo=UTC)

    assert passes_time_of_day(auction_msk) is False
    assert passes_time_of_day(mid_msk) is True
    assert passes_time_of_day(close_msk) is False
    evening_msk = datetime(2026, 5, 27, 18, 0, tzinfo=UTC)
    assert passes_time_of_day(evening_msk) is True


def test_atr_percentile_extreme_low_volatility_vetoes():
    """A bar at the tail of the trailing ATR distribution (well below the
    30th percentile) should be vetoed.

    Strategy: build 200 bars with progressively shrinking high-low range so
    the LAST bar's ATR is the minimum of the trailing window — below the
    30th percentile by construction.
    """
    n = 200
    rng_widths = np.linspace(1.0, 0.05, n)
    close = np.full(n, 100.0)
    high = close + rng_widths
    low = close - rng_widths
    open_ = close
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )
    assert passes_atr_percentile(df, n - 1, low_p=30.0, high_p=90.0) is False
