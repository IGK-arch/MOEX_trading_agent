"""
tests/unit/test_microstructure.py — OFI / Kyle / VPIN correctness.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agents.microstructure import (
    compute_kyles_lambda,
    compute_ofi,
    compute_ofi_series,
    compute_vpin,
)


def _supercandles(n: int, vb: list[float], vs: list[float], close: list[float]) -> pd.DataFrame:
    """Supercandles."""
    assert len(vb) == len(vs) == len(close) == n
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [b + s for b, s in zip(vb, vs, strict=False)],
            "vol_b": vb,
            "vol_s": vs,
        }
    )


def test_compute_ofi_balanced():
    """Test compute ofi balanced."""
    assert compute_ofi(100.0, 100.0) == pytest.approx(0.0)


def test_compute_ofi_full_buy():
    """Test compute ofi full buy."""
    assert compute_ofi(100.0, 0.0) == pytest.approx(1.0)


def test_compute_ofi_full_sell():
    """Test compute ofi full sell."""
    assert compute_ofi(0.0, 100.0) == pytest.approx(-1.0)


def test_compute_ofi_zero_total():
    """Test compute ofi zero total."""
    assert compute_ofi(0.0, 0.0) == 0.0


def test_ofi_series_aggregates_window():
    """Test ofi series aggregates window."""
    df = _supercandles(
        5,
        vb=[10, 20, 30, 40, 50],
        vs=[20, 20, 10, 10, 10],
        close=[100, 101, 102, 103, 104],
    )

    ofi = compute_ofi_series(df, window=3)
    assert ofi == pytest.approx(0.6)


def test_ofi_series_empty_df():
    """Test ofi series empty df."""
    assert compute_ofi_series(None, window=5) == 0.0
    assert compute_ofi_series(pd.DataFrame(), window=5) == 0.0


def test_ofi_series_missing_columns():
    """Test ofi series missing columns."""
    df = pd.DataFrame({"close": [100, 101, 102]})
    assert compute_ofi_series(df, window=2) == 0.0


def test_kyles_lambda_zero_when_no_movement():
    """Test kyles lambda zero when no movement."""
    df = _supercandles(10, vb=[10] * 10, vs=[10] * 10, close=[100.0] * 10)
    lam = compute_kyles_lambda(df, window=10)

    assert lam == 0.0


def test_kyles_lambda_positive_when_buy_volume_pushes_price():
    """Test kyles lambda positive when buy volume pushes price."""

    close = [100.0, 100.5, 101.2, 102.0, 103.0, 104.5, 106.0]
    vb = [50, 80, 100, 120, 150, 180, 200]
    vs = [50, 40, 30, 20, 10, 10, 10]
    df = _supercandles(7, vb=vb, vs=vs, close=close)
    lam = compute_kyles_lambda(df, window=7)
    assert lam > 0


def test_kyles_lambda_with_insufficient_data():
    """Test kyles lambda with insufficient data."""
    df = _supercandles(2, vb=[10, 20], vs=[5, 5], close=[100, 101])
    assert compute_kyles_lambda(df, window=10, min_obs=5) == 0.0


def test_vpin_zero_when_balanced_volumes():
    """Test vpin zero when balanced volumes."""
    df = _supercandles(20, vb=[50] * 20, vs=[50] * 20, close=[100] * 20)

    assert compute_vpin(df, n_buckets=20) == pytest.approx(0.0)


def test_vpin_high_when_one_sided():
    """Test vpin high when one sided."""
    df = _supercandles(20, vb=[100] * 20, vs=[0] * 20, close=[100] * 20)

    assert compute_vpin(df, n_buckets=20) == pytest.approx(1.0)


def test_vpin_fallback_without_volume_split():
    """Test vpin fallback without volume split."""
    df = pd.DataFrame(
        {
            "close": [100, 101, 100, 99, 100, 101, 102],
            "volume": [10] * 7,
        }
    )
    v = compute_vpin(df, n_buckets=7)

    assert 0.0 <= v <= 1.0
