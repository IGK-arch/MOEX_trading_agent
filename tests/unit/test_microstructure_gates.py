"""
tests/unit/test_microstructure_gates.py — Gate logic correctness.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agents.microstructure_gates import MicrostructureGates


def _normal_df(n: int = 20) -> pd.DataFrame:
    """Balanced, slightly trending volumes — should pass all gates."""
    close = [100.0 + 0.05 * i for i in range(n)]
    vb = [50.0] * n
    vs = [48.0] * n
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


@pytest.mark.asyncio
async def test_pass_through_when_no_data():
    """Test pass through when no data."""
    gates = MicrostructureGates()
    res = await gates.check("SBER", "BUY", supercandles_df=None)
    assert res.blocked is False
    assert res.weakened is False
    assert res.reason == "no_data"


@pytest.mark.asyncio
async def test_pass_through_when_short_data():
    """Test pass through when short data."""
    gates = MicrostructureGates()
    df = _normal_df(n=3)
    res = await gates.check("SBER", "BUY", supercandles_df=df)
    assert res.blocked is False
    assert res.reason == "no_data"


@pytest.mark.asyncio
async def test_clean_data_passes():
    """Test clean data passes."""
    gates = MicrostructureGates()
    df = _normal_df(n=30)
    res = await gates.check("SBER", "BUY", supercandles_df=df)
    assert res.blocked is False
    assert res.weakened is False
    assert 0.0 <= res.vpin <= 1.0


@pytest.mark.asyncio
async def test_vpin_blocks_toxic_flow():
    """Test vpin blocks toxic flow."""
    gates = MicrostructureGates()

    n = 30
    close = [100.0] * n
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [100] * n,
            "vol_b": [100] * n,
            "vol_s": [0] * n,
        }
    )
    res = await gates.check("GAZP", "BUY", supercandles_df=df)
    assert res.blocked is True
    assert "vpin" in res.reason.lower()


@pytest.mark.asyncio
async def test_ofi_opposition_weakens_buy():
    """If OFI strongly negative and direction=BUY → weaken (not block)."""
    gates = MicrostructureGates()
    n = 20
    close = [100.0 + 0.05 * i for i in range(n)]

    vb = [10.0] * n
    vs = [80.0] * n
    df = pd.DataFrame(
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
    res = await gates.check("LKOH", "BUY", supercandles_df=df)

    assert res.blocked or res.weakened


@pytest.mark.asyncio
async def test_ofi_aligned_with_direction_passes():
    """Strong positive OFI + direction=BUY → no weakening."""
    gates = MicrostructureGates()
    n = 30
    close = [100.0 + 0.05 * i for i in range(n)]

    vb = [55.0] * n
    vs = [45.0] * n
    df = pd.DataFrame(
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
    res = await gates.check("ROSN", "BUY", supercandles_df=df)
    assert res.blocked is False
    assert res.weakened is False


@pytest.mark.asyncio
async def test_disabled_gates_pass():
    """When MICROSTRUCTURE_GATES_ENABLED is False, gate always passes."""
    import app.config as cfg

    orig = cfg.MICROSTRUCTURE_GATES_ENABLED
    try:
        cfg.MICROSTRUCTURE_GATES_ENABLED = False
        gates = MicrostructureGates()
        df = _normal_df(n=20)
        res = await gates.check("SBER", "BUY", supercandles_df=df)
        assert res.blocked is False
        assert res.weakened is False
        assert res.reason == "gates_disabled"
    finally:
        cfg.MICROSTRUCTURE_GATES_ENABLED = orig
