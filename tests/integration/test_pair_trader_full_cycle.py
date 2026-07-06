"""
tests/integration/test_pair_trader_full_cycle.py — Pair trader full cycle.

Given a fitted pair with synthetic prices that yield z=+2.5, verify:
  * PairTrader.poll() emits exactly two legs (A short, B long).
  * Both legs share the pair_id metadata and z_score ≈ 2.5.
  * Beta-weighted sizing actually scales leg-B's target_quantity by |β|
    when ``sizing_mode == "beta"``.
  * Equal-dollar sizing keeps both legs at the same notional target.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

pytest.importorskip("pandas")

import pandas as pd  # noqa: E402

from app.agents.pair_trader import PairState, PairTrader
from app.dispatcher.signal import Direction, SignalSource


class _FakeCandleStore:
    """In-memory candle store; returns the same df for any (ticker, depth)."""

    def __init__(self, prices: dict[str, float]) -> None:
        """Init."""
        self.prices = prices

    def get(self, ticker: str, depth_h: int) -> pd.DataFrame:
        """Get."""
        ts = datetime.now(tz=UTC)
        return pd.DataFrame(
            {
                "open": [self.prices[ticker]],
                "high": [self.prices[ticker]],
                "low": [self.prices[ticker]],
                "close": [self.prices[ticker]],
                "volume": [100_000.0],
                "begin": [ts],
            }
        )


def _seed_qualified_state(
    pair: tuple[str, str],
    *,
    beta: float,
    mu: float = 0.0,
    sigma: float = 0.01,
) -> PairState:
    """Seed qualified state."""
    return PairState(
        ticker_a=pair[0],
        ticker_b=pair[1],
        alpha=0.0,
        beta=beta,
        mu=mu,
        sigma=sigma,
        adf_pvalue=0.001,
        qualified=True,
        last_refit_iso=datetime.now(tz=UTC).isoformat(),
        last_z_values=[],
        last_entry_iso="",
        adf_history=[{"iso": "2026-05-25", "adf_p": 0.001}],
        ttl_disqualified=False,
        ttl_reason="",
        open_entry_bar_iso="",
        open_entry_direction=0,
    )


def _prices_for_target_z(
    *, beta: float, target_z: float, sigma: float = 0.01
) -> tuple[float, float]:
    """Choose (price_a, price_b) so that
        z = (log(p_a) - α - β*log(p_b) - μ) / σ
    equals target_z. Default α=μ=0 → log(p_a) = target_z*σ + β*log(p_b).

    We use price_b = exp(1.0) ≈ 2.718 (small numeric magnitude) so the
    resulting price_a stays in the sub-1000 range even for moderate betas;
    DEFAULT_TARGET_RUB=5000 then yields a positive integer quantity on both
    legs (no degenerate qty=0 case for the sizing assertions).
    """
    log_b = 1.0
    price_b = math.exp(log_b)
    log_a = target_z * sigma + beta * log_b
    return math.exp(log_a), price_b


@pytest.mark.asyncio
async def test_pair_trader_z25_emits_two_legs(monkeypatch, tmp_path):
    """z = +2.5 (above 1.5 entry) → SELL leg A + BUY leg B."""
    monkeypatch.setattr(
        "app.agents.pair_trader.PAIR_STATE_FILE",
        tmp_path / "pair_state.json",
    )

    pair = ("PLZL", "AFLT")
    beta = 1.3
    ps = _seed_qualified_state(pair, beta=beta)
    price_a, price_b = _prices_for_target_z(beta=beta, target_z=2.5, sigma=ps.sigma)

    pt = PairTrader(pairs=[pair])
    pt.candle_store = _FakeCandleStore({pair[0]: price_a, pair[1]: price_b})
    pt.state = {ps.key(): ps}
    pt._started = True

    sigs = await pt.poll()
    assert len(sigs) == 2, f"expected 2 leg signals, got {len(sigs)}"

    by_ticker = {s.ticker: s for s in sigs}
    assert by_ticker["PLZL"].direction == Direction.SELL
    assert by_ticker["AFLT"].direction == Direction.BUY

    for s in sigs:
        assert s.source == SignalSource.PAIR
        assert s.metadata["pair_id"] == "PLZL_AFLT"
        assert s.metadata["is_pair_trade"] is True
        assert s.metadata["z_score"] == pytest.approx(2.5, abs=0.05)


@pytest.mark.asyncio
async def test_pair_trader_beta_sizing_scales_leg_b(monkeypatch, tmp_path):
    """For pairs where ``sizing_mode == "beta"``, leg-B's target_quantity must
    scale with |β| while leg-A stays at the base notional."""
    monkeypatch.setattr(
        "app.agents.pair_trader.PAIR_STATE_FILE",
        tmp_path / "pair_state.json",
    )

    pair = ("T", "GMKN")
    beta = 2.0
    ps = _seed_qualified_state(pair, beta=beta)
    price_a, price_b = _prices_for_target_z(beta=beta, target_z=2.5, sigma=ps.sigma)

    pt = PairTrader(pairs=[pair])
    pt.candle_store = _FakeCandleStore({pair[0]: price_a, pair[1]: price_b})
    pt.state = {ps.key(): ps}
    pt._started = True

    sigs = await pt.poll()
    assert len(sigs) == 2

    by_leg = {s.metadata["leg"]: s for s in sigs}
    qty_a = by_leg["A"].metadata["target_quantity"]
    qty_b = by_leg["B"].metadata["target_quantity"]
    px_a = by_leg["A"].price
    px_b = by_leg["B"].price

    notional_a = qty_a * px_a
    notional_b = qty_b * px_b

    ratio = notional_b / notional_a if notional_a > 0 else float("inf")
    assert ratio == pytest.approx(beta, rel=0.05), (
        f"expected leg-B/leg-A notional ratio ≈ |β|={beta}, got {ratio:.3f}"
    )
    assert qty_a > 0
    assert qty_b > 0
    assert by_leg["A"].metadata["sizing_mode"] == "beta"
    assert by_leg["B"].metadata["sizing_mode"] == "beta"


@pytest.mark.asyncio
async def test_pair_trader_equal_sizing_keeps_legs_balanced(monkeypatch, tmp_path):
    """For ``sizing_mode == "equal"`` pairs both notionals match
    DEFAULT_TARGET_RUB regardless of β."""
    monkeypatch.setattr(
        "app.agents.pair_trader.PAIR_STATE_FILE",
        tmp_path / "pair_state.json",
    )

    pair = ("PLZL", "AFLT")
    beta = 1.3
    ps = _seed_qualified_state(pair, beta=beta)
    price_a, price_b = _prices_for_target_z(beta=beta, target_z=2.5, sigma=ps.sigma)

    pt = PairTrader(pairs=[pair])
    pt.candle_store = _FakeCandleStore({pair[0]: price_a, pair[1]: price_b})
    pt.state = {ps.key(): ps}
    pt._started = True

    sigs = await pt.poll()
    assert len(sigs) == 2
    by_leg = {s.metadata["leg"]: s for s in sigs}
    notional_a = by_leg["A"].metadata["target_quantity"] * by_leg["A"].price
    notional_b = by_leg["B"].metadata["target_quantity"] * by_leg["B"].price
    assert notional_a == pytest.approx(notional_b, rel=0.05)
    assert by_leg["A"].metadata["sizing_mode"] == "equal"
