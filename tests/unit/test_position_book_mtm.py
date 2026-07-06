"""Unit tests for v1.2.0 PositionBook mark-to-market equity.

Background: ArenaGo uses a CFD-style margin model (confirmed via /about
"Long и Short": "при открытии short свободные средства не растут —
наоборот, замораживается обеспечение; прибыль или убыток фиксируются в
момент закрытия позиции").  So `cash_balance` from `/api/bots` is
*available* cash — neither LONG notional locked into a buy nor SHORT
margin freezing is visible there.  The correct net-liq is therefore

    equity = cash + unrealized_PnL
           = cash + Σ qty_signed × (current_price − avg_price)

That conserves equity at the moment a position opens (current == avg →
PnL == 0 → equity == cash).  These tests pin the math for LONG, SHORT,
and the graceful fallback when MTM prices are missing — the v1.1.0
LONG-only path that protects against the v1.0.x `−68k equity` death
spiral.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.risk.position_book import Position, PositionBook


def _seed_book(positions: list[Position], cash: float) -> PositionBook:
    """Build a PositionBook with no broker calls."""
    book = PositionBook(deposit_total=1_000_000.0)
    book._cash_balance = cash
    book._positions = {p.ticker: p for p in positions}
    return book


def test_mtm_equity_long_and_short_combined() -> None:
    """Mixed LONG + SHORT MTM equity matches the documented formula."""
    book = _seed_book(
        positions=[
            Position(ticker="SBER", quantity=50, avg_price=300.0, bot="t"),
            Position(ticker="GAZP", quantity=-100, avg_price=120.0, bot="t"),
        ],
        cash=133_000.0,
    )
    prices = {"SBER": 310.0, "GAZP": 115.0}

    # PnL_SBER = 50 × (310−300) = +500  (LONG up 10 rub)
    # PnL_GAZP = (−100) × (115−120) = +500  (SHORT, price fell 5 rub → win)
    # equity = 133_000 + 500 + 500 = 134_000
    assert book.total_equity_mtm(prices) == pytest.approx(134_000.0)


def test_mtm_equity_conserves_at_position_open() -> None:
    """current == avg → unrealized PnL == 0 → equity == cash exactly."""
    positions = [
        Position(ticker=f"T{i:02d}", quantity=-100, avg_price=200.0, bot="t")
        for i in range(17)
    ]
    book = _seed_book(positions=positions, cash=133_000.0)

    flat_prices = {f"T{i:02d}": 200.0 for i in range(17)}
    flat_equity = book.total_equity_mtm(flat_prices)
    assert flat_equity == pytest.approx(133_000.0)

    # v1.0.x bug would've reported cash + Σ(qty×avg) for the SHORTs which
    # is 133k + 17 × (−100) × 200 = 133k − 340k = −207k.  The MTM formula
    # stays positive.
    assert flat_equity > 0


def test_mtm_equity_short_pnl_directions() -> None:
    """Verify SHORT PnL signs: drop = win, rise = loss."""
    book = _seed_book(
        positions=[Position(ticker="GAZP", quantity=-100, avg_price=120.0, bot="t")],
        cash=200_000.0,
    )
    # Price drops 5 RUB → SHORT wins +500
    assert book.total_equity_mtm({"GAZP": 115.0}) == pytest.approx(200_500.0)
    # Price rises 5 RUB → SHORT loses -500
    assert book.total_equity_mtm({"GAZP": 125.0}) == pytest.approx(199_500.0)


def test_mtm_equity_long_pnl_directions() -> None:
    """Verify LONG PnL signs: rise = win, drop = loss."""
    book = _seed_book(
        positions=[Position(ticker="SBER", quantity=50, avg_price=300.0, bot="t")],
        cash=200_000.0,
    )
    assert book.total_equity_mtm({"SBER": 310.0}) == pytest.approx(200_500.0)
    assert book.total_equity_mtm({"SBER": 290.0}) == pytest.approx(199_500.0)


def test_mtm_fallback_long_uses_avg_price() -> None:
    """Missing price for a LONG → contribute qty × avg_price (cash matches)."""
    book = _seed_book(
        positions=[
            Position(ticker="LKOH", quantity=10, avg_price=4000.0, bot="t"),
        ],
        cash=500_000.0,
    )
    # No prices in the dict → LONG falls back to avg.
    eq = book.total_equity_mtm({})
    # cash 500k + 10×4000 = 540k (which equals what cash + LONG market_value
    # would give anyway — fallback is safe).
    assert eq == pytest.approx(540_000.0)


def test_mtm_fallback_short_drops_position() -> None:
    """Missing price for a SHORT → skip (avoid v1.0.x double-subtraction)."""
    book = _seed_book(
        positions=[
            Position(ticker="GAZP", quantity=-100, avg_price=120.0, bot="t"),
        ],
        cash=133_000.0,
    )
    eq = book.total_equity_mtm({})
    # SHORT contribution dropped → equity == cash.  This is the v1.1.0 safe
    # behaviour; it errs on the conservative side instead of going negative.
    assert eq == pytest.approx(133_000.0)


def test_total_equity_uses_cache_when_present() -> None:
    """`total_equity()` reads `_mtm_prices` automatically."""
    import time

    book = _seed_book(
        positions=[
            Position(ticker="SBER", quantity=50, avg_price=300.0, bot="t"),
            Position(ticker="GAZP", quantity=-100, avg_price=120.0, bot="t"),
        ],
        cash=133_000.0,
    )
    now = time.time()
    book._mtm_prices = {"SBER": (310.0, now), "GAZP": (115.0, now)}

    assert book.total_equity() == pytest.approx(134_000.0)


def test_total_equity_falls_back_to_v1_1_0_without_cache() -> None:
    """Empty `_mtm_prices` → v1.1.0 behaviour (cash + LONG only)."""
    book = _seed_book(
        positions=[
            Position(ticker="SBER", quantity=50, avg_price=300.0, bot="t"),
            Position(ticker="GAZP", quantity=-100, avg_price=120.0, bot="t"),
        ],
        cash=133_000.0,
    )
    # No prices cached → fall back to LONG-only sum.
    # 133k + 50×300 = 148k (SHORT ignored).
    assert book.total_equity() == pytest.approx(148_000.0)


def test_total_equity_mtm_case_insensitive() -> None:
    """Ticker lookup tolerates case differences in the prices dict."""
    book = _seed_book(
        positions=[Position(ticker="SBER", quantity=50, avg_price=300.0, bot="t")],
        cash=100_000.0,
    )
    eq = book.total_equity_mtm({"sber": 310.0})
    assert eq == pytest.approx(100_000.0 + 50 * 310.0)


def test_total_equity_mtm_zero_qty_skipped() -> None:
    """Zero-qty rows are ignored (defensive)."""
    book = _seed_book(
        positions=[
            Position(ticker="SBER", quantity=0, avg_price=300.0, bot="t"),
            Position(ticker="GAZP", quantity=-100, avg_price=120.0, bot="t"),
        ],
        cash=100_000.0,
    )
    eq = book.total_equity_mtm({"SBER": 310.0, "GAZP": 115.0})
    # Only GAZP contributes: PnL = (-100) × (115-120) = +500
    assert eq == pytest.approx(100_500.0)


@pytest.mark.asyncio
async def test_refresh_mtm_prices_populates_cache() -> None:
    """`_refresh_mtm_prices` calls `_fetch_iss_prices` and caches results."""
    book = _seed_book(
        positions=[
            Position(ticker="SBER", quantity=50, avg_price=300.0, bot="t"),
            Position(ticker="GAZP", quantity=-100, avg_price=120.0, bot="t"),
        ],
        cash=133_000.0,
    )
    with patch.object(
        book,
        "_fetch_iss_prices",
        return_value={"SBER": 310.5, "GAZP": 114.7},
    ) as mock_fetch:
        await book._refresh_mtm_prices()
        mock_fetch.assert_awaited_once()

    assert "SBER" in book._mtm_prices
    assert book._mtm_prices["SBER"][0] == pytest.approx(310.5)
    assert book._mtm_prices["GAZP"][0] == pytest.approx(114.7)

    # The cached prices feed `total_equity()` automatically:
    #   PnL = 50×(310.5−300) + (−100)×(114.7−120) = 525 + 530 = 1055
    #   equity = 133_000 + 1055 = 134_055
    expected = 133_000.0 + 50 * (310.5 - 300.0) + (-100) * (114.7 - 120.0)
    assert book.total_equity() == pytest.approx(expected)


@pytest.mark.asyncio
async def test_refresh_mtm_prices_respects_ttl() -> None:
    """Fresh cache entries are not re-fetched within TTL."""
    import time

    book = _seed_book(
        positions=[Position(ticker="SBER", quantity=50, avg_price=300.0, bot="t")],
        cash=100_000.0,
    )
    book._mtm_prices = {"SBER": (310.0, time.time())}

    with patch.object(book, "_fetch_iss_prices", return_value={}) as mock_fetch:
        await book._refresh_mtm_prices()
        mock_fetch.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_mtm_prices_swallows_errors() -> None:
    """Network failure must not crash refresh; cache stays empty."""
    book = _seed_book(
        positions=[Position(ticker="SBER", quantity=50, avg_price=300.0, bot="t")],
        cash=100_000.0,
    )

    async def _boom(_tickers):
        raise RuntimeError("ISS down")

    with patch.object(book, "_fetch_iss_prices", side_effect=_boom):
        await book._refresh_mtm_prices()  # must not raise

    assert book._mtm_prices == {}
    assert book._mtm_failures == 1
