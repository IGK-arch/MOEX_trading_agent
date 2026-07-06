"""Phase 27.5 — tests for strategy capital allocation."""

from __future__ import annotations

import json

import pytest

import app.config as cfg
from app.agents.ta_patterns.noise_blacklist import (
    is_noisy,
    magnitude_penalty,
)
from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    Direction,
    SignalSource,
    UnifiedSignal,
)
from app.risk.position_book import Position, PositionBook
from app.risk.risk_manager import RiskManager


def test_compute_optimal_allocation_balanced():
    """Test compute optimal allocation balanced."""
    from scripts.strategy_backtest import compute_optimal_allocation

    metrics = {
        "TA": {
            "sharpe": 2.0,
            "vol_pct": 1.0,
            "win_rate": 0.6,
            "avg_win_pct": 1.0,
            "avg_loss_pct": 0.8,
        },
        "ANOMALY": {
            "sharpe": 0.5,
            "vol_pct": 2.0,
            "win_rate": 0.4,
            "avg_win_pct": 1.5,
            "avg_loss_pct": 1.0,
        },
        "NEWS": {
            "sharpe": 1.0,
            "vol_pct": 0.5,
            "win_rate": 0.55,
            "avg_win_pct": 1.2,
            "avg_loss_pct": 1.0,
        },
        "MEAN_REV": {
            "sharpe": 0.2,
            "vol_pct": 1.5,
            "win_rate": 0.45,
            "avg_win_pct": 0.8,
            "avg_loss_pct": 0.8,
        },
    }
    out = compute_optimal_allocation(metrics, floor_pct=0.05)
    assert set(out.keys()) == {"sharpe_weighted", "risk_parity", "kelly_weighted", "final"}
    total = sum(out["final"].values())
    assert abs(total - 1.0) < 1e-3
    assert all(v >= 0.05 - 1e-6 for v in out["final"].values())
    assert out["sharpe_weighted"]["TA"] >= out["sharpe_weighted"]["ANOMALY"]


def test_compute_allocation_handles_zero_sharpe():
    """Test compute allocation handles zero sharpe."""
    from scripts.strategy_backtest import compute_optimal_allocation

    metrics = {
        "TA": {
            "sharpe": 0.0,
            "vol_pct": 1.0,
            "win_rate": 0.5,
            "avg_win_pct": 0.5,
            "avg_loss_pct": 0.5,
        },
        "ANOMALY": {
            "sharpe": -1.0,
            "vol_pct": 1.0,
            "win_rate": 0.3,
            "avg_win_pct": 0.5,
            "avg_loss_pct": 0.5,
        },
    }
    out = compute_optimal_allocation(metrics, floor_pct=0.10)
    for v in out["final"].values():
        assert v >= 0.10 - 1e-6


def test_exposure_by_source_sums_market_value():
    """Test exposure by source sums market value."""
    book = PositionBook(deposit_total=1_000_000)
    book._positions["SBER"] = Position(
        ticker="SBER",
        quantity=100,
        avg_price=300.0,
        bot="paper",
        source="TA",
    )
    book._positions["GAZP"] = Position(
        ticker="GAZP",
        quantity=200,
        avg_price=150.0,
        bot="paper",
        source="NEWS",
    )
    book._positions["LKOH"] = Position(
        ticker="LKOH",
        quantity=50,
        avg_price=7000.0,
        bot="paper",
        source="TA",
    )
    book._cash_balance = 100_000
    assert book.exposure_by_source("TA") == pytest.approx(380_000.0)
    assert book.exposure_by_source("NEWS") == pytest.approx(30_000.0)
    assert book.exposure_by_source("ANOMALY") == 0.0


def test_exposure_by_source_fallback_map():
    """Test exposure by source fallback map."""
    book = PositionBook(deposit_total=1_000_000)
    book._positions["SBER"] = Position(
        ticker="SBER",
        quantity=100,
        avg_price=300.0,
        bot="paper",
    )
    book._source_by_ticker["SBER"] = "TA"
    assert book.exposure_by_source("TA") == pytest.approx(30_000.0)


def test_mark_entry_with_source_updates_position():
    """Test mark entry with source updates position."""
    book = PositionBook(deposit_total=1_000_000)
    book._positions["SBER"] = Position(
        ticker="SBER",
        quantity=100,
        avg_price=300.0,
        bot="paper",
    )
    book.mark_entry_with_source("SBER", "NEWS")
    assert book._positions["SBER"].source == "NEWS"
    assert book._source_by_ticker["SBER"] == "NEWS"


def _decision(ticker: str, source: SignalSource, magnitude: float = 0.8) -> Decision:
    """Decision."""
    sig = UnifiedSignal(
        source=source,
        detector="test",
        ticker=ticker,
        direction=Direction.BUY,
        magnitude=magnitude,
        raw_confidence=magnitude,
        horizon_min=60,
        price=100.0,
        entry_level=100.0,
        stop_level=98.0,
        target_level=104.0,
        expected_rr=2.0,
        atr=1.0,
    )
    return Decision(
        decision_id="dec",
        cycle_id="c1",
        ticker=ticker,
        action=DecisionAction.EXECUTE,
        direction=Direction.BUY,
        combined_magnitude=magnitude,
        signals=[sig],
        expected_rr=2.0,
        dominant_source=source.value,
    )


def test_strategy_cap_violation_returns_reason():
    """Test strategy cap violation returns reason."""
    rm = RiskManager(deposit_total=1_000_000)
    rm.book = PositionBook(deposit_total=1_000_000)
    rm.book._cash_balance = 1_000_000
    rm.book._positions["SBER"] = Position(
        ticker="SBER",
        quantity=5000,
        avg_price=100.0,
        bot="paper",
        source="TA",
    )
    dec = _decision("GAZP", SignalSource.TA)
    reason = rm._strategy_cap_violation(dec, target_notional=30_000)
    assert reason is None
    rm.book._positions["LKOH"] = Position(
        ticker="LKOH",
        quantity=10,
        avg_price=20_000.0,
        bot="paper",
        source="TA",
    )
    rm.book._cash_balance = 100_000
    reason2 = rm._strategy_cap_violation(dec, target_notional=10_000)
    assert reason2 is not None
    assert "TA" in reason2


def test_strategy_cap_floor_keeps_minimum():
    """Test strategy cap floor keeps minimum."""
    rm = RiskManager(deposit_total=1_000_000)
    rm.book = PositionBook(deposit_total=1_000_000)
    rm.book._cash_balance = 1_000_000
    dec = _decision("SBER", SignalSource.PAIR)
    reason = rm._strategy_cap_violation(dec, target_notional=10_000)
    assert reason is None


def test_get_strategy_allocation_floor():
    """Test get strategy allocation floor."""
    assert cfg.get_strategy_allocation("UNKNOWN") == pytest.approx(
        cfg.STRATEGY_ALLOCATION_FLOOR_PCT
    )
    assert cfg.get_strategy_allocation("TA") >= cfg.STRATEGY_ALLOCATION_FLOOR_PCT


def test_noise_static_patterns_known_killers():
    """Test noise static patterns known killers."""
    for pat in ("rounding_bottom", "falling_wedge", "bull_flag", "compression_breakout_up"):
        assert is_noisy(pat) is True
    assert magnitude_penalty("rounding_bottom") == 0.3
    assert magnitude_penalty("double_top") == 1.0


def test_noise_dynamic_overrides_picked_up(tmp_path, monkeypatch):
    """Test noise dynamic overrides picked up."""
    import app.agents.ta_patterns.noise_blacklist as nb

    override_path = tmp_path / "runtime_overrides.json"
    monkeypatch.setattr(nb, "_RUNTIME_OVERRIDES_PATH", override_path)
    monkeypatch.setattr(nb, "_DYN_REFRESH_SEC", 0.0)
    nb._dynamic_cache["patterns"] = []
    override_path.write_text(json.dumps({"noise_patterns": ["secret_loser"]}))
    assert is_noisy("secret_loser") is True
    assert magnitude_penalty("secret_loser") == 0.3


def test_aggregator_picks_dominant_source_by_weighted_magnitude():
    """Test aggregator picks dominant source by weighted magnitude."""
    from app.dispatcher.aggregator import SignalAggregator

    agg = SignalAggregator()
    ta_sig = UnifiedSignal(
        source=SignalSource.TA,
        detector="d",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=0.4,
        raw_confidence=0.4,
        horizon_min=60,
        price=100.0,
    )
    news_sig = UnifiedSignal(
        source=SignalSource.NEWS,
        detector="d",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=0.6,
        raw_confidence=0.6,
        horizon_min=60,
        price=100.0,
    )
    assert agg._pick_dominant_source([ta_sig, news_sig]) == "NEWS"


def test_aggregator_dominant_source_empty():
    """Test aggregator dominant source empty."""
    from app.dispatcher.aggregator import SignalAggregator

    agg = SignalAggregator()
    assert agg._pick_dominant_source([]) is None
