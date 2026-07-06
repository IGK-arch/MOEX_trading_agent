"""
tests/integration/test_circuit_breaker_e2e.py — daily_pnl ≤ −2 % → all new
orders REJECTED at the risk layer.

Walk-through:
  1. Seed the global circuit breaker singleton with a daily P&L pointed
     below −2 % of the equity.
  2. Trigger the halt check via ``_check_halts``.
  3. Build a sane EXECUTE Decision and call ``RiskManager.evaluate``.
  4. Assert the result is ``REJECTED_CIRCUIT_BREAKER`` and that no
     TradeRequest is produced.
"""

from __future__ import annotations

import pytest

from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    DecisionTier,
    Direction,
    RiskCheckResult,
    SignalSource,
    UnifiedSignal,
)
from app.risk import circuit_breakers as cb_mod
from app.risk.circuit_breakers import CircuitBreaker, CircuitState


def _build_execute_decision(
    *, ticker: str = "SBER", mag: float = 0.80, rr: float = 2.5
) -> Decision:
    """Build execute decision."""
    sig = UnifiedSignal(
        source=SignalSource.TA,
        detector="circuit_test",
        ticker=ticker,
        direction=Direction.BUY,
        magnitude=mag,
        raw_confidence=mag,
        horizon_min=60,
        price=100.0,
        entry_level=100.0,
        stop_level=98.0,
        target_level=105.0,
        expected_rr=rr,
        atr=1.0,
    )
    return Decision(
        decision_id="test_cb_decision",
        cycle_id="cb_cycle",
        ticker=ticker,
        action=DecisionAction.EXECUTE,
        tier=DecisionTier.TIER1,
        direction=Direction.BUY,
        combined_magnitude=mag,
        signals=[sig],
        expected_rr=rr,
    )


@pytest.mark.asyncio
async def test_circuit_breaker_blocks_new_orders_on_2pct_loss(monkeypatch):
    """daily_pnl_pct = −2 % → all new EXECUTE decisions get REJECTED."""
    monkeypatch.setattr("app.risk.risk_manager.is_trading_open", lambda: True)

    cb = CircuitBreaker()
    monkeypatch.setattr(cb_mod, "_cb", cb)
    monkeypatch.setattr(cb_mod, "get_circuit_breaker", lambda: cb)
    import app.risk.risk_manager as rm_mod

    monkeypatch.setattr(rm_mod, "get_circuit_breaker", lambda: cb)

    cb.state = CircuitState(
        daily_pnl_rub=-25_000.0,
        peak_equity_rub=1_000_000.0,
        current_equity_rub=975_000.0,
        daily_pnl_pct=-0.025,
    )
    await cb._check_halts(current_equity_rub=975_000.0)
    assert cb.state.is_blocked, "Circuit breaker must engage at −2 %"
    assert "daily_loss_halt" in cb.state.block_reason

    from app.risk.risk_manager import RiskManager

    rm = RiskManager()
    rm.cb = cb

    decision = _build_execute_decision()
    result = await rm.evaluate(decision)
    assert result.result == RiskCheckResult.REJECTED_CIRCUIT_BREAKER, (
        f"Expected REJECTED_CIRCUIT_BREAKER, got {result.result}"
    )
    assert result.trade_request is None
    assert "daily_loss_halt" in result.reason


@pytest.mark.asyncio
async def test_circuit_breaker_passes_when_within_limits(monkeypatch):
    """Sanity check: when daily PnL is just −0.5 %, decisions still pass."""
    monkeypatch.setattr("app.risk.risk_manager.is_trading_open", lambda: True)

    cb = CircuitBreaker()
    cb.state = CircuitState(
        daily_pnl_rub=-5_000.0,
        peak_equity_rub=1_000_000.0,
        current_equity_rub=995_000.0,
        daily_pnl_pct=-0.005,
    )
    await cb._check_halts(current_equity_rub=995_000.0)
    assert not cb.state.is_blocked, "Below −2 % must NOT halt"

    is_blocked, _ = cb.should_block_new_trades()
    assert not is_blocked


@pytest.mark.asyncio
async def test_circuit_breaker_max_drawdown_halts_too(monkeypatch):
    """Max drawdown ≥ 10 % triggers a 24-h halt that blocks new orders."""
    monkeypatch.setattr("app.risk.risk_manager.is_trading_open", lambda: True)

    cb = CircuitBreaker()
    monkeypatch.setattr(cb_mod, "_cb", cb)
    monkeypatch.setattr(cb_mod, "get_circuit_breaker", lambda: cb)

    cb.state = CircuitState(
        peak_equity_rub=1_000_000.0,
        current_equity_rub=880_000.0,
        max_drawdown_pct=0.12,
        current_drawdown_pct=0.12,
    )
    await cb._check_halts(current_equity_rub=880_000.0)
    assert cb.state.is_blocked
    assert "max_drawdown_halt" in cb.state.block_reason
    is_blocked, reason = cb.should_block_new_trades()
    assert is_blocked
    assert "max_drawdown" in reason
