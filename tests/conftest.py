"""
tests/conftest.py — shared fixtures.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def event_loop():
    """Event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _make_synthetic_df(
    n: int = 120,
    start_price: float = 100.0,
    pattern: str = "double_top",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV that contains a recognisable chart pattern."""
    rng = np.random.default_rng(seed)
    closes = [start_price]
    for i in range(1, n):
        if pattern == "double_top":
            if i < 30:
                target = start_price + (i / 30) * 12
            elif i < 50:
                target = start_price + 12 - ((i - 30) / 20) * 10
            elif i < 75:
                target = start_price + 2 + ((i - 50) / 25) * 10
            else:
                target = start_price + 12 - ((i - 75) / (n - 75)) * 20
        elif pattern == "uptrend":
            target = start_price + i * 0.3
        elif pattern == "downtrend":
            target = start_price - i * 0.3
        elif pattern == "flat":
            target = start_price
        else:
            target = start_price
        closes.append(target + rng.standard_normal() * 0.4)

    closes = np.array(closes)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + np.abs(rng.standard_normal(n)) * 0.5
    lows = np.minimum(opens, closes) - np.abs(rng.standard_normal(n)) * 0.5
    volumes = rng.integers(50_000, 150_000, n).astype(float)
    begin = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "begin": begin,
        }
    )


@pytest.fixture
def synthetic_double_top():
    """Synthetic double top."""
    return _make_synthetic_df(pattern="double_top")


@pytest.fixture
def synthetic_uptrend():
    """Synthetic uptrend."""
    return _make_synthetic_df(pattern="uptrend")


@pytest.fixture
def synthetic_downtrend():
    """Synthetic downtrend."""
    return _make_synthetic_df(pattern="downtrend")


@pytest.fixture
def synthetic_flat():
    """Synthetic flat."""
    return _make_synthetic_df(pattern="flat")


class FakePolzaClient:
    """Minimal stand-in for PolzaClient used in tests — never calls network."""

    def __init__(self) -> None:
        """Init."""
        self._started = True
        self.calls = []

    async def startup(self):
        """Startup."""
        pass

    async def shutdown(self):
        """Shutdown."""
        pass

    async def chat_json(self, messages, model="x", max_tokens=300, purpose=""):
        """Chat json."""

        self.calls.append({"messages": messages, "purpose": purpose})
        return {
            "direction": "BUY",
            "magnitude": 0.6,
            "affected_tickers": ["SBER"],
            "horizon_min": 30,
            "reason": "fake response",
        }

    async def chat(self, messages, model="x", max_tokens=300, purpose="", use_cache=True):
        """Chat."""
        self.calls.append({"messages": messages, "purpose": purpose})
        return {
            "content": "fake content",
            "model": model,
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_rub": 0.001,
        }


@pytest.fixture
def fake_polza():
    """Fake polza."""
    return FakePolzaClient()


class FakeArenaGoClient:
    """In-memory ArenaGo for integration tests."""

    def __init__(self) -> None:
        """Init."""
        self._started = True
        self._cash = 1_000_000.0
        self.submitted_orders = []

    async def startup(self):
        """Startup."""
        pass

    async def shutdown(self):
        """Shutdown."""
        pass

    async def submit_order(self, direction, ticker, quantity, decision_id, bot=None):
        """Submit order."""
        from app.execution.arenago_client import SubmitResult

        self.submitted_orders.append(
            {
                "direction": direction,
                "ticker": ticker,
                "quantity": quantity,
                "decision_id": decision_id,
                "bot": bot,
            }
        )
        price = 100.0
        order_value = price * quantity
        self._cash -= order_value
        return SubmitResult(
            success=True,
            message="OK",
            order_value=order_value,
            price=price,
            quantity=quantity,
            remaining_cash=self._cash,
            decision_id=decision_id,
        )

    async def get_positions(self):
        """Get positions."""
        return []

    async def get_cash_balance(self):
        """Get cash balance."""
        return self._cash

    async def get_bots(self):
        """Get bots."""
        return [{"name": "test_bot", "cash_balance": self._cash}]

    async def get_trades(self):
        """Get trades."""
        return []


@pytest.fixture
def fake_arenago():
    """Fake arenago."""
    return FakeArenaGoClient()
