"""Tests for the class-based MTFConfluence (Phase 27.3 / v0.0.39).

Distinct from `test_mtf_confluence.py` which targets the existing
function-style API (compute_mtf_trend / mtf_confluence_score) used by the
aggregator.

The class-based gate is invoked inside `app/agents/ta_trader.py` AFTER the
confluence_filters check. We verify 4 representative scenarios:

  1. BUY reversal with oversold higher TF (RSI<35) + bullish lower TF → ok
  2. BUY reversal with RSI=70 on higher TF (no exhaustion) → veto
  3. BUY continuation with EMA20>EMA50 + bullish lower TF → ok
  4. BUY continuation with EMA20<EMA50 → veto
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.agents.ta_patterns.mtf_confluence import (
    MTFConfluence,
)


def _bullish_lower_tf(n: int = 30, base: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    """Lower TF with last `n` bars all clearly green (close > open) and
    positive volume — the kind of confirmation that triggers a BUY pass.
    """
    rng = np.random.default_rng(11)
    closes = base + np.cumsum(np.full(n, step) + rng.normal(0, 0.05, n))
    opens = closes - step / 2
    high = np.maximum(opens, closes) + 0.05
    low = np.minimum(opens, closes) - 0.05
    return pd.DataFrame(
        {
            "open": opens,
            "high": high,
            "low": low,
            "close": closes,
            "volume": rng.integers(1000, 5000, n),
        }
    )


def _bearish_lower_tf(n: int = 30, base: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    """Bearish lower tf."""
    rng = np.random.default_rng(22)
    closes = base - np.cumsum(np.full(n, step) + rng.normal(0, 0.05, n))
    opens = closes + step / 2
    high = np.maximum(opens, closes) + 0.05
    low = np.minimum(opens, closes) - 0.05
    return pd.DataFrame(
        {
            "open": opens,
            "high": high,
            "low": low,
            "close": closes,
            "volume": rng.integers(1000, 5000, n),
        }
    )


def _oversold_higher_tf(n: int = 80, base: float = 100.0, step: float = -0.6) -> pd.DataFrame:
    """Higher TF in a sharp down-leg → RSI very low; EMA20 < EMA50."""
    rng = np.random.default_rng(33)
    closes = base + np.cumsum(np.full(n, step) + rng.normal(0, 0.05, n))
    opens = closes - step / 2
    high = np.maximum(opens, closes) + 0.05
    low = np.minimum(opens, closes) - 0.05
    return pd.DataFrame(
        {
            "open": opens,
            "high": high,
            "low": low,
            "close": closes,
            "volume": rng.integers(1000, 5000, n),
        }
    )


def _overbought_higher_tf(n: int = 80, base: float = 100.0, step: float = 0.6) -> pd.DataFrame:
    """Higher TF in a sharp up-leg → RSI very high; EMA20 > EMA50."""
    rng = np.random.default_rng(44)
    closes = base + np.cumsum(np.full(n, step) + rng.normal(0, 0.05, n))
    opens = closes - step / 2
    high = np.maximum(opens, closes) + 0.05
    low = np.minimum(opens, closes) - 0.05
    return pd.DataFrame(
        {
            "open": opens,
            "high": high,
            "low": low,
            "close": closes,
            "volume": rng.integers(1000, 5000, n),
        }
    )


def test_mtf_reversal_buy_oversold_passes():
    """BUY reversal with oversold higher TF + bullish lower TF → ok."""
    df_higher = _oversold_higher_tf()
    df_lower = _bullish_lower_tf()
    mtf = MTFConfluence(rsi_oversold=45.0)
    ok, reason = mtf.validate("reversal", "BUY", df_higher, df_lower)
    assert ok is True, f"unexpected veto: {reason}"


def test_mtf_reversal_buy_overbought_higher_vetoes():
    """BUY reversal with RSI on higher TF in overbought territory (>65)
    has no exhaustion → veto."""
    df_higher = _overbought_higher_tf()
    df_lower = _bullish_lower_tf()
    mtf = MTFConfluence()
    ok, reason = mtf.validate("reversal", "BUY", df_higher, df_lower)
    assert ok is False
    assert "no_higher_exhaustion" in reason


def test_mtf_continuation_buy_uptrend_passes():
    """BUY continuation with EMA20>EMA50 (uptrend higher TF) + bullish
    lower TF → ok."""
    df_higher = _overbought_higher_tf()
    df_lower = _bullish_lower_tf()
    mtf = MTFConfluence()
    ok, reason = mtf.validate("continuation", "BUY", df_higher, df_lower)
    assert ok is True, f"unexpected veto: {reason}"


def test_mtf_continuation_buy_downtrend_vetoes():
    """BUY continuation with EMA20<EMA50 (downtrend higher TF) → veto."""
    df_higher = _oversold_higher_tf()
    df_lower = _bullish_lower_tf()
    mtf = MTFConfluence()
    ok, reason = mtf.validate("continuation", "BUY", df_higher, df_lower)
    assert ok is False
    assert "higher_trend" in reason and "down" in reason
