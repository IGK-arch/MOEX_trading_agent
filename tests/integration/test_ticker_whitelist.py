"""
tests/integration/test_ticker_whitelist.py — PER_TICKER_POLICY enforcement.

Phase 27 (v0.4.0): ``cfg.is_signal_allowed(ticker, detector)`` gates emit at
the source. A ticker with policy ``"DISABLED"`` (e.g. SNGSP) must yield zero
``UnifiedSignal`` instances, regardless of detector strength.

We exercise both the pure config gate and a custom emitter that wraps the
gate — proving disabled tickers leave the pipeline silent.
"""

from __future__ import annotations

import pytest

import app.config as cfg
from app.agents.base import BaseAdapter
from app.dispatcher.signal import Direction, SignalSource, UnifiedSignal


def test_per_ticker_policy_disables_sngsp():
    """SNGSP is hardcoded as DISABLED in PER_TICKER_POLICY."""
    assert cfg.PER_TICKER_POLICY.get("SNGSP") == "DISABLED"
    assert cfg.is_signal_allowed("SNGSP", "bear_flag") is False
    assert cfg.is_signal_allowed("SNGSP", "research_family") is False
    assert cfg.is_signal_allowed("SNGSP") is False


def test_per_ticker_policy_allows_gold_tickers_anything():
    """SBER is GOLD → every detector is allowed."""
    assert cfg.PER_TICKER_POLICY.get("SBER") == "GOLD"
    assert cfg.is_signal_allowed("SBER", "bear_flag") is True
    assert cfg.is_signal_allowed("SBER", "any_random_detector_name") is True
    assert cfg.is_signal_allowed("SBER") is True


def test_per_ticker_policy_unknown_ticker_defaults_to_disabled():
    """Unknown tickers fail closed (capital safety)."""
    assert cfg.is_signal_allowed("NOT_A_TICKER", "bear_flag") is False


def test_per_ticker_policy_whitelist_only_filters_detectors():
    """PIKK is WHITELIST_ONLY — only candle_hammer detector allowed."""
    assert cfg.PER_TICKER_POLICY.get("PIKK") == "WHITELIST_ONLY"
    assert cfg.is_signal_allowed("PIKK", "candle_hammer") is True
    assert cfg.is_signal_allowed("PIKK", "megaphone_top") is False


class _PolicyGatedAdapter(BaseAdapter):
    """Adapter that respects cfg.is_signal_allowed — like ta_trader does.

    Mirrors the production pattern: emit only when the policy gate says yes.
    """

    name = "POLICY_GATED"

    def __init__(self, ticker: str, detector: str) -> None:
        """Init."""
        super().__init__()
        self.ticker = ticker
        self.detector = detector

    async def startup(self) -> None:
        """Startup."""
        self._started = True

    async def shutdown(self) -> None:
        """Shutdown."""
        self._started = False

    async def poll(self) -> list[UnifiedSignal]:
        """Poll."""
        if not cfg.is_signal_allowed(self.ticker, self.detector):
            return []
        return [
            UnifiedSignal(
                source=SignalSource.TA,
                detector=self.detector,
                ticker=self.ticker,
                direction=Direction.BUY,
                magnitude=0.80,
                raw_confidence=0.80,
                horizon_min=60,
                price=100.0,
                entry_level=100.0,
                stop_level=98.0,
                target_level=105.0,
                expected_rr=2.5,
                atr=1.0,
            )
        ]


@pytest.mark.asyncio
async def test_disabled_ticker_emits_zero_signals():
    """Adapter targeting a DISABLED ticker must emit 0 signals on poll."""
    adapter = _PolicyGatedAdapter(ticker="SNGSP", detector="bear_flag")
    await adapter.startup()
    sigs = await adapter.poll()
    assert sigs == []


@pytest.mark.asyncio
async def test_gold_ticker_emits_signals_normally():
    """A GOLD ticker (SBER) emits as expected — sanity counter-test."""
    adapter = _PolicyGatedAdapter(ticker="SBER", detector="bear_flag")
    await adapter.startup()
    sigs = await adapter.poll()
    assert len(sigs) == 1
    assert sigs[0].ticker == "SBER"
    assert sigs[0].source == SignalSource.TA


@pytest.mark.asyncio
async def test_whitelist_only_ticker_non_whitelisted_detector_silenced():
    """PIKK + megaphone_top (not in whitelist) → 0 signals."""
    adapter = _PolicyGatedAdapter(ticker="PIKK", detector="megaphone_top")
    await adapter.startup()
    sigs = await adapter.poll()
    assert sigs == []


@pytest.mark.asyncio
async def test_whitelist_only_ticker_whitelisted_detector_emits():
    """PIKK + candle_hammer (in whitelist) → 1 signal."""
    adapter = _PolicyGatedAdapter(ticker="PIKK", detector="candle_hammer")
    await adapter.startup()
    sigs = await adapter.poll()
    assert len(sigs) == 1
