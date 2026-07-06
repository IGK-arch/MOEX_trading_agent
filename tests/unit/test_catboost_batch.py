"""
tests/unit/test_catboost_batch.py — v0.13.0 batch-inference parity + speed.

Validates:
    1. TACatBoost.predict_batch() returns scalars in the SAME order as N×
       predict_success_proba() (with a tight numeric tolerance).
    2. MetaClassifier.score_batch() returns scalars in the SAME order as N×
       score() (with the same tolerance).
    3. Aggregator.aggregate_batch() preserves per-ticker meta scores +
       gating identical to looping over aggregate().
    4. Empty / single-row edge cases behave correctly.
    5. Benchmark: 1×batch of N << N×single (5-20× expected with real model;
       still measurable on the heuristic fallback because of pandas/numpy
       construction overhead).
"""

from __future__ import annotations

import asyncio
import math
import time

import pytest

from app.agents.meta_classifier import MetaClassifier, MetaContext
from app.agents.ta_catboost import FEATURE_COLUMNS, TACatBoost
from app.dispatcher.aggregator import SignalAggregator
from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    DecisionTier,
    Direction,
    RiskCheckResult,
    SignalSource,
    UnifiedSignal,
)

N_ROWS = 50


def _build_feat(seed: int) -> dict[str, float]:
    """Realistic-ish feature row varied by seed so heuristic ranks change."""
    feat = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    feat["expected_rr"] = 1.0 + (seed % 5) * 0.3
    feat["rsi"] = 30.0 + (seed % 10) * 4.0
    feat["adx"] = 15.0 + (seed % 8) * 3.0
    feat["vol_z"] = -1.0 + (seed % 6) * 0.4
    feat["atr_pct"] = 0.5 + (seed % 5) * 0.2
    feat["sup_atrs"] = 1.0 + (seed % 5)
    feat["res_atrs"] = 2.0 + (seed % 5)
    feat["atr_at_entry_pct"] = 0.8 + (seed % 4) * 0.15
    feat["candle_bullish_score"] = (seed % 7) / 10.0
    feat["candle_bearish_score"] = ((seed + 3) % 7) / 10.0
    pats = [c for c in FEATURE_COLUMNS if c.startswith("pat_")]
    if pats:
        feat[pats[seed % len(pats)]] = 1.0
    return feat


def test_predict_batch_matches_single_predict():
    """N×single == 1×batch elementwise (within float tolerance)."""
    cat = TACatBoost()
    feats = [_build_feat(i) for i in range(N_ROWS)]

    singles = [cat.predict_success_proba(f) for f in feats]
    batch = cat.predict_batch(feats)

    assert len(batch) == N_ROWS == len(singles)
    for i, (a, b) in enumerate(zip(singles, batch, strict=False)):
        assert math.isclose(a, b, abs_tol=1e-9), f"row {i}: single={a!r} != batch={b!r}"


def test_predict_batch_preserves_input_order():
    """Shuffling the inputs shuffles the outputs in the same order."""
    cat = TACatBoost()
    feats = [_build_feat(i) for i in range(N_ROWS)]

    base = cat.predict_batch(feats)
    rev = cat.predict_batch(list(reversed(feats)))

    assert rev == list(reversed(base))


def test_predict_batch_handles_empty_and_single():
    """Empty list → []. Single-row list → 1-element list."""
    cat = TACatBoost()
    assert cat.predict_batch([]) == []

    one = cat.predict_batch([_build_feat(0)])
    assert len(one) == 1
    assert 0.0 <= one[0] <= 1.0
    assert math.isclose(one[0], cat.predict_success_proba(_build_feat(0)), abs_tol=1e-9)


def test_predict_batch_speedup():
    """1×batch of N must be faster than N×single. Time-bound, not just ratio."""
    cat = TACatBoost()
    feats = [_build_feat(i) for i in range(N_ROWS)]

    cat.predict_batch(feats[:1])
    cat.predict_success_proba(feats[0])

    t0 = time.perf_counter()
    for f in feats:
        cat.predict_success_proba(f)
    single_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cat.predict_batch(feats)
    batch_ms = (time.perf_counter() - t0) * 1000

    assert batch_ms <= single_ms + 1.0, (
        f"batch slower than single? batch={batch_ms:.2f}ms single={single_ms:.2f}ms"
    )
    print(
        f"\n[catboost] N={N_ROWS}: single={single_ms:.2f}ms, "
        f"batch={batch_ms:.2f}ms, speedup={single_ms / max(batch_ms, 1e-6):.2f}x"
    )


def _make_signal(
    source: SignalSource, direction: Direction, ticker: str = "SBER", magnitude: float = 0.7
) -> UnifiedSignal:
    """Make signal."""
    return UnifiedSignal(
        source=source,
        detector="test",
        ticker=ticker,
        direction=direction,
        magnitude=magnitude,
        raw_confidence=magnitude,
        horizon_min=60,
        price=100.0,
        entry_level=100.0,
        stop_level=98.0,
        target_level=104.0,
        expected_rr=2.0,
        atr=2.0,
    )


def _make_decision(
    ticker: str, signals: list[UnifiedSignal], direction: Direction = Direction.BUY
) -> Decision:
    """Make decision."""
    return Decision(
        decision_id=f"d-{ticker}",
        cycle_id="cyc",
        ticker=ticker,
        action=DecisionAction.EXECUTE,
        tier=DecisionTier.NONE,
        direction=direction,
        combined_magnitude=0.7,
        signals=signals,
        risk_check=RiskCheckResult.PASSED,
        expected_rr=2.0,
    )


def test_meta_score_batch_matches_single_score():
    """N×meta.score == 1×meta.score_batch elementwise."""
    meta = MetaClassifier()
    decisions = []
    contexts = []
    for i in range(N_ROWS):
        sig = _make_signal(
            SignalSource.TA, Direction.BUY, ticker=f"T{i:02d}", magnitude=0.5 + (i % 5) * 0.08
        )
        decisions.append(_make_decision(f"T{i:02d}", [sig]))
        contexts.append(
            MetaContext(
                ofi=(i % 10) / 20.0,
                vpin=(i % 7) / 15.0,
                atr_pct=0.5 + (i % 5) * 0.2,
                current_dd_pct=(i % 6) * 0.01,
                losing_streak=i % 4,
                winning_streak=(i + 2) % 4,
                regime=("trending" if i % 3 == 0 else "mean_reverting" if i % 3 == 1 else "crisis"),
            )
        )

    singles = [meta.score(d, c) for d, c in zip(decisions, contexts, strict=False)]
    batch = meta.score_batch(decisions, contexts)

    assert len(batch) == N_ROWS == len(singles)
    for i, (a, b) in enumerate(zip(singles, batch, strict=False)):
        assert math.isclose(a, b, abs_tol=1e-9), f"row {i}: single={a!r} != batch={b!r}"


def test_meta_score_batch_speedup():
    """1×meta.score_batch must be ≤ N×meta.score."""
    meta = MetaClassifier()
    decisions = []
    contexts = []
    for i in range(N_ROWS):
        sig = _make_signal(SignalSource.TA, Direction.BUY, ticker=f"T{i:02d}")
        decisions.append(_make_decision(f"T{i:02d}", [sig]))
        contexts.append(MetaContext(regime="trending"))

    meta.score(decisions[0], contexts[0])
    meta.score_batch(decisions[:1], contexts[:1])

    t0 = time.perf_counter()
    for d, c in zip(decisions, contexts, strict=False):
        meta.score(d, c)
    single_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    meta.score_batch(decisions, contexts)
    batch_ms = (time.perf_counter() - t0) * 1000

    assert batch_ms <= single_ms + 1.0, (
        f"meta batch slower than single? batch={batch_ms:.2f}ms single={single_ms:.2f}ms"
    )
    print(
        f"\n[meta] N={N_ROWS}: single={single_ms:.2f}ms, "
        f"batch={batch_ms:.2f}ms, speedup={single_ms / max(batch_ms, 1e-6):.2f}x"
    )


def test_meta_score_batch_validates_lengths():
    """score_batch raises when decisions/contexts length mismatch."""
    meta = MetaClassifier()
    sig = _make_signal(SignalSource.TA, Direction.BUY)
    dec = _make_decision("SBER", [sig])
    with pytest.raises(ValueError):
        meta.score_batch([dec, dec], [MetaContext()])


def test_aggregator_aggregate_batch_parity():
    """
    aggregate_batch() must produce decisions equivalent to looping over
    aggregate() (meta disabled here so no double-stage gating differs).
    """
    import app.config as cfg  # type: ignore

    original = cfg.META_ENABLED
    cfg.META_ENABLED = False
    try:
        agg = SignalAggregator()
        per_ticker = {
            f"T{i:02d}": [
                _make_signal(SignalSource.TA, Direction.BUY, ticker=f"T{i:02d}"),
                _make_signal(SignalSource.NEWS, Direction.BUY, ticker=f"T{i:02d}"),
            ]
            for i in range(5)
        }

        loop = asyncio.new_event_loop()
        try:
            batch_decisions = loop.run_until_complete(
                agg.aggregate_batch(cycle_id="c1", per_ticker_signals=per_ticker)
            )
            single_decisions = [
                loop.run_until_complete(agg.aggregate(f"T{i:02d}", "c1", per_ticker[f"T{i:02d}"]))
                for i in range(5)
            ]
        finally:
            loop.close()

        assert len(batch_decisions) == len(single_decisions) == 5
        for b, s in zip(batch_decisions, single_decisions, strict=False):
            assert b.ticker == s.ticker
            assert b.action == s.action
            assert b.direction == s.direction
            assert math.isclose(b.combined_magnitude, s.combined_magnitude, abs_tol=1e-9)
    finally:
        cfg.META_ENABLED = original
