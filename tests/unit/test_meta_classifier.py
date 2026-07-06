"""
tests/unit/test_meta_classifier.py — MetaClassifier feature extraction + heuristic.
"""

from __future__ import annotations

import pytest

from app.agents.meta_classifier import (
    META_DIRECTION_ONEHOT,
    META_FEATURE_COLUMNS,
    META_NUMERIC_FEATURES,
    META_REGIME_ONEHOT,
    META_SOURCE_ONEHOT,
    MetaClassifier,
    MetaContext,
)
from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    DecisionTier,
    Direction,
    RiskCheckResult,
    SignalSource,
    UnifiedSignal,
)


def _make_signal(
    source: SignalSource,
    direction: Direction,
    ticker: str = "SBER",
    magnitude: float = 0.7,
    detector: str = "test",
) -> UnifiedSignal:
    """Make signal."""
    return UnifiedSignal(
        source=source,
        detector=detector,
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
    signals: list[UnifiedSignal],
    direction: Direction = Direction.BUY,
    combined_magnitude: float = 0.7,
    expected_rr: float = 2.0,
) -> Decision:
    """Make decision."""
    return Decision(
        decision_id="testid",
        cycle_id="cyc",
        ticker="SBER",
        action=DecisionAction.EXECUTE,
        tier=DecisionTier.NONE,
        direction=direction,
        combined_magnitude=combined_magnitude,
        signals=signals,
        risk_check=RiskCheckResult.PASSED,
        expected_rr=expected_rr,
    )


def test_feature_columns_order_and_count():
    """Test feature columns order and count."""
    expected_count = (
        len(META_NUMERIC_FEATURES)
        + len(META_SOURCE_ONEHOT)
        + len(META_REGIME_ONEHOT)
        + len(META_DIRECTION_ONEHOT)
    )
    assert len(META_FEATURE_COLUMNS) == expected_count

    assert len(set(META_FEATURE_COLUMNS)) == len(META_FEATURE_COLUMNS)


def test_build_features_extracts_correct_one_hots():
    """Test build features extracts correct one hots."""
    sigs = [
        _make_signal(SignalSource.TA, Direction.BUY),
        _make_signal(SignalSource.NEWS, Direction.BUY),
    ]
    dec = _make_decision(sigs)
    ctx = MetaContext(regime="trending", ofi=0.3, vpin=0.2)

    feat = MetaClassifier.build_features(dec, ctx)

    assert feat["src_TA"] == 1.0
    assert feat["src_NEWS"] == 1.0
    assert feat["src_ANOMALY"] == 0.0
    assert feat["src_PAIR"] == 0.0

    assert feat["dir_BUY"] == 1.0
    assert feat["dir_SELL"] == 0.0

    assert feat["regime_trending"] == 1.0
    assert feat["regime_mean_reverting"] == 0.0
    assert feat["regime_crisis"] == 0.0

    assert feat["combined_magnitude"] == 0.7
    assert feat["n_signals"] == 2.0
    assert feat["n_sources_unique"] == 2.0
    assert feat["confluence_mult"] == 1.5
    assert feat["ofi"] == 0.3
    assert feat["vpin"] == 0.2

    for col in META_FEATURE_COLUMNS:
        assert col in feat, f"Missing column: {col}"


def test_confluence_mult_three_plus_sources():
    """Test confluence mult three plus sources."""
    sigs = [
        _make_signal(SignalSource.TA, Direction.BUY),
        _make_signal(SignalSource.NEWS, Direction.BUY),
        _make_signal(SignalSource.ANOMALY, Direction.BUY),
    ]
    dec = _make_decision(sigs)
    feat = MetaClassifier.build_features(dec, MetaContext())
    assert feat["confluence_mult"] == 2.0


def test_heuristic_in_bounds():
    """Heuristic should always return a value in [0.05, 0.95]."""
    sigs = [_make_signal(SignalSource.TA, Direction.BUY)]
    dec = _make_decision(sigs)

    ctx_bad = MetaContext(
        current_dd_pct=0.10,
        daily_pnl_pct=-0.05,
        losing_streak=5,
        regime="crisis",
        vpin=0.6,
        ofi=-0.5,
    )
    score_bad = MetaClassifier()._heuristic_score(MetaClassifier.build_features(dec, ctx_bad))
    assert 0.05 <= score_bad <= 0.95
    assert score_bad < 0.5, "Bad context should score below 0.5"

    sigs3 = [
        _make_signal(SignalSource.TA, Direction.BUY, magnitude=0.9),
        _make_signal(SignalSource.NEWS, Direction.BUY),
        _make_signal(SignalSource.ANOMALY, Direction.BUY),
    ]
    dec3 = _make_decision(sigs3, combined_magnitude=0.9, expected_rr=2.5)
    ctx_good = MetaContext(
        current_dd_pct=0.0,
        daily_pnl_pct=0.02,
        winning_streak=3,
        regime="trending",
        ofi=0.4,
        vpin=0.1,
    )
    score_good = MetaClassifier()._heuristic_score(MetaClassifier.build_features(dec3, ctx_good))
    assert 0.05 <= score_good <= 0.95
    assert score_good > 0.5, "Good context should score above 0.5"
    assert score_good > score_bad


def test_score_falls_back_to_heuristic_without_model():
    """No model loaded → score() uses heuristic, returns valid probability."""
    meta = MetaClassifier(model_path=None)

    sigs = [_make_signal(SignalSource.TA, Direction.BUY)]
    dec = _make_decision(sigs)
    ctx = MetaContext()
    score = meta.score(dec, ctx)
    assert 0.0 <= score <= 1.0
    assert meta._loaded is False


def test_score_batch_consistent_with_score():
    """Test score batch consistent with score."""
    meta = MetaClassifier(model_path=None)
    sigs = [
        [_make_signal(SignalSource.TA, Direction.BUY)],
        [_make_signal(SignalSource.NEWS, Direction.SELL, magnitude=0.4)],
    ]
    decs = [
        _make_decision(sigs[0]),
        _make_decision(sigs[1], direction=Direction.SELL, combined_magnitude=0.4),
    ]
    ctxs = [MetaContext(regime="trending"), MetaContext(regime="crisis")]

    scores_individual = [meta.score(d, c) for d, c in zip(decs, ctxs, strict=False)]
    scores_batch = meta.score_batch(decs, ctxs)
    assert scores_individual == scores_batch


def test_score_batch_empty_returns_empty():
    """Test score batch empty returns empty."""
    meta = MetaClassifier(model_path=None)
    assert meta.score_batch([], []) == []


def test_score_batch_length_mismatch_raises():
    """Test score batch length mismatch raises."""
    meta = MetaClassifier(model_path=None)
    sigs = [_make_signal(SignalSource.TA, Direction.BUY)]
    with pytest.raises(ValueError):
        meta.score_batch([_make_decision(sigs)], [])
