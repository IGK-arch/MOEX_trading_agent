"""
tests/unit/test_feature_extractor.py — meta_v2 FeatureExtractor.

Validates:
  - All 100 feature columns are returned
  - One-hot encoding is correct for ticker/source/detector/tier/direction
  - Missing values default to 0
  - Latency budget: 100 featurize() calls < 1 s (i.e. < 10ms each)
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from app.training.feature_extractor import (
    DETECTOR_TOP30,
    FEATURE_COLUMNS_V2,
    FeatureExtractor,
)


def _decision_dict(
    ticker: str = "SBER",
    direction: str = "BUY",
    tier: str = "1",
    mag: float = 0.7,
    rr: float = 2.0,
    sources: list[str] | None = None,
    detectors: list[str] | None = None,
) -> dict:
    """Build a decision dict mimicking decisions.db row + signals."""
    sources = sources or ["TA"]
    detectors = detectors or ["double_top"]
    signals = []
    for src, det in zip(sources, detectors, strict=False):
        signals.append(
            {
                "source": src,
                "detector": det,
                "ticker": ticker,
                "direction": direction,
                "magnitude": mag,
                "metadata": {"ofi": 0.2, "vpin": 0.15, "regime": "trending"},
            }
        )
    return {
        "ticker": ticker,
        "direction": direction,
        "tier": tier,
        "combined_magnitude": mag,
        "expected_rr": rr,
        "meta_score": 0.55,
        "signals": signals,
    }


def test_feature_columns_count_is_100():
    """Phase 27.8 spec says ≥ 50; current impl is 100 columns."""
    assert len(FEATURE_COLUMNS_V2) >= 50
    assert len(set(FEATURE_COLUMNS_V2)) == len(FEATURE_COLUMNS_V2), (
        "duplicate column names in FEATURE_COLUMNS_V2"
    )


def test_featurize_returns_all_columns():
    """Test featurize returns all columns."""
    fe = FeatureExtractor()
    feat = fe.featurize(_decision_dict())
    for col in FEATURE_COLUMNS_V2:
        assert col in feat, f"missing column: {col}"
    assert len(feat) == len(FEATURE_COLUMNS_V2)


def test_ticker_one_hot_correct():
    """Test ticker one hot correct."""
    fe = FeatureExtractor()
    feat = fe.featurize(_decision_dict(ticker="SBER"))
    assert feat["tkr_SBER"] == 1.0
    assert feat["tkr_GAZP"] == 0.0
    assert feat["tkr_LKOH"] == 0.0


def test_source_one_hot_correct():
    """Test source one hot correct."""
    fe = FeatureExtractor()
    feat = fe.featurize(_decision_dict(sources=["TA", "NEWS"], detectors=["x", "y"]))
    assert feat["src_TA"] == 1.0
    assert feat["src_NEWS"] == 1.0
    assert feat["src_ANOMALY"] == 0.0
    assert feat["src_PAIR"] == 0.0
    assert feat["src_MEAN_REV"] == 0.0


def test_direction_one_hot_correct():
    """Test direction one hot correct."""
    fe = FeatureExtractor()
    feat_buy = fe.featurize(_decision_dict(direction="BUY"))
    feat_sell = fe.featurize(_decision_dict(direction="SELL"))
    assert feat_buy["dir_BUY"] == 1.0
    assert feat_buy["dir_SELL"] == 0.0
    assert feat_sell["dir_SELL"] == 1.0
    assert feat_sell["dir_BUY"] == 0.0


def test_tier_one_hot_correct():
    """Test tier one hot correct."""
    fe = FeatureExtractor()
    feat = fe.featurize(_decision_dict(tier="1"))
    assert feat["tier_1"] == 1.0
    assert feat["tier_2"] == 0.0
    assert feat["tier_3"] == 0.0


def test_known_detector_set():
    """A detector in DETECTOR_TOP30 should set its column; unknown → det_OTHER."""
    fe = FeatureExtractor()
    assert "double_top" in DETECTOR_TOP30
    feat = fe.featurize(_decision_dict(detectors=["double_top"]))
    assert feat["det_double_top"] == 1.0
    assert feat["det_OTHER"] == 0.0


def test_unknown_detector_falls_into_other():
    """Test unknown detector falls into other."""
    fe = FeatureExtractor()
    feat = fe.featurize(_decision_dict(detectors=["weird_pattern_xyz"]))
    assert feat["det_OTHER"] == 1.0
    assert feat["det_double_top"] == 0.0


def test_missing_fields_default_to_zero():
    """Missing decision fields and missing broker_state values → 0."""
    fe = FeatureExtractor()
    feat = fe.featurize({"ticker": "SBER", "signals": []})
    assert feat["combined_magnitude"] == 0.0
    assert feat["expected_rr"] == 0.0
    assert feat["confluence_count"] == 0.0
    assert feat["atr_at_entry"] == 0.0


def test_unknown_ticker_keeps_zero_one_hots():
    """Test unknown ticker keeps zero one hots."""
    fe = FeatureExtractor()
    feat = fe.featurize(_decision_dict(ticker="UNKNOWN"))
    for col in FEATURE_COLUMNS_V2:
        if col.startswith("tkr_"):
            assert feat[col] == 0.0


def test_regime_one_hot_from_broker_state():
    """Test regime one hot from broker state."""
    fe = FeatureExtractor()
    feat = fe.featurize(
        _decision_dict(),
        broker_state={"regime": "crisis"},
    )
    assert feat["regime_crisis"] == 1.0
    assert feat["regime_trending"] == 0.0
    assert feat["regime_unknown"] == 0.0


def test_risk_regime_defaults_to_normal():
    """Test risk regime defaults to normal."""
    fe = FeatureExtractor()
    feat = fe.featurize(_decision_dict())
    assert feat["risk_NORMAL"] == 1.0


def test_time_features_msk_conversion():
    """ts_at_entry in UTC at 07:00 should produce hour=10 MSK."""
    fe = FeatureExtractor()
    ts = datetime(2026, 5, 26, 7, 15, tzinfo=UTC)
    feat = fe.featurize(_decision_dict(), ts_at_entry=ts)
    assert feat["hour_of_day"] == 10.0
    assert feat["minute_of_hour"] == 15.0
    assert feat["is_opening"] == 1.0


def test_evening_session_flag():
    """Test evening session flag."""
    fe = FeatureExtractor()
    ts = datetime(2026, 5, 26, 18, 30, tzinfo=UTC)
    feat = fe.featurize(_decision_dict(), ts_at_entry=ts)
    assert feat["is_evening_session"] == 1.0
    assert feat["hour_of_day"] == 21.0


def test_confluence_count_matches_n_unique_sources():
    """Test confluence count matches n unique sources."""
    fe = FeatureExtractor()
    feat = fe.featurize(
        _decision_dict(
            sources=["TA", "NEWS", "ANOMALY"],
            detectors=["a", "b", "c"],
        )
    )
    assert feat["confluence_count"] == 3.0


def test_nan_inputs_are_clamped():
    """NaN PNL / metadata should never propagate into features."""
    fe = FeatureExtractor()
    decision = _decision_dict()
    decision["combined_magnitude"] = float("nan")
    feat = fe.featurize(decision)
    assert feat["combined_magnitude"] == 0.0


def test_latency_under_10ms_per_call():
    """100 featurize() calls must complete in < 1 s."""
    fe = FeatureExtractor()
    d = _decision_dict(
        sources=["TA", "NEWS"],
        detectors=["double_top", "bull_flag"],
    )
    bs = {"ofi": 0.3, "vpin": 0.2, "regime": "trending", "current_dd_pct": 0.01}
    for _ in range(3):
        fe.featurize(d, bs)
    t0 = time.perf_counter()
    for _ in range(100):
        fe.featurize(d, bs)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"100 featurize() calls took {elapsed:.3f}s (> 1s budget)"
    assert fe.stats.avg_latency_us() < 10_000


def test_featurize_batch():
    """Test featurize batch."""
    fe = FeatureExtractor()
    feats = fe.featurize_batch(
        [
            _decision_dict(ticker="SBER"),
            _decision_dict(ticker="GAZP"),
        ]
    )
    assert len(feats) == 2
    assert feats[0]["tkr_SBER"] == 1.0
    assert feats[1]["tkr_GAZP"] == 1.0


def test_featurize_batch_length_mismatch_raises():
    """Test featurize batch length mismatch raises."""
    fe = FeatureExtractor()
    with pytest.raises(ValueError):
        fe.featurize_batch(
            [_decision_dict()],
            broker_states=[{}, {}],
        )


def test_resilient_to_garbage_decision():
    """Even with a fully garbage input, featurize() must not raise.

    Note: the function returns a dict with safe defaults (e.g. risk_NORMAL=1,
    regime_unknown=1, hour_of_day=current_hour_msk) — we do NOT assert all
    zeros, only that the call completes and yields the full column set.
    """
    fe = FeatureExtractor()
    feat = fe.featurize(None)
    assert isinstance(feat, dict)
    assert len(feat) == len(FEATURE_COLUMNS_V2)
    for v in feat.values():
        assert isinstance(v, float)
        assert v == v
        assert not (v == float("inf") or v == float("-inf"))


def test_pydantic_decision_works():
    """Smoke test using a real pydantic Decision object."""
    from app.dispatcher.signal import (
        Decision,
        DecisionAction,
        DecisionTier,
        Direction,
        RiskCheckResult,
        SignalSource,
        UnifiedSignal,
    )

    sig = UnifiedSignal(
        source=SignalSource.TA,
        detector="double_top",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=0.6,
        raw_confidence=0.6,
        horizon_min=60,
        price=300.0,
        expected_rr=2.0,
        atr=3.0,
    )
    dec = Decision(
        decision_id="abc",
        cycle_id="cyc",
        ticker="SBER",
        action=DecisionAction.EXECUTE,
        tier=DecisionTier.TIER1,
        direction=Direction.BUY,
        combined_magnitude=0.6,
        signals=[sig],
        risk_check=RiskCheckResult.PASSED,
        expected_rr=2.0,
    )
    fe = FeatureExtractor()
    feat = fe.featurize(dec)
    assert feat["tkr_SBER"] == 1.0
    assert feat["src_TA"] == 1.0
    assert feat["dir_BUY"] == 1.0
    assert feat["tier_1"] == 1.0
    assert feat["det_double_top"] == 1.0
    assert feat["combined_magnitude"] == 0.6
