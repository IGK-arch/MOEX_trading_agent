"""Phase 27.9 — MetaPerSessionWrapper tests."""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Any

import pytest

from app.agents.meta_classifier import MetaClassifier, MetaContext
from app.agents.meta_per_session import (
    MetaPerSessionWrapper,
    SessionModelEntry,
)
from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    DecisionTier,
    Direction,
    SignalSource,
    UnifiedSignal,
)


def _build_decision(mag: float = 0.5, rr: float = 2.0) -> Decision:
    """Build a stub Decision for the meta scorer."""
    sig = UnifiedSignal(
        source=SignalSource.TA,
        detector="test",
        ticker="SBER",
        direction=Direction.BUY,
        magnitude=mag,
        raw_confidence=mag,
        horizon_min=30,
        price=300.0,
        expected_rr=rr,
    )
    return Decision(
        decision_id="d1",
        cycle_id="c1",
        ticker="SBER",
        action=DecisionAction.EXECUTE,
        tier=DecisionTier.TIER2,
        direction=Direction.BUY,
        combined_magnitude=mag,
        expected_rr=rr,
        signals=[sig],
    )


def _ctx() -> MetaContext:
    """Ctx."""
    return MetaContext(
        ofi=0.1,
        vpin=0.3,
        vol_z=0.5,
        spread_bbo_bps=2.0,
        regime="trending",
        hour_of_day=13,
        minutes_to_close=240,
    )


def test_wrapper_falls_back_to_base_when_no_session_model(tmp_path: Path) -> None:
    """No per-session files on disk → wrapper still scores via base."""
    base = MetaClassifier(model_path=tmp_path / "missing.cbm")
    w = MetaPerSessionWrapper(base=base, models_dir=tmp_path)
    assert w.has_session_model("midday") is False
    score = w.score(_build_decision(), _ctx(), session_label="midday")
    assert 0.0 <= score <= 1.0


def test_wrapper_with_no_session_label_uses_base(tmp_path: Path) -> None:
    """Test wrapper with no session label uses base."""
    base = MetaClassifier(model_path=tmp_path / "missing.cbm")
    w = MetaPerSessionWrapper(base=base, models_dir=tmp_path)
    score = w.score(_build_decision(), _ctx(), session_label=None)
    assert 0.0 <= score <= 1.0


def test_wrapper_session_model_routes(tmp_path: Path) -> None:
    """If a session model is registered, calls go through that model."""
    base = MetaClassifier(model_path=tmp_path / "missing.cbm")
    w = MetaPerSessionWrapper(base=base, models_dir=tmp_path)

    class _FakeModel:
        """Fake Model."""

        feature_names_ = list(MetaClassifier.build_features(_build_decision(), _ctx()).keys())

        def predict_proba(self, X: Any) -> Any:
            """Predict proba."""
            n = len(X) if hasattr(X, "__len__") else 1
            return [[0.3, 0.7] for _ in range(n)]

    fake = _FakeModel()
    w.session_models["midday"] = SessionModelEntry(
        label="midday",
        path=tmp_path / "meta_session_midday.cbm",
        model=fake,
        feature_names=fake.feature_names_,
    )

    assert w.has_session_model("midday") is True
    score = w.score(_build_decision(), _ctx(), session_label="midday")
    assert score == pytest.approx(0.7, abs=1e-6)


def test_score_batch_dispatch(tmp_path: Path) -> None:
    """Different session labels in a batch route to different models."""
    base = MetaClassifier(model_path=tmp_path / "missing.cbm")
    w = MetaPerSessionWrapper(base=base, models_dir=tmp_path)

    class _FakeModel:
        """Fake Model."""

        feature_names_ = list(MetaClassifier.build_features(_build_decision(), _ctx()).keys())

        def __init__(self, p: float) -> None:
            """Init."""
            self.p = p

        def predict_proba(self, X: Any) -> Any:
            """Predict proba."""
            n = len(X) if hasattr(X, "__len__") else 1
            return [[1 - self.p, self.p] for _ in range(n)]

    w.session_models["midday"] = SessionModelEntry(
        label="midday",
        path=tmp_path / "meta_session_midday.cbm",
        model=_FakeModel(0.8),
        feature_names=_FakeModel.feature_names_,
    )

    decisions = [_build_decision() for _ in range(4)]
    contexts = [_ctx() for _ in range(4)]
    labels = ["midday", None, "midday", "morning"]
    scores = w.score_batch(decisions, contexts, labels)
    assert len(scores) == 4
    assert scores[0] == pytest.approx(0.8, abs=1e-6)
    assert scores[2] == pytest.approx(0.8, abs=1e-6)
    assert 0.0 <= scores[1] <= 1.0
    assert 0.0 <= scores[3] <= 1.0


def test_score_batch_empty(tmp_path: Path) -> None:
    """Test score batch empty."""
    base = MetaClassifier(model_path=tmp_path / "missing.cbm")
    w = MetaPerSessionWrapper(base=base, models_dir=tmp_path)
    assert w.score_batch([], []) == []


def test_score_batch_mismatched_lengths(tmp_path: Path) -> None:
    """Test score batch mismatched lengths."""
    base = MetaClassifier(model_path=tmp_path / "missing.cbm")
    w = MetaPerSessionWrapper(base=base, models_dir=tmp_path)
    with pytest.raises(ValueError):
        w.score_batch([_build_decision()], [])


def test_speed_100_calls_under_1_5_sec(tmp_path: Path) -> None:
    """Calling score() 100 times should complete well under 1.5 seconds."""
    base = MetaClassifier(model_path=tmp_path / "missing.cbm")
    w = MetaPerSessionWrapper(base=base, models_dir=tmp_path)
    d = _build_decision()
    c = _ctx()
    t0 = _time.monotonic()
    for _ in range(100):
        w.score(d, c, session_label="midday")
    elapsed = _time.monotonic() - t0
    assert elapsed < 1.5, f"too slow: {elapsed:.3f}s"
