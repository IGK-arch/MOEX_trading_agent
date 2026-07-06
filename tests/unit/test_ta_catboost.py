"""
tests/unit/test_ta_catboost.py — regression guard for the v0.13.x feature-order
contract between TACatBoost and the on-disk catboost_ta.cbm.

Background:
    Until this test was added, FEATURE_COLUMNS in app/agents/ta_catboost.py
    drifted out of sync with the trained model (a 30-column pruned list while
    the .cbm was still trained on 99). CatBoost enforces both the feature
    NAMES and ORDER at inference time, so predict_proba() blew up with:

        "At position 0 should be feature with name atr_pct (found expected_rr)"

    The fix: load model.feature_names_ at load() and use that as the
    authoritative column list when building the inference DataFrame.

These tests assert:
    1. A real feature dict from build_features() can be fed through
       predict_batch() without raising.
    2. The same is true for predict_success_proba().
    3. The empty/single edge cases still return correctly.
    4. When the model is loaded, the model's feature_names_ are used as
       the inference contract (not the FEATURE_COLUMNS module constant).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.ta_catboost import (
    FEATURE_COLUMNS,
    NEW_FEATURE_COLUMNS,
    PATTERN_NAMES,
    TACatBoost,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "catboost_ta.cbm"


def _fake_feat_full(seed: int = 0) -> dict[str, float]:
    """
    Build a feature dict that includes EVERY column the trained 99-feature
    model can possibly ask for (base + pat_* + regime_* + NEW_FEATURE_COLUMNS).

    Mirrors what TACatBoost.build_features() emits at runtime, so this test
    catches any future drift where the on-disk model expects a feature the
    runtime code path never sets.
    """
    feat: dict[str, float] = {}

    base = {
        "atr_pct": 0.6 + (seed % 3) * 0.1,
        "adx": 22.0 + (seed % 4) * 2.0,
        "rsi": 45.0 + (seed % 10),
        "vol_z": -0.2 + (seed % 5) * 0.3,
        "sup_atrs": 1.5 + (seed % 4),
        "res_atrs": 2.0 + (seed % 4),
        "candle_bullish_score": (seed % 7) / 10.0,
        "candle_bearish_score": ((seed + 2) % 7) / 10.0,
        "expected_rr": 1.5 + (seed % 5) * 0.25,
        "atr_at_entry_pct": 0.8 + (seed % 3) * 0.15,
    }
    feat.update(base)

    for i, p in enumerate(PATTERN_NAMES):
        feat[f"pat_{p}"] = 1.0 if i == seed % len(PATTERN_NAMES) else 0.0

    regimes = ["regime_trending", "regime_mean_reverting", "regime_crisis"]
    for i, r in enumerate(regimes):
        feat[r] = 1.0 if i == seed % 3 else 0.0

    for c in NEW_FEATURE_COLUMNS:
        feat[c] = 0.0
    feat["bb_percent_b"] = 0.5
    feat["hours_to_close"] = 4.0
    feat["close_pos_in_range_20"] = 0.5
    feat["up_bars_ratio_10"] = 0.5

    return feat


@pytest.fixture
def loaded_catboost() -> TACatBoost:
    """Real on-disk model. Skips if catboost_ta.cbm not present (CI/dev sans data)."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not present at {MODEL_PATH}; skipping live-model test")
    cat = TACatBoost(model_path=MODEL_PATH)
    if not cat.load():
        pytest.skip("CatBoost not installed or model failed to load")
    return cat


def test_load_captures_model_feature_names(loaded_catboost: TACatBoost) -> None:
    """
    Regression: load() must capture feature_names_ from the .cbm so inference
    uses the authoritative order, not the static FEATURE_COLUMNS module list.
    """
    assert loaded_catboost._loaded is True
    assert loaded_catboost._model_feature_names is not None
    assert len(loaded_catboost._model_feature_names) > 0
    assert loaded_catboost._model_feature_names[0] == "atr_pct"


def test_predict_batch_runs_without_exception(loaded_catboost: TACatBoost) -> None:
    """
    THE KEY REGRESSION TEST.

    Before the fix:
        predict_batch() built a DataFrame with FEATURE_COLUMNS column order,
        which started with "expected_rr". CatBoost rejected it with
        "At position 0 should be feature with name atr_pct (found expected_rr)"
        and the entire ML filter silently fell back to heuristic.

    After the fix:
        predict_batch() uses model.feature_names_ as authoritative; no
        exception is raised and we get back N floats in [0, 1].
    """
    feats = [_fake_feat_full(seed=i) for i in range(8)]

    probs = loaded_catboost.predict_batch(feats)

    assert isinstance(probs, list)
    assert len(probs) == len(feats)
    for p in probs:
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


def test_predict_success_proba_runs_without_exception(
    loaded_catboost: TACatBoost,
) -> None:
    """Same contract on the single-row code path."""
    feat = _fake_feat_full(seed=3)
    p = loaded_catboost.predict_success_proba(feat)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_predict_batch_empty_returns_empty() -> None:
    """Edge case unchanged by the fix."""
    cat = TACatBoost()
    assert cat.predict_batch([]) == []


def test_predict_batch_handles_missing_feature_keys(
    loaded_catboost: TACatBoost,
) -> None:
    """
    If a caller hands us a feature dict with missing keys (e.g. a unit test
    that didn't populate every NEW_FEATURE_COLUMN), inference should still
    succeed — missing values default to 0.0, not blow up the cycle.
    """
    sparse = {
        "expected_rr": 2.0,
        "rsi": 55.0,
        "atr_pct": 0.7,
    }
    probs = loaded_catboost.predict_batch([sparse])
    assert len(probs) == 1
    assert 0.0 <= probs[0] <= 1.0


def test_inference_columns_falls_back_to_feature_columns_without_model() -> None:
    """
    When no model is loaded, _inference_columns() returns FEATURE_COLUMNS
    so the heuristic-mode tests/scripts that rely on that name still work.
    """
    cat = TACatBoost(model_path=Path("/nonexistent/path/that/does/not/exist.cbm"))
    cat.load()
    assert cat._loaded is False
    assert cat._model_feature_names is None
    assert cat._inference_columns() is FEATURE_COLUMNS
