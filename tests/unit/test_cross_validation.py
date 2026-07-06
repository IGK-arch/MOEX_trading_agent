"""
tests/unit/test_cross_validation.py — PurgedKFold and time_train_test_split correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.training.cross_validation import PurgedKFold, time_train_test_split


def test_purged_kfold_basic_split():
    """Test purged kfold basic split."""

    n = 100
    t1 = np.arange(n) + 1
    cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
    folds = list(cv.split(n_samples=n, samples_t1=t1))
    assert len(folds) == 5

    for train_idx, test_idx in folds:
        assert 18 <= len(test_idx) <= 22

        assert len(np.intersect1d(train_idx, test_idx)) == 0

        assert len(np.unique(train_idx)) == len(train_idx)
        assert len(np.unique(test_idx)) == len(test_idx)


def test_purged_kfold_embargo_removes_neighbors():
    """Test purged kfold embargo removes neighbors."""

    n = 100
    t1 = np.arange(n) + 1
    cv = PurgedKFold(n_splits=5, embargo_pct=0.05)
    folds = list(cv.split(n_samples=n, samples_t1=t1))
    assert len(folds) == 5

    train_idx, test_idx = folds[0]
    assert 0 not in train_idx
    assert 19 in test_idx

    for s in [20, 21, 22, 23, 24]:
        assert s not in train_idx, f"sample {s} should be embargoed"

    assert 25 in train_idx


def test_purged_kfold_purges_overlapping_t1():
    """Test purged kfold purges overlapping t1."""

    n = 100
    t1 = np.arange(n) + 1
    t1[5] = 50
    cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
    folds = list(cv.split(n_samples=n, samples_t1=t1))

    for train_idx, test_idx in folds:
        if 40 in test_idx and 50 in test_idx:
            assert 5 not in train_idx, "Sample with overlapping t1 should be purged"
            break
    else:
        pytest.fail("Expected fold not found")


def test_purged_kfold_t0_t1_overlap_detection():
    """Test purged kfold t0 t1 overlap detection."""

    n = 3
    t0 = np.array([0, 10, 20])
    t1 = np.array([5, 15, 25])
    cv = PurgedKFold(n_splits=3, embargo_pct=0.0)
    folds = list(cv.split(n_samples=n, samples_t1=t1, samples_t0=t0))

    train_idx, test_idx = folds[1]
    assert 1 in test_idx
    assert 0 in train_idx or 2 in train_idx


def test_purged_kfold_invalid_args():
    """Test purged kfold invalid args."""
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=5, embargo_pct=0.6)
    cv = PurgedKFold(n_splits=5)
    with pytest.raises(ValueError):
        list(cv.split(n_samples=3, samples_t1=np.array([1, 2, 3])))


def test_time_train_test_split_basic():
    """Test time train test split basic."""

    train, test = time_train_test_split(n_samples=100, test_size=0.2, embargo_pct=0.0)
    assert len(test) == 20
    assert len(train) == 80
    assert max(train) < min(test)


def test_time_train_test_split_with_embargo():
    """Test time train test split with embargo."""
    train, test = time_train_test_split(
        n_samples=100,
        test_size=0.2,
        embargo_pct=0.05,
    )

    assert max(train) < 75
    assert min(test) == 80


def test_time_train_test_split_purges_overlapping_t1():
    """Test time train test split purges overlapping t1."""

    n = 100
    t1 = np.arange(n) + 1
    t1[70] = 90
    train, test = time_train_test_split(
        n_samples=n,
        test_size=0.2,
        samples_t1=t1,
        embargo_pct=0.0,
    )
    assert 70 not in train
