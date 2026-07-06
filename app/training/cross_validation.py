"""
app/training/cross_validation.py — Purged k-fold cross-validation.

Reference: Marcos Lopez de Prado, "Advances in Financial Machine Learning", Ch.7.

Problem in financial ML CV
--------------------------
Standard k-fold randomly splits samples between train/test, but in time-series
finance, this creates **lookahead bias**:

  - A training event at t0 may have its triple-barrier exit at t1 > t_test_start.
    The model implicitly "sees" the test period through the still-open trade.
  - Even after purging, neighbouring train samples right before the test interval
    have correlated noise — embargo removes the next K samples after the test
    boundary on each side.

PurgedKFold
-----------
Given:
  - `samples_t0`:  start time / start bar of each sample
  - `samples_t1`:  exit barrier time / exit bar of each sample (from labeling.py)
  - `n_splits`:    e.g. 5 contiguous time-ordered folds
  - `embargo_pct`: fraction of total samples to embargo on each side of test set

For each fold:
  1. Test = contiguous block of samples
  2. Train = all samples NOT in test, MINUS:
     a) any sample whose [t0, t1] overlaps with [test_t0_min, test_t1_max]  (purge)
     b) any sample within `embargo_n` immediately after the test block (embargo)
  3. Yield (train_idx, test_idx)

This implementation is pure numpy/pandas, no mlfinpy dependency.
"""

from __future__ import annotations

from collections.abc import Iterator

try:
    import numpy as np
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError("purged k-fold requires numpy + pandas") from e

class PurgedKFold:
    """
    Time-series-aware k-fold CV with purging and embargo.

    Parameters
    ----------
    n_splits : int
        Number of folds (default 5).
    embargo_pct : float
        Embargo size as fraction of total samples (default 0.01 = 1%).
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        """Init."""
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0.0 <= embargo_pct < 0.5:
            raise ValueError("embargo_pct must be in [0, 0.5)")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        n_samples: int,
        samples_t1: np.ndarray | pd.Series,
        samples_t0: np.ndarray | pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_idx, test_idx) tuples for each fold.

        Parameters
        ----------
        n_samples : int
            Total number of samples.
        samples_t1 : array-like[int]
            Exit bar index for each sample (from triple_barrier labeling).
            Used to determine which train samples overlap with test interval.
        samples_t0 : array-like[int] | None
            Start bar index for each sample. If None, assumes
            samples are in time order and t0 == sample index.

        Yields
        ------
        (train_idx, test_idx) : (ndarray[int], ndarray[int])
        """
        if n_samples < self.n_splits:
            raise ValueError(f"n_samples ({n_samples}) < n_splits ({self.n_splits})")

        t1 = np.asarray(samples_t1, dtype=np.int64)
        if samples_t0 is None:
            t0 = np.arange(n_samples, dtype=np.int64)
        else:
            t0 = np.asarray(samples_t0, dtype=np.int64)

        if len(t1) != n_samples or len(t0) != n_samples:
            raise ValueError("t0 / t1 length must equal n_samples")

        embargo_n = max(1, int(round(n_samples * self.embargo_pct)))

        fold_size = n_samples // self.n_splits
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            test_idx = np.arange(test_start, test_end, dtype=np.int64)

            test_t0_min = int(t0[test_idx].min()) if len(test_idx) else 0
            test_t1_max = int(t1[test_idx].max()) if len(test_idx) else 0

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False

            overlap = (t1 >= test_t0_min) & (t0 <= test_t1_max)
            train_mask[overlap] = False

            embargo_start = test_end
            embargo_end = min(n_samples, test_end + embargo_n)
            if embargo_end > embargo_start:
                train_mask[embargo_start:embargo_end] = False

            train_idx = np.where(train_mask)[0]

            if len(train_idx) == 0:
                continue
            yield train_idx, test_idx

    def __repr__(self) -> str:
        """Repr."""
        return f"PurgedKFold(n_splits={self.n_splits}, embargo_pct={self.embargo_pct})"

def time_train_test_split(
    n_samples: int,
    test_size: float = 0.2,
    samples_t1: np.ndarray | None = None,
    embargo_pct: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience: time-ordered single train/test split with embargo + purge.

    Returns (train_idx, test_idx).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    test_start = int(n_samples * (1 - test_size))
    test_idx = np.arange(test_start, n_samples, dtype=np.int64)
    train_idx = np.arange(0, test_start, dtype=np.int64)

    embargo_n = max(0, int(round(n_samples * embargo_pct)))
    if embargo_n > 0 and len(train_idx) > embargo_n:
        train_idx = train_idx[:-embargo_n]

    if samples_t1 is not None:
        t1 = np.asarray(samples_t1, dtype=np.int64)
        if len(t1) == n_samples:
            test_t0_min = test_start
            keep = t1[train_idx] < test_t0_min
            train_idx = train_idx[keep]

    return train_idx, test_idx

__all__ = ["PurgedKFold", "time_train_test_split"]
