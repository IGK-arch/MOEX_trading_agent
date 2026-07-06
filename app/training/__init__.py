"""
app/training — Training-time utilities (NOT loaded at inference).

Pure numpy/pandas implementations of:
  - Triple-barrier event labeling  (Lopez de Prado, Advances in Financial ML, Ch.3)
  - Purged k-fold cross-validation (same book, Ch.7)
  - Meta-feature extraction        (used by both train_meta.py and meta_classifier.py)

No external `mlfinpy` dependency — keeps the runtime image small.
"""
