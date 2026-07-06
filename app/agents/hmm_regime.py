"""HMM-детектор режима рынка."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore

    _HAS_HMM = True
except ImportError:
    _HAS_HMM = False

REGIME_LABELS = ["trending", "mean_reverting", "crisis"]

class HMMRegimeDetector:
    """3-state Gaussian HMM regime detector for IMOEX."""

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 200,
        random_state: int = 42,
        model_path: Path | None = None,
    ) -> None:
        """Init."""
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: Any = None
        self.state_to_label: dict[int, str] = {}
        self._current_label: str = "unknown"
        self.model_path = model_path or (cfg.DATA_DIR / "models" / "hmm.pkl")

        self._feature_mean: Any = None
        self._feature_std: Any = None

    @staticmethod
    def build_features(df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix from OHLCV daily candles.

        Returns:
            np.ndarray: shape (n_samples, 3) with [log_return, abs_return, roll_std_5d].
        """
        if not _HAS_PANDAS or df is None or df.empty:
            return np.zeros((0, 3))

        close = df["close"].astype(float)
        log_ret = np.log(close / close.shift(1))
        abs_ret = log_ret.abs()
        roll_std = log_ret.rolling(5).std()

        feat = pd.DataFrame(
            {
                "log_ret": log_ret,
                "abs_ret": abs_ret,
                "roll_std": roll_std,
            }
        ).dropna()

        return feat.values

    async def fit(self, df: pd.DataFrame) -> bool:
        """Train HMM on daily candles. Returns True on success."""
        if not _HAS_HMM:
            logger.error("hmmlearn not installed — HMM regime detection unavailable")
            return False
        if not _HAS_PANDAS or df is None or len(df) < 30:
            logger.warning(
                "HMM: insufficient data, skipping fit",
                extra={"rows": len(df) if df is not None else 0},
            )
            return False

        X = self.build_features(df)
        if len(X) < 20:
            logger.warning("HMM: too few feature rows", extra={"rows": len(X)})
            return False

        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds = np.where(stds < 1e-9, 1.0, stds)
        X_norm = (X - means) / stds

        model = None

        for cov_type in ("full", "diag", "spherical"):
            try:
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=cov_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state,
                    tol=1e-4,
                    init_params="stmc",
                )
                model.fit(X_norm)
                logger.info("HMM fitted", extra={"cov_type": cov_type})
                break
            except Exception as exc:
                logger.warning(f"HMM fit with cov={cov_type} failed", extra={"error": str(exc)})
                model = None
                continue

        if model is None:
            logger.error("HMM fit failed with all covariance types")
            return False

        self._feature_mean = means
        self._feature_std = stds

        self.model = model

        means_abs_ret = model.means_[:, 1]
        sorted_states = np.argsort(means_abs_ret)
        self.state_to_label = {
            int(sorted_states[0]): REGIME_LABELS[0],
            int(sorted_states[1]): REGIME_LABELS[1],
            int(sorted_states[2]): REGIME_LABELS[2],
        }

        logger.info(
            "HMM fitted",
            extra={
                "states": self.n_states,
                "samples": len(X),
                "state_means": means_abs_ret.tolist(),
                "label_map": self.state_to_label,
            },
        )

        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(
                    {
                        "model": model,
                        "state_to_label": self.state_to_label,
                        "feature_mean": self._feature_mean,
                        "feature_std": self._feature_std,
                    },
                    f,
                )
            logger.info("HMM saved", extra={"path": str(self.model_path)})
        except Exception as exc:
            logger.warning("HMM save failed", extra={"error": str(exc)})

        return True

    def load(self) -> bool:
        """Load model from disk. Returns True if successful."""
        if not self.model_path.exists():
            return False
        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.state_to_label = data["state_to_label"]
            self._feature_mean = data.get("feature_mean")
            self._feature_std = data.get("feature_std")
            logger.info("HMM loaded", extra={"path": str(self.model_path)})
            return True
        except Exception as exc:
            logger.error("HMM load failed", extra={"error": str(exc)})
            return False

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        """Apply the same normalisation used at fit time."""
        if self._feature_mean is None or self._feature_std is None:
            return X
        return (X - self._feature_mean) / self._feature_std

    def predict_state(self, df: pd.DataFrame) -> str:
        """Predict the current regime label from the last observation in df.

        Returns:
            str: one of 'trending', 'mean_reverting', 'crisis', 'unknown'.
        """
        if self.model is None or not _HAS_HMM:
            return "unknown"

        X = self.build_features(df)
        if len(X) == 0:
            return "unknown"

        try:
            X_norm = self._normalise(X)
            states = self.model.predict(X_norm)
            last_state = int(states[-1])
            label = self.state_to_label.get(last_state, "unknown")
            prev = self._current_label
            if label != prev:
                from app.utils.logging import get_trace_id

                logger.info(
                    "HMM regime changed",
                    extra={
                        "prev_regime": prev,
                        "regime": label,
                        "trace_id": get_trace_id(),
                    },
                )
            self._current_label = label
            return label
        except Exception as exc:
            logger.error("HMM predict failed", extra={"error": str(exc)})
            return "unknown"

    def predict_proba_last(self, df: pd.DataFrame) -> dict[str, float]:
        """Return {label: probability} for the latest observation."""
        if self.model is None:
            return dict.fromkeys(REGIME_LABELS, 0.0)

        X = self.build_features(df)
        if len(X) == 0:
            return dict.fromkeys(REGIME_LABELS, 0.0)

        try:
            X_norm = self._normalise(X)
            proba = self.model.predict_proba(X_norm)
            last_proba = proba[-1]
            return {
                self.state_to_label.get(s, f"state_{s}"): float(last_proba[s])
                for s in range(self.n_states)
            }
        except Exception as exc:
            logger.error("HMM predict_proba failed", extra={"error": str(exc)})
            return dict.fromkeys(REGIME_LABELS, 0.0)

    @property
    def current_label(self) -> str:
        """Current label."""
        return self._current_label

    def regime_size_multiplier(self) -> float:
        """Multiplier applied to position sizing based on regime."""
        return {
            "crisis": 0.7,
            "mean_reverting": 0.85,
            "trending": 1.0,
        }.get(self._current_label, 0.7)

    def regime_signal_filter(self, pattern: str, direction: str) -> float:
        """Multiplier for pattern signal magnitude based on current regime."""
        if self._current_label == "crisis":
            return 0.5

        reversal_patterns = {
            "double_top",
            "double_bottom",
            "triple_top",
            "triple_bottom",
            "head_shoulders",
            "inv_head_shoulders",
            "rising_wedge",
            "falling_wedge",
            "megaphone_buy",
            "megaphone_sell",
            "rounding_top",
            "rounding_bottom",
        }
        continuation_patterns = {
            "bull_flag",
            "bear_flag",
            "bull_pennant",
            "bear_pennant",
            "ascending_triangle",
            "descending_triangle",
            "symmetric_triangle",
            "rectangle_breakout_up",
            "rectangle_breakdown",
            "compression_breakout_up",
            "compression_breakout_down",
        }

        if self._current_label == "trending":
            if pattern in continuation_patterns:
                return 1.2
            if pattern in reversal_patterns:
                return 0.7

        if self._current_label == "mean_reverting":
            if pattern in reversal_patterns:
                return 1.2
            if pattern in continuation_patterns:
                return 0.7

        return 1.0

_hmm_detector: HMMRegimeDetector | None = None

def get_hmm_detector() -> HMMRegimeDetector:
    """Get hmm detector."""
    global _hmm_detector
    if _hmm_detector is None:
        _hmm_detector = HMMRegimeDetector()
    return _hmm_detector
