"""
app/training/feature_extractor.py — Feature extraction for meta-classifier v2.

Phase 27.8 (meta_v2). Builds a 50+ feature vector per decision suitable for
training a CatBoost classifier (4-class: big_win / small_win / small_loss /
big_loss) or a regression model that predicts pnl_pct directly.

Design goals
------------
1. Self-contained — no live dependency on broker / market data manager.
2. Tolerant — every "missing field" yields 0.0 (no NaN propagation).
3. Fast — full `featurize()` call must be < 10ms so the v2 inference path
   in `MetaClassifier.score()` is well under the 15ms budget (decision cycle
   is 30 s, so even 100 candidate decisions × 15ms = 1.5s is fine).
4. Deterministic feature order: `FEATURE_COLUMNS_V2` is the canonical
   list used everywhere (training, inference, online retrain).

Feature groups (~85 columns total before model-side feature selection):
  - Ticker one-hot                (20 cols)  — see cfg.TICKERS
  - Source strategy one-hot        (5 cols)  — TA / NEWS / ANOMALY / PAIR / MEAN_REV
  - Detector one-hot               (30 cols) — top-30 most-used detectors
  - Tier one-hot                    (6 cols) — S / A / B / 1 / 2 / 3
  - Direction one-hot               (3 cols) — BUY / SELL / NEUTRAL
  - Numeric trade context          (~15 cols) — magnitude, expected_rr,
                                                   atr_pct, vol_z, ofi, vpin,
                                                   kyle_lambda, dd, daily_pnl,
                                                   n_open, n_trades_today,
                                                   meta_v1_score, confluence,
                                                   cash_util
  - Regime one-hot                  (4 cols) — trending / mean_reverting /
                                                   crisis / unknown
  - RiskRegime one-hot              (4 cols) — NORMAL / CAUTIOUS /
                                                   DEFENSIVE / CRISIS
  - Time-of-day numeric             (~6 cols) — hour, minute, is_opening,
                                                   is_closing, is_evening,
                                                   day_of_week
  - Direction-bias                  (2 cols) — bias_match, bias_multiplier
  - RAG context                     (3 cols) — consensus_alignment,
                                                   similar_past_wr,
                                                   similar_past_count

Total = 98 columns (the CatBoost training step will rank importance and
the model itself will downweight zeros; we do NOT do hand feature selection
at extraction time — let the model decide).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

try:
    import numpy as np  # type: ignore

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

_TICKER_ONEHOT: list[str] = [f"tkr_{t}" for t in cfg.TICKERS]

_SOURCE_ONEHOT: list[str] = ["src_TA", "src_NEWS", "src_ANOMALY", "src_PAIR", "src_MEAN_REV"]

DETECTOR_TOP30: list[str] = [
    "double_top",
    "double_bottom",
    "head_shoulders",
    "inv_head_shoulders",
    "bull_flag",
    "bear_flag",
    "ascending_triangle",
    "descending_triangle",
    "rising_wedge",
    "falling_wedge",
    "hammer",
    "engulfing_bullish",
    "engulfing_bearish",
    "morning_star",
    "evening_star",
    "shooting_star",
    "smc_order_block",
    "smc_bos",
    "smc_choch",
    "fvg",
    "ofi_spike",
    "vpin_burn",
    "kyle_lambda_spike",
    "volume_z_spike",
    "spread_widening",
    "rsi_oversold",
    "rsi_overbought",
    "bollinger_squeeze",
    "news_event",
    "pair_zscore",
]
_DETECTOR_ONEHOT: list[str] = [f"det_{d}" for d in DETECTOR_TOP30] + ["det_OTHER"]

_TIER_ONEHOT: list[str] = ["tier_S", "tier_A", "tier_B", "tier_1", "tier_2", "tier_3", "tier_NONE"]

_DIRECTION_ONEHOT: list[str] = ["dir_BUY", "dir_SELL", "dir_NEUTRAL"]

_NUMERIC_CONTEXT: list[str] = [
    "magnitude",
    "expected_rr",
    "combined_magnitude",
    "atr_at_entry",
    "atr_pctile_at_entry",
    "volume_z_at_entry",
    "ofi_at_entry",
    "vpin_at_entry",
    "kyle_lambda_at_entry",
    "current_dd_pct_at_entry",
    "daily_pnl_pct_at_entry",
    "meta_v1_score",
    "confluence_count",
    "n_open_positions_at_entry",
    "cash_utilization_pct_at_entry",
]

_REGIME_ONEHOT: list[str] = [
    "regime_trending",
    "regime_mean_reverting",
    "regime_crisis",
    "regime_unknown",
]

_RISK_REGIME_ONEHOT: list[str] = [
    "risk_NORMAL",
    "risk_CAUTIOUS",
    "risk_DEFENSIVE",
    "risk_CRISIS",
]

_TIME_FEATURES: list[str] = [
    "hour_of_day",
    "minute_of_hour",
    "is_opening",
    "is_closing",
    "is_evening_session",
    "day_of_week",
]

_BIAS_FEATURES: list[str] = ["bias_match", "bias_multiplier_applied"]

_RAG_FEATURES: list[str] = [
    "consensus_alignment",
    "similar_past_trades_win_rate",
    "similar_past_trades_count",
]

FEATURE_COLUMNS_V2: list[str] = (
    _TICKER_ONEHOT
    + _SOURCE_ONEHOT
    + _DETECTOR_ONEHOT
    + _TIER_ONEHOT
    + _DIRECTION_ONEHOT
    + _NUMERIC_CONTEXT
    + _REGIME_ONEHOT
    + _RISK_REGIME_ONEHOT
    + _TIME_FEATURES
    + _BIAS_FEATURES
    + _RAG_FEATURES
)

@dataclass
class FeatureExtractorStats:
    """Operational metrics; useful for production monitoring."""

    n_extracted: int = 0
    total_time_us: int = 0
    n_missing_tickers: int = 0
    n_missing_detectors: int = 0
    n_failures: int = 0

    def avg_latency_us(self) -> float:
        """Avg latency us."""
        return (self.total_time_us / self.n_extracted) if self.n_extracted else 0.0

class FeatureExtractor:
    """
    Extract a 90+ feature dict from a Decision + broker_state snapshot.

    Caches lookup dicts at construction so the per-decision hot path is just
    O(features) — no string searches or substring comparisons.
    """

    def __init__(self) -> None:
        """Init."""
        self._ticker_set: set[str] = set(cfg.TICKERS)
        self._detector_set: set[str] = set(DETECTOR_TOP30)
        self._stats = FeatureExtractorStats()
        self._zero_template: dict[str, float] = dict.fromkeys(FEATURE_COLUMNS_V2, 0.0)

    @property
    def stats(self) -> FeatureExtractorStats:
        """Stats."""
        return self._stats

    @property
    def feature_columns(self) -> list[str]:
        """Canonical feature column order."""
        return FEATURE_COLUMNS_V2

    def featurize(
        self,
        decision: Any,
        broker_state: dict[str, Any] | None = None,
        ts_at_entry: datetime | None = None,
    ) -> dict[str, float]:
        """
        Build feature vector for a single decision.

        Args:
            decision: Decision object (pydantic) OR a dict (from decisions.db row).
                      We accept both so the same code path is used at training
                      time (where we reconstruct from DB rows) and at inference
                      (live Decision objects).
            broker_state: dict with portfolio snapshot keys (optional):
                           current_dd_pct, daily_pnl_pct, n_open_positions,
                           cash_utilization_pct, n_trades_today,
                           atr_pctile, regime, risk_regime, meta_v1_score
            ts_at_entry: datetime of decision (UTC). Defaults to now.

        Returns:
            dict[str, float]: one entry per FEATURE_COLUMNS_V2 column.
                              Missing values are 0.0 (never NaN).
        """
        t0 = time.perf_counter()
        broker_state = broker_state or {}
        if ts_at_entry is None:
            ts_at_entry = datetime.now(tz=UTC)

        try:
            return self._featurize_inner(decision, broker_state, ts_at_entry, t0)
        except Exception as exc:  # noqa: BLE001
            self._stats.n_failures += 1
            logger.warning(
                "FeatureExtractor.featurize failed — returning zeros",
                extra={"error": str(exc), "type": type(exc).__name__},
            )
            return dict(self._zero_template)

    def _featurize_inner(
        self,
        decision: Any,
        broker_state: dict[str, Any],
        ts_at_entry: datetime,
        t0: float,
    ) -> dict[str, float]:
        """Featurize inner."""
        feat = dict(self._zero_template)

        ticker = _get(decision, "ticker", "") or ""
        ticker = str(ticker).upper()
        direction = _get(decision, "direction", "")
        direction = _direction_to_str(direction)
        tier = _get(decision, "tier", "")
        tier_str = _tier_to_str(tier)
        magnitude = _safe_float(_get(decision, "combined_magnitude", 0.0))
        expected_rr = _safe_float(_get(decision, "expected_rr", 0.0))
        meta_v1_score = _safe_float(_get(decision, "meta_score", 0.0))
        signals = _get(decision, "signals", None) or []

        if ticker in self._ticker_set:
            feat[f"tkr_{ticker}"] = 1.0
        else:
            self._stats.n_missing_tickers += 1

        sources_seen: set[str] = set()
        for sig in signals:
            src = _get(sig, "source", None)
            src_str = _source_to_str(src)
            if src_str:
                col = f"src_{src_str}"
                if col in feat:
                    feat[col] = 1.0
                    sources_seen.add(src_str)

        for sig in signals:
            det = _get(sig, "detector", "") or ""
            det_str = str(det).strip()
            if not det_str:
                continue
            if det_str in self._detector_set:
                feat[f"det_{det_str}"] = 1.0
            else:
                feat["det_OTHER"] = 1.0
                self._stats.n_missing_detectors += 1

        if tier_str:
            col = f"tier_{tier_str}"
            if col in feat:
                feat[col] = 1.0
            else:
                feat["tier_NONE"] = 1.0

        if direction:
            col = f"dir_{direction}"
            if col in feat:
                feat[col] = 1.0

        feat["magnitude"] = magnitude
        feat["expected_rr"] = min(10.0, expected_rr)
        feat["combined_magnitude"] = magnitude
        feat["meta_v1_score"] = float(min(1.0, max(0.0, meta_v1_score)))
        feat["confluence_count"] = float(len(sources_seen))

        feat["atr_at_entry"] = _safe_float(broker_state.get("atr_at_entry", 0.0))
        feat["atr_pctile_at_entry"] = _safe_float(broker_state.get("atr_pctile", 0.0))
        feat["volume_z_at_entry"] = _safe_float(
            broker_state.get("vol_z", _best_metadata(signals, "vol_z"))
        )
        feat["ofi_at_entry"] = _safe_float(broker_state.get("ofi", _best_metadata(signals, "ofi")))
        feat["vpin_at_entry"] = _safe_float(
            broker_state.get("vpin", _best_metadata(signals, "vpin"))
        )
        feat["kyle_lambda_at_entry"] = _safe_float(
            broker_state.get("kyle_lambda", _best_metadata(signals, "kyles_lambda"))
        )
        feat["current_dd_pct_at_entry"] = _safe_float(broker_state.get("current_dd_pct", 0.0))
        feat["daily_pnl_pct_at_entry"] = _safe_float(broker_state.get("daily_pnl_pct", 0.0))
        feat["n_open_positions_at_entry"] = _safe_float(broker_state.get("n_open_positions", 0))
        feat["cash_utilization_pct_at_entry"] = _safe_float(
            broker_state.get("cash_utilization_pct", 0.0)
        )

        regime = broker_state.get("regime") or _best_metadata_str(signals, "regime") or "unknown"
        regime = str(regime).lower()
        col = f"regime_{regime}"
        if col in feat:
            feat[col] = 1.0
        else:
            feat["regime_unknown"] = 1.0

        risk_regime = (broker_state.get("risk_regime") or "NORMAL").upper()
        col = f"risk_{risk_regime}"
        if col in feat:
            feat[col] = 1.0
        else:
            feat["risk_NORMAL"] = 1.0

        if ts_at_entry.tzinfo is None:
            ts_msk_hour = (ts_at_entry.hour + 3) % 24
            ts_msk_min = ts_at_entry.minute
            dow = ts_at_entry.weekday()
        else:
            ts_utc = ts_at_entry.astimezone(UTC)
            ts_msk_hour = (ts_utc.hour + 3) % 24
            ts_msk_min = ts_utc.minute
            dow = ts_utc.weekday()
        feat["hour_of_day"] = float(ts_msk_hour)
        feat["minute_of_hour"] = float(ts_msk_min)
        feat["is_opening"] = 1.0 if (ts_msk_hour == 10 and ts_msk_min < 30) else 0.0
        feat["is_closing"] = (
            1.0 if (ts_msk_hour == 18 and ts_msk_min >= 30 and ts_msk_min < 50) else 0.0
        )
        feat["is_evening_session"] = (
            1.0
            if ((ts_msk_hour == 19 and ts_msk_min >= 5) or (ts_msk_hour in (20, 21, 22, 23)))
            else 0.0
        )
        feat["day_of_week"] = float(dow)

        bias_table = getattr(cfg, "PER_TICKER_DIRECTION_BIAS", {}) or {}
        if isinstance(bias_table, dict) and ticker in bias_table:
            preferred = str(bias_table[ticker]).upper()
            if direction and direction == preferred:
                feat["bias_match"] = 1.0
                feat["bias_multiplier_applied"] = float(
                    getattr(cfg, "DIRECTION_BIAS_MATCH_MULT", 1.2)
                )
            else:
                feat["bias_match"] = 0.0
                feat["bias_multiplier_applied"] = float(
                    getattr(cfg, "DIRECTION_BIAS_MISMATCH_MULT", 0.7)
                )

        feat["consensus_alignment"] = _safe_float(broker_state.get("consensus_alignment", 0.0))
        feat["similar_past_trades_win_rate"] = _safe_float(
            broker_state.get("similar_past_trades_win_rate", 0.0)
        )
        feat["similar_past_trades_count"] = _safe_float(
            broker_state.get("similar_past_trades_count", 0)
        )

        dt_us = int((time.perf_counter() - t0) * 1_000_000)
        self._stats.n_extracted += 1
        self._stats.total_time_us += dt_us

        return feat

    def featurize_batch(
        self,
        decisions: list[Any],
        broker_states: list[dict[str, Any]] | None = None,
        timestamps: list[datetime] | None = None,
    ) -> list[dict[str, float]]:
        """Convenience: vectorised wrapper."""
        n = len(decisions)
        broker_states = broker_states or [{}] * n
        timestamps = timestamps or [None] * n
        if not (len(broker_states) == n == len(timestamps)):
            raise ValueError("decisions/broker_states/timestamps length mismatch")
        return [
            self.featurize(d, bs, ts)
            for d, bs, ts in zip(decisions, broker_states, timestamps, strict=False)
        ]

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Robust dual attribute/dict access."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _safe_float(x: Any, default: float = 0.0) -> float:
    """Convert to float; never raise; NaN/Inf clamped to default."""
    try:
        v = float(x) if x is not None else default
    except (TypeError, ValueError):
        return default
    if _HAS_NUMPY and (np.isnan(v) or np.isinf(v)):  # type: ignore[name-defined]
        return default
    if v != v:
        return default
    return v

def _direction_to_str(d: Any) -> str:
    """Normalise pydantic Direction enum / str / dict-value to UPPER string."""
    if d is None:
        return ""
    if hasattr(d, "value"):
        return str(d.value).upper()
    return str(d).upper()

def _tier_to_str(t: Any) -> str:
    """Tier to str."""
    if t is None:
        return ""
    if hasattr(t, "value"):
        return str(t.value).upper()
    return str(t).upper()

def _source_to_str(s: Any) -> str:
    """Source to str."""
    if s is None:
        return ""
    if hasattr(s, "value"):
        return str(s.value).upper()
    return str(s).upper()

def _best_metadata(signals: list[Any], field: str, default: float = 0.0) -> float:
    """Return the strongest numeric value of `field` across signal metadata.

    Used to reconstruct microstructure context when broker_state doesn't carry
    it explicitly (which is the common training-time case — we only have the
    signals_json that was saved at decision time).
    """
    best = default
    for s in signals:
        md = _get(s, "metadata", None) or {}
        if not isinstance(md, dict):
            continue
        v = md.get(field)
        if isinstance(v, (int, float)) and v == v:
            best = float(v)
    return best

def _best_metadata_str(signals: list[Any], field: str) -> str:
    """Same as _best_metadata but for string fields (e.g. regime)."""
    for s in signals:
        md = _get(s, "metadata", None) or {}
        if isinstance(md, dict) and isinstance(md.get(field), str):
            return md[field]
    return ""

__all__ = [
    "FeatureExtractor",
    "FeatureExtractorStats",
    "FEATURE_COLUMNS_V2",
    "DETECTOR_TOP30",
]
