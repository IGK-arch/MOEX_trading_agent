"""Технический трейдер паттернов (TA Trader)."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC
from typing import Any

import app.config as cfg
from app.agents.base import BaseAdapter
from app.agents.hmm_regime import get_hmm_detector
from app.agents.ta_catboost import get_ta_catboost
from app.agents.ta_indicators import compute_all
from app.agents.ta_patterns.candles import latest_candle_signal
from app.agents.ta_patterns.confluence_filters import (
    passes_atr_percentile,
    passes_hmm_alignment,
    passes_time_of_day,
    passes_volume_check,
)
from app.agents.ta_patterns.continuation import (
    detect_compression_breakout,
    detect_flag,
    detect_pennant,
    detect_rectangle,
    detect_triangle,
)
from app.agents.ta_patterns.dasha_patterns import (
    detect_all_dasha_patterns,
)
from app.agents.ta_patterns.extras_chart import (
    detect_box_breakout,
    detect_cup_handle,
    detect_diamond,
    detect_wedge_continuation,
)
from app.agents.ta_patterns.harmonic import HARMONIC_DETECTORS
from app.agents.ta_patterns.levels import distance_to_nearest_atrs, find_support_resistance
from app.agents.ta_patterns.mtf_confluence import MTFConfluence
from app.agents.ta_patterns.noise_blacklist import is_noisy, magnitude_penalty
from app.agents.ta_patterns.sharpe_filter import detector_magnitude_haircut
from app.agents.ta_patterns.pivots import find_pivots
from app.agents.ta_patterns.research_patterns import (
    detect_all_research_patterns,
)
from app.agents.ta_patterns.reversal import (
    PatternSignal,
    detect_double_top_bottom,
    detect_head_shoulders,
    detect_megaphone,
    detect_rounding,
    detect_triple_top_bottom,
    detect_wedge_reversal,
)
from app.agents.ta_patterns.safe_runner import safe_detect
from app.agents.ta_patterns.smc import (
    PRODUCTION_PATTERNS as SMC_PRODUCTION_PATTERNS,
)
from app.agents.ta_patterns.smc import (
    SMC_DETECTORS,
    SMC_PRODUCTION_ENABLED,
)
from app.data.candle_store import get_candle_store
from app.dispatcher.signal import Direction, SignalSource, UnifiedSignal
from app.utils.logging import get_logger, get_trace_id

logger = get_logger(__name__)

try:
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

_TOD_MORNING = (10, 12)
_TOD_MIDDAY = (12, 15)
_TOD_AFTERNOON = (15, 19)

def _tod_bucket_for_msk_hour(h: int) -> str:
    """Tod bucket for msk hour."""
    if _TOD_MORNING[0] <= h < _TOD_MORNING[1]:
        return "morning"
    if _TOD_MIDDAY[0] <= h < _TOD_MIDDAY[1]:
        return "midday"
    if _TOD_AFTERNOON[0] <= h < _TOD_AFTERNOON[1]:
        return "afternoon"
    return "off"

def _now_msk_bucket() -> str:
    """Return current TOD bucket using server clock."""
    from datetime import datetime, timedelta

    now_utc = datetime.now(tz=UTC)
    msk_hour = (now_utc + timedelta(hours=3)).hour
    return _tod_bucket_for_msk_hour(msk_hour)

def _bar_timestamp(df: pd.DataFrame, idx: int) -> Any:
    """Return the bar's timestamp at `idx` if available.

    Tries (in order) df['timestamp'][idx], df['begin'][idx], df.index[idx].
    Returns None if nothing is parseable — callers should treat None as
    "skip TOD filter for this bar".
    """
    if df is None or not _HAS_PANDAS or not hasattr(df, "columns"):
        return None
    n = len(df)
    if n == 0:
        return None
    if idx < 0:
        idx = n + idx
    if idx < 0 or idx >= n:
        return None
    for col in ("timestamp", "begin"):
        if col in df.columns:
            try:
                return df[col].iloc[idx]
            except Exception:
                continue
    try:
        return df.index[idx]
    except Exception:
        return None

def _is_blocked(pattern: str, ticker: str, tod_bucket: str, regime: str) -> bool:
    """True if any conditional blacklist suppresses this signal."""
    if pattern in cfg.DETECTOR_GOLDLIST:
        return False
    if pattern in cfg.DETECTOR_TICKER_BLACKLIST.get(ticker, frozenset()):
        return True
    if pattern in cfg.DETECTOR_TOD_BLACKLIST.get(tod_bucket, frozenset()):
        return True
    return pattern in cfg.DETECTOR_REGIME_BLACKLIST.get(regime, frozenset())

def _gold_multiplier(pattern: str, ticker: str, tod_bucket: str, regime: str) -> float:
    """Compute the size multiplier from global + conditional gold lists."""
    in_ticker_black = pattern in cfg.DETECTOR_TICKER_BLACKLIST.get(ticker, frozenset())
    in_tod_black = pattern in cfg.DETECTOR_TOD_BLACKLIST.get(tod_bucket, frozenset())
    in_regime_black = pattern in cfg.DETECTOR_REGIME_BLACKLIST.get(regime, frozenset())

    if pattern in cfg.DETECTOR_GOLDLIST:
        if in_ticker_black or in_tod_black or in_regime_black:
            return 1.0
        return cfg.DETECTOR_GOLD_SIZE_MULT

    mult = 1.0
    if pattern in cfg.DETECTOR_TICKER_GOLDLIST.get(ticker, frozenset()):
        mult *= cfg.DETECTOR_COND_GOLD_MULT
    if pattern in cfg.DETECTOR_TOD_GOLDLIST.get(tod_bucket, frozenset()):
        mult *= cfg.DETECTOR_COND_GOLD_MULT
    if pattern in cfg.DETECTOR_REGIME_GOLDLIST.get(regime, frozenset()):
        mult *= cfg.DETECTOR_COND_GOLD_MULT
    return min(mult, cfg.DETECTOR_COND_GOLD_MAX_STACK)

REVERSAL_DETECTORS = [
    ("double_top_bottom", detect_double_top_bottom),
    ("head_shoulders", detect_head_shoulders),
    ("wedge_reversal", detect_wedge_reversal),
    ("triple_top_bottom", detect_triple_top_bottom),
    ("megaphone", detect_megaphone),
    ("rounding", detect_rounding),
    ("diamond", detect_diamond),
    ("cup_handle", detect_cup_handle),
]

CONTINUATION_DETECTORS = [
    ("flag", detect_flag),
    ("pennant", detect_pennant),
    ("triangle", detect_triangle),
    ("rectangle", detect_rectangle),
    ("compression_breakout", detect_compression_breakout),
    ("box_breakout", detect_box_breakout),
    ("wedge_continuation", detect_wedge_continuation),
]

HARMONIC_DETECTORS_REG = [(fn.__name__.replace("detect_", ""), fn) for fn in HARMONIC_DETECTORS]

SMC_DETECTORS_REG = [(fn.__name__.replace("detect_", "smc_"), fn) for fn in SMC_DETECTORS]

class TATrader(BaseAdapter):
    """Technical Pattern Trader — main adapter coordinator."""

    name = "TA"

    def __init__(
        self,
        tickers: list[str] | None = None,
        interval_min: int = 10,
        min_bars_required: int = 20,
    ) -> None:
        """Init."""
        super().__init__()
        self.tickers = tickers or cfg.TICKERS
        self.interval_min = interval_min
        self.min_bars_required = min_bars_required
        self.candle_store = get_candle_store()
        self.hmm = get_hmm_detector()
        self.catboost = get_ta_catboost()
        self.mtf = MTFConfluence(
            higher_tf=cfg.MTF_HIGHER_TF_MIN,
            lower_tf=cfg.MTF_LOWER_TF_MIN,
        )

        self._seen_patterns: set[tuple[str, int, str]] = set()
        self._seen_max_size = 5000
        self._seen_lock = threading.Lock()

        self._indicators_cache: dict[str, tuple[int, int, dict[str, Any]]] = {}

        self._poll_count = 0
        self._signal_count = 0

    async def startup(self) -> None:
        """Load HMM model (or fit on warm-up) + CatBoost."""
        if not self.hmm.load():
            logger.info("HMM model not present — first run, will fit on warm-up data")

        self.catboost.load()
        self._started = True
        logger.info(
            "TATrader started",
            extra={
                "tickers": len(self.tickers),
                "interval_min": self.interval_min,
                "regime": self.hmm.current_label,
            },
        )

    async def shutdown(self) -> None:
        """Shutdown."""
        self._started = False
        logger.info("TATrader stopped", extra={"stats": self.stats})

    async def poll(self) -> list[UnifiedSignal]:
        """One detection cycle across all tickers. Returns: list[UnifiedSignal]."""
        if not self._started:
            raise RuntimeError("TATrader not started")

        start_ts = time.monotonic()
        all_signals: list[UnifiedSignal] = []
        feats_batch: list[tuple[UnifiedSignal, dict[str, float]]] = []

        regime = self.hmm.current_label
        self.catboost.reset_extra_cache()

        ready: list[tuple[str, pd.DataFrame]] = []
        for ticker in self.tickers:
            df = self.candle_store.get(ticker, self.interval_min)
            if (
                not _HAS_PANDAS
                or not isinstance(df, pd.DataFrame)
                or len(df) < self.min_bars_required
            ):
                continue
            ready.append((ticker, df))

        async def _analyze_one(
            ticker: str,
            df: pd.DataFrame,
        ) -> list[tuple[UnifiedSignal, dict[str, float]]]:
            """Analyze one."""
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        self._analyze_ticker_sync,
                        ticker,
                        df,
                        regime,
                        True,
                    ),
                    timeout=3.0,
                )
            except TimeoutError:
                logger.warning(
                    "TATrader: per-ticker timeout",
                    extra={"ticker": ticker, "timeout_ms": 3000},
                )
                return []
            except Exception as exc:
                logger.error(
                    "TATrader: ticker analysis failed",
                    extra={"ticker": ticker, "error": str(exc)},
                )
                return []

        per_ticker_results = await asyncio.gather(
            *(_analyze_one(t, df) for t, df in ready),
            return_exceptions=False,
        )

        for signals_for_ticker in per_ticker_results:
            for unified, feat in signals_for_ticker:
                feats_batch.append((unified, feat))

        if feats_batch:
            scores = self.catboost.predict_batch([f for _, f in feats_batch])
            tod_bucket = _now_msk_bucket()
            for (unified, _), score in zip(feats_batch, scores, strict=False):
                pattern_name = unified.metadata.get("pattern", "")
                ticker_name = unified.ticker
                regime_mult = self.hmm.regime_signal_filter(pattern_name, unified.direction.value)
                gold_mult = _gold_multiplier(
                    pattern_name,
                    ticker_name,
                    tod_bucket,
                    regime,
                )
                noise_mult = magnitude_penalty(pattern_name or unified.detector)
                sharpe_mult = detector_magnitude_haircut(pattern_name or unified.detector)
                base_conf = unified.raw_confidence
                unified.magnitude = max(
                    0.0,
                    min(
                        1.0,
                        base_conf
                        * regime_mult
                        * gold_mult
                        * noise_mult
                        * sharpe_mult
                        * (0.5 + score),
                    ),
                )
                unified.metadata["catboost_score"] = round(score, 3)
                unified.metadata["regime_mult"] = round(regime_mult, 3)
                unified.metadata["gold_mult"] = round(gold_mult, 3)
                unified.metadata["noise_mult"] = round(noise_mult, 3)
                unified.metadata["sharpe_mult"] = round(sharpe_mult, 3)
                unified.metadata["regime"] = regime
                unified.metadata["tod_bucket"] = tod_bucket
                if is_noisy(pattern_name or unified.detector):
                    unified.metadata["noise_blacklisted"] = True
                all_signals.append(unified)

        elapsed_ms = round((time.monotonic() - start_ts) * 1000)
        self._poll_count += 1
        self._signal_count += len(all_signals)

        log_fn = logger.info if all_signals else logger.debug
        log_fn(
            "TATrader poll done",
            extra={
                "signals": len(all_signals),
                "tickers_scanned": len(self.tickers),
                "regime": regime,
                "latency_ms": elapsed_ms,
                "trace_id": get_trace_id(),
            },
        )

        return all_signals

    async def _analyze_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
        regime: str,
        track_seen: bool = True,
    ) -> list[tuple[UnifiedSignal, dict[str, float]]]:
        """Async wrapper around _analyze_ticker_sync."""
        return self._analyze_ticker_sync(ticker, df, regime, track_seen)

    def _analyze_ticker_sync(
        self,
        ticker: str,
        df: pd.DataFrame,
        regime: str,
        track_seen: bool = True,
    ) -> list[tuple[UnifiedSignal, dict[str, float]]]:
        """Analyze one ticker.

        Returns:
            list[tuple[UnifiedSignal, dict]]: signals + feature dicts.
        """
        cache_key = (id(df), len(df))
        cached = self._indicators_cache.get(ticker)
        if cached is not None and cached[0] == cache_key[0] and cached[1] == cache_key[1]:
            indicators = cached[2]
        else:
            indicators = compute_all(df, atr_period=14)
            self._indicators_cache[ticker] = (cache_key[0], cache_key[1], indicators)
        atr_series = indicators["atr"]

        if atr_series is None or len(atr_series) == 0:
            return []

        current_price = float(df["close"].iloc[-1])
        atr_now = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
        if atr_now <= 0:
            return []

        pivots = find_pivots(df, order=5, atr=atr_series, merge_distance_atr=0.5)
        if len(pivots) < 4:
            return []

        levels = find_support_resistance(
            df,
            pivots,
            atr_series,
            current_price=current_price,
            merge_atr=0.5,
            min_touches=2,
        )
        levels_info = distance_to_nearest_atrs(levels, current_price, atr_now)

        candle_bits = latest_candle_signal(df)

        all_patterns: list[PatternSignal] = []
        for name, detector in REVERSAL_DETECTORS:
            patterns = safe_detect(
                detector, df, pivots, atr_series, _detector_name=name, _ticker=ticker
            )
            for p in patterns:
                p.ticker = ticker
                p.metadata = p.metadata or {}
                p.metadata["detector_group"] = "reversal"
                all_patterns.append(p)

        for name, detector in CONTINUATION_DETECTORS:
            patterns = safe_detect(
                detector, df, pivots, atr_series, _detector_name=name, _ticker=ticker
            )
            for p in patterns:
                p.ticker = ticker
                p.metadata = p.metadata or {}
                p.metadata["detector_group"] = "continuation"
                all_patterns.append(p)

        for name, detector in HARMONIC_DETECTORS_REG:
            patterns = safe_detect(
                detector, df, pivots, atr_series, _detector_name=name, _ticker=ticker
            )
            for p in patterns:
                p.ticker = ticker
                p.metadata = p.metadata or {}
                p.metadata["detector_group"] = "harmonic"
                all_patterns.append(p)

        if SMC_PRODUCTION_ENABLED and SMC_PRODUCTION_PATTERNS:
            for name, detector in SMC_DETECTORS_REG:
                patterns = safe_detect(
                    detector, df, pivots, atr_series, _detector_name=name, _ticker=ticker
                )
                for p in patterns:
                    if p.pattern not in SMC_PRODUCTION_PATTERNS:
                        continue
                    p.ticker = ticker
                    p.metadata = p.metadata or {}
                    p.metadata["detector_group"] = "smc"
                    all_patterns.append(p)

        df_with_atr: pd.DataFrame | None = None
        try:
            cols_to_add: dict[str, Any] = {"atr14": atr_series.values}
            if "timestamp" not in df.columns:
                cols_to_add["timestamp"] = df.index
            df_with_atr = df.assign(**cols_to_add)
        except Exception as exc:
            logger.debug(
                "TA df.assign() for Dasha/Research failed",
                extra={"ticker": ticker, "error": str(exc)},
            )

        if df_with_atr is not None:
            dasha_patterns = safe_detect(
                detect_all_dasha_patterns,
                df_with_atr,
                atr_col="atr14",
                _detector_name="dasha_all",
                _ticker=ticker,
            )
            for dp in dasha_patterns:
                ps = PatternSignal(
                    pattern=dp.pattern,
                    direction=dp.direction,
                    confidence=dp.confidence,
                    bar_idx=dp.bar_idx,
                    entry=dp.entry,
                    stop=dp.stop,
                    target=dp.target,
                    expected_rr=dp.expected_rr,
                    atr_at_entry=dp.atr_at_entry,
                    metadata={**dp.metadata, "detector_group": "dasha"},
                )
                ps.ticker = ticker
                all_patterns.append(ps)

        if df_with_atr is None:
            try:
                cols_to_add = {"atr14": atr_series.values}
                if "timestamp" not in df.columns:
                    cols_to_add["timestamp"] = df.index
                df_with_atr = df.assign(**cols_to_add)
            except Exception as exc:
                logger.debug(
                    "TA df.assign() for Research failed",
                    extra={"ticker": ticker, "error": str(exc)},
                )

        if df_with_atr is not None:
            research_patterns = safe_detect(
                detect_all_research_patterns,
                df_with_atr,
                atr_col="atr14",
                production_only=True,
                _detector_name="research_all",
                _ticker=ticker,
            )
            for rp in research_patterns:
                ps = PatternSignal(
                    pattern=rp.pattern,
                    direction=rp.direction,
                    confidence=rp.confidence,
                    bar_idx=rp.bar_idx,
                    entry=rp.entry,
                    stop=rp.stop,
                    target=rp.target,
                    expected_rr=rp.expected_rr,
                    atr_at_entry=rp.atr_at_entry,
                    metadata={**rp.metadata, "detector_group": "research"},
                )
                ps.ticker = ticker
                all_patterns.append(ps)

        if not all_patterns:
            return []

        all_patterns = [p for p in all_patterns if p.pattern not in cfg.DETECTOR_BLACKLIST]
        if not all_patterns:
            return []

        tod_bucket_now = _now_msk_bucket()
        all_patterns = [
            p for p in all_patterns if not _is_blocked(p.pattern, ticker, tod_bucket_now, regime)
        ]
        if not all_patterns:
            return []

        results: list[tuple[UnifiedSignal, dict[str, float]]] = []
        for p in all_patterns:
            if track_seen:
                key = (ticker, p.bar_idx, p.pattern)
                with self._seen_lock:
                    if key in self._seen_patterns:
                        continue
                    self._seen_patterns.add(key)

                    if len(self._seen_patterns) > self._seen_max_size:
                        self._seen_patterns = set(
                            list(self._seen_patterns)[-self._seen_max_size // 2 :]
                        )

            family = p.metadata.get("detector_group", "") if p.metadata else ""
            if cfg.CONFLUENCE_VOLUME_CHECK and not passes_volume_check(
                df,
                p.bar_idx,
                multiplier=cfg.CONFLUENCE_VOLUME_MULTIPLIER,
            ):
                continue
            if cfg.CONFLUENCE_HMM_ALIGN and not passes_hmm_alignment(family, regime):
                continue
            if cfg.CONFLUENCE_TOD_FILTER:
                bar_ts = _bar_timestamp(df, p.bar_idx)
                if bar_ts is not None and not passes_time_of_day(bar_ts):
                    continue
            if cfg.CONFLUENCE_ATR_PCTILE and not passes_atr_percentile(
                df,
                p.bar_idx,
                low_p=cfg.CONFLUENCE_ATR_PCT_LOW,
                high_p=cfg.CONFLUENCE_ATR_PCT_HIGH,
            ):
                continue

            if cfg.MTF_CONFLUENCE_ENABLED and family in ("reversal", "continuation"):
                df_higher = self.candle_store.get(ticker, cfg.MTF_HIGHER_TF_MIN)
                df_lower = self.candle_store.get(ticker, cfg.MTF_LOWER_TF_MIN)
                if (
                    _HAS_PANDAS
                    and isinstance(df_higher, pd.DataFrame)
                    and isinstance(df_lower, pd.DataFrame)
                    and len(df_higher) >= cfg.MTF_MIN_BARS_HIGHER
                    and len(df_lower) >= cfg.MTF_MIN_BARS_LOWER
                ):
                    ok, _reason = self.mtf.validate(
                        family,
                        p.direction,
                        df_higher,
                        df_lower,
                    )
                    if not ok:
                        continue

            direction = Direction.BUY if p.direction == "BUY" else Direction.SELL

            unified = UnifiedSignal(
                source=SignalSource.TA,
                detector=p.pattern,
                ticker=ticker,
                direction=direction,
                magnitude=p.confidence,
                raw_confidence=p.confidence,
                horizon_min=120,
                price=current_price,
                entry_level=p.entry,
                stop_level=p.stop,
                target_level=p.target,
                expected_rr=p.expected_rr,
                atr=p.atr_at_entry,
                metadata={
                    "pattern": p.pattern,
                    "bar_idx": p.bar_idx,
                    **p.metadata,
                },
            )

            feat = self.catboost.build_features(
                pattern=p.pattern,
                expected_rr=p.expected_rr,
                price=current_price,
                atr_val=atr_now,
                atr_at_entry=p.atr_at_entry,
                indicators=indicators,
                levels_info=levels_info,
                candle_bits=candle_bits,
                regime=regime,
                df=df,
                bar_idx=p.bar_idx,
                pivots=pivots,
            )

            results.append((unified, feat))

        return results

    async def get_context_snapshot(self, ticker: str) -> dict[str, Any]:
        """Return the current best TA/CatBoost context for one ticker."""
        if not self._started or not _HAS_PANDAS:
            return {"ticker": ticker, "regime": self.hmm.current_label}

        df = self.candle_store.get(ticker, self.interval_min)
        if not isinstance(df, pd.DataFrame) or len(df) < self.min_bars_required:
            return {"ticker": ticker, "regime": self.hmm.current_label}

        regime = self.hmm.current_label
        candidates = await self._analyze_ticker(ticker, df, regime, track_seen=False)
        if not candidates:
            current_price = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0
            return {
                "ticker": ticker,
                "price": current_price,
                "regime": regime,
                "ta_pattern": "",
                "ta_direction": "",
                "catboost_score": 0.0,
                "ta_expected_rr": 0.0,
            }

        scores = self.catboost.predict_batch([feat for _, feat in candidates])
        best_payload: dict[str, Any] | None = None
        for (signal, _), score in zip(candidates, scores, strict=False):
            regime_mult = self.hmm.regime_signal_filter(
                str(signal.metadata.get("pattern", "")),
                signal.direction.value,
            )
            rank_score = signal.raw_confidence * regime_mult * (0.5 + score)
            payload = {
                "ticker": ticker,
                "price": signal.price,
                "regime": regime,
                "ta_pattern": str(signal.metadata.get("pattern", "")),
                "ta_direction": signal.direction.value,
                "catboost_score": float(score),
                "ta_expected_rr": signal.expected_rr,
                "ta_rank_score": rank_score,
            }
            if best_payload is None or rank_score > float(best_payload.get("ta_rank_score", 0.0)):
                best_payload = payload

        return best_payload or {"ticker": ticker, "regime": regime}

    async def fit_hmm_from_iss(self, days: int = 60) -> bool:
        """Fetch index daily candles and refresh or fit HMM regime label."""
        from datetime import datetime, timedelta

        from app.data.iss_client import get_iss_client

        iss = get_iss_client()
        if not iss._started:
            await iss.startup()

        till = datetime.now(tz=UTC)
        from_dt = till - timedelta(days=days + 10)

        df = None
        for ticker_try in ("IMOEX", "SBER"):
            try:
                df = await iss.get_candles(ticker_try, interval=24, from_dt=from_dt, till_dt=till)
                if isinstance(df, pd.DataFrame) and len(df) >= 30:
                    logger.debug("HMM data source", extra={"ticker": ticker_try, "rows": len(df)})
                    break
            except Exception as exc:
                logger.debug(f"HMM data fetch failed for {ticker_try}", extra={"error": str(exc)})
                continue

        if not isinstance(df, pd.DataFrame) or len(df) < 30:
            logger.warning(
                "HMM warm-up: too few daily candles from all sources",
                extra={"rows": len(df) if isinstance(df, pd.DataFrame) else 0},
            )
            return False

        if self.hmm.model is not None:
            self.hmm.predict_state(df)
            logger.info("HMM regime refreshed", extra={"regime": self.hmm.current_label})
            return True

        ok = await self.hmm.fit(df)
        if ok:
            self.hmm.predict_state(df)
        return ok

_ta_trader: TATrader | None = None

def get_ta_trader() -> TATrader:
    """Get ta trader."""
    global _ta_trader
    if _ta_trader is None:
        _ta_trader = TATrader()
    return _ta_trader
