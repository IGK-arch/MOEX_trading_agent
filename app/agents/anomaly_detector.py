"""Детектор микроструктурных аномалий."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any

import app.config as cfg
from app.agents.anomaly_detectors.absorption import detect_absorption
from app.agents.anomaly_detectors.atr_reversion import detect_atr_reversion
from app.agents.anomaly_detectors.base import AnomalySignal
from app.agents.anomaly_detectors.ofi_spikes import detect_ofi_spikes
from app.agents.anomaly_detectors.price_spikes import detect_price_spikes
from app.agents.anomaly_detectors.volume_zscore import detect_volume_zscore
from app.agents.anomaly_detectors.vwap_crosses import detect_vwap_crosses
from app.agents.base import BaseAdapter
from app.agents.ta_indicators import compute_atr
from app.data.algopack_client import get_algopack_client
from app.data.candle_store import get_candle_store
from app.dispatcher.signal import (
    ConfluenceResult,
    Direction,
    SignalSource,
    UnifiedSignal,
)
from app.utils.logging import get_logger, get_trace_id

logger = get_logger(__name__)

try:
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

CONTEXT_BUFFER_SIZE = 50

class AnomalyDetectorAgent(BaseAdapter):
    """Microstructure anomaly detector — produces BUY/SELL signals + market context."""

    name = "ANOMALY"

    def __init__(
        self,
        tickers: list[str] | None = None,
        interval_min: int = 10,
        min_bars_required: int = 15,
    ) -> None:
        """Init."""
        super().__init__()
        self.tickers = tickers or cfg.TICKERS
        self.interval_min = interval_min
        self.min_bars_required = min_bars_required
        self.candle_store = get_candle_store()
        self.algopack = get_algopack_client()

        self._context_buffer: dict[str, deque[AnomalySignal]] = {
            t: deque(maxlen=CONTEXT_BUFFER_SIZE) for t in self.tickers
        }

        self._seen_signals: set[tuple[str, str, int]] = set()
        self._seen_max = 5000

        self._last_signal_idx: dict[tuple[str, str], int] = {}

        self._obstats_cache: dict[str, Any] = {}
        self._obstats_cache_ts: dict[str, float] = {}
        self._obstats_ttl_sec = 90.0

        self._poll_count = 0
        self._signal_count = 0

    async def startup(self) -> None:
        """Startup."""
        if not self.algopack._started:
            await self.algopack.startup()
        self._started = True
        logger.info(
            "AnomalyDetector started",
            extra={
                "tickers": len(self.tickers),
                "interval_min": self.interval_min,
                "algopack_premium": not self.algopack._auth_failed,
            },
        )

    async def shutdown(self) -> None:
        """Shutdown."""
        self._started = False
        logger.info("AnomalyDetector stopped", extra={"stats": self.stats})

    async def poll(self) -> list[UnifiedSignal]:
        """Run one anomaly scan across all tickers. Returns: list[UnifiedSignal]."""
        if not self._started:
            raise RuntimeError("AnomalyDetector not started")

        start_ts = time.monotonic()

        ready: list[tuple[str, pd.DataFrame]] = []
        for ticker in self.tickers:
            if cfg.PER_TICKER_POLICY.get(ticker.upper(), "ENABLED") == "DISABLED":
                continue
            df = self.candle_store.get(ticker, self.interval_min)
            if (
                not _HAS_PANDAS
                or not isinstance(df, pd.DataFrame)
                or len(df) < self.min_bars_required
            ):
                continue
            ready.append((ticker, df))

        if not ready:
            self._poll_count += 1
            logger.debug(
                "AnomalyDetector poll done (no candles)",
                extra={
                    "signals": 0,
                    "tickers_scanned": 0,
                    "latency_ms": round((time.monotonic() - start_ts) * 1000),
                    "trace_id": get_trace_id(),
                },
            )
            return []

        obstats_results = await asyncio.gather(
            *(self._get_obstats(t) for t, _ in ready),
            return_exceptions=True,
        )
        obstats_by_ticker: dict[str, Any] = {}
        for (t, _), res in zip(ready, obstats_results, strict=False):
            obstats_by_ticker[t] = res if not isinstance(res, Exception) else None

        async def _analyze_one(ticker: str, df: pd.DataFrame) -> tuple[str, list[AnomalySignal]]:
            """Analyze one."""
            try:
                signals = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._analyze_ticker_sync,
                        ticker,
                        df,
                        obstats_by_ticker.get(ticker),
                    ),
                    timeout=1.0,
                )
                return ticker, signals
            except TimeoutError:
                logger.warning(
                    "AnomalyDetector: per-ticker timeout",
                    extra={"ticker": ticker, "timeout_ms": 1000},
                )
                return ticker, []
            except Exception as exc:
                logger.error(
                    "AnomalyDetector: ticker failed",
                    extra={"ticker": ticker, "error": str(exc)},
                )
                return ticker, []

        per_ticker_results = await asyncio.gather(
            *(_analyze_one(t, df) for t, df in ready),
            return_exceptions=False,
        )

        all_signals: list[UnifiedSignal] = []
        for ticker, signals in per_ticker_results:
            for s in signals:
                key = (ticker, s.detector, s.bar_idx)
                if key in self._seen_signals:
                    continue
                self._seen_signals.add(key)

                if s.is_actionable():
                    all_signals.append(self._to_unified(s))
                else:
                    self._context_buffer[ticker].append(s)

            if len(self._seen_signals) > self._seen_max:
                self._seen_signals = set(list(self._seen_signals)[-self._seen_max // 2 :])

        elapsed_ms = round((time.monotonic() - start_ts) * 1000)
        self._poll_count += 1
        self._signal_count += len(all_signals)

        log_fn = logger.info if all_signals else logger.debug
        log_fn(
            "AnomalyDetector poll done",
            extra={
                "signals": len(all_signals),
                "tickers_scanned": len(ready),
                "latency_ms": elapsed_ms,
                "trace_id": get_trace_id(),
            },
        )

        return all_signals

    async def _analyze_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
    ) -> list[AnomalySignal]:
        """Run all 6 detectors on a ticker's candles (async — for verify_signal)."""
        obstats = await self._get_obstats(ticker)
        return self._analyze_ticker_sync(ticker, df, obstats)

    def _analyze_ticker_sync(
        self,
        ticker: str,
        df: pd.DataFrame,
        obstats: pd.DataFrame | None,
    ) -> list[AnomalySignal]:
        """Synchronous detector pipeline."""
        atr = compute_atr(df, period=14)
        if atr is None or len(atr) == 0:
            return []

        results: list[AnomalySignal] = []

        try:
            results.extend(detect_volume_zscore(df, ticker))
        except Exception as exc:
            logger.warning("volume_zscore failed", extra={"ticker": ticker, "error": str(exc)})

        try:
            results.extend(detect_price_spikes(df, ticker, atr))
        except Exception as exc:
            logger.warning("price_spikes failed", extra={"ticker": ticker, "error": str(exc)})

        try:
            results.extend(detect_absorption(df, ticker, atr))
        except Exception as exc:
            logger.warning("absorption failed", extra={"ticker": ticker, "error": str(exc)})

        try:
            results.extend(detect_vwap_crosses(df, ticker, atr))
        except Exception as exc:
            logger.warning("vwap_crosses failed", extra={"ticker": ticker, "error": str(exc)})

        if obstats is not None:
            try:
                results.extend(detect_ofi_spikes(obstats, ticker))
            except Exception as exc:
                logger.warning("ofi_spikes failed", extra={"ticker": ticker, "error": str(exc)})

        try:
            cooldown_key = (ticker, "atr_reversion")
            last_idx = self._last_signal_idx.get(cooldown_key, -100)
            ar_signals = detect_atr_reversion(df, ticker, atr, last_signal_idx=last_idx)
            if ar_signals:
                self._last_signal_idx[cooldown_key] = ar_signals[-1].bar_idx
            results.extend(ar_signals)
        except Exception as exc:
            logger.warning("atr_reversion failed", extra={"ticker": ticker, "error": str(exc)})

        if cfg.DETECTOR_BLACKLIST:
            results = [r for r in results if r.detector not in cfg.DETECTOR_BLACKLIST]

        return results

    async def _get_obstats(self, ticker: str) -> pd.DataFrame | None:
        """Cached AlgoPack obstats fetch (TTL 30s)."""
        now = time.monotonic()
        cached_ts = self._obstats_cache_ts.get(ticker, 0)
        if now - cached_ts < self._obstats_ttl_sec:
            return self._obstats_cache.get(ticker)
        try:
            df = await self.algopack.get_obstats(ticker)
            self._obstats_cache[ticker] = df
            self._obstats_cache_ts[ticker] = now
            return df
        except asyncio.CancelledError:
            return None
        except Exception as exc:
            logger.debug("obstats fetch failed", extra={"ticker": ticker, "error": str(exc)})
            return None

    def _to_unified(self, s: AnomalySignal) -> UnifiedSignal:
        """Convert AnomalySignal → UnifiedSignal for Dispatcher."""
        direction = (
            Direction.BUY
            if s.direction == "BUY"
            else (Direction.SELL if s.direction == "SELL" else Direction.NEUTRAL)
        )
        damp = float(getattr(cfg, "ANOMALY_MAGNITUDE_DAMP", 0.7))
        sl_atr = float(getattr(cfg, "ANOMALY_SL_ATR_MULT", 1.5))
        tp_atr = float(getattr(cfg, "ANOMALY_TP_ATR_MULT", 2.0))
        horizon = int(getattr(cfg, "ANOMALY_SIGNAL_HORIZON_MIN", 15))
        return UnifiedSignal(
            source=SignalSource.ANOMALY,
            detector=s.detector,
            ticker=s.ticker,
            direction=direction,
            magnitude=s.confidence * damp,
            raw_confidence=s.confidence,
            horizon_min=horizon,
            price=s.price,
            entry_level=s.price,
            stop_level=s.price - (sl_atr * s.atr if direction == Direction.BUY else -sl_atr * s.atr),
            target_level=s.price + (tp_atr * s.atr if direction == Direction.BUY else -tp_atr * s.atr),
            expected_rr=tp_atr / sl_atr if sl_atr > 0 else 1.0,
            atr=s.atr,
            metadata={
                "anomaly_detector": s.detector,
                **s.metadata,
            },
        )

    async def verify_signal(
        self,
        ticker: str,
        proposed_direction: Direction | str,
        bars_window: int = 6,
    ) -> ConfluenceResult:
        """Check whether anomaly signals agree with `proposed_direction`.

        Args:
            ticker: ticker symbol.
            proposed_direction: BUY/SELL/NEUTRAL.
            bars_window: lookback in bars.
        Returns:
            ConfluenceResult: matching/opposing counts + multiplier.
        """

        if isinstance(proposed_direction, str):
            dir_str = proposed_direction.upper()
            try:
                dir_enum = Direction(dir_str)
            except ValueError:
                dir_enum = Direction.NEUTRAL
        else:
            dir_enum = proposed_direction
            dir_str = dir_enum.value

        empty = ConfluenceResult(
            ticker=ticker,
            direction=dir_enum,
            matching_count=0,
            opposing_count=0,
            multiplier=1.0,
        )

        if not self._started:
            return empty

        df = self.candle_store.get(ticker, self.interval_min)
        if not _HAS_PANDAS or not isinstance(df, pd.DataFrame) or len(df) < 30:
            return empty

        try:
            signals = await self._analyze_ticker(ticker, df)
        except Exception as exc:
            logger.warning("verify_signal scan failed", extra={"ticker": ticker, "error": str(exc)})
            return empty

        recent_bar_threshold = len(df) - bars_window
        recent = [s for s in signals if s.bar_idx >= recent_bar_threshold]

        matching = sum(1 for s in recent if s.direction == dir_str)
        opposing = sum(
            1 for s in recent if s.direction in ("BUY", "SELL") and s.direction != dir_str
        )

        if matching == 0 and opposing >= 1:
            multiplier = 0.5
        elif matching >= 3 and opposing == 0:
            multiplier = 2.0
        elif matching == 2 and opposing == 0:
            multiplier = 1.5
        elif matching == 1 and opposing == 0:
            multiplier = 1.2
        else:
            multiplier = 1.0

        result = ConfluenceResult(
            ticker=ticker,
            direction=dir_enum,
            matching_count=matching,
            opposing_count=opposing,
            multiplier=multiplier,
        )

        logger.debug(
            "verify_signal result",
            extra={
                "ticker": ticker,
                "proposed": dir_str,
                "matching": matching,
                "opposing": opposing,
                "multiplier": multiplier,
                "is_confirmed": result.is_confirmed,
                "is_vetoed": result.is_vetoed,
            },
        )

        return result

    def get_context_for_ticker(self, ticker: str, k: int = 5) -> list[dict]:
        """Return the last k NEUTRAL anomalies for a ticker."""
        buf = self._context_buffer.get(ticker)
        if not buf:
            return []
        return [
            {
                "detector": s.detector,
                "ts": str(s.ts) if s.ts is not None else None,
                "price": s.price,
                "metadata": s.metadata,
            }
            for s in list(buf)[-k:]
        ]

    def get_market_overview(self, k_per_ticker: int = 2) -> dict:
        """Cross-ticker summary of recent anomaly activity."""
        return {
            ticker: self.get_context_for_ticker(ticker, k=k_per_ticker)
            for ticker in self.tickers
            if self._context_buffer.get(ticker)
        }

_anomaly_agent: AnomalyDetectorAgent | None = None

def get_anomaly_agent() -> AnomalyDetectorAgent:
    """Get anomaly agent."""
    global _anomaly_agent
    if _anomaly_agent is None:
        _anomaly_agent = AnomalyDetectorAgent()
    return _anomaly_agent
