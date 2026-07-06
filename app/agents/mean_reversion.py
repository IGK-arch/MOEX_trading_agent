"""Mean-reversion на Bollinger Bands, gated by HMM."""

from __future__ import annotations

import time

import app.config as cfg
from app.agents.base import BaseAdapter
from app.agents.hmm_regime import get_hmm_detector
from app.agents.ta_indicators import compute_atr, compute_bollinger, compute_rsi, compute_sma
from app.data.candle_store import get_candle_store
from app.dispatcher.signal import Direction, SignalSource, UnifiedSignal
from app.utils.logging import get_logger, get_trace_id

logger = get_logger(__name__)

try:
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

class MeanReversion(BaseAdapter):
    """HMM-gated Bollinger mean-reversion."""

    name = "MEAN_REV"

    def __init__(
        self,
        tickers: list[str] | None = None,
        interval_min: int = 10,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        atr_stop_mult: float = 1.5,
        target_atr_mult: float = 1.5,
        max_hold_bars: int = 6,
        always_active: bool = True,
    ) -> None:
        """Init."""
        super().__init__()
        self.tickers = tickers or cfg.TICKERS
        self.interval_min = interval_min
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_stop_mult = atr_stop_mult
        self.target_atr_mult = target_atr_mult
        self.max_hold_bars = max_hold_bars

        self.always_active = always_active

        self.candle_store = get_candle_store()
        self.hmm = get_hmm_detector()
        self._poll_count = 0
        self._signal_count = 0

        self._seen: set[tuple[str, int, str]] = set()
        self._seen_max = 5000

    async def startup(self) -> None:
        """Startup."""
        self._started = True
        logger.info(
            "MeanReversion started",
            extra={
                "tickers": len(self.tickers),
                "bb_period": self.bb_period,
                "bb_std": self.bb_std,
                "always_active": self.always_active,
                "regime": self.hmm.current_label,
            },
        )

    async def shutdown(self) -> None:
        """Shutdown."""
        self._started = False
        logger.info("MeanReversion stopped", extra={"stats": self.stats})

    async def poll(self) -> list[UnifiedSignal]:
        """Poll."""
        if not self._started:
            raise RuntimeError("MeanReversion not started")

        if not self.always_active and self.hmm.current_label != "mean_reverting":
            return []

        start_ts = time.monotonic()
        signals: list[UnifiedSignal] = []

        for ticker in self.tickers:
            df = self.candle_store.get(ticker, self.interval_min)
            if not _HAS_PANDAS or not isinstance(df, pd.DataFrame) or len(df) < self.bb_period + 5:
                continue

            try:
                s = self._check_ticker(ticker, df)
                if s is not None:
                    key = (ticker, s.metadata["bar_idx"], s.direction.value)
                    if key in self._seen:
                        continue
                    self._seen.add(key)
                    if len(self._seen) > self._seen_max:
                        self._seen = set(list(self._seen)[-self._seen_max // 2 :])
                    signals.append(s)
            except Exception as exc:
                logger.warning(
                    "MeanReversion: ticker failed",
                    extra={"ticker": ticker, "error": str(exc), "trace_id": get_trace_id()},
                )

        elapsed_ms = round((time.monotonic() - start_ts) * 1000)
        self._poll_count += 1
        self._signal_count += len(signals)
        log_fn = logger.info if signals else logger.debug
        log_fn(
            "MeanReversion poll done",
            extra={
                "signals": len(signals),
                "regime": self.hmm.current_label,
                "latency_ms": elapsed_ms,
                "trace_id": get_trace_id(),
            },
        )
        return signals

    def _check_ticker(self, ticker: str, df: pd.DataFrame) -> UnifiedSignal | None:
        """Look for an entry signal on the last bar."""
        bb = compute_bollinger(df, period=self.bb_period, std_dev=self.bb_std)
        rsi = compute_rsi(df, period=self.rsi_period)
        atr = compute_atr(df, period=14)
        sma = compute_sma(df, period=self.bb_period)

        if bb is None or bb.empty or len(rsi) < 2 or len(atr) == 0:
            return None

        if len(df) < 2:
            return None

        prev_close = float(df["close"].iloc[-2])
        curr_close = float(df["close"].iloc[-1])
        float(df["open"].iloc[-1])

        bbu = float(bb["BBU"].iloc[-1]) if pd.notna(bb["BBU"].iloc[-1]) else None
        bbl = float(bb["BBL"].iloc[-1]) if pd.notna(bb["BBL"].iloc[-1]) else None
        bbm = float(bb["BBM"].iloc[-1]) if pd.notna(bb["BBM"].iloc[-1]) else None
        rsi_val = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None
        atr_val = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
        float(sma.iloc[-1]) if pd.notna(sma.iloc[-1]) else None

        if bbu is None or bbl is None or rsi_val is None or atr_val <= 0:
            return None

        bar_idx = len(df) - 1

        if prev_close > bbu and curr_close < bbu and rsi_val > self.rsi_overbought:
            entry = curr_close
            stop = entry + self.atr_stop_mult * atr_val
            target = bbm if bbm is not None else entry - self.target_atr_mult * atr_val
            rr = (entry - target) / (stop - entry) if stop > entry else 0
            return UnifiedSignal(
                source=SignalSource.MEAN_REV,
                detector="bollinger_short",
                ticker=ticker,
                direction=Direction.SELL,
                magnitude=min(0.85, 0.50 + (rsi_val - 70) / 30 * 0.30),
                raw_confidence=0.60,
                horizon_min=self.max_hold_bars * self.interval_min,
                price=curr_close,
                entry_level=entry,
                stop_level=stop,
                target_level=target,
                expected_rr=max(0.0, rr),
                atr=atr_val,
                metadata={
                    "bar_idx": bar_idx,
                    "rsi": round(rsi_val, 1),
                    "bb_upper": round(bbu, 3),
                    "bb_mid": round(bbm, 3) if bbm else None,
                    "regime_gated": not self.always_active,
                },
            )

        if prev_close < bbl and curr_close > bbl and rsi_val < self.rsi_oversold:
            entry = curr_close
            stop = entry - self.atr_stop_mult * atr_val
            target = bbm if bbm is not None else entry + self.target_atr_mult * atr_val
            rr = (target - entry) / (entry - stop) if entry > stop else 0
            return UnifiedSignal(
                source=SignalSource.MEAN_REV,
                detector="bollinger_long",
                ticker=ticker,
                direction=Direction.BUY,
                magnitude=min(0.85, 0.50 + (30 - rsi_val) / 30 * 0.30),
                raw_confidence=0.60,
                horizon_min=self.max_hold_bars * self.interval_min,
                price=curr_close,
                entry_level=entry,
                stop_level=stop,
                target_level=target,
                expected_rr=max(0.0, rr),
                atr=atr_val,
                metadata={
                    "bar_idx": bar_idx,
                    "rsi": round(rsi_val, 1),
                    "bb_lower": round(bbl, 3),
                    "bb_mid": round(bbm, 3) if bbm else None,
                    "regime_gated": not self.always_active,
                },
            )

        return None

_mean_rev: MeanReversion | None = None

def get_mean_reversion() -> MeanReversion:
    """Get mean reversion."""
    global _mean_rev
    if _mean_rev is None:
        _mean_rev = MeanReversion()
    return _mean_rev
