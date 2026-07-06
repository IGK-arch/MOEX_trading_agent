"""Агрегатор сигналов с confluence/veto матрицей."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import app.config as cfg
from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    DecisionTier,
    Direction,
    RiskCheckResult,
    SignalSource,
    UnifiedSignal,
)
from app.utils.logging import get_logger

if TYPE_CHECKING:
    from app.agents.anomaly_detector import AnomalyDetectorAgent
    from app.agents.meta_classifier import MetaClassifier, MetaContext
    from app.agents.microstructure_gates import MicrostructureGates

logger = get_logger(__name__)

class SignalAggregator:
    """Combine signals from multiple sources into a single Decision per ticker."""

    def __init__(
        self,
        anomaly_agent: AnomalyDetectorAgent | None = None,
        meta_classifier: MetaClassifier | None = None,
        microstructure_gates: MicrostructureGates | None = None,
    ) -> None:
        """Init."""
        self.anomaly_agent = anomaly_agent
        self.meta_classifier = meta_classifier
        self.microstructure_gates = microstructure_gates

    async def aggregate(
        self,
        ticker: str,
        cycle_id: str,
        signals: list[UnifiedSignal],
        meta_context: MetaContext | None = None,
        supercandles_df: Any | None = None,
        df_10m: Any | None = None,
        df_60m: Any | None = None,
        df_daily: Any | None = None,
        *,
        defer_meta: bool = False,
    ) -> Decision:
        """Aggregate signals for ONE ticker into a Decision.

        Args:
            ticker: instrument code
            cycle_id: dispatcher cycle id
            signals: signals to aggregate
            meta_context: optional meta context
            supercandles_df: supercandles dataframe or None
            df_10m: 10-min candles dataframe or None
            df_60m: 60-min candles dataframe or None
            df_daily: daily candles dataframe or None
            defer_meta: skip per-ticker meta scoring
        Returns:
            Decision: aggregated decision
        """
        if not signals:
            return self._no_trade_decision(ticker, cycle_id, signals, "no signals")

        buy_sigs = [s for s in signals if s.direction == Direction.BUY]
        sell_sigs = [s for s in signals if s.direction == Direction.SELL]
        [s for s in signals if s.direction == Direction.NEUTRAL]

        n_buy = len(buy_sigs)
        n_sell = len(sell_sigs)

        if n_buy == 0 and n_sell == 0:
            return self._no_trade_decision(ticker, cycle_id, signals, "all neutral")

        counter_bias_penalty = 1.0
        if getattr(cfg, "COUNTER_BIAS_GUARD_ENABLED", True):
            try:
                bias = cfg.get_ticker_bias(ticker)
            except Exception:
                bias = None
            if bias in ("BUY", "SELL"):
                proposed = Direction.BUY if n_buy > n_sell else Direction.SELL
                is_counter = (
                    (bias == "BUY" and proposed == Direction.SELL)
                    or (bias == "SELL" and proposed == Direction.BUY)
                )
                if is_counter:
                    counter_bias_penalty = float(
                        getattr(cfg, "PER_TICKER_BIAS_COUNTER_MULT", 0.70)
                    )
                    logger.info(
                        "Counter-bias: magnitude уменьшен",
                        extra={
                            "ticker": ticker,
                            "bias": bias,
                            "proposed": proposed.value,
                            "magnitude_mult": counter_bias_penalty,
                        },
                    )

        if n_buy > 0 and n_sell == 0:
            direction = Direction.BUY
            voting_sigs = buy_sigs
        elif n_sell > 0 and n_buy == 0:
            direction = Direction.SELL
            voting_sigs = sell_sigs
        else:
            return await self._resolve_conflict(ticker, cycle_id, signals, buy_sigs, sell_sigs)

        if (
            direction == Direction.BUY
            and getattr(cfg, "PRE_LONG_MOMENTUM_GUARD_ENABLED", True)
            and df_10m is not None
            and len(df_10m) >= 3
        ):
            try:
                min_momentum = float(getattr(cfg, "PRE_LONG_MIN_MOMENTUM_PCT", -0.015))
                last = float(df_10m["close"].iloc[-1])
                prev = float(df_10m["close"].iloc[-3])
                if prev > 0:
                    momentum = (last - prev) / prev
                    if momentum < min_momentum:
                        return self._no_trade_decision(
                            ticker,
                            cycle_id,
                            signals,
                            f"PRE_LONG_FALLING_KNIFE: 30m_momentum={momentum*100:.2f}% < {min_momentum*100:.1f}%",
                        )
            except Exception:
                pass

        sources = {s.source for s in voting_sigs}
        is_anomaly_only = sources == {SignalSource.ANOMALY}
        weight_factor = 0.7 if is_anomaly_only else 1.0

        unique_sources = len({s.source for s in voting_sigs})
        if getattr(cfg, "CONFLUENCE_TIERED_BOOST", True):
            if unique_sources >= 4:
                confluence_mult = 2.0
                mode = "MAX_CONFLUENCE"
            elif unique_sources == 3:
                confluence_mult = 1.7
                mode = "STRONG_CONFLUENCE"
            elif unique_sources == 2:
                confluence_mult = 1.3
                mode = "CONFLUENCE"
            else:
                confluence_mult = 1.0
                mode = "STANDALONE"
        else:
            if unique_sources >= 3:
                confluence_mult = 2.0
                mode = "STRONG_CONFLUENCE"
            elif unique_sources == 2:
                confluence_mult = 1.5
                mode = "CONFLUENCE"
            else:
                confluence_mult = 1.0
                mode = "STANDALONE"

        if (
            unique_sources == 2
            and SignalSource.TA in sources
            and SignalSource.NEWS in sources
            and SignalSource.ANOMALY not in sources
            and self.anomaly_agent is not None
        ):
            try:
                anomaly_result = await self.anomaly_agent.verify_signal(
                    ticker=ticker,
                    proposed_direction=direction.value,
                )
                if anomaly_result.is_vetoed:
                    return self._veto_decision(
                        ticker,
                        cycle_id,
                        signals,
                        reason=f"anomaly veto (opposing={anomaly_result.opposing_count})",
                    )

                confluence_mult = max(confluence_mult, anomaly_result.multiplier)
            except Exception as exc:
                logger.warning(
                    "Aggregator: anomaly verify failed",
                    extra={"ticker": ticker, "error": str(exc)},
                )

        total_weight = sum(self._signal_weight(s) for s in voting_sigs)
        avg_mag = sum(s.magnitude * self._signal_weight(s) for s in voting_sigs) / max(
            1.0, total_weight
        )
        combined_magnitude = min(
            1.0, avg_mag * confluence_mult * weight_factor * counter_bias_penalty
        )

        mtf_score: float | None = None
        mtf_trends: dict[str, int] | None = None
        if cfg.MTF_CONFLUENCE_ENABLED:
            try:
                from app.agents.ta_patterns.mtf_confluence import (
                    compute_mtf_trend,
                    mtf_confluence_score,
                    resample_ohlcv,
                )

                base_10m = df_10m if df_10m is not None else supercandles_df
                use_60m = df_60m
                use_daily = df_daily
                if use_60m is None and base_10m is not None:
                    use_60m = resample_ohlcv(base_10m, "60min")
                if use_daily is None and base_10m is not None:
                    use_daily = resample_ohlcv(base_10m, "1D")

                mtf_trends = compute_mtf_trend(
                    base_10m,
                    use_60m,
                    use_daily,
                    adx_min=cfg.MTF_CONFLUENCE_ADX_MIN,
                )
                mtf_score = mtf_confluence_score(direction.value, mtf_trends)

                if mtf_score < 0:
                    logger.info(
                        "MTF контр-тренд VETO",
                        extra={
                            "ticker": ticker,
                            "direction": direction.value,
                            "trends": mtf_trends,
                            "score": mtf_score,
                        },
                    )
                    return self._veto_decision(
                        ticker,
                        cycle_id,
                        signals,
                        reason=(f"MTF counter-trend (trends={mtf_trends}, score={mtf_score:.2f})"),
                    )

                combined_magnitude = min(1.0, combined_magnitude * mtf_score)
                logger.debug(
                    "MTF confluence applied",
                    extra={
                        "ticker": ticker,
                        "direction": direction.value,
                        "trends": mtf_trends,
                        "score": mtf_score,
                        "new_magnitude": round(combined_magnitude, 3),
                    },
                )
            except Exception as exc:
                logger.warning(
                    "MTF confluence error, allowing decision",
                    extra={"ticker": ticker, "error": str(exc)},
                )

        candidates = [
            s
            for s in voting_sigs
            if s.entry_level
            and s.stop_level
            and s.target_level
            and s.entry_level > 0
            and s.stop_level > 0
            and s.target_level > 0
        ]
        if candidates:
            best = max(candidates, key=lambda s: s.magnitude)
            entry = best.entry_level
            stop = best.stop_level
            target = best.target_level
            expected_rr = best.expected_rr

            try:
                from app.risk.sl_tp_rules import derive_sl_tp

                pattern_name = (
                    best.metadata.get("pattern") if isinstance(best.metadata, dict) else None
                ) or best.detector
                atr_val = float(best.atr or 0.0)
                if pattern_name and atr_val > 0 and entry > 0:
                    new_sl, new_tp = derive_sl_tp(
                        pattern=str(pattern_name),
                        direction=direction.value,
                        entry=float(entry),
                        atr=atr_val,
                        detector_stop=float(stop),
                        detector_target=float(target),
                        vol_ratio=None,
                    )
                    risk = abs(float(entry) - new_sl)
                    if risk > 0:
                        stop = new_sl
                        target = new_tp
                        expected_rr = abs(new_tp - float(entry)) / risk
            except Exception:
                pass
        else:
            entry = next((s.price for s in voting_sigs if s.price > 0), 0.0)
            stop = 0.0
            target = 0.0
            expected_rr = 0.0

        horizon_min = max(s.horizon_min for s in voting_sigs)

        rationale = self._build_rationale(direction, voting_sigs, mode, confluence_mult)
        if mtf_score is not None and mtf_trends is not None:
            rationale = f"{rationale} | mtf_score={mtf_score:.2f} trends={mtf_trends}"

        dominant_source = self._pick_dominant_source(voting_sigs)

        decision = Decision(
            decision_id=Decision.make_id(cycle_id, ticker, [s.signal_id for s in signals]),
            cycle_id=cycle_id,
            ticker=ticker,
            action=DecisionAction.EXECUTE,
            tier=DecisionTier.NONE,
            direction=direction,
            combined_magnitude=combined_magnitude,
            signals=signals,
            risk_check=RiskCheckResult.PASSED,
            trade_request=None,
            expected_holding_min=horizon_min,
            stop_loss=stop,
            take_profit=target,
            expected_rr=expected_rr,
            rationale=rationale,
            dominant_source=dominant_source,
        )

        if (
            cfg.MICROSTRUCTURE_GATES_ENABLED
            and self.microstructure_gates is not None
            and supercandles_df is not None
        ):
            try:
                gate = await self.microstructure_gates.check(
                    ticker=ticker,
                    direction=direction.value,
                    supercandles_df=supercandles_df,
                )

                if meta_context is not None:
                    meta_context.ofi = gate.ofi
                    meta_context.kyles_lambda = gate.kyles_lambda
                    meta_context.vpin = gate.vpin
                if gate.blocked:
                    decision.gate_reason = f"microstructure: {gate.reason}"
                    logger.info(
                        "Microstructure gate BLOCK",
                        extra={
                            "ticker": ticker,
                            "reason": gate.reason,
                            "ofi": round(gate.ofi, 3),
                            "vpin": round(gate.vpin, 3),
                        },
                    )
                    return self._no_trade_decision(
                        ticker,
                        cycle_id,
                        signals,
                        reason=f"microstructure: {gate.reason}",
                    )
                if gate.weakened:
                    combined_magnitude = min(
                        1.0,
                        combined_magnitude * gate.weakening_multiplier,
                    )
                    decision.combined_magnitude = combined_magnitude
                    decision.gate_reason = f"weakened: {gate.reason}"
                    logger.debug(
                        "Microstructure gate WEAKEN",
                        extra={
                            "ticker": ticker,
                            "reason": gate.reason,
                            "new_magnitude": round(combined_magnitude, 3),
                        },
                    )
            except Exception as exc:
                logger.warning(
                    "Microstructure gate error, allowing decision",
                    extra={"ticker": ticker, "error": str(exc)},
                )

        if (
            not defer_meta
            and cfg.META_ENABLED
            and self.meta_classifier is not None
            and meta_context is not None
        ):
            try:
                from app.agents.meta_classifier import adaptive_meta_min_proba

                meta_score = self.meta_classifier.score(decision, meta_context)
                decision.meta_score = meta_score
                meta_threshold = adaptive_meta_min_proba(meta_context.current_dd_pct)
                decision.meta_threshold = meta_threshold
                if meta_score < meta_threshold:
                    logger.debug(
                        "Meta-classifier rejected",
                        extra={
                            "ticker": ticker,
                            "meta_score": round(meta_score, 3),
                            "threshold": meta_threshold,
                            "dd_pct": round(float(meta_context.current_dd_pct), 4),
                        },
                    )
                    return self._no_trade_decision(
                        ticker,
                        cycle_id,
                        signals,
                        reason=f"meta_score={meta_score:.2f} < {meta_threshold:.2f}",
                    )
            except Exception as exc:
                logger.warning(
                    "Meta-classifier error, allowing decision",
                    extra={"ticker": ticker, "error": str(exc)},
                )

        logger.debug(
            "Aggregator decision",
            extra={
                "ticker": ticker,
                "direction": direction.value,
                "mode": mode,
                "confluence_mult": confluence_mult,
                "combined_magnitude": round(combined_magnitude, 3),
                "meta_score": round(decision.meta_score, 3)
                if decision.meta_score is not None
                else None,
                "n_signals": len(signals),
            },
        )

        return decision

    async def aggregate_batch(
        self,
        cycle_id: str,
        per_ticker_signals: dict[str, list[UnifiedSignal]],
        meta_contexts: dict[str, MetaContext] | None = None,
        supercandles_by_ticker: dict[str, Any] | None = None,
        df_10m_by_ticker: dict[str, Any] | None = None,
        df_60m_by_ticker: dict[str, Any] | None = None,
        df_daily_by_ticker: dict[str, Any] | None = None,
    ) -> list[Decision]:
        """Aggregate signals for many tickers and apply meta-classifier batched.

        Args:
            cycle_id: dispatcher cycle id
            per_ticker_signals: ticker → signals
            meta_contexts: ticker → MetaContext
            supercandles_by_ticker: ticker → supercandles df
            df_10m_by_ticker: ticker → 10-min df
            df_60m_by_ticker: ticker → 60-min df
            df_daily_by_ticker: ticker → daily df
        Returns:
            list[Decision]: aggregated decisions
        """
        meta_contexts = meta_contexts or {}
        supercandles_by_ticker = supercandles_by_ticker or {}
        df_10m_by_ticker = df_10m_by_ticker or {}
        df_60m_by_ticker = df_60m_by_ticker or {}
        df_daily_by_ticker = df_daily_by_ticker or {}

        decisions: list[Decision] = []
        for ticker, sigs in per_ticker_signals.items():
            try:
                dec = await self.aggregate(
                    ticker,
                    cycle_id,
                    sigs,
                    meta_context=meta_contexts.get(ticker),
                    supercandles_df=supercandles_by_ticker.get(ticker),
                    df_10m=df_10m_by_ticker.get(ticker),
                    df_60m=df_60m_by_ticker.get(ticker),
                    df_daily=df_daily_by_ticker.get(ticker),
                    defer_meta=True,
                )
            except Exception as exc:
                logger.error(
                    "Aggregator batch: per-ticker failed",
                    extra={"ticker": ticker, "error": str(exc)},
                )
                continue
            decisions.append(dec)

        if cfg.META_ENABLED and self.meta_classifier is not None and decisions:
            executable = [
                (i, d)
                for i, d in enumerate(decisions)
                if d.action == DecisionAction.EXECUTE and meta_contexts.get(d.ticker) is not None
            ]
            if executable:
                try:
                    from app.agents.meta_classifier import adaptive_meta_min_proba

                    decs = [d for _, d in executable]
                    ctxs = [meta_contexts[d.ticker] for _, d in executable]
                    scores = self.meta_classifier.score_batch(decs, ctxs)
                    for (idx, dec), score, ctx in zip(executable, scores, ctxs, strict=False):
                        dec.meta_score = float(score)
                        threshold = adaptive_meta_min_proba(ctx.current_dd_pct)
                        dec.meta_threshold = threshold
                        if score < threshold:
                            replaced = self._no_trade_decision(
                                dec.ticker,
                                cycle_id,
                                dec.signals,
                                reason=(f"meta_score={score:.2f} < {threshold:.2f}"),
                            )
                            replaced.meta_score = float(score)
                            replaced.meta_threshold = threshold
                            decisions[idx] = replaced
                except Exception as exc:
                    logger.warning(
                        "Meta-classifier batch error, allowing decisions",
                        extra={"error": str(exc), "n": len(executable)},
                    )

        return decisions

    async def _resolve_conflict(
        self,
        ticker: str,
        cycle_id: str,
        all_signals: list[UnifiedSignal],
        buy_sigs: list[UnifiedSignal],
        sell_sigs: list[UnifiedSignal],
    ) -> Decision:
        """Resolve BUY vs SELL conflict via anomaly verification.

        Args:
            ticker: instrument code
            cycle_id: dispatcher cycle id
            all_signals: every signal this cycle for ticker
            buy_sigs: BUY-side signals
            sell_sigs: SELL-side signals
        Returns:
            Decision: resolved decision
        """

        buy_has_anomaly = any(s.source == SignalSource.ANOMALY for s in buy_sigs)
        sell_has_anomaly = any(s.source == SignalSource.ANOMALY for s in sell_sigs)

        if buy_has_anomaly and not sell_has_anomaly:
            return self._veto_decision(
                ticker, cycle_id, all_signals, reason="anomaly on BUY side vetoes SELL"
            )
        if sell_has_anomaly and not buy_has_anomaly:
            return self._veto_decision(
                ticker, cycle_id, all_signals, reason="anomaly on SELL side vetoes BUY"
            )

        if self.anomaly_agent is not None:
            try:
                buy_check = await self.anomaly_agent.verify_signal(ticker, "BUY")
                sell_check = await self.anomaly_agent.verify_signal(ticker, "SELL")
                if buy_check.matching_count > sell_check.matching_count:
                    return self._make_winning_decision(
                        ticker,
                        cycle_id,
                        all_signals,
                        buy_sigs,
                        Direction.BUY,
                        anomaly_mult=buy_check.multiplier,
                    )
                elif sell_check.matching_count > buy_check.matching_count:
                    return self._make_winning_decision(
                        ticker,
                        cycle_id,
                        all_signals,
                        sell_sigs,
                        Direction.SELL,
                        anomaly_mult=sell_check.multiplier,
                    )

                return self._no_trade_decision(
                    ticker,
                    cycle_id,
                    all_signals,
                    reason="conflict unresolved (anomaly tie)",
                )
            except Exception as exc:
                logger.warning(
                    "Aggregator: conflict resolve failed",
                    extra={"ticker": ticker, "error": str(exc)},
                )

        return self._no_trade_decision(
            ticker,
            cycle_id,
            all_signals,
            reason=f"conflict (buy={len(buy_sigs)}, sell={len(sell_sigs)})",
        )

    def _make_winning_decision(
        self,
        ticker: str,
        cycle_id: str,
        all_signals: list[UnifiedSignal],
        winning_sigs: list[UnifiedSignal],
        direction: Direction,
        anomaly_mult: float,
    ) -> Decision:
        """Build a Decision for the winning side of a conflict.

        Args:
            ticker: instrument code
            cycle_id: dispatcher cycle id
            all_signals: every signal for the ticker
            winning_sigs: signals on the winning side
            direction: chosen direction
            anomaly_mult: anomaly multiplier
        Returns:
            Decision: aggregated decision
        """
        total_weight = sum(self._signal_weight(s) for s in winning_sigs)
        avg_mag = sum(s.magnitude * self._signal_weight(s) for s in winning_sigs) / max(
            1.0, total_weight
        )
        combined_magnitude = min(1.0, avg_mag * anomaly_mult)

        candidates = [
            s
            for s in winning_sigs
            if s.entry_level and s.stop_level and s.entry_level > 0 and s.stop_level > 0
        ]
        if candidates:
            best = max(candidates, key=lambda s: s.magnitude)
            stop = best.stop_level
            target = best.target_level
            expected_rr = best.expected_rr
        else:
            next((s.price for s in winning_sigs if s.price > 0), 0.0)
            stop = 0.0
            target = 0.0
            expected_rr = 0.0

        horizon_min = max(s.horizon_min for s in winning_sigs)

        return Decision(
            decision_id=Decision.make_id(cycle_id, ticker, [s.signal_id for s in all_signals]),
            cycle_id=cycle_id,
            ticker=ticker,
            action=DecisionAction.EXECUTE,
            tier=DecisionTier.NONE,
            direction=direction,
            combined_magnitude=combined_magnitude,
            signals=all_signals,
            risk_check=RiskCheckResult.PASSED,
            trade_request=None,
            expected_holding_min=horizon_min,
            stop_loss=stop,
            take_profit=target,
            expected_rr=expected_rr,
            rationale=f"Conflict resolved by anomaly (mult={anomaly_mult:.2f}); winning={direction.value}",
            dominant_source=self._pick_dominant_source(winning_sigs),
        )

    def _no_trade_decision(
        self,
        ticker: str,
        cycle_id: str,
        signals: list[UnifiedSignal],
        reason: str,
    ) -> Decision:
        """Build a NO_TRADE Decision.

        Args:
            ticker: instrument code
            cycle_id: cycle id
            signals: signals for the cycle
            reason: short rejection reason
        Returns:
            Decision: NO_TRADE decision
        """
        return Decision(
            decision_id=Decision.make_id(cycle_id, ticker, [s.signal_id for s in signals]),
            cycle_id=cycle_id,
            ticker=ticker,
            action=DecisionAction.NO_TRADE,
            tier=DecisionTier.NONE,
            direction=Direction.NEUTRAL,
            combined_magnitude=0.0,
            signals=signals,
            risk_check=RiskCheckResult.PASSED,
            trade_request=None,
            expected_holding_min=0,
            stop_loss=None,
            take_profit=None,
            expected_rr=0.0,
            rationale=f"NO_TRADE: {reason}",
        )

    def _veto_decision(
        self,
        ticker: str,
        cycle_id: str,
        signals: list[UnifiedSignal],
        reason: str,
    ) -> Decision:
        """Build a VETO Decision.

        Args:
            ticker: instrument code
            cycle_id: cycle id
            signals: signals for the cycle
            reason: short veto reason
        Returns:
            Decision: VETO decision
        """
        return Decision(
            decision_id=Decision.make_id(cycle_id, ticker, [s.signal_id for s in signals]),
            cycle_id=cycle_id,
            ticker=ticker,
            action=DecisionAction.VETO,
            tier=DecisionTier.NONE,
            direction=Direction.NEUTRAL,
            combined_magnitude=0.0,
            signals=signals,
            risk_check=RiskCheckResult.PASSED,
            trade_request=None,
            expected_holding_min=0,
            stop_loss=None,
            take_profit=None,
            expected_rr=0.0,
            rationale=f"VETO: {reason}",
        )

    @staticmethod
    def _build_rationale(
        direction: Direction,
        sigs: list[UnifiedSignal],
        mode: str,
        mult: float,
    ) -> str:
        """Build short rationale string.

        Args:
            direction: trade direction
            sigs: voting signals
            mode: confluence mode label
            mult: confluence multiplier
        Returns:
            str: human-readable rationale
        """
        sources = list({s.source.value for s in sigs})
        detectors = list({s.detector for s in sigs})[:3]
        return (
            f"{mode} {direction.value} | "
            f"sources={sources} | "
            f"detectors={detectors} | "
            f"mult={mult:.2f} | "
            f"n_sigs={len(sigs)}"
        )

    @staticmethod
    def _signal_weight(signal: UnifiedSignal) -> float:
        """Return weighting factor for a signal.

        Args:
            signal: signal to weight
        Returns:
            float: weight
        """
        if signal.source != SignalSource.NEWS:
            return 1.0
        tier = str(signal.metadata.get("tier", "B")).upper()
        return {"S": 1.35, "A": 1.18, "B": 1.0, "C": 0.90}.get(tier, 1.0)

    @classmethod
    def _pick_dominant_source(cls, voting_sigs: list[UnifiedSignal]) -> str | None:
        """Phase 27.5 — dominant source = highest weighted magnitude.

        Used by RiskManager to apply the per-strategy capital cap. In a
        standalone (single-source) case it returns that lone source; in a
        confluence case (e.g. TA + NEWS) the source whose
        magnitude × source-weight is largest wins. Falls back to the first
        signal's source when all weights collapse (defensive).

        Args:
            voting_sigs: signals that voted for the winning direction
        Returns:
            str | None: SignalSource value or None when input is empty.
        """
        if not voting_sigs:
            return None
        scored: dict[str, float] = {}
        for sig in voting_sigs:
            try:
                src = sig.source.value if hasattr(sig.source, "value") else str(sig.source)
            except Exception:
                src = "TA"
            weight = cls._signal_weight(sig)
            scored[src] = scored.get(src, 0.0) + float(sig.magnitude) * weight
        if not scored:
            return None
        return max(scored.items(), key=lambda kv: kv[1])[0]
