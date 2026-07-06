"""Расчёт размера позиций, 8 жёстких лимитов и circuit breakers."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import app.config as cfg
from app.dispatcher.signal import (
    Decision,
    DecisionAction,
    Direction,
    RiskCheckResult,
    TradeRequest,
)
from app.dispatcher.tier_classifier import tier_size_pct
from app.risk.circuit_breakers import get_circuit_breaker
from app.risk.position_book import SECTOR_MAP, get_position_book
from app.utils.logging import get_logger
from app.utils.sessions import is_trading_open

logger = get_logger(__name__)

KELLY_FRACTION = 0.25
HARD_CAP_PCT = cfg.MAX_POSITION_PCT
MAX_OPEN_POSITIONS = cfg.MAX_OPEN_POSITIONS
MAX_SECTOR_PCT = 0.30
MIN_CASH_RESERVE_PCT = 0.05
MIN_HOURS_BETWEEN_ENTRIES = float(
    os.getenv(
        "ENTRY_COOLDOWN_HOURS",
        "1.0" if cfg.STRICT_MODE else "0.20",
    )
)

VOL_HIGH_RATIO = 1.5
VOL_LOW_RATIO = 0.7
VOL_HIGH_MULT = 0.7
VOL_LOW_MULT = 1.2
VOL_CACHE_TTL_SEC = 24 * 60 * 60
VOL_LOOKBACK_DAYS = 60

RISK_PARITY_REF_ATR_PCT = 0.015
RISK_PARITY_ATR_PCT_FLOOR = 0.005
RISK_PARITY_MAX_BOOST = 2.0

def apply_volatility_normalization(
    target_notional: float,
    ticker: str,
    atr_pct: float,
) -> float:
    """Scale target notional inversely with the ticker's per-bar ATR%.

    Equalises ex-ante RUB risk across the book by giving low-volatility names
    (small ATR%) a larger position and damping high-volatility ones. Operates
    on a unitless ratio (ref / observed), clipped at ``RISK_PARITY_MAX_BOOST``
    so a near-zero ATR can never blow past hard caps downstream.

    Args:
        target_notional: pre-normalisation RUB notional from sizing pipeline
        ticker: instrument code (unused today, accepted for future per-ticker
            references such as sector-specific anchors)
        atr_pct: latest ATR expressed as a fraction of price (e.g. ``0.018``).

    Returns:
        float: vol-adjusted RUB notional. Returns ``target_notional`` unchanged
            when ``atr_pct`` is non-positive (no signal) so callers can safely
            wire it in front of a hard-cap clip without losing trades.
    """
    _ = ticker
    try:
        ap = float(atr_pct)
    except (TypeError, ValueError):
        return target_notional
    if not math.isfinite(ap) or ap <= 0:
        return target_notional
    ap_eff = max(ap, RISK_PARITY_ATR_PCT_FLOOR)
    ratio = RISK_PARITY_REF_ATR_PCT / ap_eff
    ratio = min(ratio, RISK_PARITY_MAX_BOOST)
    return float(target_notional) * ratio

@dataclass
class RiskDecision:
    """Outcome of RiskManager.evaluate()."""

    result: RiskCheckResult
    reason: str = ""
    trade_request: TradeRequest | None = None
    quantity: int = 0
    notional_rub: float = 0.0

class RiskManager:
    """Sizing + limits + circuit breakers."""

    def __init__(
        self,
        deposit_total: float = 1_000_000.0,
        bot_name: str | None = None,
    ) -> None:
        """Init."""
        self.deposit_total = deposit_total
        self.bot_name = bot_name or cfg.ARENAGO_BOT_NAME
        self.book = get_position_book()
        self.cb = get_circuit_breaker()
        self._vol_median_cache: dict[str, tuple[float, float]] = {}
        self._vol_cache_loader: Any | None = None

    async def evaluate(self, decision: Decision) -> RiskDecision:
        """Apply checks and return RiskDecision.

        Args:
            decision: candidate decision to evaluate
        Returns:
            RiskDecision: PASSED or REJECTED_* outcome
        """
        ticker = decision.ticker
        direction = decision.direction

        if decision.action != DecisionAction.EXECUTE:
            return RiskDecision(
                result=RiskCheckResult.PASSED,
                reason="non-execute decision, no risk check needed",
            )

        sanity_reason = self._sanity_check(decision)
        if sanity_reason:
            return RiskDecision(
                result=RiskCheckResult.REJECTED_HARD_CAP,
                reason=f"sanity: {sanity_reason}",
            )

        existing_pos_for_gate = self.book.get_position(ticker)
        is_exit_order_for_gate = bool(
            existing_pos_for_gate is not None
            and direction in (Direction.BUY, Direction.SELL)
            and (
                (existing_pos_for_gate.quantity > 0 and direction == Direction.SELL)
                or (existing_pos_for_gate.quantity < 0 and direction == Direction.BUY)
            )
        )

        if not is_exit_order_for_gate:
            try:
                from app.risk.broker_health_monitor import get_broker_health_monitor

                if get_broker_health_monitor().is_safe_mode():
                    return RiskDecision(
                        result=RiskCheckResult.REJECTED_CIRCUIT_BREAKER,
                        reason="REJECTED_SAFE_MODE: broker unreachable, entries blocked",
                    )
            except Exception:  # pragma: no cover — defensive: never crash risk eval
                pass

        if not is_exit_order_for_gate:
            try:
                from app.risk.equity_floor import get_equity_floor

                floor = get_equity_floor()
                current_equity = float(self.book.total_equity())
                ok, floor_reason = floor.check(current_equity)
                if not ok:
                    return RiskDecision(
                        result=RiskCheckResult.REJECTED_CIRCUIT_BREAKER,
                        reason=floor_reason,
                    )
            except Exception:  # pragma: no cover
                pass

        self._apply_direction_bias(decision)
        self._apply_ticker_session_bias(decision)
        self._apply_magnitude_nonlinear_scaling(decision)

        blocked, cb_reason = self.cb.should_block_new_trades()
        if blocked:
            return RiskDecision(
                result=RiskCheckResult.REJECTED_CIRCUIT_BREAKER,
                reason=cb_reason,
            )

        if not is_trading_open():
            return RiskDecision(
                result=RiskCheckResult.REJECTED_MARKET_CLOSED,
                reason="MOEX trading closed",
            )

        regime = self._compute_current_regime()

        existing_pos_for_regime = self.book.get_position(ticker)
        is_closing_now = existing_pos_for_regime is not None and (
            (existing_pos_for_regime.quantity > 0 and direction == Direction.SELL)
            or (existing_pos_for_regime.quantity < 0 and direction == Direction.BUY)
        )
        if not regime.can_open_new and not is_closing_now:
            return RiskDecision(
                result=RiskCheckResult.REJECTED_CIRCUIT_BREAKER,
                reason=f"adaptive_regime={regime.name}: {regime.reason}",
            )

        if self.book.n_open_positions >= MAX_OPEN_POSITIONS and not self.book.has_position(ticker):
            return RiskDecision(
                result=RiskCheckResult.REJECTED_MAX_POSITIONS,
                reason=f"open positions {self.book.n_open_positions}/{MAX_OPEN_POSITIONS}",
            )

        sector = SECTOR_MAP.get(ticker, "other")
        sector_exp = self.book.sector_exposure_pct(sector)

        added_pct = tier_size_pct(decision.tier)
        try:
            sector_cap_pct = float(cfg.get_sector_cap_pct(sector))
        except (AttributeError, TypeError, ValueError):
            sector_cap_pct = MAX_SECTOR_PCT
        if sector_exp + added_pct > sector_cap_pct:
            return RiskDecision(
                result=RiskCheckResult.REJECTED_SECTOR_LIMIT,
                reason=f"sector {sector} exposure would be {sector_exp + added_pct:.1%} > {sector_cap_pct:.0%}",
            )

        if getattr(cfg, "SECTOR_CONCENTRATION_HAIRCUT_ENABLED", False):
            same_sector_count = sum(
                1
                for t, pos in self.book.positions.items()
                if pos.sector == sector and t != ticker.upper()
            )
            if same_sector_count >= int(
                getattr(cfg, "SECTOR_CONCENTRATION_HAIRCUT_THRESHOLD", 3)
            ):
                haircut = float(getattr(cfg, "SECTOR_CONCENTRATION_HAIRCUT_MULT", 0.5))
                try:
                    new_mag = float(decision.combined_magnitude) * haircut
                    decision.combined_magnitude = max(0.0, min(1.0, new_mag))
                    logger.info(
                        "Sector concentration haircut applied",
                        extra={
                            "ticker": ticker,
                            "sector": sector,
                            "same_sector_count": same_sector_count,
                            "haircut": haircut,
                            "new_magnitude": decision.combined_magnitude,
                        },
                    )
                except (TypeError, ValueError):
                    pass

        if (
            getattr(cfg, "DIRECTION_CONCENTRATION_CAP_ENABLED", False)
            and not is_exit_order_for_gate
            and direction in (Direction.BUY, Direction.SELL)
        ):
            threshold = int(getattr(cfg, "DIRECTION_CONCENTRATION_THRESHOLD", 5))
            stricter_meta_floor = float(
                getattr(cfg, "DIRECTION_CONCENTRATION_META_MIN", 0.55)
            )
            n_long = sum(1 for p in self.book.positions.values() if p.quantity > 0)
            n_short = sum(1 for p in self.book.positions.values() if p.quantity < 0)
            count_same_side = n_short if direction == Direction.SELL else n_long
            if count_same_side >= threshold:
                meta = decision.meta_score
                if meta is None or float(meta) < stricter_meta_floor:
                    return RiskDecision(
                        result=RiskCheckResult.REJECTED_SECTOR_LIMIT,
                        reason=(
                            f"DIRECTION_CONCENTRATION: {count_same_side} {direction.value} "
                            f"positions already open; meta_score={meta} < "
                            f"{stricter_meta_floor:.2f} (strict floor)"
                        ),
                    )

        existing_pos = self.book.get_position(ticker)
        is_closing_long = (
            existing_pos is not None and existing_pos.quantity > 0 and direction == Direction.SELL
        )
        is_closing_short = (
            existing_pos is not None and existing_pos.quantity < 0 and direction == Direction.BUY
        )
        is_exit_order = is_closing_long or is_closing_short

        if not is_exit_order:
            cash_pct = self.book.cash_reserve_pct()
            if cash_pct - added_pct < MIN_CASH_RESERVE_PCT:
                return RiskDecision(
                    result=RiskCheckResult.REJECTED_CASH_RESERVE,
                    reason=f"cash would drop below {MIN_CASH_RESERVE_PCT:.0%} reserve",
                )

            if direction in (Direction.BUY, Direction.SELL):
                secs_since = self.book.seconds_since_last_entry(ticker)
                if secs_since < MIN_HOURS_BETWEEN_ENTRIES * 3600:
                    return RiskDecision(
                        result=RiskCheckResult.REJECTED_HOLDING_PERIOD,
                        reason=f"last entry {secs_since / 60:.1f}min ago (min {MIN_HOURS_BETWEEN_ENTRIES * 60:.0f}min)",
                    )

        price = self._signal_price(decision)
        if price <= 0:
            return RiskDecision(
                result=RiskCheckResult.REJECTED_HARD_CAP,
                reason="no valid price in signal",
            )

        atr = self._signal_atr(decision)
        vol_mult = await self._volatility_adjustment(ticker, atr)
        qty = self._compute_quantity(decision, price, atr, vol_mult=vol_mult)
        if qty <= 0:
            return RiskDecision(
                result=RiskCheckResult.REJECTED_HARD_CAP,
                reason="computed quantity 0 (likely hard cap reached)",
            )

        notional = qty * price

        if notional > self.deposit_total * HARD_CAP_PCT:
            qty = int((self.deposit_total * HARD_CAP_PCT) / price)
            notional = qty * price
            if qty <= 0:
                return RiskDecision(
                    result=RiskCheckResult.REJECTED_HARD_CAP,
                    reason=f"price {price} > hard cap notional",
                )

        if not is_exit_order:
            cap_violation = self._strategy_cap_violation(decision, notional)
            if cap_violation is not None:
                return RiskDecision(
                    result=RiskCheckResult.REJECTED_STRATEGY_ALLOCATION,
                    reason=cap_violation,
                )

        trade_request = TradeRequest(
            decision_id=decision.decision_id,
            ticker=ticker,
            direction=direction,
            quantity=qty,
            bot=self.bot_name,
            price_at_signal=price,
        )

        return RiskDecision(
            result=RiskCheckResult.PASSED,
            reason="all checks passed",
            trade_request=trade_request,
            quantity=qty,
            notional_rub=notional,
        )

    def _compute_quantity(
        self,
        decision: Decision,
        price: float,
        atr: float,
        *,
        vol_mult: float = 1.0,
    ) -> int:
        """Compute lots via tier × Kelly × volatility × streak × drawdown × regime.

        Args:
            decision: Decision being sized
            price: reference price
            atr: ATR used in vol target
            vol_mult: volatility regime multiplier
        Returns:
            int: lot count (>=0)
        """
        if price <= 0:
            return 0

        wr_tier_pct = self._wr70_size_pct(decision)
        tier_pct = wr_tier_pct if wr_tier_pct is not None else tier_size_pct(decision.tier)
        target_notional = self.deposit_total * tier_pct

        streak_mult = self.cb.state.sizing_multiplier
        target_notional *= streak_mult

        dd_mult = self.cb.state.drawdown_kelly_multiplier

        regime_mult = self._regime_size_multiplier()

        adaptive_mult = self._compute_current_regime().size_multiplier

        target_notional *= dd_mult * regime_mult * vol_mult * adaptive_mult

        risk_per_trade_rub = self.deposit_total * cfg.RISK_PER_TRADE
        vol_target_notional = risk_per_trade_rub * (price / atr) if atr > 0 else target_notional
        vol_target_notional *= dd_mult * regime_mult * vol_mult * adaptive_mult

        p = 0.50 + 0.40 * decision.combined_magnitude
        R = max(0.5, decision.expected_rr)
        kelly_f = max(0.0, (p * R - (1 - p)) / R) * KELLY_FRACTION
        kelly_notional = kelly_f * self.book.cash_balance
        kelly_notional *= dd_mult * regime_mult * vol_mult * adaptive_mult

        final_notional = min(target_notional, vol_target_notional, kelly_notional)

        if getattr(cfg, "RISK_PARITY_VOL_NORM", True):
            atr_pct = (atr / price) if (price > 0 and atr > 0) else 0.0
            if atr_pct > 0:
                final_notional = apply_volatility_normalization(
                    final_notional,
                    decision.ticker,
                    atr_pct,
                )

        final_notional = min(final_notional, self.deposit_total * HARD_CAP_PCT)

        max_cum_pct = float(
            getattr(cfg, "MAX_TICKER_NOTIONAL_PCT_CUMULATIVE", 0.08)
        )
        if max_cum_pct > 0:
            try:
                existing_pct = self.book.notional_pct_for_ticker(decision.ticker)
            except Exception:
                existing_pct = 0.0
            room_pct = max(0.0, max_cum_pct - existing_pct)
            final_notional = min(final_notional, room_pct * self.deposit_total)

        min_notional = float(getattr(cfg, "MIN_NOTIONAL_RUB", 0.0) or 0.0)
        if min_notional > 0 and final_notional < min_notional:
            hard_cap_rub = self.deposit_total * HARD_CAP_PCT
            final_notional = min(min_notional, hard_cap_rub)

        qty = int(final_notional / price) if price > 0 else 0
        return max(0, qty)

    def _compute_current_regime(self):
        """Compute current RiskRegime from CircuitBreaker state and session.

        Returns:
            RiskRegime: NORMAL / CAUTIOUS / DEFENSIVE / CRISIS
        """
        from app.risk.adaptive_regime import compute_risk_regime
        from app.utils.sessions import is_trading_open

        cb_state = self.cb.state
        deposit = max(1.0, float(self.deposit_total))
        daily_pnl_pct = float(cb_state.daily_pnl_rub) / deposit
        seconds_until_close = None
        if not is_trading_open():
            seconds_until_close = None
        return compute_risk_regime(
            current_drawdown_from_peak_pct=float(cb_state.current_drawdown_pct),
            losing_streak=int(cb_state.losing_streak),
            daily_pnl_pct=daily_pnl_pct,
            seconds_until_close=seconds_until_close,
        )

    async def _volatility_adjustment(self, ticker: str, current_atr: float) -> float:
        """Return size multiplier based on current_atr vs 60-day median.

        Args:
            ticker: instrument code
            current_atr: latest ATR value
        Returns:
            float: multiplicative size adjustment (1.0 if unknown)
        """
        if current_atr <= 0 or not ticker:
            return 1.0
        median_atr = await self._get_median_daily_atr(ticker)
        if median_atr is None or median_atr <= 0:
            return 1.0
        ratio = current_atr / median_atr
        if ratio > VOL_HIGH_RATIO:
            mult = VOL_HIGH_MULT
        elif ratio < VOL_LOW_RATIO:
            mult = VOL_LOW_MULT
        else:
            mult = 1.0
        logger.debug(
            "vol_adjustment",
            extra={
                "ticker": ticker,
                "current_atr": current_atr,
                "median_atr_60d": median_atr,
                "ratio": ratio,
                "mult": mult,
            },
        )
        return mult

    async def _get_median_daily_atr(self, ticker: str) -> float | None:
        """Return cached 60-day daily-ATR median for ticker or fetch fresh.

        Args:
            ticker: instrument code
        Returns:
            float | None: median ATR or None on failure
        """
        key = ticker.upper()
        now = time.time()
        cached = self._vol_median_cache.get(key)
        if cached is not None:
            median, fetched_at = cached
            if now - fetched_at < VOL_CACHE_TTL_SEC:
                return median

        loader = self._vol_cache_loader
        try:
            if loader is not None:
                median = await loader(key, VOL_LOOKBACK_DAYS)
            else:
                median = await self._fetch_daily_atr_median(key, VOL_LOOKBACK_DAYS)
        except Exception as exc:
            logger.debug(
                "vol median fetch failed",
                extra={"ticker": key, "error": str(exc)},
            )
            return None

        if median is None or median <= 0:
            return None
        self._vol_median_cache[key] = (median, now)
        return median

    @staticmethod
    async def _fetch_daily_atr_median(ticker: str, lookback_days: int) -> float | None:
        """Compute median True Range from daily candles via ISS.

        Args:
            ticker: instrument code
            lookback_days: history window
        Returns:
            float | None: median TR or None if unavailable
        """
        try:
            from app.data.iss_client import INTERVAL_24H, get_iss_client
        except ImportError:
            return None
        try:
            client = get_iss_client()
            if not getattr(client, "_started", False):
                return None
            till = datetime.now(tz=UTC)
            from_dt = till - timedelta(days=int(lookback_days * 1.5) + 5)
            df = await client.get_candles(
                ticker=ticker,
                interval=INTERVAL_24H,
                from_dt=from_dt,
                till_dt=till,
            )
        except Exception:
            return None
        if df is None or len(df) < 5:
            return None
        try:
            highs = [float(r) for r in df["high"].tolist()]
            lows = [float(r) for r in df["low"].tolist()]
            closes = [float(r) for r in df["close"].tolist()]
        except Exception:
            return None
        if len(closes) < 5:
            return None
        if len(closes) > lookback_days:
            highs = highs[-lookback_days:]
            lows = lows[-lookback_days:]
            closes = closes[-lookback_days:]
        trs: list[float] = []
        prev_close = closes[0]
        for i in range(1, len(closes)):
            h, l, c = highs[i], lows[i], closes[i]
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            if tr > 0:
                trs.append(tr)
            prev_close = c
        if not trs:
            return None
        trs.sort()
        mid = len(trs) // 2
        if len(trs) % 2 == 1:
            return trs[mid]
        return 0.5 * (trs[mid - 1] + trs[mid])

    @staticmethod
    def _regime_size_multiplier() -> float:
        """Return HMM regime size multiplier; 0.5 if HMM not loaded.

        Returns:
            float: regime multiplier
        """
        try:
            from app.agents.hmm_regime import get_hmm_detector

            return float(get_hmm_detector().regime_size_multiplier())
        except Exception:
            return 0.5

    @staticmethod
    def _sanity_check(decision: Decision) -> str:
        """Check decision numeric fields for NaN/inf/negative.

        Args:
            decision: decision to inspect
        Returns:
            str: empty if OK, else short violation reason
        """
        for sig in decision.signals:
            for name, val in (
                ("price", sig.price),
                ("atr", sig.atr),
                ("entry_level", sig.entry_level if sig.entry_level is not None else 1.0),
                ("stop_level", sig.stop_level if sig.stop_level is not None else 1.0),
                ("target_level", sig.target_level if sig.target_level is not None else 1.0),
                ("magnitude", sig.magnitude),
            ):
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    return f"{sig.detector}.{name}=not-a-number"
                if not math.isfinite(fval):
                    return f"{sig.detector}.{name}={fval}"
                if name in ("price", "atr") and fval < 0:
                    return f"{sig.detector}.{name}<0"
        try:
            cm = float(decision.combined_magnitude)
        except (TypeError, ValueError):
            return "combined_magnitude=not-a-number"
        if not math.isfinite(cm):
            return f"combined_magnitude={cm}"
        try:
            rr = float(decision.expected_rr)
        except (TypeError, ValueError):
            return "expected_rr=not-a-number"
        if not math.isfinite(rr):
            return f"expected_rr={rr}"
        return ""

    @staticmethod
    def _wr70_size_pct(decision: Decision) -> float | None:
        """Look up WR-driven tier sizing pct for this decision.

        Prefers WR_90 over WR_70 when both have a match (a v0.19.0 combo
        proven at WR ≥ 90% should never be down-sized to the older 70% tier).

        Args:
            decision: decision being sized
        Returns:
            float | None: largest matching WR tier pct or None
        """
        best: float | None = None
        for s in decision.signals:
            detector = getattr(s, "detector", None)
            if not detector:
                continue
            pct90 = cfg.wr90_tier_size_pct(decision.ticker, detector)
            pct70 = cfg.wr70_tier_size_pct(decision.ticker, detector)
            for pct in (pct90, pct70):
                if pct is None:
                    continue
                if best is None or pct > best:
                    best = pct
        return best

    @staticmethod
    def _apply_direction_bias(decision: Decision) -> None:
        """v0.19.0 — softly bias decision magnitude along per-ticker direction.

        Reads PER_TICKER_DIRECTION_BIAS from cfg; if the ticker has a
        STRONG/VERY_STRONG bias, multiply combined_magnitude by:
          - PER_TICKER_BIAS_ALIGN_MULT   when signal aligns with bias
          - PER_TICKER_BIAS_COUNTER_MULT when signal opposes bias
        No-op when bias is None, MIXED, or strength is WEAK. The multiplier
        is bounded to [0, 1] post-application so downstream consumers still
        see a valid magnitude.

        Args:
            decision: decision to mutate in-place
        """
        bias = cfg.get_ticker_bias(decision.ticker)
        if bias is None:
            return
        try:
            sig_dir = (
                decision.direction.value
                if hasattr(decision.direction, "value")
                else str(decision.direction)
            )
        except Exception:
            return
        if sig_dir == bias:
            mult = cfg.PER_TICKER_BIAS_ALIGN_MULT
        elif sig_dir in ("BUY", "SELL"):
            mult = cfg.PER_TICKER_BIAS_COUNTER_MULT
        else:
            return
        try:
            new_mag = float(decision.combined_magnitude) * float(mult)
            decision.combined_magnitude = max(0.0, min(1.0, new_mag))
        except (TypeError, ValueError):
            return

    @staticmethod
    def _apply_ticker_session_bias(decision: Decision) -> None:
        """v1.0.2 — soft per-(ticker, session) magnitude multiplier.

        Looks up ``cfg.PER_TICKER_SESSION_BIAS[(ticker, current_session)]``
        and multiplies ``decision.combined_magnitude`` by that factor when
        the toggle ``PER_TICKER_SESSION_BIAS_ENABLED`` is True. No-op when
        the cell is missing (returns 1.0). The result is clipped to
        ``[0, 1]`` so downstream consumers keep their invariants. Applies
        AFTER ``_apply_direction_bias`` so the two stack multiplicatively.

        Args:
            decision: decision to mutate in-place
        """
        try:
            from app.utils.session_profile import current_session
        except Exception:
            return
        if not getattr(cfg, "PER_TICKER_SESSION_BIAS_ENABLED", False):
            return
        try:
            label = current_session()
            session = label.value if hasattr(label, "value") else str(label)
            mult = cfg.get_ticker_session_mult(decision.ticker, session)
        except Exception:
            return
        if mult == 1.0:
            return
        try:
            new_mag = float(decision.combined_magnitude) * float(mult)
            decision.combined_magnitude = max(0.0, min(1.0, new_mag))
        except (TypeError, ValueError):
            return

    @staticmethod
    def _apply_magnitude_nonlinear_scaling(decision: Decision) -> None:
        """v1.0.6 — Non-linear remap of ``combined_magnitude`` for sizing.

        Applies a piece-wise power-law-style transform AFTER direction and
        session bias have already adjusted ``decision.combined_magnitude``:

        - weak signals (mag < 0.3) → ``mag × 0.7`` (further damp)
        - mid signals (0.3 ≤ mag < 0.7) → unchanged
        - strong signals (mag ≥ 0.7) → ``mag × 1.15`` (light reward)

        The result is clipped to ``[0, 1]`` so downstream callers
        (``_compute_quantity``, Kelly calc) keep their invariants. The
        eventual notional is still bounded by ``MAX_POSITION_PCT`` so a
        boosted strong signal cannot legally exceed the hard cap.

        No-op when ``cfg.MAGNITUDE_NONLINEAR_SCALING`` is False.

        Args:
            decision: decision to mutate in-place.
        """
        if not getattr(cfg, "MAGNITUDE_NONLINEAR_SCALING", False):
            return
        try:
            mag = float(decision.combined_magnitude)
        except (TypeError, ValueError):
            return
        if not math.isfinite(mag) or mag <= 0:
            return
        if mag < 0.3:
            scaled = mag * 0.7
        elif mag < 0.7:
            scaled = mag
        else:
            scaled = mag * 1.15
        decision.combined_magnitude = max(0.0, min(1.0, scaled))

    def _strategy_cap_violation(
        self,
        decision: Decision,
        target_notional: float,
    ) -> str | None:
        """Return rejection reason if strategy cap would be breached.

        Phase 27.5: every adapter has a per-strategy share of equity that
        protects the book from a single source flooding it during a bad
        regime. The dominant source is taken from `decision.dominant_source`
        (filled by the aggregator) or, lacking that, the source of the first
        signal — which works for the standalone case where there is only one
        contributing adapter.

        Args:
            decision: candidate decision under evaluation
            target_notional: post-Kelly notional in RUB
        Returns:
            str | None: rejection reason or None when the trade fits.
        """
        if target_notional <= 0:
            return None
        dominant = (decision.dominant_source or "").upper()
        if not dominant and decision.signals:
            first_src = decision.signals[0].source
            dominant = first_src.value if hasattr(first_src, "value") else str(first_src)
        dominant = dominant.upper() if dominant else "TA"
        try:
            allocation = cfg.get_strategy_allocation(dominant)
        except AttributeError:
            allocation = float(cfg.STRATEGY_CAPITAL_ALLOCATION.get(dominant, 0.10))
        equity = max(self.deposit_total, self.book.total_equity())
        strategy_max = allocation * equity
        try:
            strategy_used = self.book.exposure_by_source(dominant)
        except AttributeError:
            return None
        if strategy_used + target_notional > strategy_max:
            return (
                f"strategy {dominant} cap "
                f"{(strategy_used + target_notional) / max(1.0, equity):.1%} > "
                f"{allocation:.0%} of equity"
            )
        return None

    @staticmethod
    def _signal_price(decision: Decision) -> float:
        """Pick representative price from signals (prefer entry_level).

        Args:
            decision: decision to inspect
        Returns:
            float: chosen price or 0.0
        """
        for s in decision.signals:
            if s.entry_level and s.entry_level > 0:
                return float(s.entry_level)
        for s in decision.signals:
            if s.price > 0:
                return float(s.price)
        return 0.0

    @staticmethod
    def _signal_atr(decision: Decision) -> float:
        """Return first positive ATR from signals.

        Args:
            decision: decision to inspect
        Returns:
            float: ATR or 0.0
        """
        for s in decision.signals:
            if s.atr > 0:
                return float(s.atr)
        return 0.0

_risk_manager: RiskManager | None = None

def get_risk_manager() -> RiskManager:
    """Return process-wide RiskManager singleton.

    Returns:
        RiskManager: shared instance
    """
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager
