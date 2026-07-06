"""Парный трейдинг на коинтегрированных парах."""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import app.config as cfg
from app.agents.base import BaseAdapter
from app.data.candle_store import get_candle_store
from app.dispatcher.signal import Direction, SignalSource, UnifiedSignal
from app.utils.logging import get_logger, get_trace_id

logger = get_logger(__name__)

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import statsmodels.api as sm  # type: ignore
    from statsmodels.tsa.stattools import adfuller  # type: ignore

    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

PAIR_STATE_FILE = cfg.DATA_DIR / "models" / "pair_state.json"
PAIR_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_TARGET_RUB = 5000.0

class PairState:
    """Lightweight container for one pair's regression state."""

    __slots__ = (
        "ticker_a",
        "ticker_b",
        "alpha",
        "beta",
        "mu",
        "sigma",
        "adf_pvalue",
        "qualified",
        "last_refit_iso",
        "last_z_values",
        "last_entry_iso",
        "adf_history",
        "ttl_disqualified",
        "ttl_reason",
        "open_entry_bar_iso",
        "open_entry_direction",
    )

    def __init__(
        self,
        ticker_a: str,
        ticker_b: str,
        alpha: float = 0.0,
        beta: float = 0.0,
        mu: float = 0.0,
        sigma: float = 1.0,
        adf_pvalue: float = 1.0,
        qualified: bool = False,
        last_refit_iso: str = "",
        last_z_values: list[float] | None = None,
        last_entry_iso: str = "",
        adf_history: list[dict[str, Any]] | None = None,
        ttl_disqualified: bool = False,
        ttl_reason: str = "",
        open_entry_bar_iso: str = "",
        open_entry_direction: int = 0,
    ) -> None:
        """Init."""
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.sigma = sigma
        self.adf_pvalue = adf_pvalue
        self.qualified = qualified
        self.last_refit_iso = last_refit_iso
        self.last_z_values = last_z_values or []
        self.last_entry_iso = last_entry_iso
        self.adf_history = adf_history or []
        self.ttl_disqualified = ttl_disqualified
        self.ttl_reason = ttl_reason
        self.open_entry_bar_iso = open_entry_bar_iso
        self.open_entry_direction = open_entry_direction

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "ticker_a": self.ticker_a,
            "ticker_b": self.ticker_b,
            "alpha": self.alpha,
            "beta": self.beta,
            "mu": self.mu,
            "sigma": self.sigma,
            "adf_pvalue": self.adf_pvalue,
            "qualified": self.qualified,
            "last_refit_iso": self.last_refit_iso,
            "last_z_values": self.last_z_values[-10:],
            "last_entry_iso": self.last_entry_iso,
            "adf_history": self.adf_history[-10:],
            "ttl_disqualified": self.ttl_disqualified,
            "ttl_reason": self.ttl_reason,
            "open_entry_bar_iso": self.open_entry_bar_iso,
            "open_entry_direction": self.open_entry_direction,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PairState:
        """From dict."""
        allowed = {
            "ticker_a",
            "ticker_b",
            "alpha",
            "beta",
            "mu",
            "sigma",
            "adf_pvalue",
            "qualified",
            "last_refit_iso",
            "last_z_values",
            "last_entry_iso",
            "adf_history",
            "ttl_disqualified",
            "ttl_reason",
            "open_entry_bar_iso",
            "open_entry_direction",
        }
        clean = {k: v for k, v in d.items() if k in allowed}
        return cls(**clean)

    def key(self) -> str:
        """Key."""
        return f"{self.ticker_a}_{self.ticker_b}"

class PairTrader(BaseAdapter):
    """6-pair cointegration trader."""

    name = "PAIR"

    def __init__(
        self,
        pairs: list[tuple[str, str]] | None = None,
        z_entry: float = 1.5,
        z_exit: float = 0.3,
        z_stop: float = 3.5,
        lookback_days: int = 60,
        cooldown_hours: int = 24,
    ) -> None:
        """Init."""
        super().__init__()
        self.pairs = pairs or list(cfg.COINTEGRATED_PAIRS)
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.lookback_days = lookback_days
        self.cooldown_hours = cooldown_hours
        self.candle_store = get_candle_store()
        self.state: dict[str, PairState] = {}

        self._poll_count = 0
        self._signal_count = 0

        self._seen_entries: set[tuple[str, int]] = set()

    async def startup(self) -> None:
        """Startup."""
        if not _HAS_PANDAS or not _HAS_STATSMODELS:
            logger.error("PairTrader: pandas/statsmodels not installed")
            return

        self._load_state()
        if not self.state:
            for a, b in self.pairs:
                ps = PairState(ticker_a=a, ticker_b=b)
                self.state[ps.key()] = ps
            self._save_state()

        self._started = True
        n_qual = sum(1 for ps in self.state.values() if ps.qualified)
        logger.info(
            "PairTrader started",
            extra={
                "pairs_total": len(self.pairs),
                "pairs_qualified": n_qual,
                "z_entry": self.z_entry,
                "z_exit": self.z_exit,
            },
        )

    async def shutdown(self) -> None:
        """Shutdown."""
        self._save_state()
        self._started = False
        logger.info("PairTrader stopped", extra={"stats": self.stats})

    def _load_state(self) -> None:
        """Load state."""
        if not PAIR_STATE_FILE.exists():
            return
        try:
            with open(PAIR_STATE_FILE) as f:
                data = json.load(f)
            for key, d in data.items():
                self.state[key] = PairState.from_dict(d)
        except Exception as exc:
            logger.error("PairTrader: state load failed", extra={"error": str(exc)})

    def _save_state(self) -> None:
        """Save state."""
        try:
            data = {key: ps.to_dict() for key, ps in self.state.items()}
            with open(PAIR_STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.error("PairTrader: state save failed", extra={"error": str(exc)})

    async def refit_all_pairs(self, daily_candles: dict[str, pd.DataFrame] | None = None) -> int:
        """Refit β + ADF for all pairs. Returns: int (count of qualified pairs)."""
        if not _HAS_PANDAS or not _HAS_STATSMODELS:
            return 0

        if daily_candles is None:
            daily_candles = await self._fetch_daily_candles()

        now_iso = datetime.now(tz=UTC).isoformat()
        qualified_count = 0

        for a, b in self.pairs:
            key = f"{a}_{b}"
            ps = self.state.get(key) or PairState(ticker_a=a, ticker_b=b)

            df_a = daily_candles.get(a)
            df_b = daily_candles.get(b)

            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                logger.warning(
                    "PairTrader refit: missing data",
                    extra={
                        "pair": key,
                        "a_rows": 0 if df_a is None else len(df_a),
                        "b_rows": 0 if df_b is None else len(df_b),
                    },
                )
                ps.qualified = False
                self.state[key] = ps
                continue

            try:
                stats = self._fit_pair(df_a, df_b)
            except Exception as exc:
                logger.error("PairTrader refit failed", extra={"pair": key, "error": str(exc)})
                ps.qualified = False
                self.state[key] = ps
                continue

            ps.alpha = stats["alpha"]
            ps.beta = stats["beta"]
            ps.mu = stats["mu"]
            ps.sigma = stats["sigma"]
            ps.adf_pvalue = stats["adf_pvalue"]
            adf_pass = stats["adf_pvalue"] < cfg.PAIR_ADF_PVALUE_THRESHOLD
            ps.qualified = adf_pass and abs(stats["beta"]) > 0.01
            ps.last_refit_iso = now_iso

            ps.adf_history.append({"iso": now_iso, "adf_p": float(stats["adf_pvalue"])})
            ps.adf_history = ps.adf_history[-cfg.PAIR_ADF_DECAY_DAYS :]
            self._apply_ttl_gate(ps)

            self.state[key] = ps
            if ps.qualified and not ps.ttl_disqualified:
                qualified_count += 1

            logger.debug(
                "Pair refit",
                extra={
                    "pair": key,
                    "alpha": round(stats["alpha"], 4),
                    "beta": round(stats["beta"], 4),
                    "mu": round(stats["mu"], 5),
                    "sigma": round(stats["sigma"], 5),
                    "adf_p": round(stats["adf_pvalue"], 4),
                    "qualified": ps.qualified,
                    "ttl_disqualified": ps.ttl_disqualified,
                    "ttl_reason": ps.ttl_reason,
                    "trace_id": get_trace_id(),
                },
            )

        self._save_state()
        logger.info(
            f"Pair refit qualified={qualified_count}/{len(self.pairs)}",
            extra={
                "qualified": qualified_count,
                "total": len(self.pairs),
                "trace_id": get_trace_id(),
            },
        )
        return qualified_count

    @staticmethod
    def _apply_ttl_gate(ps: PairState) -> None:
        """TTL/auto-disqualification gate."""
        hist = ps.adf_history
        if len(hist) < cfg.PAIR_ADF_DECAY_DAYS:
            return

        recent = hist[-cfg.PAIR_ADF_DECAY_DAYS :]
        all_bad = all(rec["adf_p"] > cfg.PAIR_ADF_PVALUE_THRESHOLD for rec in recent)
        any_good = any(rec["adf_p"] <= cfg.PAIR_ADF_PVALUE_THRESHOLD for rec in recent)

        if all_bad:
            ps.ttl_disqualified = True
            worst = max(rec["adf_p"] for rec in recent)
            ps.ttl_reason = (
                f"adf>{cfg.PAIR_ADF_PVALUE_THRESHOLD} "
                f"for {cfg.PAIR_ADF_DECAY_DAYS}d (max={worst:.3f})"
            )
        elif any_good and ps.ttl_disqualified:
            ps.ttl_disqualified = False
            ps.ttl_reason = ""

    @staticmethod
    def _fit_pair(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict[str, float]:
        """OLS log-prices regression + ADF test on residuals."""

        a = df_a[["begin", "close"]].rename(columns={"close": "p_a"})
        b = df_b[["begin", "close"]].rename(columns={"close": "p_b"})
        merged = pd.merge_asof(
            a.sort_values("begin"),
            b.sort_values("begin"),
            on="begin",
            tolerance=pd.Timedelta(days=1),
        ).dropna()

        if len(merged) < 30:
            raise ValueError(f"insufficient overlapping data: {len(merged)} rows")

        log_a = np.log(merged["p_a"].astype(float))
        log_b = np.log(merged["p_b"].astype(float))

        X = sm.add_constant(log_b.values)
        model = sm.OLS(log_a.values, X).fit()
        alpha = float(model.params[0])
        beta = float(model.params[1])
        residuals = log_a.values - (alpha + beta * log_b.values)

        adf_result = adfuller(residuals, autolag="AIC")
        adf_pvalue = float(adf_result[1])

        mu = float(np.mean(residuals))
        sigma = float(np.std(residuals))

        if not np.isfinite(adf_pvalue):
            raise ValueError(f"adfuller returned non-finite p-value: {adf_pvalue}")
        if not np.isfinite(mu) or not np.isfinite(sigma):
            raise ValueError(f"residual stats non-finite: mu={mu}, sigma={sigma}")
        if sigma <= 0:
            sigma = 1e-6

        return {
            "alpha": alpha,
            "beta": beta,
            "mu": mu,
            "sigma": sigma,
            "adf_pvalue": adf_pvalue,
        }

    async def _fetch_daily_candles(self) -> dict[str, pd.DataFrame]:
        """Fetch lookback_days of daily candles for all unique tickers in pairs."""
        from app.data.iss_client import get_iss_client

        iss = get_iss_client()
        if not iss._started:
            await iss.startup()

        tickers = list({t for pair in self.pairs for t in pair})
        till = datetime.now(tz=UTC)
        from_dt = till - timedelta(days=self.lookback_days + 10)

        results = await iss.get_candles_multi(tickers, interval=24, from_dt=from_dt, till_dt=till)
        return results  # type: ignore

    async def poll(self) -> list[UnifiedSignal]:
        """Poll."""
        if not self._started:
            raise RuntimeError("PairTrader not started")

        start_ts = time.monotonic()
        all_signals: list[UnifiedSignal] = []

        for a, b in self.pairs:
            key = f"{a}_{b}"
            ps = self.state.get(key)
            if not ps or not ps.qualified:
                continue
            if ps.ttl_disqualified:
                continue

            df_a = self.candle_store.get(a, 10)
            df_b = self.candle_store.get(b, 10)
            if not _HAS_PANDAS:
                continue
            if not isinstance(df_a, pd.DataFrame) or not isinstance(df_b, pd.DataFrame):
                continue
            if df_a.empty or df_b.empty:
                continue

            price_a = float(df_a["close"].iloc[-1])
            price_b = float(df_b["close"].iloc[-1])
            if price_a <= 0 or price_b <= 0:
                continue

            try:
                signals = self._evaluate_pair(ps, price_a, price_b)
                all_signals.extend(signals)
            except Exception as exc:
                logger.error("PairTrader: pair eval failed", extra={"pair": key, "error": str(exc)})

        elapsed_ms = round((time.monotonic() - start_ts) * 1000)
        self._poll_count += 1
        self._signal_count += len(all_signals)
        log_fn = logger.info if all_signals else logger.debug
        log_fn(
            "PairTrader poll done",
            extra={
                "signals": len(all_signals),
                "pairs_checked": sum(1 for ps in self.state.values() if ps.qualified),
                "latency_ms": elapsed_ms,
                "trace_id": get_trace_id(),
            },
        )
        return all_signals

    def _evaluate_pair(
        self,
        ps: PairState,
        price_a: float,
        price_b: float,
    ) -> list[UnifiedSignal]:
        """Compute z-score and emit entry/exit signals."""
        params = cfg.get_pair_params(ps.key())
        z_entry = float(params["z_entry"])
        z_exit = float(params["z_exit"])
        z_stop = float(params["z_stop"])
        max_hold_bars = int(params["max_hold_bars"])

        e_t = math.log(price_a) - ps.alpha - ps.beta * math.log(price_b)
        z_t = (e_t - ps.mu) / ps.sigma

        ps.last_z_values.append(z_t)
        if len(ps.last_z_values) > 10:
            ps.last_z_values = ps.last_z_values[-10:]

        signals: list[UnifiedSignal] = []
        now = datetime.now(tz=UTC)

        if ps.open_entry_bar_iso:
            try:
                entry_dt = datetime.fromisoformat(ps.open_entry_bar_iso)
                held_hours = (now - entry_dt).total_seconds() / 3600.0
                if held_hours >= max_hold_bars:
                    ps.open_entry_bar_iso = ""
                    ps.open_entry_direction = 0
                    self._save_state()
                    return self._make_exit_signals(
                        ps,
                        price_a,
                        price_b,
                        z_t,
                        reason="time_stop",
                    )
            except Exception:
                pass

        if ps.last_entry_iso:
            try:
                last_entry = datetime.fromisoformat(ps.last_entry_iso)
                if (now - last_entry).total_seconds() < self.cooldown_hours * 3600:
                    return []
            except Exception:
                pass

        abs_z = abs(z_t)

        if abs_z > z_stop:
            ps.open_entry_bar_iso = ""
            ps.open_entry_direction = 0
            return self._make_exit_signals(ps, price_a, price_b, z_t, reason="z_stop")

        if abs_z < z_exit:
            if ps.open_entry_bar_iso:
                ps.open_entry_bar_iso = ""
                ps.open_entry_direction = 0
                self._save_state()
            return self._make_exit_signals(ps, price_a, price_b, z_t, reason="z_target")

        if abs_z >= z_entry:
            recent = ps.last_z_values[-3:]
            if len(recent) >= 3:
                if z_t > 0:
                    monotone = recent[0] < recent[1] < recent[2]
                else:
                    monotone = recent[0] > recent[1] > recent[2]
            else:
                monotone = True

            if monotone:
                ps.last_entry_iso = now.isoformat()
                ps.open_entry_bar_iso = now.isoformat()
                ps.open_entry_direction = -1 if z_t > 0 else +1
                self._save_state()

                if z_t > 0:
                    signals.append(
                        self._make_leg_signal(
                            ticker=ps.ticker_a,
                            direction=Direction.SELL,
                            price=price_a,
                            pair=ps,
                            z=z_t,
                            is_leg_a=True,
                        )
                    )
                    signals.append(
                        self._make_leg_signal(
                            ticker=ps.ticker_b,
                            direction=Direction.BUY,
                            price=price_b,
                            pair=ps,
                            z=z_t,
                            is_leg_a=False,
                        )
                    )
                else:
                    signals.append(
                        self._make_leg_signal(
                            ticker=ps.ticker_a,
                            direction=Direction.BUY,
                            price=price_a,
                            pair=ps,
                            z=z_t,
                            is_leg_a=True,
                        )
                    )
                    signals.append(
                        self._make_leg_signal(
                            ticker=ps.ticker_b,
                            direction=Direction.SELL,
                            price=price_b,
                            pair=ps,
                            z=z_t,
                            is_leg_a=False,
                        )
                    )

        return signals

    def _make_leg_signal(
        self,
        ticker: str,
        direction: Direction,
        price: float,
        pair: PairState,
        z: float,
        is_leg_a: bool,
    ) -> UnifiedSignal:
        """Make leg signal."""

        params = cfg.get_pair_params(pair.key())
        sizing_mode = str(params.get("sizing_mode", "equal"))
        pair_z_entry = float(params.get("z_entry", self.z_entry))

        target_rub = DEFAULT_TARGET_RUB
        beta_floor = float(getattr(cfg, "PAIR_BETA_FLOOR", 0.1))
        weight = (
            (1.0 if is_leg_a else max(beta_floor, abs(pair.beta)))
            if sizing_mode == "beta"
            else 1.0
        )
        target_qty = int(weight * target_rub / price) if price > 0 else 0

        sl_pct = float(getattr(cfg, "PAIR_LEG_SL_PCT", 0.03))
        tp_pct = float(getattr(cfg, "PAIR_LEG_TP_PCT", 0.03))
        if direction == Direction.BUY:
            stop = price * (1.0 - sl_pct)
            target = price * (1.0 + tp_pct)
        else:
            stop = price * (1.0 + sl_pct)
            target = price * (1.0 - tp_pct)

        mag_cap = float(getattr(cfg, "PAIR_MAG_CAP", 0.85))
        mag_base = float(getattr(cfg, "PAIR_MAG_BASE", 0.45))
        mag_z_scale = float(getattr(cfg, "PAIR_MAG_Z_SCALE", 0.15))

        return UnifiedSignal(
            source=SignalSource.PAIR,
            detector=f"pair_{pair.key()}_{'A' if is_leg_a else 'B'}",
            ticker=ticker,
            direction=direction,
            magnitude=min(mag_cap, mag_base + (abs(z) - pair_z_entry) * mag_z_scale),
            raw_confidence=float(getattr(cfg, "PAIR_RAW_CONFIDENCE", 0.65)),
            horizon_min=int(getattr(cfg, "PAIR_HORIZON_MIN", 180)),
            price=price,
            entry_level=price,
            stop_level=stop,
            target_level=target,
            expected_rr=1.0,
            atr=0.0,
            metadata={
                "pair_id": pair.key(),
                "z_score": round(z, 3),
                "beta": round(pair.beta, 4),
                "adf_pvalue": round(pair.adf_pvalue, 4),
                "target_quantity": target_qty,
                "sizing_mode": sizing_mode,
                "z_entry": pair_z_entry,
                "leg": "A" if is_leg_a else "B",
                "is_pair_trade": True,
            },
        )

    def _make_exit_signals(
        self,
        pair: PairState,
        price_a: float,
        price_b: float,
        z: float,
        reason: str,
    ) -> list[UnifiedSignal]:
        """Emit closing signals for both legs (cooperative with position book)."""

        sig_a = UnifiedSignal(
            source=SignalSource.PAIR,
            detector=f"pair_exit_{pair.key()}_A",
            ticker=pair.ticker_a,
            direction=Direction.NEUTRAL,
            magnitude=0.50,
            raw_confidence=0.65,
            horizon_min=0,
            price=price_a,
            entry_level=price_a,
            stop_level=price_a,
            target_level=price_a,
            expected_rr=0.0,
            atr=0.0,
            metadata={
                "pair_id": pair.key(),
                "exit_reason": reason,
                "z_score": round(z, 3),
                "leg": "A",
            },
        )
        sig_b = UnifiedSignal(
            source=SignalSource.PAIR,
            detector=f"pair_exit_{pair.key()}_B",
            ticker=pair.ticker_b,
            direction=Direction.NEUTRAL,
            magnitude=0.50,
            raw_confidence=0.65,
            horizon_min=0,
            price=price_b,
            entry_level=price_b,
            stop_level=price_b,
            target_level=price_b,
            expected_rr=0.0,
            atr=0.0,
            metadata={
                "pair_id": pair.key(),
                "exit_reason": reason,
                "z_score": round(z, 3),
                "leg": "B",
            },
        )
        return [sig_a, sig_b]

_pair_trader: PairTrader | None = None

def get_pair_trader() -> PairTrader:
    """Get pair trader."""
    global _pair_trader
    if _pair_trader is None:
        _pair_trader = PairTrader()
    return _pair_trader
