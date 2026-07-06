"""CatBoost для оценки success probability паттернов."""

from __future__ import annotations

import math
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
    from catboost import CatBoostClassifier  # type: ignore

    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

PATTERN_NAMES: list[str] = [
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
    "diamond_top",
    "diamond_bottom",
    "cup_and_handle",
    "box_breakout_up",
    "box_breakout_down",
    "wedge_continuation_up",
    "wedge_continuation_down",
    "gartley",
    "bat",
    "butterfly",
    "crab",
    "cypher",
    "shark",
    "smc_order_block_bull",
    "smc_order_block_bear",
    "smc_fvg_bull",
    "smc_fvg_bear",
    "smc_sweep_high",
    "smc_sweep_low",
    "smc_bos_bull",
    "smc_bos_bear",
    "smc_choch_bull",
    "smc_choch_bear",
]

NEW_FEATURE_COLUMNS: list[str] = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_friday",
    "is_monday",
    "hours_to_close",
    "is_morning",
    "is_close_period",
    "roc_5",
    "roc_10",
    "roc_20",
    "momentum_10",
    "gap_pct",
    "direction_consistency_5",
    "bb_width",
    "bb_percent_b",
    "vol_z_60",
    "vol_change_pct_5",
    "volume_trend_10",
    "obv_slope_10",
    "ema20_dist_pct",
    "ema50_dist_pct",
    "ema20_slope",
    "vwap_dist_pct",
    "close_above_ema_count_20",
    "close_pos_in_range_20",
    "atr_ratio_short_long",
    "atr_change_pct_10",
    "price_volatility_20",
    "intraday_range_pct",
    "body_to_range_avg_5",
    "body_ratio_last",
    "rsi_slope_5",
    "adx_slope_5",
    "up_bars_ratio_10",
    "pivot_density_30",
    "bars_since_last_pivot",
    "swing_high_dist_pct",
    "swing_low_dist_pct",
]

FEATURE_COLUMNS: list[str] = [
    "expected_rr",
    "rsi",
    "vol_z",
    "atr_at_entry_pct",
    "sup_atrs",
    "atr_pct",
    "res_atrs",
    "adx",
    "pat_rounding_top",
    "candle_bullish_score",
    "candle_bearish_score",
    "pat_ascending_triangle",
    "pat_descending_triangle",
    "pat_compression_breakout_up",
    "pat_head_shoulders",
    "pat_compression_breakout_down",
    "pat_rounding_bottom",
    "pat_double_bottom",
    "pat_rectangle_breakdown",
    "pat_bear_flag",
    "pat_inv_head_shoulders",
    "pat_double_top",
    "pat_rising_wedge",
    "pat_smc_order_block_bull",
    "pat_smc_fvg_bull",
    "pat_shark",
    "pat_cypher",
    "pat_crab",
    "pat_smc_order_block_bear",
    "pat_smc_sweep_low",
]

class TACatBoost:
    """CatBoost calibrated success-probability model for chart patterns."""

    def __init__(self, model_path: Path | None = None) -> None:
        """Init."""
        self.model: Any = None
        self.model_path = model_path or (cfg.DATA_DIR / "models" / "catboost_ta.cbm")
        self._loaded = False
        self._extra_cache: dict[tuple[int, int], dict[str, float]] = {}
        self._model_feature_names: list[str] | None = None

    def load(self) -> bool:
        """Try to load a previously-trained model. Returns: bool."""
        if not _HAS_CATBOOST:
            logger.warning("CatBoost not installed — using heuristic fallback")
            return False
        if not self.model_path.exists():
            logger.info(
                "CatBoost model file not found — heuristic mode",
                extra={"path": str(self.model_path)},
            )
            return False
        try:
            model = CatBoostClassifier()
            model.load_model(str(self.model_path))
            self.model = model
            self._loaded = True
            try:
                names = list(getattr(model, "feature_names_", []) or [])
                self._model_feature_names = names or None
            except Exception:
                self._model_feature_names = None
            logger.info(
                "CatBoost loaded",
                extra={
                    "path": str(self.model_path),
                    "n_features": (
                        len(self._model_feature_names) if self._model_feature_names else 0
                    ),
                },
            )
            return True
        except Exception as exc:
            logger.error("CatBoost load failed", extra={"error": str(exc)})
            return False

    def build_features(
        self,
        pattern: str,
        expected_rr: float,
        price: float,
        atr_val: float,
        atr_at_entry: float,
        indicators: dict[str, Any],
        levels_info: dict[str, float],
        candle_bits: dict[str, int],
        regime: str,
        df: Any = None,
        bar_idx: int | None = None,
        pivots: Any = None,
    ) -> dict[str, float]:
        """Build a single feature row (dict) for inference.

        Returns:
            dict[str, float]: aligned with FEATURE_COLUMNS.
        """

        def _last_scalar(s: Any, default: float = 0.0) -> float:
            """Last scalar."""
            if s is None:
                return default
            try:
                if hasattr(s, "iloc"):
                    v = s.iloc[-1]
                elif hasattr(s, "__getitem__"):
                    v = s[-1] if len(s) > 0 else default
                else:
                    return default
                if v is None or (_HAS_PANDAS and pd.isna(v)):
                    return default
                return float(v)
            except Exception:
                return default

        adx_val = _last_scalar(
            indicators.get("adx", {}).get("ADX")
            if isinstance(indicators.get("adx"), pd.DataFrame)
            else None
        )
        rsi_val = _last_scalar(indicators.get("rsi"))
        vol_z_val = _last_scalar(indicators.get("vol_z"))

        atr_pct = (atr_val / price * 100) if price > 0 else 0.0
        atr_at_entry_pct = (atr_at_entry / price * 100) if price > 0 else 0.0

        bull_bits = sum(max(0, v) for v in candle_bits.values()) / 1300
        bear_bits = sum(max(0, -v) for v in candle_bits.values()) / 1300

        feat: dict[str, float] = {
            "atr_pct": atr_pct,
            "adx": adx_val,
            "rsi": rsi_val,
            "vol_z": vol_z_val,
            "sup_atrs": min(10.0, float(levels_info.get("support_atrs", 10.0))),
            "res_atrs": min(10.0, float(levels_info.get("resistance_atrs", 10.0))),
            "candle_bullish_score": float(bull_bits),
            "candle_bearish_score": float(bear_bits),
            "expected_rr": min(10.0, float(expected_rr)),
            "atr_at_entry_pct": atr_at_entry_pct,
        }

        for p in PATTERN_NAMES:
            feat[f"pat_{p}"] = 1.0 if p == pattern else 0.0

        feat["regime_trending"] = 1.0 if regime == "trending" else 0.0
        feat["regime_mean_reverting"] = 1.0 if regime == "mean_reverting" else 0.0
        feat["regime_crisis"] = 1.0 if regime == "crisis" else 0.0

        cache = self._extra_cache
        cache_key = (id(df) if df is not None else 0, bar_idx if bar_idx is not None else -1)
        extra = cache.get(cache_key)
        if extra is None:
            extra = self._build_extra_features(
                df=df,
                bar_idx=bar_idx,
                pivots=pivots,
                indicators=indicators,
                price=price,
                atr_val=atr_val,
            )
            cache[cache_key] = extra
        feat.update(extra)

        return feat

    def reset_extra_cache(self) -> None:
        """Clear the per-cycle (df, bar_idx) memoization for _build_extra_features."""
        self._extra_cache = {}

    @staticmethod
    def _build_extra_features(
        df: Any,
        bar_idx: int | None,
        pivots: Any,
        indicators: dict[str, Any],
        price: float,
        atr_val: float,
    ) -> dict[str, float]:
        """Compute v0.0.38+ feature additions (calendar, momentum, BB, vol, EMA, pivots).

        Returns:
            dict[str, float]: prepopulated with neutral defaults.
        """
        feats: dict[str, float] = dict.fromkeys(NEW_FEATURE_COLUMNS, 0.0)

        feats["bb_percent_b"] = 0.5
        feats["hours_to_close"] = 4.0
        feats["close_pos_in_range_20"] = 0.5
        feats["up_bars_ratio_10"] = 0.5

        if not _HAS_PANDAS or df is None:
            return feats

        try:
            if not isinstance(df, pd.DataFrame) or len(df) == 0:
                return feats
            n = len(df)
            if bar_idx is None or bar_idx < 0 or bar_idx >= n:
                bar_idx = n - 1

            ts = None
            if "begin" in df.columns:
                try:
                    ts = df["begin"].iloc[bar_idx]
                    if not isinstance(ts, pd.Timestamp):
                        ts = pd.to_datetime(ts, errors="coerce")
                except Exception:
                    ts = None
            if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
                hour = float(ts.hour)
                dow = float(ts.dayofweek)
                feats["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
                feats["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
                feats["dow_sin"] = math.sin(2 * math.pi * dow / 7.0)
                feats["dow_cos"] = math.cos(2 * math.pi * dow / 7.0)
                feats["is_friday"] = 1.0 if dow == 4 else 0.0
                feats["is_monday"] = 1.0 if dow == 0 else 0.0

                hour_msk = (hour + 3.0) % 24.0
                close_msk = 18.75
                feats["hours_to_close"] = max(-12.0, min(12.0, close_msk - hour_msk))
                feats["is_morning"] = 1.0 if 9.5 <= hour_msk <= 13.0 else 0.0
                feats["is_close_period"] = 1.0 if hour_msk >= 17.5 else 0.0

            close = df["close"]
            high = df["high"]
            low = df["low"]
            vol = df["volume"] if "volume" in df.columns else None

            c_now = float(close.iloc[bar_idx]) if not pd.isna(close.iloc[bar_idx]) else price

            def _roc(periods: int) -> float:
                """Roc."""
                if bar_idx - periods < 0:
                    return 0.0
                prev = float(close.iloc[bar_idx - periods])
                if prev <= 0:
                    return 0.0
                return float((c_now / prev - 1.0) * 100.0)

            feats["roc_5"] = _roc(5)
            feats["roc_10"] = _roc(10)
            feats["roc_20"] = _roc(20)
            feats["momentum_10"] = feats["roc_10"]

            if bar_idx >= 1 and "open" in df.columns:
                prev_close = float(close.iloc[bar_idx - 1])
                if prev_close > 0:
                    open_now = float(df["open"].iloc[bar_idx])
                    feats["gap_pct"] = float((open_now / prev_close - 1.0) * 100.0)

            if bar_idx >= 5:
                last5 = close.iloc[bar_idx - 4 : bar_idx + 1].to_numpy()
                if len(last5) >= 2:
                    diffs = np.sign(np.diff(last5))
                    if len(diffs) > 0:
                        feats["direction_consistency_5"] = float(abs(diffs.sum()) / len(diffs))

            bb = indicators.get("bb") if isinstance(indicators, dict) else None
            if isinstance(bb, pd.DataFrame) and len(bb) > 0:
                bb_row = bb.iloc[min(bar_idx, len(bb) - 1)]
                bbb = bb_row.get("BBB", np.nan) if hasattr(bb_row, "get") else np.nan
                bbp = bb_row.get("BBP", np.nan) if hasattr(bb_row, "get") else np.nan
                if not pd.isna(bbb):
                    feats["bb_width"] = float(bbb)
                if not pd.isna(bbp):
                    feats["bb_percent_b"] = float(np.clip(bbp, -1.0, 2.0))

            if vol is not None and bar_idx >= 1:
                window = max(0, bar_idx - 59)
                vol_win = vol.iloc[window : bar_idx + 1].astype(float).to_numpy()
                v_now = float(vol_win[-1])
                if len(vol_win) >= 5:
                    med = float(np.median(vol_win))
                    mad = float(np.median(np.abs(vol_win - med))) or 1.0
                    feats["vol_z_60"] = float((v_now - med) / (1.4826 * mad))

                if bar_idx >= 5:
                    prev_v = float(vol.iloc[bar_idx - 5])
                    if prev_v > 0:
                        feats["vol_change_pct_5"] = float((v_now / prev_v - 1.0) * 100.0)

                if bar_idx >= 10:
                    vol_trend_win = np.log1p(
                        np.maximum(
                            vol.iloc[bar_idx - 9 : bar_idx + 1].astype(float).to_numpy(),
                            0.0,
                        )
                    )
                    if len(vol_trend_win) >= 2 and np.std(vol_trend_win) > 0:
                        x = np.arange(len(vol_trend_win), dtype=float)
                        slope = float(np.polyfit(x, vol_trend_win, 1)[0])
                        feats["volume_trend_10"] = slope

            obv = indicators.get("obv") if isinstance(indicators, dict) else None
            if isinstance(obv, pd.Series) and bar_idx >= 10:
                obv_win = obv.iloc[bar_idx - 9 : bar_idx + 1].astype(float).to_numpy()
                obv_win = obv_win[~np.isnan(obv_win)]
                if len(obv_win) >= 2:
                    denom = max(abs(obv_win[0]), 1.0)
                    x = np.arange(len(obv_win), dtype=float)
                    slope = float(np.polyfit(x, obv_win, 1)[0]) / denom
                    feats["obv_slope_10"] = slope

            def _series_at(name: str) -> float | None:
                """Series at."""
                s = indicators.get(name) if isinstance(indicators, dict) else None
                if isinstance(s, pd.Series) and len(s) > bar_idx:
                    v = s.iloc[bar_idx]
                    if not pd.isna(v):
                        return float(v)
                return None

            ema20 = _series_at("ema20")
            ema50 = _series_at("ema50")
            vwap = _series_at("vwap")

            if ema20 and c_now:
                feats["ema20_dist_pct"] = float((c_now / ema20 - 1.0) * 100.0)
            if ema50 and c_now:
                feats["ema50_dist_pct"] = float((c_now / ema50 - 1.0) * 100.0)
            if vwap and c_now:
                feats["vwap_dist_pct"] = float((c_now / vwap - 1.0) * 100.0)

            ema20_series = indicators.get("ema20") if isinstance(indicators, dict) else None
            if isinstance(ema20_series, pd.Series) and bar_idx >= 5:
                ema_win = ema20_series.iloc[bar_idx - 4 : bar_idx + 1].astype(float).to_numpy()
                ema_win = ema_win[~np.isnan(ema_win)]
                if len(ema_win) >= 2 and ema_win[0] > 0:
                    feats["ema20_slope"] = float((ema_win[-1] - ema_win[0]) / ema_win[0] * 100.0)

            if isinstance(ema20_series, pd.Series) and bar_idx >= 20:
                window_close = close.iloc[bar_idx - 19 : bar_idx + 1].astype(float).to_numpy()
                window_ema = ema20_series.iloc[bar_idx - 19 : bar_idx + 1].astype(float).to_numpy()
                mask = ~(np.isnan(window_close) | np.isnan(window_ema))
                if mask.sum() > 0:
                    feats["close_above_ema_count_20"] = float(
                        (window_close[mask] > window_ema[mask]).mean()
                    )

            if bar_idx >= 20:
                hi20 = float(high.iloc[bar_idx - 19 : bar_idx + 1].max())
                lo20 = float(low.iloc[bar_idx - 19 : bar_idx + 1].min())
                rng = hi20 - lo20
                if rng > 0:
                    feats["close_pos_in_range_20"] = float((c_now - lo20) / rng)

            atr = indicators.get("atr") if isinstance(indicators, dict) else None
            if isinstance(atr, pd.Series):
                if bar_idx >= 50:
                    atr_recent = atr.iloc[bar_idx - 13 : bar_idx + 1].astype(float)
                    atr_long = atr.iloc[bar_idx - 49 : bar_idx + 1].astype(float)
                    a_short = float(atr_recent.mean()) if len(atr_recent) > 0 else 0.0
                    a_long = float(atr_long.mean()) if len(atr_long) > 0 else 0.0
                    if a_long > 0:
                        feats["atr_ratio_short_long"] = a_short / a_long

                if bar_idx >= 10:
                    a_prev = (
                        float(atr.iloc[bar_idx - 10])
                        if not pd.isna(atr.iloc[bar_idx - 10])
                        else 0.0
                    )
                    a_now = float(atr.iloc[bar_idx]) if not pd.isna(atr.iloc[bar_idx]) else 0.0
                    if a_prev > 0:
                        feats["atr_change_pct_10"] = float((a_now / a_prev - 1.0) * 100.0)

            if bar_idx >= 20:
                rets = close.iloc[bar_idx - 19 : bar_idx + 1].pct_change().dropna().to_numpy()
                if len(rets) > 1:
                    feats["price_volatility_20"] = float(np.std(rets) * 100.0)

            if "open" in df.columns:
                h = float(high.iloc[bar_idx])
                l_ = float(low.iloc[bar_idx])
                o = float(df["open"].iloc[bar_idx])
                rng = max(1e-9, h - l_)
                feats["intraday_range_pct"] = float(rng / max(1e-9, c_now) * 100.0)
                feats["body_ratio_last"] = float(abs(c_now - o) / rng)

                if bar_idx >= 4:
                    ratios = []
                    for i in range(bar_idx - 4, bar_idx + 1):
                        hi = float(high.iloc[i])
                        lo = float(low.iloc[i])
                        op = float(df["open"].iloc[i])
                        cl = float(close.iloc[i])
                        r = max(1e-9, hi - lo)
                        ratios.append(abs(cl - op) / r)
                    feats["body_to_range_avg_5"] = float(np.mean(ratios))

            rsi_s = indicators.get("rsi") if isinstance(indicators, dict) else None
            if isinstance(rsi_s, pd.Series) and bar_idx >= 5:
                r_now = rsi_s.iloc[bar_idx]
                r_prev = rsi_s.iloc[bar_idx - 5]
                if not pd.isna(r_now) and not pd.isna(r_prev):
                    feats["rsi_slope_5"] = float(r_now - r_prev)

            adx_df = indicators.get("adx") if isinstance(indicators, dict) else None
            if isinstance(adx_df, pd.DataFrame) and "ADX" in adx_df.columns and bar_idx >= 5:
                a_now = adx_df["ADX"].iloc[bar_idx]
                a_prev = adx_df["ADX"].iloc[bar_idx - 5]
                if not pd.isna(a_now) and not pd.isna(a_prev):
                    feats["adx_slope_5"] = float(a_now - a_prev)

            if bar_idx >= 10:
                rets_dir = close.iloc[bar_idx - 9 : bar_idx + 1].pct_change().dropna().to_numpy()
                if len(rets_dir) > 0:
                    feats["up_bars_ratio_10"] = float((rets_dir > 0).mean())

            if pivots is not None:
                try:
                    pivot_list = list(pivots)
                    pivot_idxs = [int(getattr(p, "idx", -1)) for p in pivot_list]
                    recent = [i for i in pivot_idxs if 0 <= i <= bar_idx and (bar_idx - i) <= 30]
                    feats["pivot_density_30"] = float(len(recent))
                    valid_idxs = [i for i in pivot_idxs if 0 <= i <= bar_idx]
                    if valid_idxs:
                        last_pivot = max(valid_idxs)
                        feats["bars_since_last_pivot"] = float(min(50, bar_idx - last_pivot))

                    highs = [
                        float(getattr(p, "price", 0.0))
                        for p in pivot_list
                        if getattr(p, "kind", "") == "H"
                        and 0 <= int(getattr(p, "idx", -1)) <= bar_idx
                    ]
                    lows = [
                        float(getattr(p, "price", 0.0))
                        for p in pivot_list
                        if getattr(p, "kind", "") == "L"
                        and 0 <= int(getattr(p, "idx", -1)) <= bar_idx
                    ]
                    if highs and c_now > 0:
                        feats["swing_high_dist_pct"] = float((highs[-1] / c_now - 1.0) * 100.0)
                    if lows and c_now > 0:
                        feats["swing_low_dist_pct"] = float((c_now / lows[-1] - 1.0) * 100.0)
                except Exception:
                    pass

        except Exception as exc:
            logger.debug("extra feature compute failed", extra={"error": str(exc)})

        return feats

    def _inference_columns(self) -> list[str]:
        """Return the column list the model expects, in trained order."""
        return self._model_feature_names or FEATURE_COLUMNS

    def predict_success_proba(self, feat: dict[str, float]) -> float:
        """Return calibrated success probability for one feature row."""
        if self._loaded and self.model is not None and _HAS_PANDAS:
            try:
                cols = self._inference_columns()
                row = [feat.get(c, 0.0) for c in cols]
                X = pd.DataFrame([row], columns=cols)
                proba = self.model.predict_proba(X)[0, 1]
                return float(proba)
            except Exception as exc:
                logger.warning("CatBoost predict failed, fallback", extra={"error": str(exc)})

        return self._heuristic_score(feat)

    def predict_batch(self, feats: list[dict[str, float]]) -> list[float]:
        """Batched predict for speed. Returns: list[float]."""
        if not feats:
            return []
        if self._loaded and self.model is not None and _HAS_PANDAS:
            try:
                cols = self._inference_columns()
                rows = [[f.get(c, 0.0) for c in cols] for f in feats]
                X = pd.DataFrame(rows, columns=cols)
                proba = self.model.predict_proba(X)[:, 1]
                return [float(p) for p in proba]
            except Exception as exc:
                logger.warning("CatBoost batch predict failed", extra={"error": str(exc)})
        return [self._heuristic_score(f) for f in feats]

    @staticmethod
    def _heuristic_score(feat: dict[str, float]) -> float:
        """Heuristic used when no trained CatBoost model is available yet."""
        base = 0.50

        rr = feat.get("expected_rr", 1.0)
        rr_score = min(0.20, (rr - 1.0) * 0.10)

        adx = feat.get("adx", 20.0)
        adx_score = min(0.10, max(-0.05, (adx - 25.0) * 0.005))

        vol_z = feat.get("vol_z", 0.0)
        vol_score = max(-0.05, min(0.10, vol_z * 0.03))

        regime_score = 0.0
        if feat.get("regime_crisis", 0) > 0.5:
            regime_score = -0.15
        elif feat.get("regime_trending", 0) > 0.5:
            cont_flags = [
                feat.get(f"pat_{p}", 0)
                for p in (
                    "bull_flag",
                    "bear_flag",
                    "bull_pennant",
                    "bear_pennant",
                    "ascending_triangle",
                    "descending_triangle",
                    "compression_breakout_up",
                    "compression_breakout_down",
                )
            ]
            if any(f > 0.5 for f in cont_flags):
                regime_score = 0.08
        elif feat.get("regime_mean_reverting", 0) > 0.5:
            rev_flags = [
                feat.get(f"pat_{p}", 0)
                for p in (
                    "double_top",
                    "double_bottom",
                    "head_shoulders",
                    "inv_head_shoulders",
                    "rounding_top",
                    "rounding_bottom",
                )
            ]
            if any(f > 0.5 for f in rev_flags):
                regime_score = 0.08

        score = base + rr_score + adx_score + vol_score + regime_score
        return max(0.05, min(0.95, score))

_ta_catboost: TACatBoost | None = None

def get_ta_catboost() -> TACatBoost:
    """Get ta catboost."""
    global _ta_catboost
    if _ta_catboost is None:
        _ta_catboost = TACatBoost()
        _ta_catboost.load()
    return _ta_catboost
