"""Candlestick patterns (TA-Lib + manual)."""

from __future__ import annotations

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
    import pandas_ta as ta  # type: ignore

    _HAS_TA = True
except ImportError:
    try:
        import pandas_ta_classic as ta  # type: ignore

        _HAS_TA = True
    except ImportError:
        _HAS_TA = False

try:
    import talib  # type: ignore

    _HAS_TALIB = True
    _TALIB_PATTERNS: list[str] = talib.get_function_groups().get("Pattern Recognition", [])
except ImportError:
    _HAS_TALIB = False
    _TALIB_PATTERNS = []

PATTERN_BIAS: dict[str, int] = {
    "ENGULFING": 0,
    "HAMMER": +1,
    "INVERTED_HAMMER": +1,
    "SHOOTING_STAR": -1,
    "HANGING_MAN": -1,
    "MORNING_STAR": +1,
    "EVENING_STAR": -1,
    "DOJI": 0,
    "HARAMI": 0,
    "DARK_CLOUD_COVER": -1,
    "PIERCING": +1,
    "THREE_WHITE_SOLDIERS": +1,
    "THREE_BLACK_CROWS": -1,
}

def detect_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame with one column per pattern; values ∈ {-100, 0, +100}.
    Positive = bullish signal at that bar, Negative = bearish, 0 = no signal.

    Uses TA-Lib when available (61 patterns). Falls back to manual (13 patterns)
    if TA-Lib not installed.

    v0.0.39 — Output is filtered through cfg.CANDLE_WHITELIST when non-empty
    (a small set of patterns with PF >= 1.0 on 90d × 20-ticker MOEX backtest).
    Empty whitelist → no filtering, all patterns pass through.
    """
    if not _HAS_PANDAS or df is None or df.empty:
        return pd.DataFrame() if _HAS_PANDAS else {}

    if not {"open", "high", "low", "close"}.issubset(set(df.columns)):
        return pd.DataFrame()

    if _HAS_TALIB:
        try:
            result = _detect_via_talib(df)
        except Exception as exc:
            logger.warning(
                "TA-Lib candle detection failed, fallback to manual", extra={"error": str(exc)}
            )
            result = _detect_manual(df)
    else:
        result = _detect_manual(df)

    return _apply_whitelist(result)

def _apply_whitelist(result: pd.DataFrame) -> pd.DataFrame:
    """Filter detected pattern columns by cfg.CANDLE_WHITELIST.

    Empty whitelist (or absent attribute) → pass-through. This is intentional
    so unit tests / probes that don't initialise config still see all patterns.
    """
    whitelist = getattr(cfg, "CANDLE_WHITELIST", None)
    if not whitelist:
        return result
    if not _HAS_PANDAS or result is None or not hasattr(result, "columns"):
        return result
    kept = [c for c in result.columns if c in whitelist]
    if not kept:
        return pd.DataFrame(index=result.index)
    return result[kept]

def _detect_via_talib(df: pd.DataFrame) -> pd.DataFrame:
    """Run all 61 TA-Lib candle pattern functions; return one column per pattern."""
    result = pd.DataFrame(index=df.index)
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values

    for name in _TALIB_PATTERNS:
        try:
            fn = getattr(talib, name)
            arr = fn(o, h, l, c)
            short_name = name[3:] if name.startswith("CDL") else name
            result[short_name] = arr
        except Exception:
            continue
    return result

def _detect_via_pandas_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Use pandas-ta candle-pattern functions where available."""
    result = pd.DataFrame(index=df.index)

    pattern_names = ["engulfing", "hammer", "doji"]

    import contextlib
    import io

    for name in pattern_names:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                series = ta.cdl_pattern(
                    df["open"],
                    df["high"],
                    df["low"],
                    df["close"],
                    name=name,
                )
            if series is not None and not series.empty:
                col = (
                    name.upper()
                    .replace("3", "THREE_")
                    .replace("INVERTEDHAMMER", "INVERTED_HAMMER")
                    .replace("SHOOTINGSTAR", "SHOOTING_STAR")
                    .replace("HANGINGMAN", "HANGING_MAN")
                    .replace("MORNINGSTAR", "MORNING_STAR")
                    .replace("EVENINGSTAR", "EVENING_STAR")
                    .replace("DARKCLOUDCOVER", "DARK_CLOUD_COVER")
                    .replace("WHITESOLDIERS", "WHITE_SOLDIERS")
                    .replace("BLACKCROWS", "BLACK_CROWS")
                )

                if hasattr(series, "columns"):
                    result[col] = series.iloc[:, 0]
                else:
                    result[col] = series
        except Exception:
            continue

    return result

def _detect_manual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Manual pure-Python implementations of the most useful patterns.
    Vectorised via pandas/numpy where possible.
    """
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    bullish = c > o
    bearish = c < o

    result = pd.DataFrame(index=df.index)

    doji = (body / rng < 0.10).astype(int) * 100
    result["DOJI"] = doji.where(doji > 0, 0)

    hammer = ((lower_wick >= 2 * body) & (upper_wick < body) & (body > 0)).astype(int) * 100
    result["HAMMER"] = hammer

    inv_hammer = ((upper_wick >= 2 * body) & (lower_wick < body) & (body > 0)).astype(int) * 100
    result["INVERTED_HAMMER"] = inv_hammer

    result["SHOOTING_STAR"] = -inv_hammer

    result["HANGING_MAN"] = -hammer

    prev_o = o.shift(1)
    prev_c = c.shift(1)
    prev_bullish = prev_c > prev_o
    prev_bearish = prev_c < prev_o
    bull_engulf = (prev_bearish & bullish & (c > prev_o) & (o < prev_c)).astype(int) * 100
    bear_engulf = (prev_bullish & bearish & (o > prev_c) & (c < prev_o)).astype(int) * 100
    result["ENGULFING"] = bull_engulf - bear_engulf

    inside_bull = (prev_bearish & bullish & (c < prev_o) & (o > prev_c)).astype(int) * 100
    inside_bear = (prev_bullish & bearish & (o < prev_c) & (c > prev_o)).astype(int) * 100
    result["HARAMI"] = inside_bull - inside_bear

    prev_mid = (prev_o + prev_c) / 2
    piercing = (prev_bearish & bullish & (o < l.shift(1)) & (c > prev_mid)).astype(int) * 100
    result["PIERCING"] = piercing

    dark_cloud = (prev_bullish & bearish & (o > h.shift(1)) & (c < prev_mid)).astype(int) * 100
    result["DARK_CLOUD_COVER"] = -dark_cloud

    prev2_o = o.shift(2)
    prev2_c = c.shift(2)
    prev2_body = (prev2_c - prev2_o).abs()
    prev_body = (prev_c - prev_o).abs()
    morning_star = (
        (prev2_c < prev2_o)
        & (prev_body < 0.3 * prev2_body)
        & (bullish)
        & (body > prev_body * 2)
        & (c > (prev2_o + prev2_c) / 2)
    ).astype(int) * 100
    result["MORNING_STAR"] = morning_star

    evening_star = (
        (prev2_c > prev2_o)
        & (prev_body < 0.3 * prev2_body)
        & (bearish)
        & (body > prev_body * 2)
        & (c < (prev2_o + prev2_c) / 2)
    ).astype(int) * 100
    result["EVENING_STAR"] = -evening_star

    bull_seq = (
        bullish
        & bullish.shift(1).fillna(False).astype(bool)
        & bullish.shift(2).fillna(False).astype(bool)
        & (c > c.shift(1))
        & (c.shift(1) > c.shift(2))
        & (o > o.shift(1))
        & (o.shift(1) > o.shift(2))
        & (body > 0)
    ).astype(int) * 100
    result["THREE_WHITE_SOLDIERS"] = bull_seq

    bear_seq = (
        bearish
        & bearish.shift(1).fillna(False).astype(bool)
        & bearish.shift(2).fillna(False).astype(bool)
        & (c < c.shift(1))
        & (c.shift(1) < c.shift(2))
        & (o < o.shift(1))
        & (o.shift(1) < o.shift(2))
        & (body > 0)
    ).astype(int) * 100
    result["THREE_BLACK_CROWS"] = -bear_seq

    result = result.fillna(0).astype(int)
    return result

_LATEST_TAIL_BARS = 20

def latest_candle_signal(df: pd.DataFrame) -> dict[str, int]:
    """
    Return dict {pattern: value} for the LAST bar only.
    Used by TATrader to add candle-pattern features without re-computing each cycle.
    """
    if df is None or len(df) == 0:
        return {}
    tail = df.tail(_LATEST_TAIL_BARS) if len(df) > _LATEST_TAIL_BARS else df
    result = detect_candle_patterns(tail)
    if result.empty:
        return {}
    last_row = result.iloc[-1]
    return {k: int(v) for k, v in last_row.items() if v != 0}

PRODUCTION_PATTERNS: set[str] = {
    "XSIDEGAP3METHODS",
    "STALLEDPATTERN",
    "EVENINGDOJISTAR",
    "SHOOTINGSTAR",
    "DOJISTAR",
}
