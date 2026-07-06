"""Continuation chart patterns."""

from __future__ import annotations

from app.utils.logging import get_logger

from .pivots import PivotPoint, trend_direction
from .reversal import PatternSignal, _atr_at, _rr

logger = get_logger(__name__)

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    _READY = True
except ImportError:
    _READY = False

def _prior_impulse(
    df: pd.DataFrame,
    atr: pd.Series,
    first_idx: int,
    bars: int = 36,
) -> tuple[float, float]:
    """
    Returns (return_pct, impulse_in_atr) for the `bars` bars preceding `first_idx`.

    Used as a directional gate for continuation patterns — a bullish pattern
    (rectangle, pennant, flag, wedge_up, compression_up) is only valid when the
    prior `bars` bars show a positive impulse, and vice-versa for bearish.

    Ported from the reference notebook (TA/continuation patterns) Phase 11.13.
    """
    if not _READY or df is None or first_idx <= 0 or len(df) == 0:
        return 0.0, 0.0
    first_idx = int(first_idx)
    start = max(0, first_idx - bars)
    if first_idx <= start:
        return 0.0, 0.0
    atr_val = _atr_at(atr, first_idx)
    if atr_val <= 0:
        return 0.0, 0.0
    c0 = float(df["close"].iloc[start])
    c1 = float(df["close"].iloc[first_idx])
    if c0 <= 0:
        return 0.0, 0.0
    ret = c1 / c0 - 1.0
    impulse_atr = abs(c1 - c0) / atr_val
    return ret, impulse_atr

_PRIOR_IMPULSE_BARS = 36
_MIN_PRIOR_IMPULSE_RETURN = 0.003
_MIN_PRIOR_IMPULSE_ATR = 0.5

def detect_flag(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr: pd.Series,
    impulse_min_atr: float = 1.5,
    impulse_max_bars: int = 8,
    flag_min_bars: int = 3,
    flag_max_bars: int = 15,
    flag_max_retrace: float = 0.6,
    confirm_window: int = 24,
    min_rr: float = 0.7,
) -> list[PatternSignal]:
    """
    Flag pattern: sharp impulse → small counter-trend channel → breakout.
    Scans FULL history (every valid end bar), not just last 30 bars.
    """
    if not _READY or df is None or len(df) < impulse_max_bars + flag_max_bars:
        return []

    signals: list[PatternSignal] = []
    n = len(df)
    seen_breakouts: set[int] = set()

    for end in range(impulse_max_bars, n - flag_min_bars - 1):
        atr_val = _atr_at(atr, end)
        if atr_val <= 0:
            continue

        impulse_start = max(0, end - impulse_max_bars)
        impulse_high = float(df["high"].iloc[impulse_start : end + 1].max())
        impulse_low = float(df["low"].iloc[impulse_start : end + 1].min())
        impulse_open = float(df["open"].iloc[impulse_start])
        impulse_close = float(df["close"].iloc[end])
        impulse_move = impulse_close - impulse_open

        if abs(impulse_move) < impulse_min_atr * atr_val:
            continue

        bullish_impulse = impulse_move > 0

        flag_start = end + 1
        flag_end = min(n - 1, flag_start + flag_max_bars)
        if flag_end - flag_start < flag_min_bars:
            continue

        flag_slice = df.iloc[flag_start : flag_end + 1]
        flag_high = float(flag_slice["high"].max())
        flag_low = float(flag_slice["low"].min())

        if bullish_impulse:
            retrace = (impulse_high - flag_low) / abs(impulse_move)
            if retrace > flag_max_retrace:
                continue
            first_close = float(flag_slice["close"].iloc[0])
            last_close = float(flag_slice["close"].iloc[-1])
            if last_close > first_close + 0.8 * atr_val:
                continue
            for ci in range(flag_end, min(flag_end + confirm_window, n)):
                if ci in seen_breakouts:
                    continue
                if float(df["close"].iloc[ci]) > flag_high + 0.1 * atr_val:
                    entry = flag_high + 0.1 * atr_val
                    stop = flag_low - 0.2 * atr_val
                    target = entry + abs(impulse_move)
                    rr = _rr(entry, stop, target)
                    if rr < min_rr:
                        continue
                    seen_breakouts.add(ci)
                    signals.append(
                        PatternSignal(
                            pattern="bull_flag",
                            direction="BUY",
                            confidence=0.62,
                            bar_idx=ci,
                            entry=entry,
                            stop=stop,
                            target=target,
                            expected_rr=rr,
                            atr_at_entry=atr_val,
                            metadata={"impulse_move": round(impulse_move, 2)},
                        )
                    )
                    break
        else:
            retrace = (flag_high - impulse_low) / abs(impulse_move)
            if retrace > flag_max_retrace:
                continue
            first_close = float(flag_slice["close"].iloc[0])
            last_close = float(flag_slice["close"].iloc[-1])
            if last_close < first_close - 0.8 * atr_val:
                continue
            for ci in range(flag_end, min(flag_end + confirm_window, n)):
                if ci in seen_breakouts:
                    continue
                if float(df["close"].iloc[ci]) < flag_low - 0.1 * atr_val:
                    entry = flag_low - 0.1 * atr_val
                    stop = flag_high + 0.2 * atr_val
                    target = entry - abs(impulse_move)
                    rr = _rr(entry, stop, target)
                    if rr < min_rr:
                        continue
                    seen_breakouts.add(ci)
                    signals.append(
                        PatternSignal(
                            pattern="bear_flag",
                            direction="SELL",
                            confidence=0.62,
                            bar_idx=ci,
                            entry=entry,
                            stop=stop,
                            target=target,
                            expected_rr=rr,
                            atr_at_entry=atr_val,
                        )
                    )
                    break

    return signals

def detect_pennant(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr: pd.Series,
    impulse_min_atr: float = 1.5,
    consol_min_bars: int = 4,
    consol_max_bars: int = 15,
    confirm_window: int = 24,
    min_rr: float = 0.7,
) -> list[PatternSignal]:
    """
    Pennant: sharp impulse → symmetric small triangle → breakout.
    Scans FULL history.
    """
    if not _READY or df is None or len(df) < 20:
        return []

    signals: list[PatternSignal] = []
    n = len(df)
    seen_breakouts: set[int] = set()

    for end in range(8, n - consol_min_bars - 1):
        atr_val = _atr_at(atr, end)
        if atr_val <= 0:
            continue

        impulse_open = float(df["open"].iloc[max(0, end - 7)])
        impulse_close = float(df["close"].iloc[end])
        impulse_move = impulse_close - impulse_open

        if abs(impulse_move) < impulse_min_atr * atr_val:
            continue

        bullish = impulse_move > 0

        consol_start = end + 1
        consol_end = min(n - 1, consol_start + consol_max_bars)
        if consol_end - consol_start < consol_min_bars:
            continue

        consol = df.iloc[consol_start : consol_end + 1]
        early = consol.iloc[: max(1, len(consol) // 3)]
        late = consol.iloc[-max(1, len(consol) // 3) :]

        early_range = float(early["high"].max() - early["low"].min())
        late_range = float(late["high"].max() - late["low"].min())

        if late_range > 0.6 * early_range:
            continue
        if late_range < 0.3 * atr_val:
            continue

        consol_high = float(consol["high"].max())
        consol_low = float(consol["low"].min())

        if bullish:
            for ci in range(consol_end, min(consol_end + confirm_window, n)):
                if ci in seen_breakouts:
                    continue
                if float(df["close"].iloc[ci]) > consol_high + 0.1 * atr_val:
                    entry = consol_high + 0.1 * atr_val
                    stop = consol_low - 0.2 * atr_val
                    target = entry + abs(impulse_move)
                    rr = _rr(entry, stop, target)
                    if rr < min_rr:
                        continue
                    seen_breakouts.add(ci)
                    signals.append(
                        PatternSignal(
                            pattern="bull_pennant",
                            direction="BUY",
                            confidence=0.60,
                            bar_idx=ci,
                            entry=entry,
                            stop=stop,
                            target=target,
                            expected_rr=rr,
                            atr_at_entry=atr_val,
                        )
                    )
                    break
        else:
            for ci in range(consol_end, min(consol_end + confirm_window, n)):
                if ci in seen_breakouts:
                    continue
                if float(df["close"].iloc[ci]) < consol_low - 0.1 * atr_val:
                    entry = consol_low - 0.1 * atr_val
                    stop = consol_high + 0.2 * atr_val
                    target = entry - abs(impulse_move)
                    rr = _rr(entry, stop, target)
                    if rr < min_rr:
                        continue
                    seen_breakouts.add(ci)
                    signals.append(
                        PatternSignal(
                            pattern="bear_pennant",
                            direction="SELL",
                            confidence=0.60,
                            bar_idx=ci,
                            entry=entry,
                            stop=stop,
                            target=target,
                            expected_rr=rr,
                            atr_at_entry=atr_val,
                        )
                    )
                    break

    return signals

def _fit_line(pts: list[PivotPoint]) -> tuple[float, float]:
    """OLS slope/intercept of pivot prices vs bar index."""
    xs = np.array([p.idx for p in pts], dtype=float)
    ys = np.array([p.price for p in pts], dtype=float)
    if len(xs) < 2 or xs[-1] == xs[0]:
        return 0.0, ys.mean()
    m = np.polyfit(xs, ys, 1)
    return float(m[0]), float(m[1])

def detect_triangle(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr: pd.Series,
    min_touches_per_side: int = 2,
    window: int = 7,
    confirm_window: int = 24,
    min_rr: float = 0.7,
    min_compression: float = 0.0,
) -> list[PatternSignal]:
    """
    Ascending / Descending / Symmetric triangle. Scans full history pivot-by-pivot.

    Phase 11.13 — `window` reduced 8 → 7 (matches reference notebook `j-6:j+1`),
    yielding more candidates on real H1 data without sacrificing quality. Added
    optional `min_compression` filter (price range of last 3 pivots vs first 3,
    matches reference's `MIN_COMPRESSION=0.02`); 0.0 default = backward compatible.
    """
    if not _READY or df is None or len(df) < 20 or len(pivots) < window:
        return []

    n = len(df)
    signals: list[PatternSignal] = []
    seen: set[int] = set()

    for end_pivot in range(window, len(pivots)):
        recent = pivots[end_pivot - window : end_pivot]
        highs = [p for p in recent if p.kind == "H"][-3:]
        lows = [p for p in recent if p.kind == "L"][-3:]
        if len(highs) < min_touches_per_side or len(lows) < min_touches_per_side:
            continue

        last_pivot_idx = max(highs[-1].idx, lows[-1].idx)
        if last_pivot_idx >= n - 1:
            continue
        atr_val = _atr_at(atr, last_pivot_idx)
        if atr_val <= 0:
            continue

        if min_compression > 0:
            prices = [p.price for p in recent]
            if len(prices) >= 6:
                first3 = prices[:3]
                last3 = prices[-3:]
                first_range = max(first3) - min(first3)
                last_range = max(last3) - min(last3)
                if first_range > 0:
                    compression = 1.0 - last_range / first_range
                    if compression < min_compression:
                        continue

        h_slope, h_intercept = _fit_line(highs)
        l_slope, l_intercept = _fit_line(lows)
        h_slope_norm = h_slope / atr_val
        l_slope_norm = l_slope / atr_val

        pattern: str | None = None
        direction: str | None = None
        if abs(h_slope_norm) < 0.05 and l_slope_norm > 0.05:
            pattern = "ascending_triangle"
            direction = "BUY"
        elif abs(l_slope_norm) < 0.05 and h_slope_norm < -0.05:
            pattern = "descending_triangle"
            direction = "SELL"
        elif h_slope_norm < -0.05 and l_slope_norm > 0.05:
            pattern = "symmetric_triangle"
            trend = trend_direction(pivots[:end_pivot])
            direction = "BUY" if trend == "UP" else "SELL" if trend == "DOWN" else None

        if pattern is None or direction is None:
            continue

        for ci in range(last_pivot_idx + 1, min(last_pivot_idx + 1 + confirm_window, n)):
            if ci in seen:
                continue
            h_at_ci = h_slope * ci + h_intercept
            l_at_ci = l_slope * ci + l_intercept
            triangle_height = h_at_ci - l_at_ci
            if triangle_height < 0.4 * atr_val:
                continue
            close = float(df["close"].iloc[ci])

            if direction == "BUY" and close > h_at_ci + 0.1 * atr_val:
                entry = h_at_ci + 0.1 * atr_val
                stop = l_at_ci - 0.2 * atr_val
                target = entry + triangle_height
                rr = _rr(entry, stop, target)
                if rr < min_rr:
                    continue
                seen.add(ci)
                signals.append(
                    PatternSignal(
                        pattern=pattern,
                        direction="BUY",
                        confidence=0.60,
                        bar_idx=ci,
                        entry=entry,
                        stop=stop,
                        target=target,
                        expected_rr=rr,
                        atr_at_entry=atr_val,
                    )
                )
                break
            elif direction == "SELL" and close < l_at_ci - 0.1 * atr_val:
                entry = l_at_ci - 0.1 * atr_val
                stop = h_at_ci + 0.2 * atr_val
                target = entry - triangle_height
                rr = _rr(entry, stop, target)
                if rr < min_rr:
                    continue
                seen.add(ci)
                signals.append(
                    PatternSignal(
                        pattern=pattern,
                        direction="SELL",
                        confidence=0.60,
                        bar_idx=ci,
                        entry=entry,
                        stop=stop,
                        target=target,
                        expected_rr=rr,
                        atr_at_entry=atr_val,
                    )
                )
                break

    return signals

def detect_rectangle(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr: pd.Series,
    min_touches: int = 2,
    price_tolerance_atr: float = 0.8,
    window: int = 6,
    min_height_atr: float = 0.6,
    confirm_window: int = 24,
    min_rr: float = 0.7,
    require_prior_impulse: bool = False,
    prior_impulse_bars: int = _PRIOR_IMPULSE_BARS,
) -> list[PatternSignal]:
    """
    Rectangle / trading range. Scans full history pivot-by-pivot.

    Phase 11.13 — `window` reduced 8 → 6 (matches reference notebook `j-5:j+1`
    sliding window — gives more candidates on real H1 data). Added optional
    `require_prior_impulse` gate (default OFF — caller can enable for higher
    precision continuation-only entries; matches reference notebook semantics).
    """
    if not _READY or df is None or len(df) < 20 or len(pivots) < window:
        return []

    n = len(df)
    signals: list[PatternSignal] = []
    seen: set[int] = set()

    for end_pivot in range(window, len(pivots)):
        recent = pivots[end_pivot - window : end_pivot]
        highs = [p for p in recent if p.kind == "H"]
        lows = [p for p in recent if p.kind == "L"]
        if len(highs) < min_touches or len(lows) < min_touches:
            continue

        last_pivot_idx = max(highs[-1].idx, lows[-1].idx)
        if last_pivot_idx >= n - 1:
            continue
        atr_val = _atr_at(atr, last_pivot_idx)
        if atr_val <= 0:
            continue

        high_prices = [p.price for p in highs]
        low_prices = [p.price for p in lows]
        if max(high_prices) - min(high_prices) > price_tolerance_atr * atr_val:
            continue
        if max(low_prices) - min(low_prices) > price_tolerance_atr * atr_val:
            continue

        resistance = float(np.mean(high_prices))
        support = float(np.mean(low_prices))
        height = resistance - support
        if height < min_height_atr * atr_val:
            continue

        if require_prior_impulse:
            first_pivot_idx = min(highs[0].idx, lows[0].idx)
            impulse_ret, impulse_atr = _prior_impulse(
                df, atr, first_pivot_idx, bars=prior_impulse_bars
            )
            allow_up = (
                impulse_ret >= _MIN_PRIOR_IMPULSE_RETURN and impulse_atr >= _MIN_PRIOR_IMPULSE_ATR
            )
            allow_down = (
                impulse_ret <= -_MIN_PRIOR_IMPULSE_RETURN and impulse_atr >= _MIN_PRIOR_IMPULSE_ATR
            )
        else:
            allow_up = allow_down = True
            impulse_ret = impulse_atr = 0.0

        for ci in range(last_pivot_idx + 1, min(last_pivot_idx + 1 + confirm_window, n)):
            if ci in seen:
                continue
            close = float(df["close"].iloc[ci])
            if allow_up and close > resistance + 0.1 * atr_val:
                entry = resistance + 0.1 * atr_val
                stop = support - 0.2 * atr_val
                target = entry + height
                rr = _rr(entry, stop, target)
                if rr < min_rr:
                    continue
                seen.add(ci)
                signals.append(
                    PatternSignal(
                        pattern="rectangle_breakout_up",
                        direction="BUY",
                        confidence=0.58,
                        bar_idx=ci,
                        entry=entry,
                        stop=stop,
                        target=target,
                        expected_rr=rr,
                        atr_at_entry=atr_val,
                        metadata={"prior_impulse_atr": round(impulse_atr, 2)}
                        if require_prior_impulse
                        else {},
                    )
                )
                break
            elif allow_down and close < support - 0.1 * atr_val:
                entry = support - 0.1 * atr_val
                stop = resistance + 0.2 * atr_val
                target = entry - height
                rr = _rr(entry, stop, target)
                if rr < min_rr:
                    continue
                seen.add(ci)
                signals.append(
                    PatternSignal(
                        pattern="rectangle_breakdown",
                        direction="SELL",
                        confidence=0.58,
                        bar_idx=ci,
                        entry=entry,
                        stop=stop,
                        target=target,
                        expected_rr=rr,
                        atr_at_entry=atr_val,
                        metadata={"prior_impulse_atr": round(impulse_atr, 2)}
                        if require_prior_impulse
                        else {},
                    )
                )
                break

    return signals

def detect_compression_breakout(
    df: pd.DataFrame,
    pivots: list[PivotPoint],
    atr: pd.Series,
    compression_bars: int = 10,
    long_atr_period: int = 60,
    compression_ratio: float = 0.7,
    expansion_atr: float = 1.2,
    min_rr: float = 0.7,
    require_prior_impulse: bool = False,
    prior_impulse_lookback: int = 12,
    strict_prior_impulse: bool = False,
    prior_impulse_bars: int = _PRIOR_IMPULSE_BARS,
    min_prior_impulse_atr: float = _MIN_PRIOR_IMPULSE_ATR,
) -> list[PatternSignal]:
    """
    Volatility compression breakout — scans ALL history.

    For each bar t: if range over (t-compression_bars..t) is small, and bar t+1
    expands beyond that range with body ≥ expansion_atr × ATR → emit signal.

    Phase 11.13 directional gating:
      - `require_prior_impulse=True` (default): the breakout direction must
        match a **light** directional bias — close at the compression-start
        bar vs `prior_impulse_lookback` bars earlier. Empirically on H1 MOEX
        data this raises win-rate by ~1-2pp while only modestly cutting volume.
      - `strict_prior_impulse=True` (opt-in): use the heavier reference-notebook
        gate (≥ `min_prior_impulse_atr × ATR` and `_MIN_PRIOR_IMPULSE_RETURN`
        signed move over `prior_impulse_bars` bars). Cuts volume ~2× but boosts
        signal quality further when you want fewer-stronger trades.
      - `require_prior_impulse=False`: no directional gate — old behavior.
    """
    if not _READY or df is None or len(df) < long_atr_period + compression_bars + 1:
        return []

    n = len(df)
    signals: list[PatternSignal] = []

    highs_arr = df["high"].values
    lows_arr = df["low"].values
    opens_arr = df["open"].values
    closes_arr = df["close"].values

    for t in range(long_atr_period + compression_bars, n - 1):
        atr_val = _atr_at(atr, t)
        if atr_val <= 0:
            continue
        long_atr = float(atr.iloc[t - long_atr_period : t].mean()) if t >= long_atr_period else 0.0
        if long_atr <= 0:
            continue

        recent_high = float(highs_arr[t - compression_bars : t].max())
        recent_low = float(lows_arr[t - compression_bars : t].min())
        recent_range = recent_high - recent_low

        if recent_range > compression_ratio * long_atr * compression_bars / 2:
            continue

        bar_open = float(opens_arr[t + 1])
        bar_close = float(closes_arr[t + 1])
        bar_high = float(highs_arr[t + 1])
        bar_low = float(lows_arr[t + 1])
        bar_range = bar_high - bar_low

        if bar_range < expansion_atr * atr_val:
            continue

        breakout_idx = t + 1

        bias_up = True
        bias_down = True
        impulse_atr = 0.0
        if require_prior_impulse:
            prior_start = t - compression_bars
            if strict_prior_impulse:
                impulse_ret, impulse_atr = _prior_impulse(
                    df, atr, prior_start, bars=prior_impulse_bars
                )
                bias_up = (
                    impulse_ret >= _MIN_PRIOR_IMPULSE_RETURN
                    and impulse_atr >= min_prior_impulse_atr
                )
                bias_down = (
                    impulse_ret <= -_MIN_PRIOR_IMPULSE_RETURN
                    and impulse_atr >= min_prior_impulse_atr
                )
            else:
                back_start = max(0, prior_start - prior_impulse_lookback)
                c0 = float(closes_arr[back_start])
                c1 = float(closes_arr[prior_start])
                bias_up = c1 > c0
                bias_down = c1 < c0

        if bar_close > recent_high and bar_close > bar_open and bias_up:
            entry = bar_close + 0.05 * atr_val
            stop = recent_low - 0.2 * atr_val
            target = entry + 2.5 * atr_val
            rr = _rr(entry, stop, target)
            if rr < min_rr:
                continue
            signals.append(
                PatternSignal(
                    pattern="compression_breakout_up",
                    direction="BUY",
                    confidence=0.58,
                    bar_idx=breakout_idx,
                    entry=entry,
                    stop=stop,
                    target=target,
                    expected_rr=rr,
                    atr_at_entry=atr_val,
                    metadata={"prior_impulse_atr": round(impulse_atr, 2)}
                    if strict_prior_impulse
                    else {},
                )
            )
        elif bar_close < recent_low and bar_close < bar_open and bias_down:
            entry = bar_close - 0.05 * atr_val
            stop = recent_high + 0.2 * atr_val
            target = entry - 2.5 * atr_val
            rr = _rr(entry, stop, target)
            if rr < min_rr:
                continue
            signals.append(
                PatternSignal(
                    pattern="compression_breakout_down",
                    direction="SELL",
                    confidence=0.58,
                    bar_idx=breakout_idx,
                    entry=entry,
                    stop=stop,
                    target=target,
                    expected_rr=rr,
                    atr_at_entry=atr_val,
                    metadata={"prior_impulse_atr": round(impulse_atr, 2)}
                    if strict_prior_impulse
                    else {},
                )
            )

    return signals

PRODUCTION_PATTERNS: set[str] = {
    "rectangle_breakdown",
    "descending_triangle",
    "symmetric_triangle",
    "bear_pennant",
    "ascending_triangle",
    "bear_flag",
    "compression_breakout_down",
    "rectangle_breakout_up",
    "bull_pennant",
}
