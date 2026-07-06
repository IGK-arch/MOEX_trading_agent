"""tests/unit/test_detectors_edge_cases.py — Detector safety against
degenerate data.

The trading bot calls ~30 chart-pattern detectors per ticker per cycle.
A single crash there would take down the whole TA scan loop. This suite
verifies that every detector — reversal, continuation, harmonic, SMC,
research, Dasha, VPVR, candles, extras — returns an empty list (or empty
DataFrame for `detect_candle_patterns`) instead of raising on the
classes of data we've actually seen in production:

    1. Empty DataFrame (len == 0)             — warm-up / boot before
                                                  data arrives
    2. Single-row DataFrame (len == 1)        — first bar of session
    3. Below-minimum DataFrame                — illiquid ticker
    4. All-NaN OHLCV                          — MOEX feed outage
    5. Flat candles (high == low ∀ bars)      — auction freeze / halt
    6. Massive gap mid-series                 — earnings / news shock
    7. ATR == 0 series                        — no movement / freeze
    8. ATR with NaN / Inf                     — broken indicator pipeline
    9. Garbage pivots (OOB idx / NaN price)   — upstream regression
    10. Missing OHLCV columns                 — schema drift
    11. Negative / zero close prices          — bad data
    12. Mismatched ATR length                 — pipeline bug

It also verifies that ``safe_detect`` swallows ANY detector exception
and logs at DEBUG level, never at WARNING/ERROR — failed detectors are
expected edge cases, not bugs to escalate.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from app.agents.ta_patterns.candles import (
    detect_candle_patterns,
    latest_candle_signal,
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
    detect_double_patterns_dasha,
    detect_hs_patterns_dasha,
    zigzag_atr_pivots_hilo,
)
from app.agents.ta_patterns.extras_chart import (
    CHART_EXTRA_DETECTORS,
    detect_box_breakout,
    detect_cup_handle,
    detect_diamond,
    detect_wedge_continuation,
)
from app.agents.ta_patterns.harmonic import (
    HARMONIC_DETECTORS,
    detect_bat,
    detect_butterfly,
    detect_crab,
    detect_cypher,
    detect_gartley,
    detect_shark,
)
from app.agents.ta_patterns.pivots import PivotPoint, find_pivots
from app.agents.ta_patterns.research_patterns import (
    detect_all_research_patterns,
    detect_bb_squeeze_breakout,
    detect_inside_bar_breakout,
    detect_pivot_reversal,
    detect_three_soldiers_volume,
    detect_vcp,
)
from app.agents.ta_patterns.reversal import (
    detect_double_top_bottom,
    detect_head_shoulders,
    detect_megaphone,
    detect_rounding,
    detect_triple_top_bottom,
    detect_wedge_reversal,
)
from app.agents.ta_patterns.safe_runner import safe_detect
from app.agents.ta_patterns.smc import (
    SMC_DETECTORS,
    detect_all_smc_patterns,
    detect_bos,
    detect_choch,
    detect_fair_value_gap,
    detect_liquidity_sweep,
    detect_order_block,
)
from app.agents.ta_patterns.volume_profile import (
    compute_vpvr,
    detect_vpvr_signal,
    find_hvn,
    find_lvn,
)


def _empty_df() -> pd.DataFrame:
    """Zero-row OHLCV DataFrame with the canonical column set + atr14."""
    return pd.DataFrame(
        {
            "open": pd.Series(dtype=float),
            "high": pd.Series(dtype=float),
            "low": pd.Series(dtype=float),
            "close": pd.Series(dtype=float),
            "volume": pd.Series(dtype=float),
            "atr14": pd.Series(dtype=float),
        }
    )


def _one_row_df() -> pd.DataFrame:
    """One row df."""
    return pd.DataFrame(
        {
            "open": [100.0],
            "high": [100.5],
            "low": [99.5],
            "close": [100.0],
            "volume": [1000.0],
            "atr14": [0.5],
        }
    )


def _short_df(n: int = 5) -> pd.DataFrame:
    """Short df."""
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [100.5] * n,
            "low": [99.5] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
            "atr14": [0.5] * n,
        }
    )


def _all_nan_df(n: int = 80) -> pd.DataFrame:
    """All nan df."""
    return pd.DataFrame(
        {
            "open": [np.nan] * n,
            "high": [np.nan] * n,
            "low": [np.nan] * n,
            "close": [np.nan] * n,
            "volume": [np.nan] * n,
            "atr14": [np.nan] * n,
        }
    )


def _flat_df(n: int = 100) -> pd.DataFrame:
    """high == low == close == open ∀ bars. ATR is zero. Auction freeze."""
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
            "atr14": [0.0] * n,
        }
    )


def _gap_df(n: int = 100) -> pd.DataFrame:
    """50%/50% split with a 20-rouble crash in the middle."""
    closes = np.concatenate([np.linspace(100, 110, n // 2), np.linspace(20, 25, n - n // 2)])
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "volume": [1000.0] * n,
        }
    )
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)
    return df


def _normal_df(n: int = 120, seed: int = 7) -> pd.DataFrame:
    """Normal df."""
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0, 0.5, n).cumsum()
    df = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.1, n),
            "high": close + np.abs(rng.normal(0.5, 0.2, n)),
            "low": close - np.abs(rng.normal(0.5, 0.2, n)),
            "close": close,
            "volume": rng.integers(1000, 5000, n).astype(float),
        }
    )
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)
    return df


def _df_no_high() -> pd.DataFrame:
    """Df no high."""
    return pd.DataFrame(
        {
            "open": [100.0] * 50,
            "low": [99.0] * 50,
            "close": [100.0] * 50,
            "volume": [1000.0] * 50,
            "atr14": [0.5] * 50,
        }
    )


def _df_no_close() -> pd.DataFrame:
    """Df no close."""
    return pd.DataFrame(
        {
            "open": [100.0] * 50,
            "high": [101.0] * 50,
            "low": [99.0] * 50,
            "volume": [1000.0] * 50,
            "atr14": [0.5] * 50,
        }
    )


def _atr(df: pd.DataFrame) -> pd.Series:
    """Atr."""
    if "atr14" in df.columns:
        return df["atr14"]
    return pd.Series([0.0] * len(df))


def _garbage_pivots() -> list[PivotPoint]:
    """Indices out of range, prices NaN/Inf — what a buggy upstream might emit."""
    return [
        PivotPoint(idx=-1, price=100.0, kind="H", label="HH"),
        PivotPoint(idx=99_999, price=110.0, kind="L", label="LL"),
        PivotPoint(idx=0, price=float("nan"), kind="H", label="HH"),
        PivotPoint(idx=1, price=float("inf"), kind="L", label="LL"),
    ]


PIVOTAL_DETECTORS: list[tuple[str, object]] = [
    ("detect_double_top_bottom", detect_double_top_bottom),
    ("detect_triple_top_bottom", detect_triple_top_bottom),
    ("detect_head_shoulders", detect_head_shoulders),
    ("detect_wedge_reversal", detect_wedge_reversal),
    ("detect_megaphone", detect_megaphone),
    ("detect_rounding", detect_rounding),
    ("detect_flag", detect_flag),
    ("detect_pennant", detect_pennant),
    ("detect_triangle", detect_triangle),
    ("detect_rectangle", detect_rectangle),
    ("detect_compression_breakout", detect_compression_breakout),
    ("detect_gartley", detect_gartley),
    ("detect_bat", detect_bat),
    ("detect_butterfly", detect_butterfly),
    ("detect_crab", detect_crab),
    ("detect_cypher", detect_cypher),
    ("detect_shark", detect_shark),
    ("detect_order_block", detect_order_block),
    ("detect_fair_value_gap", detect_fair_value_gap),
    ("detect_liquidity_sweep", detect_liquidity_sweep),
    ("detect_bos", detect_bos),
    ("detect_choch", detect_choch),
    ("detect_diamond", detect_diamond),
    ("detect_cup_handle", detect_cup_handle),
    ("detect_box_breakout", detect_box_breakout),
    ("detect_wedge_continuation", detect_wedge_continuation),
]

DF_ONLY_DETECTORS: list[tuple[str, object]] = [
    ("detect_vcp", detect_vcp),
    ("detect_bb_squeeze_breakout", detect_bb_squeeze_breakout),
    ("detect_inside_bar_breakout", detect_inside_bar_breakout),
    ("detect_three_soldiers_volume", detect_three_soldiers_volume),
    ("detect_pivot_reversal", detect_pivot_reversal),
    ("detect_all_research_patterns", detect_all_research_patterns),
    ("detect_all_smc_patterns", detect_all_smc_patterns),
    ("detect_all_dasha_patterns", detect_all_dasha_patterns),
]


def test_pivotal_detectors_handle_empty_df():
    """Test pivotal detectors handle empty df."""
    df = _empty_df()
    atr = _atr(df)
    for name, fn in PIVOTAL_DETECTORS:
        out = safe_detect(fn, df, [], atr, _detector_name=name)
        assert out == [], f"{name} crashed/non-empty on empty df: {out!r}"


def test_df_only_detectors_handle_empty_df():
    """Test df only detectors handle empty df."""
    df = _empty_df()
    for name, fn in DF_ONLY_DETECTORS:
        out = safe_detect(fn, df, _detector_name=name)
        assert out == [], f"{name} non-empty on empty df: {out!r}"


def test_candle_detectors_handle_empty_df():
    """Test candle detectors handle empty df."""
    df = _empty_df()
    result = detect_candle_patterns(df)
    assert isinstance(result, (pd.DataFrame, dict))
    assert len(result) == 0
    sig = latest_candle_signal(df)
    assert sig == {}


def test_compute_vpvr_handles_empty_df():
    """Test compute vpvr handles empty df."""
    df = _empty_df()
    out = compute_vpvr(df)
    assert out == []
    out2 = detect_vpvr_signal(df, current_price=100.0)
    assert out2 == []


def test_pivotal_detectors_handle_one_row():
    """Test pivotal detectors handle one row."""
    df = _one_row_df()
    atr = _atr(df)
    for name, fn in PIVOTAL_DETECTORS:
        out = safe_detect(fn, df, [], atr, _detector_name=name)
        assert out == [], f"{name} crashed/non-empty on 1-row df: {out!r}"


def test_df_only_detectors_handle_one_row():
    """Test df only detectors handle one row."""
    df = _one_row_df()
    for name, fn in DF_ONLY_DETECTORS:
        out = safe_detect(fn, df, _detector_name=name)
        assert out == [], f"{name} non-empty on 1-row df: {out!r}"


def test_pivotal_detectors_handle_short_df():
    """Test pivotal detectors handle short df."""
    df = _short_df(5)
    atr = _atr(df)
    for name, fn in PIVOTAL_DETECTORS:
        out = safe_detect(fn, df, [], atr, _detector_name=name)
        assert out == [], f"{name} returned signals on 5-bar df: {out!r}"


def test_pivotal_detectors_handle_all_nan():
    """Test pivotal detectors handle all nan."""
    df = _all_nan_df()
    atr = _atr(df)
    for name, fn in PIVOTAL_DETECTORS:
        out = safe_detect(fn, df, [], atr, _detector_name=name)
        assert out == [], f"{name} crashed/non-empty on all-NaN df: {out!r}"


def test_df_only_detectors_handle_all_nan():
    """Test df only detectors handle all nan."""
    df = _all_nan_df()
    for name, fn in DF_ONLY_DETECTORS:
        out = safe_detect(fn, df, _detector_name=name)
        assert out == [], f"{name} non-empty on all-NaN df: {out!r}"


def test_candle_detector_handles_all_nan():
    """Test candle detector handles all nan."""
    df = _all_nan_df()
    result = detect_candle_patterns(df)
    assert isinstance(result, (pd.DataFrame, dict))


def test_pivotal_detectors_handle_flat():
    """Test pivotal detectors handle flat."""
    df = _flat_df(100)
    atr = _atr(df)
    for name, fn in PIVOTAL_DETECTORS:
        out = safe_detect(fn, df, [], atr, _detector_name=name)
        assert out == [], f"{name} returned signals on auction-freeze data: {out!r}"


def test_df_only_detectors_handle_flat():
    """Test df only detectors handle flat."""
    df = _flat_df(100)
    for name, fn in DF_ONLY_DETECTORS:
        out = safe_detect(fn, df, _detector_name=name)
        assert out == [], f"{name} returned signals on flat df: {out!r}"


def test_vpvr_handles_flat_range():
    """VPVR has special-case for p_max == p_min — one bin, all volume."""
    df = _flat_df(80)
    profile = compute_vpvr(df)
    assert isinstance(profile, list)
    out = detect_vpvr_signal(df, current_price=100.0, profile=profile)
    assert out == []


def test_pivotal_detectors_handle_gap():
    """Test pivotal detectors handle gap."""
    df = _gap_df(100)
    atr = _atr(df)
    pivots = find_pivots(df, atr=atr, method="zigzag")
    for name, fn in PIVOTAL_DETECTORS:
        out = safe_detect(fn, df, pivots, atr, _detector_name=name)
        assert isinstance(out, list), f"{name} returned non-list on gap df"


def test_df_only_detectors_handle_gap():
    """Test df only detectors handle gap."""
    df = _gap_df(100)
    for name, fn in DF_ONLY_DETECTORS:
        out = safe_detect(fn, df, _detector_name=name)
        assert isinstance(out, list)


def test_detectors_handle_zero_atr():
    """Test detectors handle zero atr."""
    df = _normal_df(120)
    zero_atr = pd.Series([0.0] * len(df))
    for name, fn in PIVOTAL_DETECTORS:
        out = safe_detect(fn, df, [], zero_atr, _detector_name=name)
        assert out == [], f"{name} returned signals with zero ATR: {out!r}"


def test_detectors_handle_nan_atr():
    """Test detectors handle nan atr."""
    df = _normal_df(120)
    nan_atr = pd.Series([np.nan] * len(df))
    for name, fn in PIVOTAL_DETECTORS:
        out = safe_detect(fn, df, [], nan_atr, _detector_name=name)
        assert isinstance(out, list)


def test_detectors_handle_inf_atr():
    """Test detectors handle inf atr."""
    df = _normal_df(120)
    inf_atr = pd.Series([np.inf] * len(df))
    for name, fn in PIVOTAL_DETECTORS:
        out = safe_detect(fn, df, [], inf_atr, _detector_name=name)
        assert isinstance(out, list)


def test_detectors_survive_garbage_pivots():
    """Test detectors survive garbage pivots."""
    df = _normal_df(120)
    atr = _atr(df)
    garbage = _garbage_pivots()
    for name, fn in PIVOTAL_DETECTORS:
        out = safe_detect(fn, df, garbage, atr, _detector_name=name)
        assert isinstance(out, list), f"{name} crashed on garbage pivots"


def test_safe_detect_catches_missing_high_column():
    """Without safe_detect, detectors like detect_flag would raise KeyError."""
    df = _df_no_high()
    atr = pd.Series([0.5] * len(df))
    for name, fn in PIVOTAL_DETECTORS:
        out = safe_detect(fn, df, [], atr, _detector_name=name, _ticker="TEST")
        assert out == [], f"{name} did not return [] on missing 'high'"


def test_safe_detect_catches_missing_close_column():
    """Test safe detect catches missing close column."""
    df = _df_no_close()
    atr = pd.Series([0.5] * len(df))
    for name, fn in PIVOTAL_DETECTORS:
        out = safe_detect(fn, df, [], atr, _detector_name=name)
        assert out == []


def test_safe_detect_catches_missing_columns_in_df_only_detectors():
    """Test safe detect catches missing columns in df only detectors."""
    df = _df_no_high()
    for name, fn in DF_ONLY_DETECTORS:
        out = safe_detect(fn, df, _detector_name=name)
        assert out == []


def test_dasha_handles_empty_df():
    """Test dasha handles empty df."""
    df = _empty_df()
    piv = pd.DataFrame(columns=["pivot_i", "time", "price", "type"])
    assert detect_double_patterns_dasha(df, piv) == []
    assert detect_hs_patterns_dasha(df, piv) == []
    assert detect_all_dasha_patterns(df) == []


def test_dasha_handles_flat_df():
    """Test dasha handles flat df."""
    df = _flat_df(100)
    out = detect_all_dasha_patterns(df)
    assert out == []
    piv = zigzag_atr_pivots_hilo(df)
    assert isinstance(piv, pd.DataFrame)


def test_dasha_handles_df_missing_atr_col():
    """No atr14 column → detectors return []."""
    df = pd.DataFrame(
        {
            "open": [100.0] * 50,
            "high": [101.0] * 50,
            "low": [99.0] * 50,
            "close": [100.0] * 50,
            "volume": [1000.0] * 50,
        }
    )
    out = detect_all_dasha_patterns(df)
    assert out == []


def test_dasha_handles_garbage_pivot_df():
    """Test dasha handles garbage pivot df."""
    df = _normal_df(120)
    df["atr14"] = _atr(df)
    garbage_piv = pd.DataFrame(
        {
            "pivot_i": [-1, 99_999, 0],
            "time": [None, None, None],
            "price": [float("nan"), float("inf"), 100.0],
            "type": [1, -1, 1],
        }
    )
    out_dt = safe_detect(detect_double_patterns_dasha, df, garbage_piv, _detector_name="dasha_dt")
    out_hs = safe_detect(detect_hs_patterns_dasha, df, garbage_piv, _detector_name="dasha_hs")
    assert out_dt == [] and out_hs == []


def test_vpvr_helpers_handle_empty_profile():
    """Test vpvr helpers handle empty profile."""
    assert find_hvn([]) == []
    assert find_lvn([]) == []


def test_vpvr_signal_handles_negative_price():
    """Test vpvr signal handles negative price."""
    df = _normal_df(60)
    profile = compute_vpvr(df)
    out = detect_vpvr_signal(df, current_price=-50.0, profile=profile)
    assert out == []


def test_vpvr_signal_handles_nan_price():
    """Test vpvr signal handles nan price."""
    df = _normal_df(60)
    profile = compute_vpvr(df)
    out = safe_detect(detect_vpvr_signal, df, float("nan"), profile, _detector_name="vpvr")
    assert out == []


def test_candle_patterns_handle_garbage_inputs():
    """detect_candle_patterns has its own try/except — verify the contract."""
    bad_dfs = [
        _empty_df(),
        _all_nan_df(),
        _flat_df(50),
        _one_row_df(),
        pd.DataFrame({}),
    ]
    for df in bad_dfs:
        out = detect_candle_patterns(df)
        assert isinstance(out, (pd.DataFrame, dict))


def test_latest_candle_signal_handles_garbage_inputs():
    """Test latest candle signal handles garbage inputs."""
    for df in [_empty_df(), _all_nan_df(), _flat_df(50), _one_row_df()]:
        sig = latest_candle_signal(df)
        assert isinstance(sig, dict)


def test_safe_detect_returns_list_on_success():
    """Test safe detect returns list on success."""
    df = _normal_df(80)
    atr = _atr(df)
    out = safe_detect(detect_double_top_bottom, df, [], atr)
    assert isinstance(out, list)


def test_safe_detect_swallows_exceptions():
    """Test safe detect swallows exceptions."""

    def boom(*_args, **_kwargs):
        """Boom."""
        raise RuntimeError("simulated crash")

    out = safe_detect(boom, _detector_name="boom", _ticker="SBER")
    assert out == []


def test_safe_detect_swallows_keyerror():
    """Test safe detect swallows keyerror."""

    def boom(*_args, **_kwargs):
        """Boom."""
        raise KeyError("high")

    assert safe_detect(boom, _detector_name="boom_keyerror") == []


def test_safe_detect_swallows_zerodiv():
    """Test safe detect swallows zerodiv."""

    def boom(*_args, **_kwargs):
        """Boom."""
        return 1 / 0

    assert safe_detect(boom) == []


def test_safe_detect_coerces_none_to_empty():
    """Test safe detect coerces none to empty."""

    def returns_none(*_args, **_kwargs):
        """Returns none."""
        return None

    assert safe_detect(returns_none) == []


def test_safe_detect_coerces_non_list_to_empty():
    """A scalar / dict / generator return is treated as zero signals."""

    def returns_scalar(*_args, **_kwargs):
        """Returns scalar."""
        return 42

    def returns_dict(*_args, **_kwargs):
        """Returns dict."""
        return {"a": 1}

    assert safe_detect(returns_scalar) == []
    assert safe_detect(returns_dict) == []


def test_safe_detect_iterable_materialized():
    """An iterable (generator) is materialised into a list."""

    def returns_gen(*_args, **_kwargs):
        """Returns gen."""
        return (i for i in range(3))

    out = safe_detect(returns_gen)
    assert out == [0, 1, 2]


def test_safe_detect_propagates_keyboard_interrupt():
    """KeyboardInterrupt is a BaseException — MUST propagate."""

    def boom(*_args, **_kwargs):
        """Boom."""
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        safe_detect(boom)


def test_safe_detect_propagates_system_exit():
    """Test safe detect propagates system exit."""

    def boom(*_args, **_kwargs):
        """Boom."""
        raise SystemExit

    with pytest.raises(SystemExit):
        safe_detect(boom)


def test_safe_detect_logs_at_debug_not_error(caplog):
    """Detector failures MUST log at DEBUG — they are expected edge cases,
    not bugs to escalate to ops."""

    def boom(*_args, **_kwargs):
        """Boom."""
        raise ValueError("simulated")

    with caplog.at_level(logging.DEBUG, logger="app.agents.ta_patterns.safe_runner"):
        out = safe_detect(boom, _detector_name="test_boom", _ticker="X")

    assert out == []
    warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warn_records == [], (
        f"safe_detect must not log at WARNING/ERROR — got: {[r.message for r in warn_records]}"
    )
    debug_records = [
        r for r in caplog.records if r.levelno == logging.DEBUG and "safe_detect" in r.message
    ]
    assert debug_records, "expected at least one DEBUG record from safe_detect"


def test_safe_detect_forwards_args_and_kwargs():
    """Positional + keyword args are passed through unchanged."""
    received = {}

    def detector(df, pivots, atr, *, threshold=0.5):
        """Detector."""
        received["df"] = df
        received["pivots"] = pivots
        received["atr"] = atr
        received["threshold"] = threshold
        return [1, 2, 3]

    df = _normal_df(20)
    atr = _atr(df)
    out = safe_detect(detector, df, ["pivot"], atr, threshold=0.7)
    assert out == [1, 2, 3]
    assert received["pivots"] == ["pivot"]
    assert received["threshold"] == 0.7


def test_all_registered_detectors_run_through_safe_detect():
    """Test all registered detectors run through safe detect."""
    df = _normal_df(150)
    atr = _atr(df)
    pivots = find_pivots(df, atr=atr, method="zigzag")
    for name, fn in PIVOTAL_DETECTORS:
        out = safe_detect(fn, df, pivots, atr, _detector_name=name)
        assert isinstance(out, list)
    for name, fn in DF_ONLY_DETECTORS:
        out = safe_detect(fn, df, _detector_name=name)
        assert isinstance(out, list)


def test_smc_and_chart_extra_registries_are_safe():
    """Verify the module-level *_DETECTORS lists work through safe_detect."""
    df = _normal_df(120)
    atr = _atr(df)
    pivots = find_pivots(df, atr=atr, method="zigzag")
    for fn in SMC_DETECTORS:
        out = safe_detect(fn, df, pivots, atr)
        assert isinstance(out, list)
    for fn in HARMONIC_DETECTORS:
        out = safe_detect(fn, df, pivots, atr)
        assert isinstance(out, list)
    for fn in CHART_EXTRA_DETECTORS:
        out = safe_detect(fn, df, pivots, atr)
        assert isinstance(out, list)
