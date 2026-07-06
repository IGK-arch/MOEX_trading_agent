"""Per-pattern-family SL/TP rules + time-stop (Phase 28 / v0.3.0)."""

from __future__ import annotations

import pytest

from app.risk.sl_tp_rules import (
    RULES,
    FamilyRule,
    adaptive_sl_atr,
    derive_sl_tp,
    family_of,
    rule_for,
)
from app.risk.trailing_stop import (
    DEFAULT_LADDER,
    LADDER_BY_FAMILY,
    compute_trailing_stop,
    ladder_for_family,
    should_time_stop,
)


def test_family_of_smc_prefix():
    """Test family of smc prefix."""
    assert family_of("smc_order_block_bull") == "smc"
    assert family_of("smc_fvg_bear") == "smc"
    assert family_of("smc_sweep_high") == "smc"
    assert family_of("smc_choch_bull") == "smc"


def test_family_of_chart_patterns():
    """Test family of chart patterns."""
    assert family_of("double_top") == "chart"
    assert family_of("head_shoulders") == "chart"
    assert family_of("bull_flag") == "chart"
    assert family_of("ascending_triangle") == "chart"
    assert family_of("rising_wedge") == "chart"
    assert family_of("rounding_bottom") == "chart"


def test_family_of_candle_patterns():
    """Test family of candle patterns."""
    assert family_of("candle_cdldoji") == "candle"
    assert family_of("candle_cdlhammer") == "candle"
    assert family_of("three_white_soldiers_vol") == "candle"
    assert family_of("three_black_crows_vol") == "candle"


def test_family_of_research_patterns():
    """Test family of research patterns."""
    assert family_of("vcp") == "research"
    assert family_of("bb_squeeze_breakout") == "research"
    assert family_of("inside_bar_breakout") == "research"
    assert family_of("pivot_reversal_long") == "research"


def test_family_of_unknown_returns_other():
    """Test family of unknown returns other."""
    assert family_of("nonexistent_pattern") == "other"
    assert family_of("") == "other"
    assert family_of(None) == "other"  # type: ignore[arg-type]


def test_rule_for_known_family():
    """Test rule for known family."""
    chart = rule_for("double_top")
    assert isinstance(chart, FamilyRule)
    assert chart.sl_atr == 3.0
    assert chart.rr == 1.5
    assert chart.time_stop_bars == 32


def test_rule_for_smc():
    """Test rule for smc."""
    smc = rule_for("smc_order_block_bull")
    assert smc.sl_atr == 2.0
    assert smc.time_stop_bars == 12


def test_rule_for_unknown_uses_other_default():
    """Test rule for unknown uses other default."""
    other = rule_for("some_unknown_pattern_xyz")
    assert other is RULES["other"]


def test_adaptive_sl_no_vol_returns_base():
    """Test adaptive sl no vol returns base."""
    assert adaptive_sl_atr("double_top") == 3.0
    assert adaptive_sl_atr("double_top", vol_ratio=None) == 3.0
    assert adaptive_sl_atr("double_top", vol_ratio=1.0) == 3.0


def test_adaptive_sl_high_vol_widens():
    """When current ATR is 2× the median, widen stops by 20%."""
    chart_rule = RULES["chart"]
    expected = chart_rule.sl_atr * chart_rule.regime_high_sl_mult
    assert adaptive_sl_atr("double_top", vol_ratio=2.0) == pytest.approx(expected)


def test_adaptive_sl_low_vol_tightens():
    """Test adaptive sl low vol tightens."""
    chart_rule = RULES["chart"]
    expected = chart_rule.sl_atr * chart_rule.regime_low_sl_mult
    assert adaptive_sl_atr("double_top", vol_ratio=0.5) == pytest.approx(expected)


def test_adaptive_sl_within_band_no_change():
    """vol_ratio between 0.7 and 1.5 → no adjustment."""
    assert adaptive_sl_atr("vcp", vol_ratio=1.2) == RULES["research"].sl_atr
    assert adaptive_sl_atr("vcp", vol_ratio=0.8) == RULES["research"].sl_atr


def test_derive_sl_tp_buy_uses_family_default():
    """No detector stop → derive from family defaults."""
    sl, tp = derive_sl_tp(
        pattern="vcp",
        direction="BUY",
        entry=100.0,
        atr=1.0,
    )
    assert sl == pytest.approx(98.0)
    assert tp == pytest.approx(106.0)


def test_derive_sl_tp_sell_uses_family_default():
    """Test derive sl tp sell uses family default."""
    sl, tp = derive_sl_tp(
        pattern="vcp",
        direction="SELL",
        entry=100.0,
        atr=1.0,
    )
    assert sl == pytest.approx(102.0)
    assert tp == pytest.approx(94.0)


def test_derive_sl_tp_buy_clips_detector_stop_to_safer():
    """Detector stop tighter than family → keep the safer one (in-band)."""
    sl, tp = derive_sl_tp(
        pattern="vcp",
        direction="BUY",
        entry=100.0,
        atr=1.0,
        detector_stop=99.0,
    )
    assert sl == pytest.approx(99.0)
    risk = 100.0 - sl
    assert tp == pytest.approx(100.0 + 3.0 * risk)


def test_derive_sl_tp_buy_rejects_too_tight_detector_stop():
    """Detector stop tighter than 1/3 of family default → use tight_bound."""
    sl, _tp = derive_sl_tp(
        pattern="vcp",
        direction="BUY",
        entry=100.0,
        atr=1.0,
        detector_stop=99.8,
    )
    assert sl == pytest.approx(99.333, abs=1e-3)


def test_derive_sl_tp_buy_ignores_absurdly_loose_detector_stop():
    """Detector stop 5 ATR away → clamp to 1.5× family."""
    sl, tp = derive_sl_tp(
        pattern="vcp",
        direction="BUY",
        entry=100.0,
        atr=1.0,
        detector_stop=80.0,
    )
    assert sl == pytest.approx(98.0)


def test_derive_sl_tp_zero_atr_returns_detector_levels():
    """Defensive: missing ATR → fall back to detector."""
    sl, tp = derive_sl_tp(
        pattern="vcp",
        direction="BUY",
        entry=100.0,
        atr=0.0,
        detector_stop=97.0,
        detector_target=110.0,
    )
    assert sl == 97.0
    assert tp == 110.0


def test_derive_sl_tp_unknown_direction_returns_detector():
    """Test derive sl tp unknown direction returns detector."""
    sl, tp = derive_sl_tp(
        pattern="vcp",
        direction="FLAT",
        entry=100.0,
        atr=1.0,
        detector_stop=98.5,
        detector_target=104.0,
    )
    assert sl == 98.5
    assert tp == 104.0


def test_ladder_for_chart_locks_earlier_than_default():
    """chart family locks half at +1.5 ATR, not +2.0."""
    chart_ladder = ladder_for_family("chart")
    assert chart_ladder is LADDER_BY_FAMILY["chart"]
    assert chart_ladder[1].profit_atr == 1.5
    assert chart_ladder[1].lock_atr == 0.75


def test_ladder_for_research_keeps_default():
    """Research RR=3 → keep the 1/2/3 ladder."""
    assert ladder_for_family("research") is DEFAULT_LADDER


def test_ladder_for_unknown_falls_back_to_default():
    """Test ladder for unknown falls back to default."""
    assert ladder_for_family("nonexistent") is DEFAULT_LADDER
    assert ladder_for_family(None) is DEFAULT_LADDER
    assert ladder_for_family("") is DEFAULT_LADDER


def test_chart_ladder_triggers_lock_at_1_5_atr_buy():
    """At +1.5 ATR a chart-family trade locks +0.75 ATR — better than the
    default ladder which would still be at break-even."""
    trail = compute_trailing_stop(
        direction="BUY",
        entry_price=100.0,
        current_price=101.5,
        atr=1.0,
        ladder=ladder_for_family("chart"),
    )
    assert trail == pytest.approx(100.75)


def test_time_stop_disabled_when_max_bars_none():
    """Test time stop disabled when max bars none."""
    assert (
        should_time_stop(
            direction="BUY",
            entry_price=100.0,
            current_price=99.0,
            bars_held=999,
            max_bars=None,
        )
        is False
    )


def test_time_stop_disabled_when_not_yet_reached():
    """Test time stop disabled when not yet reached."""
    assert (
        should_time_stop(
            direction="BUY",
            entry_price=100.0,
            current_price=99.5,
            bars_held=5,
            max_bars=10,
        )
        is False
    )


def test_time_stop_fires_when_unprofitable_buy():
    """BUY trade held past max_bars and below entry → close."""
    assert (
        should_time_stop(
            direction="BUY",
            entry_price=100.0,
            current_price=99.9,
            bars_held=16,
            max_bars=16,
        )
        is True
    )


def test_time_stop_blocks_when_in_profit():
    """If the trade is currently in profit, let it run."""
    assert (
        should_time_stop(
            direction="BUY",
            entry_price=100.0,
            current_price=101.0,
            bars_held=20,
            max_bars=16,
        )
        is False
    )


def test_time_stop_fires_sell_above_entry():
    """Test time stop fires sell above entry."""
    assert (
        should_time_stop(
            direction="SELL",
            entry_price=100.0,
            current_price=100.5,
            bars_held=20,
            max_bars=12,
        )
        is True
    )


def test_time_stop_blocks_sell_in_profit():
    """Test time stop blocks sell in profit."""
    assert (
        should_time_stop(
            direction="SELL",
            entry_price=100.0,
            current_price=99.5,
            bars_held=20,
            max_bars=12,
        )
        is False
    )


def test_time_stop_handles_zero_price():
    """Defensive: a zero current_price is a data anomaly → don't fire."""
    assert (
        should_time_stop(
            direction="BUY",
            entry_price=100.0,
            current_price=0.0,
            bars_held=20,
            max_bars=10,
        )
        is False
    )
