"""Тесты для adaptive risk regime (v0.18.0).

Проверяем что compute_risk_regime() правильно определяет режим
из runtime-метрик: drawdown, losing_streak, daily_pnl, секунды до закрытия.

Принцип: система должна сама подстраиваться, никаких env-флагов.
"""

from __future__ import annotations

import pytest

from app.risk.adaptive_regime import compute_risk_regime


class TestNormalRegime:
    """NORMAL — все метрики в норме, обычное поведение."""

    def test_zero_drawdown_zero_streak(self):
        """Test zero drawdown zero streak."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.0,
            losing_streak=0,
            daily_pnl_pct=0.0,
        )
        assert r.name == "NORMAL"
        assert r.size_multiplier == 1.0
        assert r.hard_sl_pct is None
        assert r.hard_tp_pct is None
        assert r.can_open_new is True

    def test_small_positive_pnl(self):
        """Прибыль за день +1% → всё ещё NORMAL."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.0,
            losing_streak=0,
            daily_pnl_pct=0.01,
        )
        assert r.name == "NORMAL"


class TestCautiousRegime:
    """CAUTIOUS — небольшие проблемы или конец сессии."""

    def test_drawdown_0p5_pct(self):
        """Test drawdown 0p5 pct."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.005,
            losing_streak=0,
            daily_pnl_pct=0.0,
        )
        assert r.name == "CAUTIOUS"
        assert r.size_multiplier == 0.7
        assert r.hard_sl_pct == 0.012
        assert r.can_open_new is True

    def test_losing_streak_2(self):
        """Test losing streak 2."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.0,
            losing_streak=2,
            daily_pnl_pct=0.0,
        )
        assert r.name == "CAUTIOUS"

    def test_end_of_session_triggers_cautious(self):
        """Last 30 min before close — затягиваем стопы даже без drawdown."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.0,
            losing_streak=0,
            daily_pnl_pct=0.0,
            seconds_until_close=1500.0,
        )
        assert r.name == "CAUTIOUS"

    def test_session_open_no_cautious(self):
        """Много времени до закрытия → НЕ CAUTIOUS."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.0,
            losing_streak=0,
            daily_pnl_pct=0.0,
            seconds_until_close=14400.0,
        )
        assert r.name == "NORMAL"


class TestDefensiveRegime:
    """DEFENSIVE — серьёзная просадка, режем size в 2.5×."""

    def test_drawdown_2_pct(self):
        """Test drawdown 2 pct."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.02,
            losing_streak=0,
            daily_pnl_pct=0.0,
        )
        assert r.name == "DEFENSIVE"
        assert r.size_multiplier == 0.4
        assert r.hard_sl_pct == 0.010
        assert r.hard_tp_pct == 0.008
        assert r.can_open_new is True

    def test_losing_streak_3(self):
        """Test losing streak 3."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.0,
            losing_streak=3,
            daily_pnl_pct=0.0,
        )
        assert r.name == "DEFENSIVE"

    def test_daily_pnl_minus_1_pct(self):
        """Test daily pnl minus 1 pct."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.0,
            losing_streak=0,
            daily_pnl_pct=-0.01,
        )
        assert r.name == "DEFENSIVE"


class TestCrisisRegime:
    """CRISIS — small-size mode (v0.19.6 raised thresholds, kept new entries).

    Triggers now: DD >= 6%, losing_streak >= 5, daily_pnl <= -2.5%.
    Size mult = 0.25× (was 0.0 in v0.18.0). can_open_new = True.
    """

    def test_drawdown_7_pct(self):
        """Test drawdown 7 pct."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.07,
            losing_streak=0,
            daily_pnl_pct=0.0,
        )
        assert r.name == "CRISIS"
        assert r.size_multiplier == 0.25
        assert r.hard_sl_pct == 0.005
        assert r.hard_tp_pct == 0.005
        assert r.can_open_new is True

    def test_losing_streak_5(self):
        """Test losing streak 5."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.0,
            losing_streak=5,
            daily_pnl_pct=0.0,
        )
        assert r.name == "CRISIS"
        assert r.can_open_new is True

    def test_daily_pnl_minus_3_pct(self):
        """-3% daily — критический убыток, small-size mode."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.0,
            losing_streak=0,
            daily_pnl_pct=-0.03,
        )
        assert r.name == "CRISIS"

    def test_drawdown_10_pct_extreme(self):
        """Экстремальная просадка — всё равно CRISIS, не выше."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.10,
            losing_streak=10,
            daily_pnl_pct=-0.05,
        )
        assert r.name == "CRISIS"
        assert r.size_multiplier == 0.25


class TestRegimeOrdering:
    """Проверяем что более серьёзный триггер побеждает менее серьёзный."""

    def test_crisis_overrides_defensive(self):
        """v0.19.6: DD=7% И losing_streak=4 → CRISIS, не DEFENSIVE."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.07,
            losing_streak=4,
            daily_pnl_pct=-0.01,
        )
        assert r.name == "CRISIS"

    def test_defensive_overrides_cautious(self):
        """Test defensive overrides cautious."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.02,
            losing_streak=2,
            daily_pnl_pct=0.0,
        )
        assert r.name == "DEFENSIVE"

    def test_size_mult_monotonic(self):
        """size_multiplier должен убывать с увеличением риска (с >= для crisis vs defensive)."""
        normal = compute_risk_regime(
            current_drawdown_from_peak_pct=0.0, losing_streak=0, daily_pnl_pct=0.0
        )
        cautious = compute_risk_regime(
            current_drawdown_from_peak_pct=0.005, losing_streak=0, daily_pnl_pct=0.0
        )
        defensive = compute_risk_regime(
            current_drawdown_from_peak_pct=0.02, losing_streak=0, daily_pnl_pct=0.0
        )
        crisis = compute_risk_regime(
            current_drawdown_from_peak_pct=0.07, losing_streak=0, daily_pnl_pct=0.0
        )
        assert normal.size_multiplier > cautious.size_multiplier
        assert cautious.size_multiplier > defensive.size_multiplier
        assert defensive.size_multiplier > crisis.size_multiplier


class TestRegimeEdgeCases:
    """Граничные случаи и устойчивость."""

    def test_negative_drawdown_treated_as_abs(self):
        """DD=-0.07 (отрицательное число) — тоже триггерит CRISIS (abs(-0.07) = 0.07 >= 0.06)."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=-0.07,
            losing_streak=0,
            daily_pnl_pct=0.0,
        )
        assert r.name == "CRISIS"

    def test_negative_losing_streak_clamped_to_zero(self):
        """Test negative losing streak clamped to zero."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.0,
            losing_streak=-5,
            daily_pnl_pct=0.0,
        )
        assert r.name == "NORMAL"

    def test_positive_daily_pnl_irrelevant_to_crisis(self):
        """Положительный daily_pnl не триггерит CRISIS даже при большой DD."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.06,
            losing_streak=0,
            daily_pnl_pct=0.05,
        )
        assert r.name == "CRISIS"

    def test_reason_is_human_readable(self):
        """Test reason is human readable."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.05,
            losing_streak=5,
            daily_pnl_pct=-0.02,
        )
        assert "CRISIS" in r.reason
        assert "dd" in r.reason.lower()

    def test_immutability(self):
        """RiskRegime — frozen dataclass, нельзя мутировать."""
        r = compute_risk_regime(
            current_drawdown_from_peak_pct=0.0, losing_streak=0, daily_pnl_pct=0.0
        )
        with pytest.raises((AttributeError, Exception)):
            r.name = "CRISIS"  # type: ignore[misc]


class TestNoEnvFlagDependency:
    """Регрессионный тест: compute_risk_regime НЕ зависит от env vars.

    Это и есть adaptive system — поведение определяется runtime-метриками,
    не FIRST_DAY_PROTECTION/STRICT_MODE флагами.
    """

    def test_same_inputs_same_output_regardless_of_env(self, monkeypatch):
        """Test same inputs same output regardless of env."""
        monkeypatch.setenv("STRICT_MODE", "0")
        r1 = compute_risk_regime(
            current_drawdown_from_peak_pct=0.02, losing_streak=0, daily_pnl_pct=0.0
        )
        monkeypatch.setenv("STRICT_MODE", "1")
        r2 = compute_risk_regime(
            current_drawdown_from_peak_pct=0.02, losing_streak=0, daily_pnl_pct=0.0
        )
        assert r1.name == r2.name == "DEFENSIVE"
        assert r1.size_multiplier == r2.size_multiplier
