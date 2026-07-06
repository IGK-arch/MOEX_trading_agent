"""Адаптивный риск-режим — вычисляет уровень риска из метрик выполнения."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import app.config as cfg

RegimeName = Literal["NORMAL", "CAUTIOUS", "DEFENSIVE", "CRISIS"]

@dataclass(frozen=True)
class RiskRegime:
    """Current risk regime, computed on the fly."""

    name: RegimeName
    size_multiplier: float
    hard_sl_pct: float | None
    hard_tp_pct: float | None
    can_open_new: bool
    reason: str

    def __str__(self) -> str:
        """Str."""
        return (
            f"RiskRegime({self.name}, size×{self.size_multiplier}, "
            f"hard_sl={self.hard_sl_pct}, hard_tp={self.hard_tp_pct}, "
            f"can_open_new={self.can_open_new})"
        )

def compute_risk_regime(
    *,
    current_drawdown_from_peak_pct: float,
    losing_streak: int,
    daily_pnl_pct: float,
    seconds_until_close: float | None = None,
) -> RiskRegime:
    """Compute current risk regime from runtime metrics.

    Args:
        current_drawdown_from_peak_pct: drawdown from peak equity (positive)
        losing_streak: consecutive losing trades
        daily_pnl_pct: today's PnL as % of deposit
        seconds_until_close: seconds until market close (None if unknown)
    Returns:
        RiskRegime: regime parameters for current cycle
    """
    if getattr(cfg, "DISABLE_REGIME_DD_CHECK", False):
        dd = 0.0
    else:
        dd = abs(current_drawdown_from_peak_pct)
    ls = max(0, int(losing_streak))
    dpnl = float(daily_pnl_pct)
    secs_close = float(seconds_until_close) if seconds_until_close is not None else 1e9

    force_name = getattr(cfg, "FORCE_REGIME_NAME", "")
    if force_name in ("NORMAL", "CAUTIOUS", "DEFENSIVE", "CRISIS"):
        size_mult_map = {
            "NORMAL": cfg.ADAPTIVE_NORMAL_SIZE_MULT,
            "CAUTIOUS": cfg.ADAPTIVE_CAUTIOUS_SIZE_MULT,
            "DEFENSIVE": cfg.ADAPTIVE_DEFENSIVE_SIZE_MULT,
            "CRISIS": cfg.ADAPTIVE_CRISIS_SIZE_MULT,
        }
        sl_map = {
            "NORMAL": cfg.ADAPTIVE_NORMAL_HARD_SL_PCT,
            "CAUTIOUS": cfg.ADAPTIVE_CAUTIOUS_HARD_SL_PCT,
            "DEFENSIVE": cfg.ADAPTIVE_DEFENSIVE_HARD_SL_PCT,
            "CRISIS": cfg.ADAPTIVE_CRISIS_HARD_SL_PCT,
        }
        tp_map = {
            "NORMAL": cfg.ADAPTIVE_NORMAL_HARD_TP_PCT,
            "CAUTIOUS": cfg.ADAPTIVE_CAUTIOUS_HARD_TP_PCT,
            "DEFENSIVE": cfg.ADAPTIVE_DEFENSIVE_HARD_TP_PCT,
            "CRISIS": cfg.ADAPTIVE_CRISIS_HARD_TP_PCT,
        }
        return RiskRegime(
            name=force_name,  # type: ignore[arg-type]
            size_multiplier=size_mult_map[force_name],
            hard_sl_pct=sl_map[force_name],
            hard_tp_pct=tp_map[force_name],
            can_open_new=size_mult_map[force_name] > 0.0,
            reason=f"FORCE_REGIME_NAME={force_name} (env override)",
        )

    if (
        dd >= cfg.ADAPTIVE_CRISIS_DD_PCT
        or ls >= cfg.ADAPTIVE_CRISIS_LOSING_STREAK
        or dpnl <= cfg.ADAPTIVE_CRISIS_DAILY_PNL_PCT
    ):
        return RiskRegime(
            name="CRISIS",
            size_multiplier=cfg.ADAPTIVE_CRISIS_SIZE_MULT,
            hard_sl_pct=cfg.ADAPTIVE_CRISIS_HARD_SL_PCT,
            hard_tp_pct=cfg.ADAPTIVE_CRISIS_HARD_TP_PCT,
            can_open_new=cfg.ADAPTIVE_CRISIS_SIZE_MULT > 0.0,
            reason=(f"CRISIS: dd={dd:.1%} losing_streak={ls} daily_pnl={dpnl:.2%}"),
        )

    if (
        dd >= cfg.ADAPTIVE_DEFENSIVE_DD_PCT
        or ls >= cfg.ADAPTIVE_DEFENSIVE_LOSING_STREAK
        or dpnl <= cfg.ADAPTIVE_DEFENSIVE_DAILY_PNL_PCT
    ):
        return RiskRegime(
            name="DEFENSIVE",
            size_multiplier=cfg.ADAPTIVE_DEFENSIVE_SIZE_MULT,
            hard_sl_pct=cfg.ADAPTIVE_DEFENSIVE_HARD_SL_PCT,
            hard_tp_pct=cfg.ADAPTIVE_DEFENSIVE_HARD_TP_PCT,
            can_open_new=True,
            reason=(f"DEFENSIVE: dd={dd:.1%} losing_streak={ls} daily_pnl={dpnl:.2%}"),
        )

    if (
        dd >= cfg.ADAPTIVE_CAUTIOUS_DD_PCT
        or ls >= cfg.ADAPTIVE_CAUTIOUS_LOSING_STREAK
        or dpnl <= cfg.ADAPTIVE_CAUTIOUS_DAILY_PNL_PCT
        or secs_close <= 1800.0
    ):
        eoc = "end-of-session" if secs_close <= 1800.0 else ""
        return RiskRegime(
            name="CAUTIOUS",
            size_multiplier=cfg.ADAPTIVE_CAUTIOUS_SIZE_MULT,
            hard_sl_pct=cfg.ADAPTIVE_CAUTIOUS_HARD_SL_PCT,
            hard_tp_pct=cfg.ADAPTIVE_CAUTIOUS_HARD_TP_PCT,
            can_open_new=True,
            reason=(f"CAUTIOUS: dd={dd:.1%} losing_streak={ls} daily_pnl={dpnl:.2%} {eoc}"),
        )

    return RiskRegime(
        name="NORMAL",
        size_multiplier=cfg.ADAPTIVE_NORMAL_SIZE_MULT,
        hard_sl_pct=cfg.ADAPTIVE_NORMAL_HARD_SL_PCT,
        hard_tp_pct=cfg.ADAPTIVE_NORMAL_HARD_TP_PCT,
        can_open_new=True,
        reason="NORMAL — все метрики в норме",
    )
