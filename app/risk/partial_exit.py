"""Partial profit-taking ladder."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class PartialExitPlan:
    """How to split a position across TP1 + TP2."""

    qty_total: int
    qty_tp1: int
    qty_tp2: int
    tp1_price: float | None
    tp2_price: float | None

    @property
    def has_partial(self) -> bool:
        """Return True when partial-exit logic should be used.

        Returns:
            bool: True if partial exit is configured
        """
        return self.qty_tp1 > 0 and self.tp1_price is not None

def plan_partial_exit(
    quantity: int,
    take_profit: float | None,
    take_profit_1: float | None = None,
    take_profit_2: float | None = None,
    entry_price: float | None = None,
) -> PartialExitPlan:
    """Build the (qty_tp1, qty_tp2, tp1, tp2) plan for a position.

    Args:
        quantity: total position size
        take_profit: full TP price or None
        take_profit_1: explicit TP1 or None
        take_profit_2: explicit TP2 or None
        entry_price: entry price for midpoint derivation
    Returns:
        PartialExitPlan: split plan
    """
    if quantity <= 0:
        return PartialExitPlan(0, 0, 0, None, None)

    tp1 = take_profit_1
    tp2 = take_profit_2 if take_profit_2 is not None else take_profit

    if tp1 is None and take_profit is not None and entry_price is not None and entry_price > 0:
        tp1 = entry_price + 0.5 * (take_profit - entry_price)
        if tp2 is None:
            tp2 = take_profit

    if tp1 is None or quantity < 2:
        return PartialExitPlan(
            qty_total=quantity,
            qty_tp1=0,
            qty_tp2=quantity,
            tp1_price=tp1,
            tp2_price=tp2,
        )

    qty_tp1 = quantity // 2
    qty_tp2 = quantity - qty_tp1
    return PartialExitPlan(
        qty_total=quantity,
        qty_tp1=qty_tp1,
        qty_tp2=qty_tp2,
        tp1_price=tp1,
        tp2_price=tp2,
    )

@dataclass(frozen=True)
class PartialExitPlan3(PartialExitPlan):
    """3-tier partial-exit plan with a trailing leg."""

    qty_tp3: int = 0
    tp3_price: float | None = None

    @property
    def has_three_tier(self) -> bool:
        """Return True if a third trailing leg is configured.

        Returns:
            bool: True if 3-tier active
        """
        return self.qty_tp3 > 0

def plan_partial_exit_3tier(
    quantity: int,
    entry_price: float,
    atr: float,
    direction: str,
) -> PartialExitPlan3:
    """Build a 33% / 33% / trailing partial-exit plan.

    Args:
        quantity: total position size
        entry_price: entry price
        atr: ATR value
        direction: BUY or SELL
    Returns:
        PartialExitPlan3: 3-tier split plan
    """
    if quantity <= 0 or entry_price <= 0 or atr <= 0:
        return PartialExitPlan3(0, 0, 0, None, None, qty_tp3=0, tp3_price=None)

    dir_u = (direction or "").upper()
    if dir_u == "BUY":
        tp1_p = entry_price + 1.0 * atr
        tp2_p = entry_price + 2.0 * atr
    elif dir_u == "SELL":
        tp1_p = entry_price - 1.0 * atr
        tp2_p = entry_price - 2.0 * atr
    else:
        return PartialExitPlan3(quantity, 0, quantity, None, None, qty_tp3=0, tp3_price=None)

    if quantity == 1:
        return PartialExitPlan3(
            qty_total=1,
            qty_tp1=0,
            qty_tp2=0,
            tp1_price=tp1_p,
            tp2_price=tp2_p,
            qty_tp3=1,
            tp3_price=None,
        )
    if quantity == 2:
        return PartialExitPlan3(
            qty_total=2,
            qty_tp1=1,
            qty_tp2=0,
            tp1_price=tp1_p,
            tp2_price=tp2_p,
            qty_tp3=1,
            tp3_price=None,
        )

    q1 = quantity // 3
    q2 = quantity // 3
    q3 = quantity - q1 - q2
    return PartialExitPlan3(
        qty_total=quantity,
        qty_tp1=q1,
        qty_tp2=q2,
        tp1_price=tp1_p,
        tp2_price=tp2_p,
        qty_tp3=q3,
        tp3_price=None,
    )

def tp1_hit(direction: str, price: float, tp1: float | None) -> bool:
    """Return True when price crosses TP1 in profit direction.

    Args:
        direction: BUY or SELL
        price: current price
        tp1: TP1 level or None
    Returns:
        bool: True if hit
    """
    if tp1 is None or price <= 0:
        return False
    if direction.upper() == "BUY":
        return price >= tp1
    return price <= tp1

def tp2_hit(direction: str, price: float, tp2: float | None) -> bool:
    """Return True when price crosses TP2 in profit direction.

    Args:
        direction: BUY or SELL
        price: current price
        tp2: TP2 level or None
    Returns:
        bool: True if hit
    """
    if tp2 is None or price <= 0:
        return False
    if direction.upper() == "BUY":
        return price >= tp2
    return price <= tp2
