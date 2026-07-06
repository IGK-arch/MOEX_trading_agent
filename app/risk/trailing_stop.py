"""ATR-laddered trailing stop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

@dataclass(frozen=True)
class TrailingRung:
    """One step in the trailing ladder."""

    profit_atr: float
    lock_atr: float

DEFAULT_LADDER: tuple[TrailingRung, ...] = (
    TrailingRung(profit_atr=1.0, lock_atr=0.0),
    TrailingRung(profit_atr=2.0, lock_atr=1.0),
    TrailingRung(profit_atr=3.0, lock_atr=2.0),
)

LADDER_BY_FAMILY: dict[str, tuple[TrailingRung, ...]] = {
    "chart": (
        TrailingRung(profit_atr=0.75, lock_atr=0.0),
        TrailingRung(profit_atr=1.50, lock_atr=0.75),
        TrailingRung(profit_atr=2.50, lock_atr=1.50),
    ),
    "research": DEFAULT_LADDER,
    "smc": (
        TrailingRung(profit_atr=0.75, lock_atr=0.0),
        TrailingRung(profit_atr=1.50, lock_atr=0.75),
        TrailingRung(profit_atr=2.50, lock_atr=1.25),
    ),
    "candle": (
        TrailingRung(profit_atr=0.75, lock_atr=0.0),
        TrailingRung(profit_atr=1.50, lock_atr=0.75),
        TrailingRung(profit_atr=2.50, lock_atr=1.25),
    ),
    "other": DEFAULT_LADDER,
}

def ladder_for_family(family: str | None) -> tuple[TrailingRung, ...]:
    """Pick calibrated ladder for a family.

    Args:
        family: family name or None
    Returns:
        tuple[TrailingRung, ...]: trailing ladder rungs
    """
    if not family:
        return DEFAULT_LADDER
    return LADDER_BY_FAMILY.get(family, DEFAULT_LADDER)

HIGH_WR_LADDER: tuple[TrailingRung, ...] = (
    TrailingRung(profit_atr=0.5, lock_atr=0.0),
    TrailingRung(profit_atr=1.5, lock_atr=0.75),
    TrailingRung(profit_atr=2.5, lock_atr=1.5),
)

_REGIME_TRAIL_MULT: dict[str, float] = {
    "trending": 1.20,
    "mean_reverting": 0.80,
    "crisis": 0.60,
    "unknown": 1.00,
}

def _scale_ladder(
    base: tuple[TrailingRung, ...],
    profit_mult: float,
) -> tuple[TrailingRung, ...]:
    """Multiply rung profit/lock thresholds by `profit_mult`.

    Args:
        base: source ladder
        profit_mult: scaling factor
    Returns:
        tuple[TrailingRung, ...]: scaled ladder
    """
    if profit_mult <= 0 or profit_mult == 1.0:
        return base
    return tuple(
        TrailingRung(
            profit_atr=r.profit_atr * profit_mult,
            lock_atr=r.lock_atr * profit_mult,
        )
        for r in base
    )

def ladder_for_regime(
    *,
    family: str | None = None,
    ticker: str | None = None,
    hmm_regime: str | None = None,
) -> tuple[TrailingRung, ...]:
    """Pick trailing ladder for (family, ticker, regime).

    Args:
        family: pattern family or None
        ticker: instrument code or None
        hmm_regime: regime label or None
    Returns:
        tuple[TrailingRung, ...]: trailing ladder rungs
    """
    try:
        import app.config as cfg

        high_wr = set(getattr(cfg, "HIGH_WR_TICKERS", frozenset()))
    except Exception:
        high_wr = set()

    base = HIGH_WR_LADDER if ticker and ticker.upper() in high_wr else ladder_for_family(family)

    regime = (hmm_regime or "unknown").lower()
    mult = _REGIME_TRAIL_MULT.get(regime, 1.0)
    scaled = _scale_ladder(base, mult)

    if regime == "crisis" and len(scaled) >= 2:
        scaled = scaled[:2]
    return scaled

def compute_trailing_stop(
    direction: str,
    entry_price: float,
    current_price: float,
    atr: float,
    *,
    prev_trailing: float | None = None,
    ladder: tuple[TrailingRung, ...] = DEFAULT_LADDER,
) -> float | None:
    """Return new trailing-stop price or None if no rung triggered.

    Args:
        direction: BUY or SELL
        entry_price: position entry price
        current_price: latest market price
        atr: ATR value
        prev_trailing: previously computed trail
        ladder: ladder of rungs to evaluate
    Returns:
        float | None: new trailing stop price
    """
    if atr <= 0 or entry_price <= 0 or current_price <= 0:
        return prev_trailing

    direction_u = direction.upper()
    if direction_u == "BUY":
        profit = current_price - entry_price
    elif direction_u == "SELL":
        profit = entry_price - current_price
    else:
        return prev_trailing

    if profit <= 0:
        return prev_trailing

    profit_in_atr = profit / atr
    if profit_in_atr < ladder[0].profit_atr:
        return prev_trailing

    triggered_rung: TrailingRung | None = None
    for rung in ladder:
        if profit_in_atr >= rung.profit_atr:
            triggered_rung = rung
        else:
            break

    if triggered_rung is None:
        return prev_trailing

    if direction_u == "BUY":
        new_trail = entry_price + triggered_rung.lock_atr * atr
        if prev_trailing is not None:
            new_trail = max(new_trail, prev_trailing)
    else:
        new_trail = entry_price - triggered_rung.lock_atr * atr
        if prev_trailing is not None:
            new_trail = min(new_trail, prev_trailing)

    return new_trail

def should_time_stop(
    *,
    direction: str,
    entry_price: float,
    current_price: float,
    bars_held: int,
    max_bars: int | None,
) -> bool:
    """Return True when a stalled position should be time-stopped.

    Args:
        direction: BUY or SELL
        entry_price: position entry price
        current_price: latest price
        bars_held: number of bars held
        max_bars: time-stop threshold or None
    Returns:
        bool: True if time-stop applies
    """
    if max_bars is None or max_bars <= 0:
        return False
    if bars_held < max_bars:
        return False
    if entry_price <= 0 or current_price <= 0:
        return False
    direction_u = direction.upper()
    if direction_u == "BUY":
        return current_price <= entry_price
    if direction_u == "SELL":
        return current_price >= entry_price
    return False

_MSK_TZ = timezone(timedelta(hours=3))

def should_force_close_eod(
    now: datetime | None = None,
    *,
    close_hhmm: tuple[int, int] | None = None,
    minutes_before_close: int | None = None,
) -> bool:
    """Return True near MOEX main session close.

    Args:
        now: wall-clock (defaults to now MSK)
        close_hhmm: session close hour/min in MSK
        minutes_before_close: window size; <=0 disables
    Returns:
        bool: True if within force-close window
    """
    try:
        import app.config as cfg

        cfg_close = getattr(cfg, "MAIN_SESSION_CLOSE_MSK", (18, 50))
        cfg_mins = int(getattr(cfg, "FORCE_CLOSE_BEFORE_CLOSE_MIN", 30))
        cfg_open = getattr(cfg, "MAIN_SESSION_OPEN_MSK", (10, 0))
    except Exception:
        cfg_close = (18, 50)
        cfg_mins = 30
        cfg_open = (10, 0)

    if close_hhmm is None:
        close_hhmm = cfg_close
    if minutes_before_close is None:
        minutes_before_close = cfg_mins
    if minutes_before_close <= 0:
        return False

    if now is None:
        now = datetime.now(_MSK_TZ)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_MSK_TZ)
    else:
        now = now.astimezone(_MSK_TZ)

    time(close_hhmm[0], close_hhmm[1])
    open_t = time(cfg_open[0], cfg_open[1])
    cur_t = now.time()

    if cur_t < open_t:
        return False

    close_dt = now.replace(
        hour=close_hhmm[0],
        minute=close_hhmm[1],
        second=0,
        microsecond=0,
    )
    trigger_dt = close_dt - timedelta(minutes=minutes_before_close)
    return now >= trigger_dt

def effective_stop(
    direction: str,
    initial_stop: float | None,
    trailing_stop: float | None,
) -> float | None:
    """Combine static SL with trailing SL — always the more protective.

    Args:
        direction: BUY or SELL
        initial_stop: initial SL or None
        trailing_stop: trailing SL or None
    Returns:
        float | None: effective SL
    """
    if initial_stop is None and trailing_stop is None:
        return None
    if initial_stop is None:
        return trailing_stop
    if trailing_stop is None:
        return initial_stop
    if direction.upper() == "BUY":
        return max(initial_stop, trailing_stop)
    return min(initial_stop, trailing_stop)

def compute_r_trailing_stop(
    *,
    direction: str,
    entry_price: float,
    current_price: float,
    original_sl: float,
    r_to_be: float = 1.0,
    r_to_r1: float = 2.0,
) -> float | None:
    """Return an updated stop based on R-multiples of price advance.

    R-based laddering (independent of ATR), v1.0.4:

      * After ``+r_to_be × R`` of unrealized profit, move the stop to
        break-even (i.e. ``entry_price``).
      * After ``+r_to_r1 × R``, move the stop to ``+1 × R`` (locks in one
        unit of risk).

    Here ``R = |entry_price - original_sl|`` — the distance the trader
    was willing to lose per share at order submission. If R cannot be
    computed (``original_sl`` invalid / same as entry), the function
    returns ``None`` and the caller should fall back to its previous
    trailing logic (typically the ATR ladder above).

    The returned price is *only* a candidate — callers must combine it
    with the existing stop via :func:`effective_stop` so we never relax
    the SL.

    Args:
        direction: ``BUY`` or ``SELL``.
        entry_price: position entry price (must be > 0).
        current_price: latest market price (must be > 0).
        original_sl: the SL set at entry (must be > 0).
        r_to_be: profit in R-multiples after which SL → break-even.
        r_to_r1: profit in R-multiples after which SL → entry + 1 R.
    Returns:
        float | None: candidate SL, or None when no rung triggered /
        inputs invalid.
    """
    if entry_price <= 0 or current_price <= 0 or original_sl <= 0:
        return None
    dir_u = direction.upper()
    if dir_u not in ("BUY", "SELL"):
        return None

    risk = abs(entry_price - original_sl)
    if risk <= 0:
        return None

    if dir_u == "BUY":
        profit = current_price - entry_price
        if profit <= 0:
            return None
        profit_r = profit / risk
        if profit_r >= max(r_to_be, r_to_r1) and profit_r >= r_to_r1:
            return entry_price + risk
        if profit_r >= r_to_be:
            return entry_price
        return None
    profit = entry_price - current_price
    if profit <= 0:
        return None
    profit_r = profit / risk
    if profit_r >= max(r_to_be, r_to_r1) and profit_r >= r_to_r1:
        return entry_price - risk
    if profit_r >= r_to_be:
        return entry_price
    return None
