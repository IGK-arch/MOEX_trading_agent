"""Кэш открытых позиций."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import app.config as cfg
from app.execution.arenago_client import get_arenago_client
from app.utils.logging import get_logger

logger = get_logger(__name__)

MTM_PRICE_TTL_SEC: float = 60.0

try:
    import aiosqlite  # type: ignore

    _HAS_AIOSQLITE = True
except ImportError:
    _HAS_AIOSQLITE = False

TRADES_DB = cfg.DATA_DIR / "trades.db"

SECTOR_MAP: dict[str, str] = {
    "SBER": "banks",
    "SBERP": "banks",
    "VTBR": "banks",
    "GAZP": "oil_gas",
    "LKOH": "oil_gas",
    "ROSN": "oil_gas",
    "NVTK": "oil_gas",
    "SNGS": "oil_gas",
    "SNGSP": "oil_gas",
    "TATN": "oil_gas",
    "TATNP": "oil_gas",
    "GMKN": "metals",
    "NLMK": "metals",
    "CHMF": "metals",
    "PLZL": "metals",
    "MGNT": "retail",
    "AFLT": "transport",
    "MTSS": "telecom",
    "PIKK": "construction",
    "YDEX": "it",
}

@dataclass
class Position:
    """Single open position."""

    ticker: str
    quantity: int
    avg_price: float
    bot: str
    sector: str = field(init=False)
    entry_ts: float = field(default_factory=time.time)
    source: str | None = None

    def __post_init__(self) -> None:
        """Post init."""
        self.sector = SECTOR_MAP.get(self.ticker, "other")

    @property
    def market_value(self) -> float:
        """Return notional value of the position.

        Returns:
            float: quantity × avg_price
        """
        return self.quantity * self.avg_price

    @property
    def age_seconds(self) -> float:
        """Return seconds since the position was opened.

        Returns:
            float: age in seconds
        """
        return time.time() - self.entry_ts

class PositionBook:
    """Live cache of open positions + cash balance."""

    def __init__(
        self,
        deposit_total: float = 1_000_000.0,
        refresh_interval_sec: int = 30,
    ) -> None:
        """Init."""
        self.deposit_total = deposit_total
        self.refresh_interval_sec = refresh_interval_sec
        self.arenago = get_arenago_client()

        self._positions: dict[str, Position] = {}
        self._cash_balance: float = deposit_total
        self._last_refresh_ts: float = 0
        self._stop_event = asyncio.Event()
        self._refresh_task: asyncio.Task | None = None

        self._last_entry_ts: dict[str, float] = {}

        self._source_by_ticker: dict[str, str] = {}

        self._mtm_prices: dict[str, tuple[float, float]] = {}
        self._last_mtm_fetch_ts: float = 0.0
        self._mtm_failures: int = 0

    async def start(self) -> None:
        """Start background refresh loop after initial refresh."""

        await self.refresh()

        self._refresh_task = asyncio.create_task(self._refresh_loop(), name="position_book_refresh")
        logger.info(
            "PositionBook запущен",
            extra={
                "positions": len(self._positions),
                "cash": self._cash_balance,
                "deposit_total": self.deposit_total,
            },
        )

    async def stop(self) -> None:
        """Stop refresh task."""
        self._stop_event.set()
        if self._refresh_task:
            await asyncio.gather(self._refresh_task, return_exceptions=True)
        logger.info("PositionBook остановлен")

    async def _refresh_loop(self) -> None:
        """Periodic refresh worker."""
        while not self._stop_event.is_set():
            try:
                await self.refresh()
            except Exception as exc:
                logger.warning("PositionBook: ошибка обновления", extra={"error": str(exc)})
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.refresh_interval_sec)

    async def refresh(self) -> None:
        """Pull positions + cash from ArenaGo with trades.db fallback."""
        positions = await self.arenago.get_positions()
        new_positions: dict[str, Position] = {}

        if positions:
            for p in positions:
                ticker = str(p.get("secid", "")).upper()
                qty = int(p.get("position", 0))
                price = float(p.get("average_price", 0))
                bot = str(p.get("bot", ""))
                if qty == 0 or not ticker:
                    continue

                existing = self._positions.get(ticker)
                entry_ts = existing.entry_ts if existing else time.time()
                new_positions[ticker] = Position(
                    ticker=ticker,
                    quantity=qty,
                    avg_price=price,
                    bot=bot,
                    entry_ts=entry_ts,
                )
        else:
            local = await self._derive_positions_from_trades()
            for ticker, (qty, vwap) in local.items():
                if qty == 0:
                    continue
                existing = self._positions.get(ticker)
                entry_ts = existing.entry_ts if existing else time.time()
                new_positions[ticker] = Position(
                    ticker=ticker,
                    quantity=qty,
                    avg_price=vwap,
                    bot=cfg.ARENAGO_BOT_NAME,
                    entry_ts=entry_ts,
                )
            if local:
                logger.debug(
                    "PositionBook derived from trades.db fallback",
                    extra={"n_positions": len(local)},
                )

        self._positions = new_positions

        cash = await self.arenago.get_cash_balance()
        if cash > 0:
            self._cash_balance = cash
            self._cash_ever_fetched = True
        elif not getattr(self, "_cash_ever_fetched", False):
            logger.warning(
                "PositionBook: ArenaGo cash=0 and no prior fetch — using deposit_total",
                extra={"deposit_total": self.deposit_total},
            )

        self._last_refresh_ts = time.time()

        with contextlib.suppress(Exception):
            await self._refresh_mtm_prices()

        try:
            from app.risk.circuit_breakers import get_circuit_breaker

            cb = get_circuit_breaker()
            equity = self.total_equity()
            await cb.on_equity_update(equity)
        except Exception as exc:
            logger.debug("CB equity update skipped", extra={"error": str(exc)})

    async def _derive_positions_from_trades(self) -> dict[str, tuple[int, float]]:
        """Aggregate net qty + VWAP per ticker from today's trades.db.

        Returns:
            dict[str, tuple[int, float]]: ticker → (net_qty, vwap)
        """
        if not _HAS_AIOSQLITE:
            return {}
        out: dict[str, tuple[int, float]] = {}
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        try:
            async with (
                aiosqlite.connect(TRADES_DB) as db,
                db.execute(
                    "SELECT ticker, direction, quantity, price FROM trades WHERE trade_date = ?",
                    (today,),
                ) as cur,
            ):
                rows = await cur.fetchall()
            agg: dict[str, dict[str, float]] = defaultdict(
                lambda: {"net_qty": 0.0, "buy_qty": 0.0, "buy_notional": 0.0}
            )
            for ticker, direction, qty, price in rows:
                if not ticker or qty is None:
                    continue
                t = str(ticker).upper()
                q = int(qty)
                p = float(price or 0)
                if str(direction).upper() == "BUY":
                    agg[t]["net_qty"] += q
                    agg[t]["buy_qty"] += q
                    agg[t]["buy_notional"] += q * p
                else:
                    agg[t]["net_qty"] -= q
            for t, a in agg.items():
                net = int(a["net_qty"])
                if net <= 0:
                    continue
                vwap = (a["buy_notional"] / a["buy_qty"]) if a["buy_qty"] > 0 else 0.0
                out[t] = (net, vwap)
        except Exception as exc:
            logger.error("Trades.db position derivation failed", extra={"error": str(exc)})
        return out

    @property
    def positions(self) -> dict[str, Position]:
        """Return current positions map.

        Returns:
            dict[str, Position]: ticker → Position
        """
        return self._positions

    @property
    def cash_balance(self) -> float:
        """Return current cash balance.

        Returns:
            float: cash in RUB
        """
        return self._cash_balance

    @property
    def n_open_positions(self) -> int:
        """Return number of open positions.

        Returns:
            int: open position count
        """
        return len(self._positions)

    def has_position(self, ticker: str) -> bool:
        """Return True if ticker has an open position.

        Args:
            ticker: instrument code
        Returns:
            bool: True if position exists
        """
        return ticker.upper() in self._positions

    def get_position(self, ticker: str) -> Position | None:
        """Return Position for ticker if any.

        Args:
            ticker: instrument code
        Returns:
            Position | None: position or None
        """
        return self._positions.get(ticker.upper())

    def total_market_value(self) -> float:
        """Return sum of all open position market values.

        Returns:
            float: total notional value
        """
        return sum(p.market_value for p in self._positions.values())

    def total_exposure_pct(self) -> float:
        """Return total positions value / deposit_total.

        Returns:
            float: exposure as fraction
        """
        return self.total_market_value() / self.deposit_total if self.deposit_total > 0 else 0.0

    def sector_exposure_pct(self, sector: str) -> float:
        """Return % of deposit invested in one sector.

        Args:
            sector: sector code
        Returns:
            float: exposure as fraction
        """
        value = sum(p.market_value for p in self._positions.values() if p.sector == sector)
        return value / self.deposit_total if self.deposit_total > 0 else 0.0

    def exposure_by_source(self, source: str) -> float:
        """Return total notional of open positions opened by `source`.

        Falls back to looking up the dominant-source map populated at entry
        time via `mark_entry_with_source` so legacy positions (without an
        in-memory `source` attribute) still resolve correctly after a
        process restart.

        Args:
            source: SignalSource value (TA / NEWS / ANOMALY / MEAN_REV / PAIR)
        Returns:
            float: summed notional in RUB
        """
        if not source:
            return 0.0
        target = source.upper()
        total = 0.0
        for ticker, pos in self._positions.items():
            tagged = (pos.source or self._source_by_ticker.get(ticker, "")).upper()
            if tagged == target:
                total += abs(pos.market_value)
        return total

    def exposure_by_source_pct(self, source: str) -> float:
        """Return per-strategy exposure as fraction of deposit_total.

        Args:
            source: SignalSource value
        Returns:
            float: fraction of deposit (0.0–1.0)
        """
        if self.deposit_total <= 0:
            return 0.0
        return self.exposure_by_source(source) / self.deposit_total

    def total_equity(self) -> float:
        """Return ArenaGo-aware equity.

        Preferred (v1.2.0): mark-to-market via ISS spot-prices cached in
        `self._mtm_prices`.  Confirmed via ArenaGo /about docs ("Long и
        Short" section) that the platform uses a CFD model — opening a
        SHORT does NOT credit `cash_balance`, instead collateral is frozen
        invisibly and profit/loss is realized only at close.  Therefore
        the correct equity formula is

            equity = cash + Σ qty_signed × (current_price − avg_price)
                   = cash + unrealized_PnL

        This conserves equity at the moment a position opens
        (current=avg → PnL=0 → equity unchanged), which is the
        sanity-test that disqualified the earlier "cash + qty×current"
        formula.

        Fallback (v1.1.0 safe): without an MTM cache (startup, network
        outage, ISS empty) we still use `cash + Σ LONG.market_value`
        because crediting LONG market_value matches the cash that was
        paid out at entry, while SHORTs contribute 0 unrealized PnL
        when we lack a current price.

        Returns:
            float: equity in RUB
        """
        if self._mtm_prices:
            prices = {t: v[0] for t, v in self._mtm_prices.items()}
            mtm = self.total_equity_mtm(prices)
        else:
            long_value = sum(
                p.market_value for p in self._positions.values() if p.quantity > 0
            )
            mtm = float(self._cash_balance) + float(long_value)
        try:
            floor_frac = float(
                getattr(__import__("app.config", fromlist=["x"]),
                        "EQUITY_SANITY_FLOOR_FRACTION", 0.7)
            )
        except Exception:
            floor_frac = 0.7
        floor = float(self.deposit_total) * floor_frac
        if mtm < floor:
            return floor
        return mtm

    def total_equity_mtm(self, current_prices: dict[str, float]) -> float:
        """Return mark-to-market equity given a current_prices dict.

        Math: under ArenaGo's CFD model the only thing that changes
        equity once positions are open is unrealized PnL —

            equity = cash + Σ qty_signed × (current_price − avg_price)

        which works for LONG and SHORT alike:
            * LONG  qty=+50, avg=300, current=310 → PnL = +500
            * SHORT qty=−100, avg=120, current=115 → PnL = −100 × (−5)
                                                         = +500

        Worked example:
            cash=133_000, SBER LONG 50 @300 (cur 310),
            GAZP SHORT −100 @120 (cur 115).
            equity = 133_000 + 50×(310−300) + (−100)×(115−120)
                   = 133_000 + 500 + 500 = 134_000

        Sanity at open (current==avg) → PnL=0 → equity=cash.

        Per-ticker fallback (current_price missing):
            * LONG  → contribute qty × avg_price as a notional proxy so
                      `cash + LONG_market_value` still approximates
                      equity in the v1.1.0 sense.
            * SHORT → contribute 0 (no PnL).  Using avg_price would
                      double-subtract the original sale value that
                      ArenaGo never actually credited.

        Args:
            current_prices: ticker → spot RUB.  Case-insensitive lookup.

        Returns:
            float: equity in RUB.
        """
        upper = {str(k).upper(): float(v) for k, v in (current_prices or {}).items() if v}
        pnl = 0.0
        long_notional_fallback = 0.0
        for ticker, pos in self._positions.items():
            qty = int(pos.quantity)
            if qty == 0:
                continue
            avg = float(pos.avg_price)
            px = upper.get(ticker.upper())
            if px is not None and px > 0:
                pnl += qty * (px - avg)
            elif qty > 0:
                long_notional_fallback += qty * avg
        return float(self._cash_balance) + pnl + long_notional_fallback

    async def _refresh_mtm_prices(self) -> None:
        """Fetch fresh spot prices from MOEX ISS for all open tickers.

        TTL-cached — re-fetch only every `MTM_PRICE_TTL_SEC` seconds, and
        only for tickers that don't have a fresh cached price yet.  Updates
        `self._mtm_prices` in-place.  Failures degrade gracefully — entries
        are left stale, and `total_equity()` falls back to v1.1.0 maths.
        """
        if not self._positions:
            return

        now = time.time()
        stale: list[str] = []
        for ticker in self._positions:
            cached = self._mtm_prices.get(ticker)
            if not cached or (now - cached[1]) > MTM_PRICE_TTL_SEC:
                stale.append(ticker)

        if not stale:
            return

        try:
            prices = await self._fetch_iss_prices(stale)
        except Exception as exc:
            self._mtm_failures += 1
            logger.warning(
                "PositionBook MTM fetch failed",
                extra={"error": str(exc), "n_stale": len(stale), "fails": self._mtm_failures},
            )
            return

        if not prices:
            self._mtm_failures += 1
            return

        self._mtm_failures = 0
        ts = time.time()
        for t, p in prices.items():
            if p > 0:
                self._mtm_prices[t.upper()] = (float(p), ts)
        self._last_mtm_fetch_ts = ts
        logger.debug(
            "PositionBook MTM prices refreshed",
            extra={"n_prices": len(prices), "tickers": list(prices.keys())[:10]},
        )

    async def _fetch_iss_prices(self, tickers: list[str]) -> dict[str, float]:
        """Hit MOEX ISS securities.json for batch spot LAST prices.

        Args:
            tickers: list of MOEX SECIDs.
        Returns:
            dict[str, float]: ticker → last/legal close price (RUB).  Empty
                on transport / parsing failure.
        """
        if not tickers:
            return {}
        try:
            import httpx  # type: ignore
        except ImportError:
            return {}

        uniq = ",".join(sorted({t.upper() for t in tickers if t}))
        url = (
            f"{cfg.ISS_BASE_URL}/engines/stock/markets/shares/boards/"
            f"{cfg.ISS_BOARD}/securities.json"
        )
        params = {
            "securities": uniq,
            "iss.meta": "off",
            "iss.only": "marketdata,securities",
            "marketdata.columns": "SECID,LAST,LCURRENTPRICE,LCLOSEPRICE",
            "securities.columns": "SECID,PREVPRICE,PREVADMITTEDQUOTE",
        }

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.HTTP_TIMEOUT),
            headers={"User-Agent": "MoexML-Trader/1.2 (mtm)"},
        ) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            payload: dict[str, Any] = r.json()

        out: dict[str, float] = {}

        md = payload.get("marketdata", {}) or {}
        md_cols: list[str] = md.get("columns", []) or []
        md_rows: list[list[Any]] = md.get("data", []) or []
        for row in md_rows:
            rec = dict(zip(md_cols, row))
            sec = str(rec.get("SECID", "")).upper()
            if not sec:
                continue
            for fld in ("LAST", "LCURRENTPRICE", "LCLOSEPRICE"):
                val = rec.get(fld)
                if val is not None and float(val or 0) > 0:
                    out[sec] = float(val)
                    break

        sec_section = payload.get("securities", {}) or {}
        sec_cols: list[str] = sec_section.get("columns", []) or []
        sec_rows: list[list[Any]] = sec_section.get("data", []) or []
        for row in sec_rows:
            rec = dict(zip(sec_cols, row))
            sec = str(rec.get("SECID", "")).upper()
            if not sec or sec in out:
                continue
            for fld in ("PREVPRICE", "PREVADMITTEDQUOTE"):
                val = rec.get(fld)
                if val is not None and float(val or 0) > 0:
                    out[sec] = float(val)
                    break
        return out

    def sector_breakdown(self) -> dict[str, float]:
        """Return per-sector exposure breakdown.

        Returns:
            dict[str, float]: sector → pct_of_deposit
        """
        breakdown: dict[str, float] = {}
        for p in self._positions.values():
            breakdown[p.sector] = breakdown.get(p.sector, 0.0) + p.market_value / self.deposit_total
        return breakdown

    def cash_reserve_pct(self) -> float:
        """Return % of deposit currently in cash.

        Returns:
            float: cash fraction
        """
        return self._cash_balance / self.deposit_total if self.deposit_total > 0 else 0.0

    def seconds_since_last_entry(self, ticker: str) -> float:
        """Return seconds since last entry for ticker.

        Args:
            ticker: instrument code
        Returns:
            float: seconds (inf if never entered)
        """
        ticker_u = ticker.upper()
        last = self._last_entry_ts.get(ticker_u)
        if last is None:
            return float("inf")
        return time.time() - last

    def notional_pct_for_ticker(self, ticker: str) -> float:
        """v1.4.0 — return current |notional| as % of deposit_total.

        Used by risk_manager's MAX_TICKER_NOTIONAL_PCT_CUMULATIVE check
        to prevent SBER LONG 288 (9.3% of deposit) style over-concentration
        observed in 27 May production.  Uses MTM price if available
        (self._mtm_prices), else avg_price.

        Args:
            ticker: instrument code
        Returns:
            float: abs(qty * price) / deposit_total, or 0.0 if no position
        """
        ticker_u = ticker.upper()
        pos = self._positions.get(ticker_u)
        if pos is None or int(pos.quantity) == 0 or self.deposit_total <= 0:
            return 0.0
        qty_abs = abs(int(pos.quantity))
        cached = self._mtm_prices.get(ticker_u) if self._mtm_prices else None
        price = float(cached[0]) if cached else float(pos.avg_price)
        if price <= 0:
            price = float(pos.avg_price)
        notional = qty_abs * price
        return notional / float(self.deposit_total)

    def mark_entry(self, ticker: str) -> None:
        """Record entry timestamp for ticker.

        Args:
            ticker: instrument code
        """
        self._last_entry_ts[ticker.upper()] = time.time()

    def mark_entry_with_source(self, ticker: str, source: str | None) -> None:
        """Record entry + dominant signal source (Phase 27.5).

        Args:
            ticker: instrument code
            source: SignalSource value (TA/NEWS/ANOMALY/MEAN_REV/PAIR)
        """
        ticker_u = ticker.upper()
        self._last_entry_ts[ticker_u] = time.time()
        if source:
            self._source_by_ticker[ticker_u] = source.upper()
            pos = self._positions.get(ticker_u)
            if pos is not None:
                pos.source = source.upper()

_position_book: PositionBook | None = None

def get_position_book() -> PositionBook:
    """Return process-wide PositionBook singleton.

    Returns:
        PositionBook: shared instance
    """
    global _position_book
    if _position_book is None:
        _position_book = PositionBook()
    return _position_book
