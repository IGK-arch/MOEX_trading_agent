"""Парсер календаря предстоящих дивидендов с Smart-Lab.

Источник: https://smart-lab.ru/dividends/ — открытая HTML-таблица с upcoming
record dates по российским акциям. Парсится раз в сутки (06:00 МСК) через
selectolax, результаты сохраняются в feeds.db в таблицу scheduled_dividend
для ahead-of-time позиционирования.

В отличие от reactive NewsLLM (ловит событие после публикации), календарь
даёт **proactive** знание: за 1-7 дней до record date система может
boost magnitude для соответствующего тикера.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from selectolax.parser import HTMLParser

    _HAS_SELECTOLAX = True
except ImportError:
    _HAS_SELECTOLAX = False

try:
    from bs4 import BeautifulSoup

    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

SMARTLAB_DIVIDENDS_URL = "https://smart-lab.ru/dividends/"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
FEEDS_DB_PATH = cfg.DATA_DIR / "feeds.db"

@dataclass(frozen=True)
class DividendEvent:
    """Запланированное дивидендное событие.

    Attributes:
        ticker: тикер MOEX (e.g. SBER, GAZP)
        period: период за который дивиденды (e.g. "4кв 2025")
        amount_rub: размер дивиденда в рублях
        yield_pct: дивидендная доходность в %
        buy_before: последний день покупки для получения дивов (DD.MM.YYYY)
        record_date: дата закрытия реестра (ex-dividend, DD.MM.YYYY)
        payment_date: дата выплаты
        current_price: текущая цена акции
    """

    ticker: str
    period: str
    amount_rub: float | None
    yield_pct: float | None
    buy_before: str | None
    record_date: str | None
    payment_date: str | None
    current_price: float | None

def _parse_float(text: str) -> float | None:
    """Парсит число из российского формата (запятая → точка).

    Args:
        text: строка с числом, e.g. "9,71" или "11,7%"
    Returns:
        float или None если не удалось распарсить
    """
    if not text:
        return None
    cleaned = re.sub(r"[^\d,.\-]", "", text).replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None

def _parse_date_ru(text: str) -> str | None:
    """DD.MM.YYYY → YYYY-MM-DD (ISO).

    Args:
        text: дата в формате DD.MM.YYYY
    Returns:
        YYYY-MM-DD или None
    """
    if not text or not re.match(r"^\d{2}\.\d{2}\.\d{4}$", text):
        return None
    try:
        dt = datetime.strptime(text, "%d.%m.%Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None

def parse_smartlab_dividends_html(html: str) -> list[DividendEvent]:
    """Парсит HTML страницы smart-lab.ru/dividends/.

    Args:
        html: полный HTML response
    Returns:
        list[DividendEvent]: только тикеры в cfg.TICKERS, упорядоченные по
                             record_date asc (ближайшие первыми)
    """
    events: list[DividendEvent] = []
    allowed = {t.upper() for t in cfg.TICKERS}

    if _HAS_SELECTOLAX:
        tree = HTMLParser(html)
        for table in tree.css("table.simple-little-table"):
            rows = table.css("tr")
            if len(rows) < 3:
                continue
            for row in rows[1:]:
                cells = [c.text(strip=True) for c in row.css("td")]
                if len(cells) < 9:
                    continue
                ticker = cells[1].upper()
                if ticker not in allowed:
                    continue
                events.append(
                    DividendEvent(
                        ticker=ticker,
                        period=cells[2] or "",
                        amount_rub=_parse_float(cells[3]),
                        yield_pct=_parse_float(cells[4]),
                        buy_before=_parse_date_ru(cells[6]),
                        record_date=_parse_date_ru(cells[7]),
                        payment_date=_parse_date_ru(cells[8]),
                        current_price=_parse_float(cells[9]),
                    )
                )
            break
    elif _HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        for table in soup.find_all("table", class_="simple-little-table"):
            rows = table.find_all("tr")
            if len(rows) < 3:
                continue
            for row in rows[1:]:
                cells = [c.get_text(strip=True) for c in row.find_all("td")]
                if len(cells) < 9:
                    continue
                ticker = cells[1].upper()
                if ticker not in allowed:
                    continue
                events.append(
                    DividendEvent(
                        ticker=ticker,
                        period=cells[2] or "",
                        amount_rub=_parse_float(cells[3]),
                        yield_pct=_parse_float(cells[4]),
                        buy_before=_parse_date_ru(cells[6]),
                        record_date=_parse_date_ru(cells[7]),
                        payment_date=_parse_date_ru(cells[8]),
                        current_price=_parse_float(cells[9]),
                    )
                )
            break
    else:
        logger.warning("Neither selectolax nor bs4 installed — dividend calendar disabled")
    events.sort(key=lambda e: e.record_date or "9999-12-31")
    return events

async def fetch_smartlab_dividends() -> list[DividendEvent]:
    """Скачивает + парсит smart-lab dividend calendar.

    Returns:
        list[DividendEvent]: dividend events для тикеров MOEX в cfg.TICKERS
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.get(
                SMARTLAB_DIVIDENDS_URL,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            )
            if r.status_code != 200:
                logger.warning(
                    "Smart-Lab dividends fetch non-200",
                    extra={"status": r.status_code, "size": len(r.text)},
                )
                return []
            return parse_smartlab_dividends_html(r.text)
        except Exception as exc:
            logger.warning("Smart-Lab dividends fetch failed", extra={"error": str(exc)})
            return []

def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Создаёт таблицу scheduled_dividend если её ещё нет."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scheduled_dividend (
            ticker TEXT NOT NULL,
            record_date TEXT,
            buy_before TEXT,
            payment_date TEXT,
            amount_rub REAL,
            yield_pct REAL,
            period TEXT,
            current_price REAL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (ticker, record_date, period)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sched_div_ticker_date "
        "ON scheduled_dividend(ticker, record_date)"
    )

def save_to_feeds_db(events: list[DividendEvent]) -> int:
    """Сохраняет события в feeds.db (UPSERT по PK).

    Args:
        events: список DividendEvent
    Returns:
        int: количество добавленных/обновлённых строк
    """
    if not events:
        return 0
    now_iso = datetime.utcnow().isoformat()
    with sqlite3.connect(FEEDS_DB_PATH) as conn:
        _ensure_schema(conn)
        n = 0
        for e in events:
            conn.execute(
                """
                INSERT OR REPLACE INTO scheduled_dividend
                (ticker, record_date, buy_before, payment_date, amount_rub,
                 yield_pct, period, current_price, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    e.ticker,
                    e.record_date,
                    e.buy_before,
                    e.payment_date,
                    e.amount_rub,
                    e.yield_pct,
                    e.period,
                    e.current_price,
                    now_iso,
                ),
            )
            n += 1
        conn.commit()
    return n

def get_upcoming_dividends(days_ahead: int = 7) -> list[DividendEvent]:
    """Возвращает запланированные события на ближайшие N дней.

    Args:
        days_ahead: горизонт в днях
    Returns:
        list[DividendEvent]: события где record_date в [today, today+days_ahead]
    """
    if not FEEDS_DB_PATH.exists():
        return []
    today = datetime.utcnow().date()
    until = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")
    results: list[DividendEvent] = []
    with sqlite3.connect(FEEDS_DB_PATH) as conn:
        _ensure_schema(conn)
        cur = conn.execute(
            """
            SELECT ticker, period, amount_rub, yield_pct, buy_before,
                   record_date, payment_date, current_price
            FROM scheduled_dividend
            WHERE record_date IS NOT NULL
              AND record_date BETWEEN ? AND ?
            ORDER BY record_date ASC
            """,
            (today_str, until),
        )
        for row in cur.fetchall():
            results.append(DividendEvent(*row))
    return results

def get_dividend_proximity_mult(ticker: str) -> float:
    """Возвращает магнитудный множитель если ticker близок к ex-div date.

    Pre-event позиционирование: за 1-3 дня до record_date цена обычно
    растёт (anticipation). После — gap down на размер дивиденда (mean
    reversion target).

    Args:
        ticker: тикер MOEX
    Returns:
        float множитель в [0.8, 1.3]:
            - 1.30 за 1-2 дня до record date
            - 1.15 за 3-7 дней до record date
            - 0.85 первые 1-2 дня ПОСЛЕ record date (ожидаемый gap down)
            - 1.0 в остальных случаях
    """
    upcoming = get_upcoming_dividends(days_ahead=14)
    today = datetime.utcnow().date()
    for ev in upcoming:
        if ev.ticker != ticker.upper() or not ev.record_date:
            continue
        try:
            rec_date = datetime.strptime(ev.record_date, "%Y-%m-%d").date()
        except ValueError:
            continue
        diff = (rec_date - today).days
        if 1 <= diff <= 2:
            return 1.30
        if 3 <= diff <= 7:
            return 1.15
        if -2 <= diff <= 0:
            return 0.85
    return 1.0
