"""Фильтр материальности новостей."""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.utils.logging import get_logger

logger = get_logger(__name__)

KEYWORDS_MATERIAL: set[str] = {
    "дивиденд",
    "dividend",
    "выплат",
    "payout",
    "buyback",
    "выкуп",
    "отчет",
    "отчёт",
    "earnings",
    "квартальн",
    "quarterly",
    "прибыл",
    "profit",
    "убыт",
    "loss",
    "выручк",
    "revenue",
    "ebitda",
    "ebit",
    "margin",
    "маржа",
    "guidance",
    "прогноз",
    "outlook",
    "ipo",
    "spo",
    "сплит",
    "split",
    "merger",
    "поглощен",
    "consolidat",
    "консолидац",
    "acquisition",
    "сделк",
    "ceo",
    "генеральный директор",
    "увольнен",
    "resignation",
    "отставк",
    "rating",
    "рейтинг",
    "downgrade",
    "upgrade",
    "повыс",
    "пониж",
    "лиценз",
    "license",
    "halt",
    "приостанов",
    "suspension",
    "штраф",
    "fine",
    "investigation",
    "расследован",
    "запрет",
    "ban",
    "санкци",
    "sanction",
    "ставка цб",
    "key rate",
    "ставка фрс",
    "fomc",
    "fed funds",
    "инфляц",
    "inflation",
    "cpi",
    "ppi",
    "ввп",
    "gdp",
    "промпроизводств",
    "industrial production",
    "pmi",
    "labor market",
    "рынок труда",
    "opec",
    "опек",
    "опек+",
    "opec+",
    "petroleum",
    "wpsr",
    "запасы нефти",
    "production cut",
    "сокращен добыч",
    "discovery",
    "месторожден",
    "поставк",
    "supply",
    "экспорт",
    "export",
    "импорт",
    "import",
    "лизинг",
    "leasing",
    "забастовк",
    "strike",
    "авари",
    "accident",
    "капвлож",
    "capex",
    "investment",
    "инвестиц",
}

KEYWORDS_ALWAYS: set[str] = {
    "ofac",
    "офак",
    "sdn list",
    "санкционный список",
    "fomc decision",
    "решение фрс",
    "ecb decision",
    "решение ецб",
    "cbr decision",
    "решение цб рф",
    "war",
    "военные действия",
    "военная операция",
    "ядерн",
    "nuclear",
    "atc clearing",
    "trade halt all",
}

def _compile(words: Iterable[str]) -> re.Pattern:
    """
    Build regex matching keywords with prefix-boundary only.
    Russian inflects suffixes ("дивиденд" → "дивиденды", "дивидендам") so we
    anchor on \b at the start but allow free suffix on the right.
    English keywords stay safe because Latin letter classes still terminate at
    word boundaries on right (followed by space/punct).
    """
    escaped = sorted({re.escape(w) for w in words}, key=len, reverse=True)
    pattern = r"\b(" + "|".join(escaped) + r")"
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)

_MATERIAL_RE = _compile(KEYWORDS_MATERIAL)
_ALWAYS_RE = _compile(KEYWORDS_ALWAYS)

def is_material(text: str, has_ticker: bool = False) -> tuple[bool, list[str], bool]:
    """
    Decide whether a news item is material enough to warrant LLM analysis.

    Returns:
        (is_material: bool, matched_keywords: list[str], is_always: bool)

    Rules:
        - If KEYWORDS_ALWAYS hit → always material (regardless of ticker).
        - If KEYWORDS_MATERIAL hit AND has_ticker → material.
        - Otherwise: skip.
    """
    if not text:
        return False, [], False

    matches_always = _ALWAYS_RE.findall(text)
    if matches_always:
        return True, list({m.lower() for m in matches_always}), True

    if not has_ticker:
        return False, [], False

    matches = _MATERIAL_RE.findall(text)
    if matches:
        return True, list({m.lower() for m in matches}), False

    return False, [], False
