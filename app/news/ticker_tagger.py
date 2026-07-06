"""Маппинг новостей на тикеры MOEX."""

from __future__ import annotations

import re

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from natasha import Doc, MorphVocab, NewsEmbedding, NewsNERTagger, Segmenter  # type: ignore

    _HAS_NATASHA = True
except ImportError:
    _HAS_NATASHA = False

TICKER_ALIASES: dict[str, str] = {
    "sber": "SBER",
    "sberbank": "SBER",
    "сбер": "SBER",
    "сбербанк": "SBER",
    "сбера": "SBER",
    "сбербанка": "SBER",
    "сбербанком": "SBER",
    "sberp": "SBERP",
    "сбербанк-п": "SBERP",
    "сберпреф": "SBERP",
    "gazprom": "GAZP",
    "газпром": "GAZP",
    "газпрома": "GAZP",
    "газпрому": "GAZP",
    "gazp": "GAZP",
    "lukoil": "LKOH",
    "лукойл": "LKOH",
    "лукойла": "LKOH",
    "лукойлу": "LKOH",
    "lkoh": "LKOH",
    "rosneft": "ROSN",
    "роснефть": "ROSN",
    "роснефти": "ROSN",
    "rosn": "ROSN",
    "novatek": "NVTK",
    "новатэк": "NVTK",
    "новатэка": "NVTK",
    "nvtk": "NVTK",
    "surgutneftegas": "SNGS",
    "сургутнефтегаз": "SNGS",
    "sngs": "SNGS",
    "sngsp": "SNGSP",
    "сургут-п": "SNGSP",
    "norilsk nickel": "GMKN",
    "норникель": "GMKN",
    "норильский никель": "GMKN",
    "nornickel": "GMKN",
    "gmkn": "GMKN",
    "nlmk": "NLMK",
    "нлмк": "NLMK",
    "новолипецкий": "NLMK",
    "novolipetsk": "NLMK",
    "severstal": "CHMF",
    "северсталь": "CHMF",
    "chmf": "CHMF",
    "magnit": "MGNT",
    "магнит": "MGNT",
    "магнита": "MGNT",
    "mgnt": "MGNT",
    "vtb": "VTBR",
    "втб": "VTBR",
    "vtbr": "VTBR",
    "vtb bank": "VTBR",
    "aeroflot": "AFLT",
    "аэрофлот": "AFLT",
    "аэрофлота": "AFLT",
    "aflt": "AFLT",
    "mts": "MTSS",
    "мтс": "MTSS",
    "мтсс": "MTSS",
    "mtss": "MTSS",
    "tatneft": "TATN",
    "татнефть": "TATN",
    "tatn": "TATN",
    "tatnp": "TATNP",
    "татнефть-п": "TATNP",
    "polyus": "PLZL",
    "полюс": "PLZL",
    "plzl": "PLZL",
    "polyus gold": "PLZL",
    "pik": "PIKK",
    "пик": "PIKK",
    "pikk": "PIKK",
    "pik group": "PIKK",
    "yandex": "YDEX",
    "яндекс": "YDEX",
    "ydex": "YDEX",
    "t-банк": "T",
    "тинькофф": "T",
    "тинькоф": "T",
    "tinkoff": "T",
    "т-банк": "T",
    "ткс": "T",
    "tcs group": "T",
    "tcs": "T",
    "tinkoff bank": "T",
    "тинкофф": "T",
    "т банк": "T",
    "x5": "X5",
    "x5 retail": "X5",
    "икс пять": "X5",
    "перекрёсток": "X5",
    "перекресток": "X5",
    "пятёрочка": "X5",
    "пятерочка": "X5",
    "x5 group": "X5",
    "икс 5": "X5",
    "x5 retail group": "X5",
    "alrosa": "ALRS",
    "алроса": "ALRS",
    "алросы": "ALRS",
    "алросу": "ALRS",
    "alrs": "ALRS",
    "moex": "MOEX",
    "мосбиржа": "MOEX",
    "московская биржа": "MOEX",
    "мосбиржи": "MOEX",
    "мосбирже": "MOEX",
    "moscow exchange": "MOEX",
}

COMMODITY_TO_TICKERS: dict[str, list[str]] = {
    "brent": ["LKOH", "ROSN", "SNGS", "SNGSP", "NVTK", "TATN", "TATNP"],
    "crude": ["LKOH", "ROSN", "SNGS", "SNGSP", "NVTK", "TATN", "TATNP"],
    "oil": ["LKOH", "ROSN", "SNGS", "SNGSP", "NVTK", "TATN", "TATNP"],
    "нефть": ["LKOH", "ROSN", "SNGS", "SNGSP", "NVTK", "TATN", "TATNP"],
    "wti": ["LKOH", "ROSN", "SNGS", "SNGSP", "NVTK", "TATN", "TATNP"],
    "opec": ["LKOH", "ROSN", "SNGS", "SNGSP", "NVTK", "TATN", "TATNP"],
    "опек": ["LKOH", "ROSN", "SNGS", "SNGSP", "NVTK", "TATN", "TATNP"],
    "natural gas": ["GAZP", "NVTK"],
    "газ": ["GAZP", "NVTK"],
    "lng": ["GAZP", "NVTK"],
    "спг": ["GAZP", "NVTK"],
    "nickel": ["GMKN"],
    "никель": ["GMKN"],
    "palladium": ["GMKN"],
    "палладий": ["GMKN"],
    "gold": ["PLZL"],
    "золото": ["PLZL"],
    "diamond": ["ALRS"],
    "diamonds": ["ALRS"],
    "алмаз": ["ALRS"],
    "алмазы": ["ALRS"],
    "бриллиант": ["ALRS"],
    "iron ore": ["NLMK", "CHMF"],
    "железная руда": ["NLMK", "CHMF"],
    "steel": ["NLMK", "CHMF"],
    "сталь": ["NLMK", "CHMF"],
    "ставка цб": ["SBER", "SBERP", "VTBR", "MGNT", "PIKK"],
    "key rate": ["SBER", "SBERP", "VTBR", "MGNT", "PIKK"],
    "cbr": ["SBER", "SBERP", "VTBR", "MGNT", "PIKK"],
    "central bank of russia": ["SBER", "SBERP", "VTBR", "MGNT", "PIKK"],
}

_WORD_RE = re.compile(r"[\w\-]+", re.UNICODE)

def _normalise(text: str) -> str:
    """Normalise."""
    return text.lower().strip()

def _tokens(text: str) -> list[str]:
    """Tokens."""
    return [_normalise(w) for w in _WORD_RE.findall(text)]

class TickerTagger:
    """Map a text to a list of MOEX tickers."""

    def __init__(self, enable_ner: bool = True) -> None:
        """Init."""
        self.enable_ner = enable_ner and _HAS_NATASHA
        if self.enable_ner:
            try:
                self._segmenter = Segmenter()
                self._morph = MorphVocab()
                self._emb = NewsEmbedding()
                self._ner_tagger = NewsNERTagger(self._emb)
                logger.info("Natasha NER loaded")
            except Exception as exc:
                logger.warning("Natasha NER failed to load, disabling", extra={"error": str(exc)})
                self.enable_ner = False

        self._sorted_aliases = sorted(TICKER_ALIASES.keys(), key=len, reverse=True)
        self._sorted_commodities = sorted(COMMODITY_TO_TICKERS.keys(), key=len, reverse=True)

    def tag(self, text: str, max_tickers: int = 10) -> list[str]:
        """
        Return list of MOEX tickers mentioned in text.
        Combines direct alias matches + commodity-derived tickers + (optional) NER.
        """
        if not text:
            return []

        tickers: set[str] = set()
        " " + text.lower() + " "

        for alias in self._sorted_aliases:
            pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE | re.UNICODE)
            if pattern.search(text):
                tickers.add(TICKER_ALIASES[alias])
                if len(tickers) >= max_tickers:
                    break

        for commodity in self._sorted_commodities:
            pattern = re.compile(r"\b" + re.escape(commodity) + r"\b", re.IGNORECASE | re.UNICODE)
            if pattern.search(text):
                for t in COMMODITY_TO_TICKERS[commodity]:
                    tickers.add(t)
                    if len(tickers) >= max_tickers:
                        break

        if self.enable_ner and len(tickers) == 0:
            try:
                doc = Doc(text[:5000])
                doc.segment(self._segmenter)
                doc.tag_ner(self._ner_tagger)
                for span in doc.spans:
                    if span.type == "ORG":
                        org_norm = _normalise(span.text)

                        for alias in self._sorted_aliases:
                            if alias in org_norm or org_norm in alias:
                                tickers.add(TICKER_ALIASES[alias])
                                break
            except Exception as exc:
                logger.debug("NER tag failed", extra={"error": str(exc)})

        valid = [t for t in tickers if t in cfg.TICKERS]
        return valid[:max_tickers]

_tagger: TickerTagger | None = None

def get_ticker_tagger() -> TickerTagger:
    """Get ticker tagger."""
    global _tagger
    if _tagger is None:
        _tagger = TickerTagger()
    return _tagger
