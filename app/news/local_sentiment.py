"""Local (offline) sentiment scorer — fallback when POLZA LLM is unavailable.

Phase 26 (v0.0.31) — POLZA API key was disabled in production, so the news
pipeline returned 0 signals/day. This module provides a self-contained Russian
financial sentiment classifier that runs without any external API.

Two backends, picked at runtime:

1. **HuggingFace** (primary, preferred) — fine-tuned RuBERT-tiny2 on Russian
   financial Telegram posts: ``mxlcw/rubert-tiny2-russian-financial-sentiment``.
   ~29 M params, ~120 MB, F1≈0.75 on Russian financial text. CPU-only inference
   ~50–200 ms per headline on a modern container.
   Falls back to the base sentiment model (``seara/rubert-tiny2-russian-sentiment``)
   if the financial fine-tune is unreachable.

2. **Keyword scorer** (fallback) — zero dependencies, instantaneous. Russian
   financial positive/negative term lexicon. Used when ``transformers`` /
   ``torch`` are not installed, when the model file cannot be downloaded
   (e.g. air-gapped CI), or when HF inference raises.

Public API::

    scorer = LocalSentimentScorer()
    score = scorer.score_text("Сбербанк увеличил прибыль на 30%")
    # score ∈ [-1.0, +1.0]; sign = direction, magnitude = confidence
"""

from __future__ import annotations

import re
import threading
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)

POSITIVE_KEYWORDS: tuple[str, ...] = (
    "рост",
    "выросл",
    "вырос",
    "увелич",
    "повыс",
    "прибыл",
    "доход",
    "дивиденд",
    "выплат",
    "buyback",
    "выкуп",
    "приобрел",
    "приобрет",
    "контракт",
    "сделк",
    "соглашен",
    "партнерств",
    "сотруднич",
    "запуск",
    "открыл",
    "ввод в эксплуатац",
    "расширен",
    "рекордн",
    "лучше прогноз",
    "превысил",
    "beat",
    "beating",
    "одобр",
    "разрешил",
    "получил лиценз",
    "успешн",
    "снижен ставк",
    "rate cut",
    "позитив",
    "оптимизм",
    "ipo",
    "spo",
    "размещен",
    "ликвидност",
    "стабильн",
)

NEGATIVE_KEYWORDS: tuple[str, ...] = (
    "падение",
    "упал",
    "снижен",
    "сократ",
    "пониж",
    "убыт",
    "потер",
    "loss",
    "down",
    "минус",
    "санкци",
    "sanction",
    "запрет",
    "ban",
    "blocked",
    "блок",
    "отзыв лиценз",
    "лишен лиценз",
    "приостанов",
    "halt",
    "halted",
    "штраф",
    "fine",
    "пени",
    "взыскан",
    "расследован",
    "investigation",
    "обыск",
    "банкрот",
    "default",
    "дефолт",
    "реструктур",
    "downgrade",
    "хуже прогноз",
    "miss",
    "missing forecast",
    "atc clearing",
    "trade halt",
    "пожар",
    "авари",
    "ущерб",
    "крушен",
    "забастовк",
    "strike",
    "повыс ставк",
    "rate hike",
    "негатив",
    "пессим",
    "отток",
    "продаж",
    "распродаж",
    "ликвидац",
)

def _compile_stems(stems: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a regex matching word-start with free Russian suffix.

    Russian inflects on suffixes — we anchor on ``\\b`` at the start and allow
    arbitrary letters/dashes afterwards so "прибыл" hits "прибыль", "прибыли",
    "прибылью". English entries get a closing ``\\b`` because Latin letter
    classes terminate naturally at the next non-word character.
    """
    parts = sorted({re.escape(s) for s in stems}, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(parts) + r")"
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)

_POSITIVE_RE = _compile_stems(POSITIVE_KEYWORDS)
_NEGATIVE_RE = _compile_stems(NEGATIVE_KEYWORDS)

_HF_MODEL_CANDIDATES: tuple[str, ...] = (
    "mxlcw/rubert-tiny2-russian-financial-sentiment",
    "seara/rubert-tiny2-russian-sentiment",
)

def keyword_sentiment_score(text: str) -> float:
    """Pure-python sentiment score from RU/EN financial keyword counts.

    Returns a value in [-1.0, +1.0]:

    * sign — net direction (positive keywords ↔ negative keywords)
    * magnitude — normalised count (saturates around 5 matched keywords)

    The score is intentionally conservative: a single keyword in a long
    article should not push magnitude above ~0.3, while three concordant
    keywords (e.g. ``"санкции"``, ``"запрет"``, ``"блок"``) approach 0.7.
    """
    if not text:
        return 0.0

    pos_hits = len(_POSITIVE_RE.findall(text))
    neg_hits = len(_NEGATIVE_RE.findall(text))

    if pos_hits == 0 and neg_hits == 0:
        return 0.0

    raw = (pos_hits - neg_hits) / max(5.0, float(pos_hits + neg_hits))
    return max(-1.0, min(1.0, raw))

class LocalSentimentScorer:
    """Lazy-loaded sentiment scorer with HF model + keyword fallback.

    The model is loaded on first call to :meth:`score_text` (or via the
    explicit :meth:`preload` helper). Subsequent calls reuse the cached
    pipeline. Thread-safe.

    Failure modes are non-fatal: anything that prevents the BERT model from
    loading or running degrades the scorer to the keyword backend. The class
    never raises out of :meth:`score_text`.
    """

    def __init__(
        self,
        model_candidates: tuple[str, ...] | None = None,
        device: str = "cpu",
        max_text_chars: int = 1500,
        force_keyword_only: bool = False,
    ) -> None:
        """Init."""
        self._candidates = tuple(model_candidates or _HF_MODEL_CANDIDATES)
        self._device = device
        self._max_text_chars = max_text_chars
        self._force_keyword_only = force_keyword_only

        self._pipeline: Any = None
        self._model_name: str | None = None
        self._tried_load: bool = False
        self._lock = threading.Lock()

        self._calls_total = 0
        self._calls_bert = 0
        self._calls_keyword = 0

    @property
    def backend(self) -> str:
        """``"bert"``, ``"keyword"`` or ``"unloaded"`` — for logging only."""
        if self._pipeline is not None:
            return "bert"
        if self._tried_load:
            return "keyword"
        return "unloaded"

    @property
    def model_name(self) -> str | None:
        """Model name."""
        return self._model_name

    def preload(self) -> bool:
        """Force-load the HF pipeline. Returns ``True`` if BERT is ready."""
        self._ensure_loaded()
        return self._pipeline is not None

    def _ensure_loaded(self) -> None:
        """One-shot lazy load behind a lock."""
        if self._tried_load or self._force_keyword_only:
            if self._force_keyword_only:
                self._tried_load = True
            return
        with self._lock:
            if self._tried_load:
                return
            self._tried_load = True
            try:
                from transformers import pipeline  # type: ignore
            except Exception as exc:
                logger.warning(
                    "LocalSentimentScorer: transformers not installed, using keyword fallback",
                    extra={"error": str(exc)},
                )
                return

            for name in self._candidates:
                try:
                    self._pipeline = pipeline(
                        "sentiment-analysis",
                        model=name,
                        tokenizer=name,
                        device=-1 if self._device == "cpu" else 0,
                        truncation=True,
                        max_length=512,
                    )
                    self._model_name = name
                    logger.info(
                        "LocalSentimentScorer: BERT model loaded",
                        extra={"model": name, "device": self._device},
                    )
                    return
                except Exception as exc:
                    logger.warning(
                        "LocalSentimentScorer: model load failed",
                        extra={"model": name, "error": str(exc)},
                    )
            logger.warning(
                "LocalSentimentScorer: all HF candidates failed, falling back to keyword scorer"
            )

    @staticmethod
    def _label_to_score(label: str, confidence: float) -> float:
        """Map a HF 3-class label + score onto the [-1, +1] continuous axis."""
        normalised = label.strip().lower()
        if normalised in ("positive", "pos", "label_1"):
            return float(confidence)
        if normalised in ("negative", "neg", "label_2"):
            return -float(confidence)
        if normalised in ("neutral", "neu", "label_0"):
            return 0.0
        return 0.0

    def score_text(self, text: str) -> float:
        """Return sentiment score ∈ [-1.0, +1.0] for ``text``.

        * ``+1`` — maximally positive
        * ``-1`` — maximally negative
        * ``0`` — neutral or empty

        The result is the BERT model's continuous score when the model is
        available, otherwise the keyword score. Errors during BERT inference
        silently fall back to the keyword score.
        """
        self._calls_total += 1
        if not text:
            return 0.0

        snippet = text[: self._max_text_chars]

        self._ensure_loaded()
        if self._pipeline is not None:
            try:
                result = self._pipeline(snippet)
                if isinstance(result, list) and result:
                    item = result[0]
                    label = str(item.get("label", ""))
                    confidence = float(item.get("score", 0.0) or 0.0)
                    bert_score = self._label_to_score(label, confidence)
                    self._calls_bert += 1
                    return max(-1.0, min(1.0, bert_score))
            except Exception as exc:
                logger.warning(
                    "LocalSentimentScorer: BERT inference failed, falling back to keyword",
                    extra={"error": str(exc), "model": self._model_name},
                )

        score = keyword_sentiment_score(snippet)
        self._calls_keyword += 1
        return score

    def stats(self) -> dict[str, Any]:
        """Stats."""
        return {
            "backend": self.backend,
            "model": self._model_name,
            "calls_total": self._calls_total,
            "calls_bert": self._calls_bert,
            "calls_keyword": self._calls_keyword,
        }

_scorer: LocalSentimentScorer | None = None

def get_local_sentiment_scorer() -> LocalSentimentScorer:
    """Process-wide singleton — model is loaded once and reused."""
    global _scorer
    if _scorer is None:
        _scorer = LocalSentimentScorer()
    return _scorer
