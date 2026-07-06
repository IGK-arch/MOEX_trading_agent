"""Утренний план в 09:30 МСК."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import app.config as cfg
from app.agents.hmm_regime import get_hmm_detector
from app.llm.polza_client import get_polza_client
from app.utils.logging import get_logger

logger = get_logger(__name__)

MORNING_PROMPT = """Ты — старший трейдер MOEX. Сегодня {today} ({day_of_week}).

КОНТЕКСТ С ПРЕДЫДУЩЕГО ДНЯ:
{yesterday_summary}

УРОКИ ВЧЕРАШНЕГО ДНЯ:
{lessons_text}

ТЕКУЩИЙ РЕЖИМ РЫНКА (HMM):
{regime} (proba={regime_proba})

ОБОРОТ ЗА НЕДЕЛЮ:
{turnover_status}

ОТКРЫТЫЕ ПОЗИЦИИ:
{positions_summary}

ЗАДАЧА:
Составь краткий утренний брифинг (markdown, 200-400 слов). Включи:
- На что обращать внимание сегодня (тикеры, секторы, события)
- Какие сигналы фильтровать строже из-за вчерашних ошибок
- Стратегические корректировки sizing (если есть отстание по обороту)
- 3-5 конкретных setup'ов которые ожидаются

Стиль: трейдерский, кратко, по делу. Без воды.
"""

class MorningPlanner:
    """Generate the daily morning brief at 09:30 MSK."""

    def __init__(self) -> None:
        """Init."""
        self.polza = get_polza_client()
        self.hmm = get_hmm_detector()

    async def generate(self) -> str:
        """Generate today's morning brief markdown."""

        import app.config as cfg

        if cfg.DISABLE_LLM:
            logger.info("morning_plan: DISABLE_LLM=1, skipping LLM brief")
            return ""
        if not self.polza._started:
            await self.polza.startup()

        today = datetime.now(tz=UTC)
        today_str = today.strftime("%Y-%m-%d")
        yesterday_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")

        lessons_text = self._load_lessons(yesterday_str)

        regime = self.hmm.current_label
        regime_proba = "n/a"

        turnover_status = "Week-to-date turnover not yet computed (needs turnover_tracker)"

        yesterday_summary = self._load_yesterday_summary(yesterday_str)

        prompt = MORNING_PROMPT.format(
            today=today_str,
            day_of_week=today.strftime("%A"),
            yesterday_summary=yesterday_summary,
            lessons_text=lessons_text,
            regime=regime,
            regime_proba=regime_proba,
            turnover_status=turnover_status,
            positions_summary="(будет заполнено position_book.py)",
        )

        morning_model = (
            "qwen/qwen3.7-max"
            if getattr(cfg, "USE_QWEN_FOR_MORNING_BRIEF", False)
            else cfg.POLZA_MODEL_MORNING_BRIEF
        )
        max_t = int(getattr(cfg, "MORNING_PLAN_MAX_TOKENS", 3000))
        try:
            result = await self.polza.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Ты опытный трейдер MOEX. Пиши кратко и конкретно.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=morning_model,
                purpose="morning_plan",
                max_tokens=max_t,
                use_cache=False,
                use_reasoning=True,
                reasoning_effort="low",
                include_reasoning=False,
            )
        except Exception as exc:
            logger.error("Morning plan LLM call failed", extra={"error": str(exc)})
            return ""

        brief = result.get("content", "")
        if not brief:
            return ""

        plans_dir = cfg.DATA_DIR / "morning_plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        path = plans_dir / f"{today_str}.md"
        try:
            with open(path, "w") as f:
                f.write(f"# Morning Plan — {today_str}\n\n")
                f.write(f"**Regime:** {regime}\n\n")
                f.write(brief)
            logger.info("Morning plan saved", extra={"path": str(path), "chars": len(brief)})
        except Exception as exc:
            logger.error("Morning plan save failed", extra={"error": str(exc)})

        return brief

    def _load_lessons(self, date_str: str) -> str:
        """Read yesterday's lessons from JSON file."""
        path = cfg.DATA_DIR / "lessons" / f"{date_str}.json"
        if not path.exists():
            return "(нет уроков — первый день)"
        try:
            with open(path) as f:
                data = json.load(f)
            lessons = data.get("lessons", [])
            if not lessons:
                return "(нет уроков)"
            lines = []
            for i, l in enumerate(lessons, 1):
                lines.append(
                    f"{i}. [{l.get('category', '?')}] (важность={l.get('importance', 0):.2f}) "
                    f"{l.get('lesson', '')}"
                )
            return "\n".join(lines)
        except Exception as exc:
            logger.error("Lessons load failed", extra={"error": str(exc)})
            return "(ошибка загрузки уроков)"

    def _load_yesterday_summary(self, date_str: str) -> str:
        """Compact summary of yesterday's P&L and trade count."""

        return "(будет реализовано в Phase 9 после первого торгового дня)"

_morning_planner: MorningPlanner | None = None

def get_morning_planner() -> MorningPlanner:
    """Get morning planner."""
    global _morning_planner
    if _morning_planner is None:
        _morning_planner = MorningPlanner()
    return _morning_planner
