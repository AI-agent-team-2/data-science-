from __future__ import annotations

import logging
import re

from app.config import settings
from app.observability.langfuse_client import get_langfuse_client
from app.observability.sanitize import has_pii

logger = logging.getLogger(__name__)

WORD_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]{3,}")
STOP_WORDS = {
    "это",
    "как",
    "что",
    "для",
    "или",
    "про",
    "под",
    "есть",
    "нет",
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
}


def _normalize_score(value: float) -> float:
    """Ограничивает score диапазоном [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def _extract_terms(text: str) -> set[str]:
    """Извлекает осмысленные токены для простой relevance-оценки."""
    raw = str(text or "").lower()
    if not raw:
        return set()

    terms = {token for token in WORD_PATTERN.findall(raw) if token not in STOP_WORDS}
    return terms


def compute_relevance(user_text: str, assistant_text: str) -> float:
    """Считает простую overlap-метрику релевантности (0..1)."""
    query_terms = _extract_terms(user_text)
    answer_terms = _extract_terms(assistant_text)
    if not query_terms or not answer_terms:
        return 0.0

    overlap = len(query_terms & answer_terms)
    return _normalize_score(overlap / max(1, len(query_terms)))


def compute_auto_scores(user_text: str, assistant_text: str, blocked: bool = False) -> dict[str, float]:
    """Формирует стандартные auto-scores для Langfuse trace."""
    relevance = compute_relevance(user_text=user_text, assistant_text=assistant_text)
    contains_pii = 1.0 if has_pii(assistant_text) else 0.0
    blocked_value = 1.0 if blocked else 0.0
    safety = 1.0 if (blocked or contains_pii == 0.0) else 0.0

    return {
        "relevance": relevance,
        "safety": safety,
        "contains_pii": contains_pii,
        "blocked": blocked_value,
    }


def submit_trace_scores(trace_id: str, scores: dict[str, float]) -> None:
    """Отправляет набор score в Langfuse и не бросает исключения наружу."""
    if not settings.langfuse_enabled or not settings.langfuse_auto_scoring_enabled:
        return

    clean_trace_id = str(trace_id or "").strip()
    if not clean_trace_id:
        return

    client = get_langfuse_client()
    if client is None:
        return

    try:
        for name, value in scores.items():
            client.score(trace_id=clean_trace_id, name=name, value=float(value))
    except Exception:
        logger.exception("Не удалось отправить auto-scores в Langfuse.")


def score_response_trace(
    trace_id: str,
    user_text: str,
    assistant_text: str,
    blocked: bool = False,
) -> dict[str, float]:
    """
    Считает и отправляет auto-scores для trace.

    Возвращает рассчитанные значения для удобства локальной диагностики.
    """
    scores = compute_auto_scores(
        user_text=user_text,
        assistant_text=assistant_text,
        blocked=blocked,
    )
    submit_trace_scores(trace_id=trace_id, scores=scores)
    return scores
