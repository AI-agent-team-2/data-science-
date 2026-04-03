from __future__ import annotations

import re

from app.routing import is_domain_query

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)ignore (all|previous|prior) instructions"),
    re.compile(r"(?i)system prompt"),
    re.compile(r"(?i)developer message"),
    re.compile(r"(?i)игнорируй\s+предыдущ"),
    re.compile(r"(?i)раскрой\s+системн"),
)


def detect_prompt_injection(query: str) -> tuple[bool, list[str]]:
    """Определяет признаки prompt injection в пользовательском запросе."""
    matched: list[str] = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(query):
            matched.append(pattern.pattern)
    return bool(matched), matched


def rewrite_suspicious_query(query: str) -> str:
    """Убирает из подозрительного запроса управляющие инъекционные фрагменты."""
    rewritten = query
    for pattern in INJECTION_PATTERNS:
        rewritten = pattern.sub(" ", rewritten)
    rewritten = re.sub(r"\s+", " ", rewritten).strip(" ,.;:")
    return rewritten


def apply_guard(query: str) -> tuple[str, str, list[str]]:
    """
    Применяет легкий guard и возвращает:
    safe_query, action (allow|rewrite|block), risk_flags.
    """
    is_injection, matched = detect_prompt_injection(query)
    if not is_injection:
        return query, "allow", []

    risk_flags = ["prompt_injection"] + [f"pattern:{value}" for value in matched[:3]]
    rewritten = rewrite_suspicious_query(query)
    if rewritten and is_domain_query(rewritten.lower()):
        return rewritten, "rewrite", risk_flags
    return query, "block", risk_flags


def known_domain_constraint_response(query: str) -> str:
    """Возвращает детерминированный ответ для критичных доменных ограничений."""
    lowered = str(query or "").lower()
    if "мультифлекс" in lowered and any(marker in lowered for marker in ("душить поток", "регулировать поток", "регулиров", "придушить")):
        return (
            "Нет, шаровыми кранами мультифлекса нельзя регулировать (душить) поток. "
            "Они предназначены для полного открытия/закрытия. "
            "Для регулирования используйте штатные регулирующие элементы контура."
        )
    return ""
