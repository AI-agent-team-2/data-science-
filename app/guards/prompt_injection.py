from __future__ import annotations

import re

from app.routing import is_domain_query

ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\u2060\uFEFF\u00AD]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_for_guard(value: str) -> str:
    normalized = ZERO_WIDTH_PATTERN.sub("", str(value or ""))
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # English (direct)
    re.compile(r"(?i)\bignore\b.{0,40}\b(instructions?|system|developer|assistant|prompt)\b"),
    re.compile(r"(?i)\b(disregard|forget|override)\b.{0,40}\b(instructions?|system|developer|assistant|prompt)\b"),
    re.compile(r"(?i)\b(system prompt|developer message|assistant instructions?)\b"),
    re.compile(r"(?i)\b(reveal|print|show|leak)\b.{0,50}\b(system prompt|developer message|secret|policy)\b"),
    re.compile(r"(?i)\b(do anything now|jailbreak|developer mode)\b"),
    re.compile(r"(?i)\brole\s*[:=]\s*(system|developer)\b"),
    re.compile(r"(?i)\b(begin|end)\b.{0,10}\b(system prompt|developer message)\b"),
    # English (obfuscated spacing)
    re.compile(r"(?i)\bi\s*g\s*n\s*o\s*r\s*e\b.{0,40}\b(instructions?|system|developer|assistant|prompt)\b"),
    # Russian (direct)
    re.compile(
        r"(?i)\bигнориру(?:й|йте)\b.{0,40}\b(?:инструкц\w*|предыдущ\w*|систем\w*|промпт\w*|разработчик\w*|ассистент\w*)\b"
    ),
    re.compile(r"(?i)\b(не\s+следуй|не\s+учитывай|не\s+обращай\s+внимания)\b.{0,40}\b(?:инструкц\w*|предыдущ\w*|систем\w*)\b"),
    re.compile(r"(?i)\b(забудь|забудьте)\b.{0,40}\b(?:инструкц\w*|предыдущ\w*|систем\w*)\b"),
    re.compile(r"(?i)\b(системн\w*\s+промпт|сообщени\w*\s+разработчик\w*)\b"),
    re.compile(r"(?i)\b(раскрой|покажи|выведи)\b.{0,50}\b(системн|промпт|сообщени\w*\s+разработчик\w*|секрет|политик)\b"),
    re.compile(r"(?i)\b(джейлбрейк|jailbreak|режим\s+разработчика|developer\s+mode)\b"),
    # Russian (obfuscated spacing)
    re.compile(
        r"(?i)\bи\s*г\s*н\s*о\s*р\s*и\s*р\s*у\s*(?:й|йте)\b.{0,40}\b(?:инструкц\w*|предыдущ\w*|систем\w*|промпт\w*|разработчик\w*)\b"
    ),
)


def detect_prompt_injection(query: str) -> tuple[bool, list[str]]:
    """Определяет признаки prompt injection в пользовательском запросе."""
    normalized = _normalize_for_guard(query)
    matched: list[str] = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(normalized):
            matched.append(pattern.pattern)
    return bool(matched), matched


def rewrite_suspicious_query(query: str) -> str:
    """Убирает из подозрительного запроса управляющие инъекционные фрагменты."""
    rewritten = _normalize_for_guard(query)
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

