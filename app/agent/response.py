from __future__ import annotations

import re


TRUNCATED_MARKER_PATTERN = re.compile(r"[\[\(\{]?\s*\.{0,3}\s*truncated\s*[\]\)\}]?", re.IGNORECASE)


def prepare_user_answer(raw_assistant_text: str) -> str:
    """Очищает только технический мусор и сохраняет валидные пользовательские ссылки."""
    value = str(raw_assistant_text or "").strip()
    if not value:
        return "Не удалось получить ответ."
    cleaned = TRUNCATED_MARKER_PATTERN.sub(" ", value)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned
