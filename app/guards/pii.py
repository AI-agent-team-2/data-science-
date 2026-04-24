from __future__ import annotations

from dataclasses import dataclass

from app.observability.sanitize import EMAIL_PATTERN, PHONE_PATTERN, BEARER_PATTERN, API_KEY_PATTERN


@dataclass(frozen=True)
class RedactionResult:
    text: str
    redacted: bool


def redact_pii(text: str) -> RedactionResult:
    """
    Редактирует наиболее типичные PII/секреты в тексте, без принудительной обрезки длины.

    NB: Это для ответа пользователю, поэтому намеренно НЕ используем LONG_SECRET_PATTERN,
    чтобы не портить артикулы/модели/идентификаторы.
    """
    value = str(text or "")
    if not value:
        return RedactionResult(text=value, redacted=False)

    redacted = value
    redacted = EMAIL_PATTERN.sub("[email]", redacted)
    redacted = PHONE_PATTERN.sub("[phone]", redacted)
    redacted = BEARER_PATTERN.sub("[bearer]", redacted)
    redacted = API_KEY_PATTERN.sub("[secret]", redacted)
    return RedactionResult(text=redacted, redacted=(redacted != value))
