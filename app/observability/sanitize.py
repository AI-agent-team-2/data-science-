from __future__ import annotations

import hashlib
import re

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)")
BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-+/=]{12,}\b")
API_KEY_PATTERN = re.compile(r"(?i)\b(?:sk|rk|pk|api[_-]?key|token)[-_:=\s]*[A-Za-z0-9._\-]{12,}\b")
LONG_SECRET_PATTERN = re.compile(r"\b[A-Za-z0-9_\-]{24,}\b")
MAX_TEXT_LEN = 400


def hash_user_id(user_id: str) -> str:
    """Возвращает стабильный хэш для user/session идентификатора."""
    raw = str(user_id or "unknown")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"u_{digest[:16]}"


def sanitize_text(text: str) -> str:
    """Маскирует PII/секреты в текстовом payload."""
    value = str(text or "")
    if not value:
        return value

    sanitized = EMAIL_PATTERN.sub("[email]", value)
    sanitized = PHONE_PATTERN.sub("[phone]", sanitized)
    sanitized = BEARER_PATTERN.sub("[bearer]", sanitized)
    sanitized = API_KEY_PATTERN.sub("[secret]", sanitized)
    sanitized = LONG_SECRET_PATTERN.sub("[secret]", sanitized)

    if len(sanitized) > MAX_TEXT_LEN:
        return f"{sanitized[:MAX_TEXT_LEN]}...[truncated]"
    return sanitized

