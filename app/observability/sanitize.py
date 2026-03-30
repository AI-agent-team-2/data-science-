from __future__ import annotations

import hashlib
import re
from typing import Callable, Literal

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RU_PATTERN = re.compile(r"(?<!\w)(?:\+7|8)\s*\(?\d{3}\)?[\s-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\w)")
CARD_PATTERN = re.compile(r"(?<!\w)(?:\d[\s-]*){13,19}(?!\w)")
INN_PATTERN = re.compile(r"(?<!\w)(?:\d{10}|\d{12})(?!\w)")
SNILS_PATTERN = re.compile(r"(?<!\w)\d{3}-\d{3}-\d{3}\s?\d{2}(?!\w)")
PASSPORT_RU_PATTERN = re.compile(r"(?i)\bпаспорт(?:\s+рф)?\b[^\d]{0,20}(\d{2}\s?\d{2}\s?\d{6})")
BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-+/=]{12,}\b")
API_KEY_PATTERN = re.compile(
    r"(?i)\b(?:api[_-]?key|token|secret|password|pwd|sk|rk|pk)\b[_:=\s-]*[A-Za-z0-9._\-+/=]{12,}\b"
)
ADDRESS_PATTERN = re.compile(
    r"(?i)\b(?:г\.?\s*[а-яa-z\-]+,\s*)?(?:ул\.?|улица|проспект|пр-т|пер\.?|переулок|б-р|бульвар)\s+"
    r"[а-яa-z0-9\- ]{2,40},?\s*(?:д\.?|дом)\s*\d+[а-яa-z]?(?:,?\s*(?:кв\.?|квартира)\s*\d+)?"
)
MAX_TEXT_LEN = 400
SANITIZE_MODE = Literal["observability", "model"]


def _digits_only(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _is_luhn_valid(number: str) -> bool:
    digits = _digits_only(number)
    if len(digits) < 13 or len(digits) > 19:
        return False

    total = 0
    reverse_digits = digits[::-1]
    for index, char in enumerate(reverse_digits):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _is_inn_valid(number: str) -> bool:
    digits = _digits_only(number)
    if len(digits) == 10:
        coeffs = (2, 4, 10, 3, 5, 9, 4, 6, 8)
        checksum = sum(int(digits[i]) * coeffs[i] for i in range(9)) % 11 % 10
        return checksum == int(digits[9])

    if len(digits) == 12:
        coeffs_11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
        coeffs_12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
        checksum_11 = sum(int(digits[i]) * coeffs_11[i] for i in range(10)) % 11 % 10
        checksum_12 = sum(int(digits[i]) * coeffs_12[i] for i in range(11)) % 11 % 10
        return checksum_11 == int(digits[10]) and checksum_12 == int(digits[11])

    return False


def _is_snils_valid(number: str) -> bool:
    digits = _digits_only(number)
    if len(digits) != 11:
        return False

    base = digits[:9]
    control = int(digits[9:])
    weighted_sum = sum(int(base[i]) * (9 - i) for i in range(9))

    if weighted_sum < 100:
        expected = weighted_sum
    elif weighted_sum in (100, 101):
        expected = 0
    else:
        expected = weighted_sum % 101
        if expected == 100:
            expected = 0

    return expected == control


def _replace_valid_matches(
    text: str,
    pattern: re.Pattern[str],
    tag: str,
    detector: Callable[[str], bool] | None = None,
) -> tuple[str, bool]:
    found = False

    def _sub(match: re.Match[str]) -> str:
        nonlocal found
        value = match.group(0)
        if detector is not None and not detector(value):
            return value

        found = True
        return f"[{tag}_REDACTED]"

    return pattern.sub(_sub, text), found


def hash_user_id(user_id: str) -> str:
    """Возвращает стабильный хэш для user/session идентификатора."""
    raw = str(user_id or "unknown")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"u_{digest[:16]}"


def detect_pii(text: str, mode: SANITIZE_MODE = "observability") -> list[str]:
    """Возвращает список найденных PII/секретов в тексте."""
    value = str(text or "")
    if not value:
        return []

    tags: list[str] = []
    if EMAIL_PATTERN.search(value):
        tags.append("EMAIL")
    if PHONE_RU_PATTERN.search(value):
        tags.append("PHONE_RU")
    if any(_is_luhn_valid(match.group(0)) for match in CARD_PATTERN.finditer(value)):
        tags.append("CARD")
    if any(_is_inn_valid(match.group(0)) for match in INN_PATTERN.finditer(value)):
        tags.append("INN")
    if any(_is_snils_valid(match.group(0)) for match in SNILS_PATTERN.finditer(value)):
        tags.append("SNILS")
    if PASSPORT_RU_PATTERN.search(value):
        tags.append("PASSPORT_RU")
    if BEARER_PATTERN.search(value):
        tags.append("BEARER")
    if API_KEY_PATTERN.search(value):
        tags.append("SECRET")
    if mode == "observability" and ADDRESS_PATTERN.search(value):
        tags.append("ADDRESS_RU")
    return tags


def sanitize_text(text: str, mode: SANITIZE_MODE = "observability") -> str:
    """Маскирует PII/секреты в текстовом payload."""
    value = str(text or "")
    if not value:
        return value

    sanitized = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", value)
    sanitized = PHONE_RU_PATTERN.sub("[PHONE_RU_REDACTED]", sanitized)
    sanitized, _ = _replace_valid_matches(sanitized, CARD_PATTERN, "CARD", detector=_is_luhn_valid)
    sanitized, _ = _replace_valid_matches(sanitized, INN_PATTERN, "INN", detector=_is_inn_valid)
    sanitized, _ = _replace_valid_matches(sanitized, SNILS_PATTERN, "SNILS", detector=_is_snils_valid)
    sanitized = PASSPORT_RU_PATTERN.sub("паспорт [PASSPORT_RU_REDACTED]", sanitized)
    sanitized = BEARER_PATTERN.sub("[BEARER_REDACTED]", sanitized)
    sanitized = API_KEY_PATTERN.sub("[SECRET_REDACTED]", sanitized)
    if mode == "observability":
        sanitized = ADDRESS_PATTERN.sub("[ADDRESS_RU_REDACTED]", sanitized)

    if mode == "observability" and len(sanitized) > MAX_TEXT_LEN:
        return f"{sanitized[:MAX_TEXT_LEN]}...[truncated]"
    return sanitized


def has_pii(text: str, mode: SANITIZE_MODE = "observability") -> bool:
    """Возвращает `True`, если в тексте найдены PII/секреты."""
    return bool(detect_pii(text=text, mode=mode))

