from __future__ import annotations

import re

NON_ALNUM_PATTERN = re.compile(r"[^A-Z0-9]+")
SKU_PATTERN = re.compile(r"\b[A-Z0-9][A-Z0-9\-_]{4,}\b")
SKU_SPLIT_PATTERN = re.compile(r"\b([A-Z]{2,}[A-Z0-9]{1,})[\s\-_/]+([0-9]{2,}[A-Z0-9]*)\b")

# Паттерны российских идентификаторов (НЕ должны быть SKU)
INN_PATTERN = re.compile(r"\b\d{10}\b|\b\d{12}\b")  # ИНН: 10 или 12 цифр
PASSPORT_PATTERN = re.compile(r"\b\d{2}[\s\-]?\d{6}\b")  # Паспорт: 45 1234567 или 451234567
SNILS_PATTERN = re.compile(r"\b\d{3}-\d{3}-\d{3}-\d{2}\b")  # СНИЛС: 123-456-789-01
PHONE_PATTERN = re.compile(r"\b[78]\d{10}\b")  # Телефон: 8XXXXXXXXXX или 7XXXXXXXXXX


def canonical_sku(value: str) -> str:
    """Нормализует SKU к каноническому виду для надежного сравнения."""
    return NON_ALNUM_PATTERN.sub("", str(value or "").upper())


def is_russian_identifier(query: str) -> bool:
    """
    Проверяет, содержит ли строка российский идентификатор (ИНН, паспорт, СНИЛС, телефон).
    Возвращает True, если найден хотя бы один российский идентификатор.
    """
    if not query:
        return False
    
    # Очищаем от лишних пробелов
    query_clean = str(query).strip()
    
    # Проверка ИНН (10 или 12 цифр подряд)
    if INN_PATTERN.search(query_clean):
        return True
    
    # Проверка номера паспорта (2 цифры + пробел/дефис/ничего + 6 цифр)
    if PASSPORT_PATTERN.search(query_clean):
        return True
    
    # Проверка СНИЛС (формат XXX-XXX-XXX-XX)
    if SNILS_PATTERN.search(query_clean):
        return True
    
    # Проверка телефона (11 цифр, начинается с 7 или 8)
    if PHONE_PATTERN.search(query_clean):
        return True
    
    return False


def extract_sku_candidates(query: str, *, require_digit: bool = True) -> set[str]:
    """
    Извлекает набор канонических SKU из строки, включая "разбитые" форматы.
    Игнорирует российские идентификаторы (ИНН, паспорт, СНИЛС, телефон).
    """
    # Если запрос содержит российский идентификатор — не извлекаем SKU
    if is_russian_identifier(query):
        return set()
    
    query_upper = str(query or "").upper()
    candidates: set[str] = set()

    for raw_sku in SKU_PATTERN.findall(query_upper):
        normalized = canonical_sku(raw_sku)
        if not normalized:
            continue
        if require_digit and not any(char.isdigit() for char in normalized):
            continue
        candidates.add(normalized)

    for left, right in SKU_SPLIT_PATTERN.findall(query_upper):
        normalized = canonical_sku(f"{left}{right}")
        if not normalized:
            continue
        if require_digit and not any(char.isdigit() for char in normalized):
            continue
        candidates.add(normalized)

    return candidates


def contains_sku_candidate(query: str, *, require_digit: bool = False) -> bool:
    """Проверяет, содержит ли строка хотя бы один SKU-кандидат."""
    return bool(extract_sku_candidates(query, require_digit=require_digit))

