from __future__ import annotations

import re

NON_ALNUM_PATTERN = re.compile(r"[^A-Z0-9]+")
SKU_PATTERN = re.compile(r"\b[A-Z0-9][A-Z0-9\-_]{4,}\b")
SKU_SPLIT_PATTERN = re.compile(r"\b([A-Z]{2,}[A-Z0-9]{1,})[\s\-_/]+([0-9]{2,}[A-Z0-9]*)\b")


def canonical_sku(value: str) -> str:
    """Нормализует SKU к каноническому виду для надежного сравнения."""
    return NON_ALNUM_PATTERN.sub("", str(value or "").upper())


def extract_sku_candidates(query: str, *, require_digit: bool = True) -> set[str]:
    """Извлекает набор канонических SKU из строки, включая "разбитые" форматы."""
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

