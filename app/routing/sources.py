from __future__ import annotations

import re
from typing import Literal

from app.routing.intents import has_sku_signal, is_domain_query, is_noise_query

ToolName = Literal["lookup", "rag", "web"]

WEB_YEAR_PATTERN = re.compile(r"\bв\s+20\d{2}\b")

WEB_PRIORITY_MARKERS: tuple[str, ...] = (
    "сейчас",
    "сегодня",
    "новые требования",
    "что изменилось",
    "по отзывам",
    "отзывы",
    "где купить",
    "в москве",
    "в россии",
    "средняя цена",
    "цена",
    "аналоги",
    "сравни",
    "новости",
    "лучший",
    "рейтинг",
    "топ",
    "популярный",
    "актуальный",
    "последний",
    "новинки",
    "тренды",
    "тенденции",
    "современные решения",
    "чаще используют",
    "стандарты",
    "бренды",
)

LOOKUP_PRIORITY_MARKERS: tuple[str, ...] = (
    "артикул",
    "sku",
    "модель",
    "код",
    "товар",
    "бренд",
    "серия",
    "позици",
)


def resolve_source_order(query: str) -> list[ToolName]:
    """Возвращает порядок источников в зависимости от типа запроса."""
    if should_prefer_web(query):
        return ["web", "rag", "lookup"]
    if should_prefer_lookup(query):
        return ["lookup", "rag", "web"]
    return ["rag", "lookup", "web"]


def should_use_web_source(query: str, web_mode: str) -> bool:
    """Ограничивает WEB: как fallback используем только для явных внешних запросов."""
    lowered_query = query.lower()
    if is_noise_query(query):
        return False
    if web_mode == "fallback":
        return should_prefer_web(query)
    return should_prefer_web(query) or is_domain_query(lowered_query)


def should_prefer_web(query: str) -> bool:
    """Определяет, когда WEB должен быть первым источником."""
    lowered_query = query.lower()
    words = lowered_query.split()
    has_web_marker = any(marker in lowered_query for marker in WEB_PRIORITY_MARKERS)
    has_web_year_marker = WEB_YEAR_PATTERN.search(lowered_query) is not None
    has_lookup_marker = any(marker in lowered_query for marker in LOOKUP_PRIORITY_MARKERS)
    has_sku = has_sku_signal(query)

    if (has_web_marker or has_web_year_marker) and not has_lookup_marker and not has_sku:
        return True

    if len(words) < 3:
        return (has_web_marker or has_web_year_marker) and not has_sku

    if is_domain_query(lowered_query):
        return False

    return has_web_marker or has_web_year_marker


def should_prefer_lookup(query: str) -> bool:
    """Определяет, когда LOOKUP должен быть первым источником."""
    if has_sku_signal(query):
        return True

    lowered_query = query.lower()
    return any(marker in lowered_query for marker in LOOKUP_PRIORITY_MARKERS)
