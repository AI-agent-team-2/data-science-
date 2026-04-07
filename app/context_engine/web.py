from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from app.config import settings
from app.routing.intents import SANITARY_KEYWORDS

logger = logging.getLogger(__name__)

YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
CLEAN_TRUNCATED_MARKER_PATTERN = re.compile(r"[\[\(\{]?\s*\.{0,3}\s*truncated\s*[\]\)\}]?", re.IGNORECASE)
WEB_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(system prompt|developer message|assistant instructions?)\b"),
    re.compile(r"(?i)\b(ignore|disregard|override|forget)\b.{0,40}\b(instruction|previous|system|prompt|developer|assistant)\b"),
    re.compile(r"(?i)\b(follow|repeat|reveal|print)\b.{0,40}\b(instruction|prompt|system|developer|secret)\b"),
    re.compile(r"(?i)\byou are (chatgpt|an ai|assistant)\b"),
    re.compile(r"(?i)игнориру(?:й|йте).{0,40}(инструкц|предыдущ|систем|промпт|разработчик)"),
    re.compile(r"(?i)(системн\w*\s+промпт|сообщени\w*\s+разработчик\w*)"),
)


def enhance_search_query(original_query: str, search_type: str = "general") -> str:
    """Улучшает формулировку запроса для внешнего web-поиска."""
    lowered_query = original_query.lower()

    if any(marker in lowered_query for marker in ("сантехник", "унитаз", "смеситель")):
        return original_query

    if "новинк" in lowered_query or "новые" in lowered_query:
        year_match = YEAR_PATTERN.search(lowered_query)
        target_year = year_match.group(1) if year_match else str(datetime.now().year)
        return f"новинки сантехники {target_year} каталог"

    if any(marker in lowered_query for marker in ("бюджет", "цен", "стоит", "сколько")):
        return f"{original_query} сантехника"

    if "купить" in lowered_query or "где" in lowered_query:
        return f"{original_query} сантехника интернет магазин"

    if search_type == "fallback":
        return f"{original_query} сантехника"

    return f"{original_query} сантехника товары"


def is_web_useful(items: list[dict[str, Any]]) -> bool:
    """Проверяет, что WEB-результаты содержат валидные ссылки."""
    for item in items:
        url = str(item.get("url", "")).strip().lower()
        if url.startswith("https://") or url.startswith("http://"):
            return True
    return False


def has_minimum_web_evidence(items: list[dict[str, Any]]) -> bool:
    """Проверяет, что для web-ответа найдено минимум валидных источников."""
    urls = extract_web_urls(items)
    return len(urls) >= max(1, settings.web_min_sources)


def is_sanitary_relevant(items: list[dict[str, Any]]) -> bool:
    """Проверяет, что WEB-результаты действительно относятся к сантехнике."""
    for item in items:
        title = str(item.get("title", "")).lower()
        snippet = str(item.get("snippet", "")).lower()
        combined = f"{title} {snippet}"
        if any(keyword in combined for keyword in SANITARY_KEYWORDS):
            return True
    return False


def filter_safe_web_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Отбрасывает WEB-результаты с инструкционными/инъекционными фрагментами."""
    safe_items: list[dict[str, Any]] = []
    dropped = 0
    for item in items:
        title = str(item.get("title", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        combined = f"{title}\n{snippet}"
        if contains_instruction_like_text(combined):
            dropped += 1
            continue
        safe_items.append(item)
    if dropped:
        logger.warning("Dropped %s suspicious web result(s) before prompt assembly", dropped)
    return safe_items


def filter_trusted_web_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ограничивает web-результаты allowlist доменов, если список задан в конфиге."""
    trusted_domains = [value.lower() for value in settings.web_trusted_domains if value]
    if not trusted_domains:
        return items

    safe: list[dict[str, Any]] = []
    for item in items:
        raw_url = str(item.get("url", "")).strip()
        if not raw_url:
            continue
        host = (urlparse(raw_url).hostname or "").lower()
        if any(host == domain or host.endswith(f".{domain}") for domain in trusted_domains):
            safe.append(item)
    return safe


def contains_instruction_like_text(value: str) -> bool:
    """Определяет instruction-like текст, который не должен попадать в LLM-контекст из WEB."""
    normalized = str(value or "").strip()
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in WEB_INJECTION_PATTERNS)


def clean_web_text(value: str) -> str:
    """Удаляет служебные маркеры обрезки из web-фрагментов и нормализует пробелы."""
    cleaned = CLEAN_TRUNCATED_MARKER_PATTERN.sub(" ", str(value or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;|")
    return cleaned


def extract_web_urls(items: list[dict[str, Any]]) -> list[str]:
    """Собирает уникальные URL из WEB-результатов."""
    urls: list[str] = []
    seen: set[str] = set()
    for item in items:
        url = str(item.get("url", "")).strip()
        if not (url.startswith("https://") or url.startswith("http://")):
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls
