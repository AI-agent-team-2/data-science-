from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

from langchain_core.tools import tool

from app.config import settings
from app.observability import sanitize_text
from app.tools.response_utils import empty_results_payload

logger = logging.getLogger(__name__)

CACHE_DIR: Final[Path] = Path(__file__).resolve().parents[2] / ".web_cache"
MAX_RESULTS_HARD_LIMIT: Final[int] = 10


def _get_cache_ttl() -> timedelta:
    """Возвращает TTL кэша веб-поиска на основе настроек."""
    if settings.web_cache_enabled:
        return timedelta(hours=settings.web_cache_ttl_hours)
    return timedelta(hours=0)


def _get_cache_key(query: str, max_results: int) -> str:
    """Создает стабильный ключ кэша по параметрам поиска."""
    key_source = f"{query}_{max_results}"
    return hashlib.md5(key_source.encode("utf-8")).hexdigest()


def _load_from_cache(key: str) -> dict[str, Any] | None:
    """Возвращает кэшированный результат, если кэш включен и не протух."""
    if not settings.web_cache_enabled:
        return None

    cache_file = CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        return None

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        cache_time = datetime.fromisoformat(str(payload.get("cached_at", "")))

        if datetime.now() - cache_time > _get_cache_ttl():
            cache_file.unlink(missing_ok=True)
            return None

        result = payload.get("result")
        return result if isinstance(result, dict) else None
    except Exception:
        logger.exception("Не удалось загрузить кэш WEB-поиска из %s", cache_file)
        return None


def _save_to_cache(key: str, result: dict[str, Any]) -> None:
    """Сохраняет результат поиска в файловый кэш."""
    if not settings.web_cache_enabled:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{key}.json"
    payload = {
        "cached_at": datetime.now().isoformat(),
        "result": result,
    }

    try:
        cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Не удалось сохранить кэш WEB-поиска в %s", cache_file)


def _normalize_results(query: str, items: list[dict[str, Any]], provider: str) -> dict[str, Any]:
    """Приводит результаты разных провайдеров к единому формату."""
    normalized_items: list[dict[str, str]] = []

    for item in items:
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        url = str(item.get("url") or "").strip()

        if not (title or snippet or url):
            continue

        normalized_items.append(
            {
                "title": title,
                "snippet": snippet,
                "url": url,
            }
        )

    return {
        "query": query,
        "provider": provider,
        "count": len(normalized_items),
        "results": normalized_items,
        "error": "",
    }


def _error_object(query: str, provider: str, message: str) -> dict[str, Any]:
    """Формирует JSON-ответ с ошибкой в едином формате."""
    payload = empty_results_payload(query=query)
    payload.update(
        {
            "provider": provider,
            "error": message,
        }
    )
    return payload


def _duckduckgo_search(query: str, max_results: int) -> dict[str, Any]:
    """Выполняет web-поиск через DDGS/DuckDuckGo."""
    try:
        try:
            from ddgs import DDGS  # type: ignore
        except Exception:
            from duckduckgo_search import DDGS  # type: ignore
    except Exception as exc:
        return _error_object(
            query=query,
            provider="duckduckgo",
            message=f"Бэкенд DuckDuckGo недоступен. Установите зависимость ddgs. Детали: {exc}",
        )

    items: list[dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            for result in ddgs.text(
                query,
                max_results=max_results,
                region="ru-ru",
                safesearch="moderate",
            ):
                items.append(
                    {
                        "title": result.get("title", ""),
                        "snippet": result.get("body", "") or result.get("snippet", ""),
                        "url": result.get("href", "") or result.get("url", ""),
                    }
                )
    except Exception as exc:
        return _error_object(
            query=query,
            provider="duckduckgo",
            message=f"Ошибка поиска DuckDuckGo. Детали: {exc}",
        )

    return _normalize_results(query=query, items=items, provider="duckduckgo")


def _tavily_search(query: str, max_results: int, api_key: str) -> dict[str, Any]:
    """Выполняет веб-поиск через Tavily."""
    try:
        from tavily import TavilyClient  # type: ignore
    except Exception as exc:
        return _error_object(
            query=query,
            provider="tavily",
            message=f"Бэкенд Tavily недоступен. Установите зависимость tavily-python. Детали: {exc}",
        )

    try:
        client = TavilyClient(api_key=api_key)
        response: dict[str, Any] = client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )
        results = response.get("results") or []
        items = [
            {
                "title": item.get("title", ""),
                "snippet": item.get("content", "") or item.get("snippet", ""),
                "url": item.get("url", ""),
            }
            for item in results
            if isinstance(item, dict)
        ]
    except Exception as exc:
        return _error_object(
            query=query,
            provider="tavily",
            message=f"Ошибка поиска Tavily. Детали: {exc}",
        )

    return _normalize_results(query=query, items=items, provider="tavily")


@tool
def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Ищет актуальную внешнюю информацию в интернете.

    Parameters
    ----------
    query : str
        Поисковый запрос.
    max_results : int, default=5
        Максимальное число результатов.

    Returns
    -------
    str
        JSON-ответ с результатами внешнего поиска.
    """
    if not settings.enable_web_search:
        return _error_object(query, "disabled", "Веб-поиск отключен в настройках.")

    normalized_max_results = min(max_results, settings.web_search_max_results)
    normalized_max_results = int(max(1, min(normalized_max_results, MAX_RESULTS_HARD_LIMIT)))

    cache_key = _get_cache_key(query, normalized_max_results)
    cached_result = _load_from_cache(cache_key)
    if cached_result is not None:
        return cached_result

    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key:
        result = _tavily_search(query=query, max_results=normalized_max_results, api_key=tavily_key)
    else:
        result = _duckduckgo_search(query=query, max_results=normalized_max_results)

    try:
        if isinstance(result, dict) and result.get("results"):
            _save_to_cache(cache_key, result)
    except Exception as exc:
        logger.exception("Не удалось разобрать ответ web_search для кэширования")
        logger.debug("Детали ошибки web_search: %s", sanitize_text(str(exc)))

    return result
