from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

from langchain_core.tools import tool

from app.config import settings
from app.tools.response_utils import build_tool_payload, error_payload

def filter_by_trusted_domains(results: list[dict]) -> list[dict]:
    """Фильтрует результаты поиска по доверенным доменам."""
    if not settings.web_trusted_domains_enabled:
        return results

    trusted = [domain.lower().strip() for domain in settings.web_trusted_domains if domain.strip()]
    if not trusted:
        return results

    filtered: list[dict] = []
    for item in results:
        raw_url = str(item.get("url", "")).strip()
        if not raw_url:
            continue
        host = (urlparse(raw_url).hostname or "").lower()
        if any(host == domain or host.endswith(f".{domain}") for domain in trusted):
            filtered.append(item)

    return filtered

logger = logging.getLogger(__name__)

CACHE_DIR: Final[Path] = Path(__file__).resolve().parents[2] / ".web_cache"
MAX_RESULTS_HARD_LIMIT: Final[int] = 10


def _acquire_lock(lock_file: Path, timeout_sec: float = 2.0, stale_after_sec: float = 60.0) -> int | None:
    """Пытается атомарно захватить lock-файл (межпроцессный).

    Возвращает fd на lock-файл (его нужно закрыть), либо None если захватить не удалось.
    """
    deadline = time.monotonic() + max(0.0, timeout_sec)
    while True:
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"pid={os.getpid()} ts={datetime.now().isoformat()}\n".encode("utf-8"))
            except Exception:
                pass
            return fd
        except FileExistsError:
            try:
                stat = lock_file.stat()
                if time.time() - stat.st_mtime > stale_after_sec:
                    lock_file.unlink(missing_ok=True)
                    continue
            except Exception:
                pass

            if time.monotonic() >= deadline:
                return None
            time.sleep(0.01 + random.random() * 0.03)
        except Exception:
            return None


def _release_lock(lock_file: Path, fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except Exception:
        pass
    try:
        lock_file.unlink(missing_ok=True)
    except Exception:
        pass


def _coerce_web_payload(payload: dict[str, Any], provider: str = "") -> dict[str, Any]:
    """Приводит web-payload к единой runtime-схеме независимо от источника/версии кэша."""
    normalized_provider = str(payload.get("provider") or provider or "unknown")
    results = payload.get("results")
    normalized_results = results if isinstance(results, list) else []
    error = str(payload.get("error") or "")
    note = str(payload.get("note") or ("" if normalized_results else "Ничего не найдено во внешнем поиске."))
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        meta = {"tool": "web", "provider": normalized_provider}
    status = str(payload.get("status") or ("failed" if error else "ok"))

    return build_tool_payload(
        status=status,
        query=str(payload.get("query") or ""),
        results=[item for item in normalized_results if isinstance(item, dict)],
        note=note,
        meta=meta,
        provider=normalized_provider,
        error=error,
    )


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
        return _coerce_web_payload(result) if isinstance(result, dict) else None
    except Exception:
        logger.exception("Не удалось загрузить кэш WEB-поиска из %s", cache_file)
        return None


def _save_to_cache(key: str, result: dict[str, Any]) -> None:
    """Сохраняет результат поиска в файловый кэш."""
    if not settings.web_cache_enabled:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{key}.json"
    lock_file = CACHE_DIR / f"{key}.lock"
    payload = {
        "cached_at": datetime.now().isoformat(),
        "result": result,
    }

    try:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Не удалось сериализовать WEB-кэш для %s", cache_file)
        return

    lock_fd: int | None = None
    tmp_path: str | None = None
    try:
        lock_fd = _acquire_lock(lock_file)
        if lock_fd is None:
            return

        fd, tmp_path = tempfile.mkstemp(prefix=f"{key}.", suffix=".tmp", dir=str(CACHE_DIR))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as tmp:
                tmp.write(serialized)
                tmp.flush()
                os.fsync(tmp.fileno())
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            raise

        os.replace(tmp_path, cache_file)
    except Exception:
        logger.exception("Не удалось сохранить кэш WEB-поиска в %s", cache_file)
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
        _release_lock(lock_file, lock_fd)


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

    return build_tool_payload(
        query=query,
        results=normalized_items,
        note="" if normalized_items else "Ничего не найдено во внешнем поиске.",
        meta={"tool": "web", "provider": provider},
        provider=provider,
        error="",
    )


def _error_object(query: str, provider: str, message: str) -> dict[str, Any]:
    """Формирует JSON-ответ с ошибкой в едином формате."""
    payload = error_payload(
        query=query,
        note="Внешний поиск временно недоступен.",
        error=message,
        meta={"tool": "web", "provider": provider},
        provider=provider,
    )
    payload.update(
        {
            "provider": provider,
            "meta": {"tool": "web", "provider": provider},
        }
    )
    return _coerce_web_payload(payload, provider=provider)


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
        return _coerce_web_payload(cached_result)

    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key:
        result = _tavily_search(query=query, max_results=normalized_max_results, api_key=tavily_key)
    else:
        result = _duckduckgo_search(query=query, max_results=normalized_max_results)

    try:
        if isinstance(result, dict) and result.get("results"):
            _save_to_cache(cache_key, result)
    except Exception:
        logger.exception("Не удалось разобрать ответ web_search для кэширования")
    
    if isinstance(result, dict) and "results" in result:
        result["results"] = filter_by_trusted_domains(result["results"])
    
    return _coerce_web_payload(result)
