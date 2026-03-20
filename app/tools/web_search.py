from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict

from langchain_core.tools import tool

# Кэш для результатов поиска
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".web_cache")
CACHE_TTL = timedelta(hours=24)


def _get_cache_key(query: str, max_results: int) -> str:
    """Создать ключ кэша"""
    key_str = f"{query}_{max_results}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _load_from_cache(key: str) -> Dict[str, Any] | None:
    """Загрузить из кэша"""
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cache_time = datetime.fromisoformat(data['cached_at'])
        if datetime.now() - cache_time > CACHE_TTL:
            os.remove(cache_file)
            return None
        
        return data['result']
    except Exception:
        return None


def _save_to_cache(key: str, result: Dict[str, Any]) -> None:
    """Сохранить в кэш"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    data = {
        'cached_at': datetime.now().isoformat(),
        'result': result
    }
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Поиск актуальной внешней информации в интернете.
    Используй только если вопрос требует внешних или изменяющихся данных.
    """
    max_results = int(max(1, min(max_results, 10)))
    
    # Проверяем кэш
    cache_key = _get_cache_key(query, max_results)
    cached = _load_from_cache(cache_key)
    if cached:
        return json.dumps(cached, ensure_ascii=False, indent=2)

    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    
    # Приоритет отдаем Tavily, если ключ задан в окружении.
    if tavily_key:
        result = _tavily_search(query=query, max_results=max_results, api_key=tavily_key)
    else:
        # Fallback-путь: публичный поиск через DuckDuckGo.
        result = _duckduckgo_search(query=query, max_results=max_results)
    
    # Сохраняем в кэш
    try:
        result_dict = json.loads(result) if isinstance(result, str) else result
        if result_dict.get('results'):
            _save_to_cache(cache_key, result_dict)
    except Exception:
        pass
    
    return result


def _normalize_results(query: str, items: list[dict[str, Any]], provider: str) -> str:
    # Приводим разные форматы провайдеров к единой схеме title/snippet/url.
    normalized: list[dict[str, str]] = []
    for it in items:
        title = str(it.get("title") or "").strip()
        snippet = str(it.get("snippet") or "").strip()
        url = str(it.get("url") or "").strip()
        # Пустые записи отбрасываем, чтобы не засорять ответ инструмента.
        if not (title or snippet or url):
            continue
        normalized.append({"title": title, "snippet": snippet, "url": url})
    return json.dumps(
        {
            "query": query,
            "provider": provider,
            "count": len(normalized),
            "results": normalized,
            "error": "",
        },
        ensure_ascii=False,
        indent=2,
    )


def _error_object(query: str, provider: str, message: str) -> str:
    # Возвращаем JSON-объект единого формата, даже при ошибке.
    return json.dumps(
        {
            "query": query,
            "provider": provider,
            "count": 0,
            "results": [],
            "error": message,
        },
        ensure_ascii=False,
        indent=2,
    )


def _duckduckgo_search(query: str, max_results: int) -> str:
    try:
        try:
            # Актуальное имя пакета.
            from ddgs import DDGS  # type: ignore
        except Exception:
            # Обратная совместимость для старого названия библиотеки.
            from duckduckgo_search import DDGS  # type: ignore
    except Exception as e:
        # Возвращаем ошибку в формате массива, без смены типа ответа инструмента.
        return _error_object(
            query=query,
            provider="duckduckgo",
            message=f"DuckDuckGo backend недоступен. Установите зависимость ddgs. Details: {e}",
        )

    items: list[dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            # Забираем короткую текстовую выдачу и нормализуем ключи в единый формат.
            for r in ddgs.text(
                query,
                max_results=max_results,
                region="ru-ru",
                safesearch="moderate",
            ):
                items.append(
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("body", "") or r.get("snippet", ""),
                        "url": r.get("href", "") or r.get("url", ""),
                    }
                )
    except Exception as e:
        # Ошибку транспорта/API также возвращаем в едином формате.
        return _error_object(
            query=query,
            provider="duckduckgo",
            message=f"DuckDuckGo search failed. Details: {e}",
        )

    return _normalize_results(query=query, items=items, provider="duckduckgo")


def _tavily_search(query: str, max_results: int, api_key: str) -> str:
    try:
        from tavily import TavilyClient  # type: ignore
    except Exception as e:
        # Подсказываем, какую зависимость установить, сохраняя единый формат ответа.
        return _error_object(
            query=query,
            provider="tavily",
            message=f"Tavily backend недоступен. Установите зависимость tavily-python. Details: {e}",
        )

    try:
        client = TavilyClient(api_key=api_key)
        resp: dict[str, Any] = client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )
        results = resp.get("results") or []
        # Приводим поля Tavily к той же схеме, что и у DuckDuckGo.
        items = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("content", "") or r.get("snippet", ""),
                "url": r.get("url", ""),
            }
            for r in results
        ]
    except Exception as e:
        # Возвращаем структуру ошибки в едином формате.
        return _error_object(
            query=query,
            provider="tavily",
            message=f"Tavily search failed. Details: {e}",
        )

    return _normalize_results(query=query, items=items, provider="tavily")
