from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Поиск актуальной внешней информации в интернете.
    Используй только если вопрос требует внешних или изменяющихся данных.
    """
    # Реализация:
    # - Если задан TAVILY_API_KEY, используем Tavily (стабильный API).
    # - Иначе используем DuckDuckGo через duckduckgo_search (без токена).
    max_results = int(max(1, min(max_results, 10)))

    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key:
        return _tavily_search(query=query, max_results=max_results, api_key=tavily_key)
    return _duckduckgo_search(query=query, max_results=max_results)


def _normalize_results(items: List[Dict[str, Any]]) -> str:
    normalized: List[Dict[str, str]] = []
    for it in items:
        title = str(it.get("title") or "").strip()
        snippet = str(it.get("snippet") or "").strip()
        url = str(it.get("url") or "").strip()
        if not (title or snippet or url):
            continue
        normalized.append({"title": title, "snippet": snippet, "url": url})
    return json.dumps(normalized, ensure_ascii=False, indent=2)


def _duckduckgo_search(query: str, max_results: int) -> str:
    try:
        try:
            # New package name (preferred).
            from ddgs import DDGS  # type: ignore
        except Exception:
            # Backward compatibility.
            from duckduckgo_search import DDGS  # type: ignore
    except Exception as e:
        return json.dumps(
            {
                "error": "DuckDuckGo search backend is not installed.",
                "hint": "Install dependency: pip install ddgs",
                "details": str(e),
            },
            ensure_ascii=False,
            indent=2,
        )

    items: List[Dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
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
        return json.dumps(
            {"error": "DuckDuckGo search failed.", "details": str(e)},
            ensure_ascii=False,
            indent=2,
        )

    return _normalize_results(items)


def _tavily_search(query: str, max_results: int, api_key: str) -> str:
    try:
        from tavily import TavilyClient  # type: ignore
    except Exception as e:
        return json.dumps(
            {
                "error": "Tavily backend is not installed.",
                "hint": "Install dependency: pip install tavily-python",
                "details": str(e),
            },
            ensure_ascii=False,
            indent=2,
        )

    try:
        client = TavilyClient(api_key=api_key)
        resp: Dict[str, Any] = client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )
        results = resp.get("results") or []
        items = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("content", "") or r.get("snippet", ""),
                "url": r.get("url", ""),
            }
            for r in results
        ]
    except Exception as e:
        return json.dumps(
            {"error": "Tavily search failed.", "details": str(e)},
            ensure_ascii=False,
            indent=2,
        )

    return _normalize_results(items)
