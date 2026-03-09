from __future__ import annotations

import json
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """
    Поиск актуальной внешней информации в интернете.
    Используй только если вопрос требует внешних или изменяющихся данных.
    """
    # TODO: подключить реальный поиск (SerpAPI, Tavily, SearxNG и т.п.) и нормальные источники.
    # Сейчас возвращаем заглушку, чтобы пайплайн tool-calling оставался рабочим.
    mock_results = [
        {
            "title": "Заглушка web_search",
            "snippet": f"Найдено по запросу: {query}",
            "url": "https://example.com",
            "source": "example",
        }
    ]
    # Возвращаем JSON-строку, которую LLM может прочитать и пересказать пользователю.
    return json.dumps(mock_results, ensure_ascii=False, indent=2)
