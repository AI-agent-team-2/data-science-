"""
Веб-поиск для RAG/бота: DuckDuckGo и опционально Tavily.

Использование:
    from web_search import search, web_search_tool

    # Простая функция (строка результатов)
    text = search("редуктор давления ONDO")

    # Инструмент для LangChain-агента
    tools = [web_search_tool]  # может быть None, если библиотека не установлена
"""
import os
from typing import Optional

# DuckDuckGo (без API-ключа)
_duckduckgo_available = False
web_search_tool = None

try:
    from langchain_community.tools import DuckDuckGoSearchResults

    _duckduckgo_available = True
    web_search_tool = DuckDuckGoSearchResults(
        num_results=5,
        output_format="string",
        name="web_search",
        description=(
            "Поиск актуальной информации в интернете через DuckDuckGo. "
            "Используй для актуальных новостей, фреймворков 2025–2026 гг., "
            "последних исследований и событий, которых нет в базе знаний."
        ),
    )
except ImportError:
    pass

# Tavily (нужен TAVILY_API_KEY в .env)
_tavily_available = False
try:
    from langchain_community.tools.tavily_search import TavilySearchResults

    if os.getenv("TAVILY_API_KEY"):
        _tavily_available = True
        tavily_search_tool = TavilySearchResults(
            max_results=5,
            name="tavily_search",
            description="Поиск в интернете (Tavily). Для актуальной информации.",
        )
    else:
        tavily_search_tool = None
except ImportError:
    tavily_search_tool = None


def search(
    query: str,
    num_results: int = 5,
    backend: Optional[str] = None,
) -> str:
    """
    Выполняет веб-поиск и возвращает результаты одной строкой.

    Args:
        query: Поисковый запрос.
        num_results: Сколько результатов вернуть (для DuckDuckGo).
        backend: "duckduckgo" | "tavily" | None (авто: сначала Tavily, иначе DuckDuckGo).

    Returns:
        Текст с заголовками и сниппетами найденных страниц или сообщение об ошибке.
    """
    query = (query or "").strip()
    if not query:
        return "Запрос пустой."

    use_tavily = (
        (backend == "tavily" or (backend is None and _tavily_available))
        and tavily_search_tool is not None
    )

    if use_tavily:
        try:
            return tavily_search_tool.invoke({"query": query})
        except Exception as e:
            return f"Ошибка Tavily: {e}. Попробуйте backend='duckduckgo'."

    if not _duckduckgo_available or web_search_tool is None:
        return (
            "Веб-поиск недоступен. Установите: pip install duckduckgo-search. "
            "Либо задайте TAVILY_API_KEY для Tavily."
        )

    try:
        return web_search_tool.invoke({"query": query})
    except Exception as e:
        return f"Ошибка поиска: {e}"


def is_available() -> bool:
    """Есть ли хотя бы один рабочий бэкенд (DuckDuckGo или Tavily)."""
    return _duckduckgo_available or _tavily_available


if __name__ == "__main__":
    # Тест: python web_search.py   или   python web_search.py "ваш запрос"
    import sys
    if not is_available():
        print("❌ Веб-поиск недоступен. Установите: pip install duckduckgo-search")
        print("   (или задайте TAVILY_API_KEY в .env для Tavily)")
        sys.exit(1)
    query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else "python langchain 2025"
    print("Запрос:", repr(query))
    print("-" * 50)
    print(search(query, num_results=3))
