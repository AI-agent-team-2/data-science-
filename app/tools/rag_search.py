from __future__ import annotations

import logging

from langchain_core.tools import tool

from app.config import settings
from app.observability import sanitize_text
from app.rag.retriever import ChromaRetriever
from app.tools.response_utils import build_tool_payload, empty_results_payload

logger = logging.getLogger(__name__)

_retriever: ChromaRetriever | None = None


def _get_retriever() -> ChromaRetriever:
    """Ленивая инициализация retriever для устойчивого импорта модуля."""
    global _retriever
    if _retriever is None:
        _retriever = ChromaRetriever()
    return _retriever


@tool
def rag_search(query: str) -> dict[str, object]:
    """
    Выполняет поиск по внутренней базе знаний.

    Parameters
    ----------
    query : str
        Текст запроса пользователя.

    Returns
    -------
    str
        JSON-ответ с найденными фрагментами.
    """
    if not settings.enable_rag:
        return empty_results_payload(query=query, note="RAG-поиск отключен в настройках.")

    try:
        retriever = _get_retriever()
        results = retriever.search(query=query)
    except Exception as exc:
        logger.exception("Ошибка RAG-поиска для запроса: %s", query)
        logger.debug("Детали ошибки rag_search: %s", sanitize_text(str(exc)))
        return empty_results_payload(query=query, note="Внутренняя ошибка RAG-поиска.")

    return build_tool_payload(
        query=query,
        results=results,
        note="" if results else "Ничего не найдено во внутренней базе знаний.",
        meta={"tool": "rag"},
    )
