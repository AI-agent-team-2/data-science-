from __future__ import annotations

import logging

from langchain_core.tools import tool

from app.config import settings
from app.observability import sanitize_text
from app.rag.retriever import ChromaRetriever
from app.tools.response_utils import empty_results_payload, to_json

logger = logging.getLogger(__name__)

retriever = ChromaRetriever()


@tool
def rag_search(query: str) -> str:
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
        return to_json(empty_results_payload(query=query, note="RAG-поиск отключен в настройках."))

    try:
        results = retriever.search(query=query)
    except Exception as exc:
        logger.exception("Ошибка RAG-поиска для запроса: %s", query)
        logger.debug("Детали ошибки rag_search: %s", sanitize_text(str(exc)))
        return to_json(empty_results_payload(query=query, note="Внутренняя ошибка RAG-поиска."))

    return to_json(
        {
            "query": query,
            "count": len(results),
            "results": results,
            "note": "" if results else "Ничего не найдено во внутренней базе знаний.",
        }
    )
