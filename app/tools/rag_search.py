from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from app.config import settings
from app.rag.retriever import ChromaRetriever

logger = logging.getLogger(__name__)

retriever = ChromaRetriever()


def _to_json(payload: dict[str, Any]) -> str:
    """Сериализует payload в JSON единого формата."""
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool
def rag_search(query: str) -> str:
    """Выполняет поиск по внутренней базе знаний и возвращает JSON-ответ."""
    if not settings.enable_rag:
        return _to_json(
            {
                "query": query,
                "count": 0,
                "results": [],
                "note": "RAG-поиск отключен в настройках.",
            }
        )

    try:
        results = retriever.search(query=query)
    except Exception:
        logger.exception("Ошибка RAG-поиска для запроса: %s", query)
        return _to_json(
            {
                "query": query,
                "count": 0,
                "results": [],
                "note": "Внутренняя ошибка RAG-поиска.",
            }
        )

    return _to_json(
        {
            "query": query,
            "count": len(results),
            "results": results,
            "note": "" if results else "Ничего не найдено во внутренней базе знаний.",
        }
    )
