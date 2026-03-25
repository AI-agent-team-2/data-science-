from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from app.config import settings
from app.observability import (
    capture_error,
    create_span,
    end_observation,
    get_observability_parent,
    get_observability_trace,
    sanitize_text,
)
from app.rag.retriever import ChromaRetriever

logger = logging.getLogger(__name__)

retriever = ChromaRetriever()


def _to_json(payload: dict[str, Any]) -> str:
    """Сериализует payload в JSON единого формата."""
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _score_summary(items: list[dict[str, Any]]) -> dict[str, float]:
    """Возвращает агрегированные метрики релевантности для списка результатов."""
    scores = [float(item.get("score", 0.0) or 0.0) for item in items if isinstance(item, dict)]
    if not scores:
        return {"avg_score": 0.0, "max_score": 0.0}
    avg_score = sum(scores) / len(scores)
    return {"avg_score": round(avg_score, 4), "max_score": round(max(scores), 4)}


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
    span = create_span(
        parent=get_observability_parent() or get_observability_trace(),
        name="rag_search_exec",
        input_payload={"query": sanitize_text(query), "top_k": settings.top_k},
    )
    if not settings.enable_rag:
        payload = {
            "query": query,
            "count": 0,
            "results": [],
            "note": "RAG-поиск отключен в настройках.",
        }
        end_observation(span, output_payload={"retrieved_count": 0, "status": "disabled"})
        return _to_json(payload)

    try:
        results = retriever.search(query=query)
        summary = _score_summary(results)
        end_observation(
            span,
            output_payload={
                "retrieved_count": len(results),
                "top_k": settings.top_k,
                **summary,
            },
        )
    except Exception as exc:
        logger.exception("Ошибка RAG-поиска для запроса: %s", query)
        capture_error(span, exc, metadata={"tool": "rag_search", "query": sanitize_text(query)})
        end_observation(span, output_payload={"retrieved_count": 0, "status": "error"})
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
