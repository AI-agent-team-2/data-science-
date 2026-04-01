from __future__ import annotations

import logging
from typing import Any, Final, TypedDict

from langchain_core.tools import tool

from app.observability import sanitize_text
from app.rag.retriever import ProductRetriever
from app.tools.response_utils import empty_results_payload

logger = logging.getLogger(__name__)

MIN_RESULT_LIMIT: Final[int] = 1
MAX_RESULT_LIMIT: Final[int] = 20
MAX_SKU_IN_RESULT: Final[int] = 20


class SearchResult(TypedDict):
    """Структура одной карточки товара в JSON-ответе инструмента."""

    name: str
    brand: str
    category: str
    sku_list: list[str]
    source: str
    score: float


_retriever: ProductRetriever | None = None


def _get_retriever() -> ProductRetriever:
    """Ленивая инициализация retriever для устойчивого импорта модуля."""
    global _retriever
    if _retriever is None:
        _retriever = ProductRetriever()
    return _retriever


def _normalize(text: str) -> str:
    """Приводит строку к нижнему регистру и нормализует пробелы."""
    return " ".join(str(text or "").lower().split())


def _clamp_limit(limit: int) -> int:
    """Ограничивает размер выдачи допустимым диапазоном."""
    return max(MIN_RESULT_LIMIT, min(int(limit), MAX_RESULT_LIMIT))


def _build_empty_response(query: str, note: str) -> dict[str, Any]:
    """Формирует типовой пустой ответ для ошибок/пустых результатов."""
    return empty_results_payload(query=query, note=note)


def _serialize_item(item: dict[str, Any]) -> SearchResult:
    """Преобразует внутренний результат retriever в единый формат ответа."""
    sku_list_raw = item.get("sku_list")
    if isinstance(sku_list_raw, list):
        sku_list = [str(value).strip() for value in sku_list_raw if str(value).strip()]
    else:
        sku_list = []

    return {
        "name": str(item.get("name", "")).strip(),
        "brand": str(item.get("brand", "")).strip(),
        "category": str(item.get("category", "")).strip(),
        "sku_list": sku_list[:MAX_SKU_IN_RESULT],
        "source": str(item.get("source", "")).strip(),
        "score": round(float(item.get("score", 0.0) or 0.0), 4),
    }


@tool
def product_lookup(query: str, limit: int = 5) -> dict[str, Any]:
    """
    Ищет карточки товаров в product-level коллекции Chroma.

    Сначала делает exact SKU-match, затем fallback semantic-поиск по `lookup_text`.
    """
    normalized_query = _normalize(query)
    if not normalized_query:
        return _build_empty_response(normalized_query, "Пустой поисковый запрос.")

    top_n = _clamp_limit(limit)
    mode = "semantic"
    retriever = _get_retriever()
    query_skus = retriever.extract_query_skus(normalized_query)

    try:
        if query_skus:
            exact_matches = retriever.find_exact_sku_matches(query=normalized_query, limit=top_n)
            if exact_matches:
                mode = "sku_first"
                results = [_serialize_item(item) for item in exact_matches[:top_n]]
                return {
                    "query": normalized_query,
                    "count": len(results),
                    "mode": mode,
                    "results": results,
                }

            # Для SKU-запросов не подставляем «похожие» товары, если exact не найден.
            return empty_results_payload(
                query=normalized_query,
                note="Точный артикул не найден в базе товаров.",
                mode="sku_not_found",
            )

        semantic_matches = retriever.semantic_search(query=normalized_query, limit=top_n)
        results = [_serialize_item(item) for item in semantic_matches[:top_n]]
        return {
            "query": normalized_query,
            "count": len(results),
            "mode": mode,
            "results": results,
        }
    except Exception as exc:
        logger.exception("Ошибка product_lookup для запроса: %s", normalized_query)
        logger.debug("Детали ошибки product_lookup: %s", sanitize_text(str(exc)))
        return _build_empty_response(
            normalized_query,
            "Внутренняя ошибка поиска. Попробуйте повторить запрос позже.",
        )
