from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, TypedDict

from langchain_core.tools import tool

from app.observability import (
    capture_error,
    create_span,
    end_observation,
    sanitize_text,
)

logger = logging.getLogger(__name__)

SKU_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b[A-Z0-9][A-Z0-9\-_]{4,}\b")
WORD_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-zA-Zа-яА-Я0-9]+")
NON_ALNUM_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^A-Z0-9]+")

MAX_SEARCH_DOC_LENGTH: Final[int] = 3500
MAX_SKU_IN_RESULT: Final[int] = 20
MIN_RESULT_LIMIT: Final[int] = 1
MAX_RESULT_LIMIT: Final[int] = 20

EXACT_SKU_BASE_SCORE: Final[float] = 100.0
EXACT_SKU_MATCH_BONUS: Final[float] = 25.0
TOKEN_OVERLAP_SCORE: Final[float] = 5.0
TOKEN_COVERAGE_BONUS: Final[float] = 20.0
SUBSTRING_MATCH_BONUS: Final[float] = 15.0
SKU_FIRST_MATCH_SCORE: Final[float] = 100.0
SKU_FIRST_TOKEN_BONUS: Final[float] = 3.0


class SearchResult(TypedDict):
    """Структура одного результата поиска в JSON-ответе."""

    name: str
    brand: str
    category: str
    sku_list: list[str]
    source: str
    score: float


@dataclass(frozen=True)
class CatalogItem:
    """Элемент локального каталога с предрассчитанными полями для ранжирования."""

    name: str
    brand: str
    category: str
    sku_list: list[str]
    source: str
    searchable: str
    tokens: set[str]


def _normalize(text: str) -> str:
    """Приводит строку к нижнему регистру и нормализует пробелы."""
    return " ".join(text.lower().split())


def _tokenize(text: str) -> set[str]:
    """Разбивает текст на набор токенов для простого лексического поиска."""
    return {token.lower() for token in WORD_PATTERN.findall(text)}


def _extract_field(text: str, field_name: str) -> str:
    """Извлекает значение поля вида `FIELD_NAME: value` из текстового документа."""
    pattern = re.compile(rf"^\s*{field_name}\s*:\s*(.+?)\s*$", re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _extract_skus(text: str) -> list[str]:
    """
    Извлекает SKU из текста.

    Возвращает только нормализованные латинские коды с цифрами,
    чтобы уменьшить шум от технических обозначений.
    """
    normalized_skus: set[str] = set()
    for raw_token in (match.group(0) for match in SKU_PATTERN.finditer(text.upper())):
        canonical_token = _canonical_sku(raw_token)
        if canonical_token and any(char.isdigit() for char in canonical_token):
            normalized_skus.add(canonical_token)
    return sorted(normalized_skus)


def _canonical_sku(value: str) -> str:
    """Нормализует SKU: убирает разделители (`-`, `_`, пробелы и т.д.)."""
    return NON_ALNUM_PATTERN.sub("", value.upper())


def _build_title(doc_name: str, product: str, product_type: str, document: str) -> str:
    """Выбирает наиболее информативный заголовок для карточки товара."""
    for candidate in (product, product_type, document):
        if candidate:
            return candidate
    return doc_name


@lru_cache(maxsize=1)
def _load_catalog() -> list[CatalogItem]:
    """Загружает локальный каталог из `data/knowledge_base` и кеширует результат."""
    root = Path(__file__).resolve().parents[2]
    kb_root = root / "data" / "knowledge_base"
    items: list[CatalogItem] = []
    if not kb_root.exists():
        logger.warning("Каталог базы знаний не найден: %s", kb_root)
        return items

    source_paths = sorted(kb_root.glob("*.txt"))

    for path in source_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            logger.exception("Не удалось прочитать файл каталога: %s", path)
            continue

        brand = _extract_field(text, "BRAND")
        product = _extract_field(text, "PRODUCT")
        category = _extract_field(text, "CATEGORY")
        product_type = _extract_field(text, "PRODUCT TYPE")
        document = _extract_field(text, "DOCUMENT")
        title = _build_title(path.stem, product, product_type, document)
        skus = _extract_skus(text)

        searchable = _normalize(
            " ".join(
                [title, brand, category, product, product_type, document, " ".join(skus), text[:MAX_SEARCH_DOC_LENGTH]]
            )
        )
        item = CatalogItem(
            name=title,
            brand=brand,
            category=category or product_type or product,
            sku_list=skus,
            source=str(path.relative_to(root)).replace("\\", "/"),
            searchable=searchable,
            tokens=_tokenize(searchable),
        )
        items.append(item)

    logger.info("Каталог загружен: %d позиций", len(items))
    return items


def _score_item(
    query: str,
    query_tokens: set[str],
    query_skus: set[str],
    item: CatalogItem,
) -> float:
    """Считает итоговый score документа для текстового ранжирования."""
    score = 0.0

    item_skus = {_canonical_sku(sku) for sku in item.sku_list}
    matched_skus = sorted(query_skus.intersection(item_skus))
    if matched_skus:
        score += EXACT_SKU_BASE_SCORE + EXACT_SKU_MATCH_BONUS * len(matched_skus)

    overlap = len(query_tokens.intersection(item.tokens))
    if overlap:
        score += overlap * TOKEN_OVERLAP_SCORE
        score += (overlap / max(1, len(query_tokens))) * TOKEN_COVERAGE_BONUS

    if query in item.searchable:
        score += SUBSTRING_MATCH_BONUS

    return score


def _clamp_limit(limit: int) -> int:
    """Ограничивает размер выдачи допустимым диапазоном."""
    return max(MIN_RESULT_LIMIT, min(limit, MAX_RESULT_LIMIT))


def _serialize_item(item: CatalogItem, score: float) -> SearchResult:
    """Преобразует элемент каталога в формат JSON-ответа."""
    return {
        "name": item.name,
        "brand": item.brand,
        "category": item.category,
        "sku_list": item.sku_list[:MAX_SKU_IN_RESULT],
        "source": item.source,
        "score": round(score, 2),
    }


def _to_json(payload: dict[str, Any]) -> str:
    """Сериализует ответ в JSON с единым форматом."""
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_empty_response(query: str, note: str) -> str:
    """Формирует типовой пустой ответ для ошибок валидации/данных."""
    return _to_json({"query": query, "count": 0, "results": [], "note": note})


def _rank_sku_matches(
    catalog: list[CatalogItem],
    query_tokens: set[str],
    query_skus: set[str],
) -> list[tuple[float, CatalogItem]]:
    """Ранжирует элементы только по SKU-совпадениям (режим `sku_first`)."""
    ranked: list[tuple[float, CatalogItem]] = []
    for item in catalog:
        item_skus = {_canonical_sku(sku) for sku in item.sku_list}
        matched_count = len(query_skus.intersection(item_skus))
        if matched_count == 0:
            continue

        token_overlap = len(query_tokens.intersection(item.tokens))
        score = matched_count * SKU_FIRST_MATCH_SCORE + token_overlap * SKU_FIRST_TOKEN_BONUS
        ranked.append((score, item))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return ranked


def _rank_text_matches(
    catalog: list[CatalogItem], query: str, query_tokens: set[str], query_skus: set[str]
) -> list[tuple[float, CatalogItem]]:
    """Ранжирует элементы по текстовой релевантности."""
    ranked: list[tuple[float, CatalogItem]] = []
    for item in catalog:
        score = _score_item(query, query_tokens, query_skus, item)
        if score > 0:
            ranked.append((score, item))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return ranked


@tool
def product_lookup(query: str, limit: int = 5) -> str:
    """
    Ищет товары по локальному каталогу.

    Parameters
    ----------
    query : str
        Пользовательский запрос.
    limit : int, default=5
        Максимальное число результатов.

    Returns
    -------
    str
        JSON-ответ с результатами поиска.
    """
    normalized_query = _normalize(query)
    span = create_span(
        parent=None,
        name="product_lookup_exec",
        input_payload={"query": sanitize_text(normalized_query), "limit": limit},
    )
    if not normalized_query:
        end_observation(span, output_payload={"result_count": 0, "status": "empty_query"})
        return _build_empty_response(normalized_query, "Пустой поисковый запрос.")

    try:
        catalog = _load_catalog()
        if not catalog:
            end_observation(span, output_payload={"result_count": 0, "status": "catalog_missing"})
            return _build_empty_response(
                normalized_query,
                "Каталог не найден. Проверьте данные в data/knowledge_base.",
            )

        query_tokens = _tokenize(normalized_query)
        query_skus = {_canonical_sku(sku) for sku in SKU_PATTERN.findall(normalized_query.upper())}
        top_n = _clamp_limit(limit)

        # Режим sku_first: при наличии артикула сначала возвращаем точные SKU-совпадения.
        if query_skus:
            sku_ranked = _rank_sku_matches(catalog, query_tokens, query_skus)
            if sku_ranked:
                top_sku_matches = sku_ranked[:top_n]
                results = [_serialize_item(item, score) for score, item in top_sku_matches]
                end_observation(
                    span,
                    output_payload={
                        "normalized_query": sanitize_text(normalized_query),
                        "detected_sku_count": len(query_skus),
                        "result_count": len(results),
                        "mode": "sku_first",
                    },
                )
                return _to_json(
                    {
                        "query": normalized_query,
                        "count": len(results),
                        "mode": "sku_first",
                        "results": results,
                    }
                )

        ranked = _rank_text_matches(catalog, normalized_query, query_tokens, query_skus)
        top_text_matches = ranked[:top_n]
        results = [_serialize_item(item, score) for score, item in top_text_matches]
        end_observation(
            span,
            output_payload={
                "normalized_query": sanitize_text(normalized_query),
                "detected_sku_count": len(query_skus),
                "result_count": len(results),
                "mode": "text_ranked",
            },
        )
        return _to_json(
            {
                "query": normalized_query,
                "count": len(results),
                "mode": "text_ranked",
                "results": results,
            }
        )
    except Exception as exc:
        logger.exception("Ошибка product_lookup для запроса: %s", normalized_query)
        capture_error(
            span,
            exc,
            metadata={"tool": "product_lookup", "query": sanitize_text(normalized_query)},
        )
        end_observation(span, output_payload={"result_count": 0, "status": "error"})
        return _build_empty_response(
            normalized_query,
            "Внутренняя ошибка поиска. Попробуйте повторить запрос позже.",
        )
