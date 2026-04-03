from __future__ import annotations

from typing import Any

from app.config import settings


def extract_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Возвращает список объектов результата из payload."""
    items = payload.get("results")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def is_rag_useful(items: list[dict[str, Any]]) -> bool:
    """Проверяет, что RAG-результаты содержат полезный текст с приемлемым score."""
    for item in items:
        text = str(item.get("text", "")).strip()
        score = float(item.get("score", 0.0) or 0.0)
        if text and score >= settings.min_rag_score:
            return True
    return False


def format_rag_context(items: list[dict[str, Any]]) -> str:
    """Форматирует контекстный блок из результатов RAG."""
    lines: list[str] = []
    for index, item in enumerate(items[:settings.max_rag_context_items], start=1):
        metadata = item.get("metadata") or {}
        source = str(metadata.get("source", "unknown"))
        section = str(metadata.get("section", "")).strip()
        doc_id = str(metadata.get("doc_id", "")).strip()
        product = str(metadata.get("product", "")).strip()
        score = float(item.get("score", 0.0) or 0.0)
        text = str(item.get("text", "")).strip().replace("\n", " ")
        lines.append(
            f"[RAG {index}] source={source} section={section} doc_id={doc_id} "
            f"product={product} score={score:.3f} text={text[:800]}"
        )
    return "\n".join(lines).strip()


def format_lookup_context(payload: dict[str, Any], items: list[dict[str, Any]]) -> str:
    """Форматирует контекстный блок из результатов LOOKUP."""
    mode = str(payload.get("mode", "lookup"))
    lines = [f"[LOOKUP] mode={mode} count={len(items)}"]
    for index, item in enumerate(items[:settings.max_lookup_context_items], start=1):
        name = str(item.get("name", "")).strip()
        brand = str(item.get("brand", "")).strip()
        category = str(item.get("category", "")).strip()
        source = str(item.get("source", "")).strip()
        score = item.get("score", "")
        sku_list = item.get("sku_list") or []
        sku_preview = ", ".join(str(value) for value in sku_list[:5])
        lines.append(
            f"[LOOKUP {index}] {name} | brand={brand} | category={category} | "
            f"sku={sku_preview} | source={source} | score={score}"
        )
    return "\n".join(lines).strip()


def format_web_context(items: list[dict[str, Any]], clean_web_text_fn) -> str:
    """Форматирует контекстный блок из результатов WEB-поиска."""
    lines: list[str] = []
    for index, item in enumerate(items[:settings.max_web_context_items], start=1):
        title = clean_web_text_fn(str(item.get("title", "")).strip())
        snippet = clean_web_text_fn(str(item.get("snippet", "")).strip().replace("\n", " "))
        url = str(item.get("url", "")).strip()
        lines.append(f"[WEB {index}] {title} | {snippet} | {url}")
    return "\n".join(lines).strip()


def is_strict_sku_existence_query(query: str) -> bool:
    """Возвращает True для запросов, где нужен только факт наличия exact SKU."""
    lowered = str(query or "").lower()
    markers = (
        "что за товар",
        "что это за артикул",
        "найди товар",
        "найди артикул",
        "есть ли товар",
    )
    return any(marker in lowered for marker in markers)


def needs_technical_context(query: str) -> bool:
    """Определяет запросы, где после lookup полезно подтянуть RAG-факты."""
    lowered = str(query or "").lower()
    markers = (
        "характерист",
        "давление",
        "температур",
        "размер",
        "совместим",
        "срок службы",
        "для чего",
        "отлич",
        "пропуск",
        "монтаж",
        "подходит",
    )
    return any(marker in lowered for marker in markers)
