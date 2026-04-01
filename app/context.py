from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, TypeAlias

from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.routing import SANITARY_KEYWORDS, ToolName, should_use_web_source
from app.tools.product_lookup import product_lookup
from app.tools.rag_search import rag_search
from app.tools.web_search import web_search

logger = logging.getLogger(__name__)

YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
MIN_RAG_SCORE = 0.2
MAX_RAG_CONTEXT_ITEMS = 4
MAX_LOOKUP_CONTEXT_ITEMS = 5
MAX_WEB_CONTEXT_ITEMS = 5


@dataclass(frozen=True)
class ContextBuildResult:
    """Результат получения контекста из инструментов."""

    context_text: str
    web_urls: list[str]
    used_web: bool
    used_source: str


EMPTY_CONTEXT_RESULT = ContextBuildResult(context_text="", web_urls=[], used_web=False, used_source="none")

ToolCallable: TypeAlias = Callable[[dict[str, Any]], Any]
ToolPayload: TypeAlias = dict[str, Any]
InvokeToolFn: TypeAlias = Callable[[ToolCallable, ToolPayload, str, RunnableConfig | None], ToolPayload]


def build_context(
    query: str,
    source_order: list[ToolName],
    invoke_tool: InvokeToolFn,
    config: RunnableConfig | None = None,
) -> ContextBuildResult:
    """Подбирает контекст из доступных источников по приоритету."""
    for index, source in enumerate(source_order):
        web_mode = "primary" if index == 0 else "fallback"

        if source == "lookup" and not settings.enable_product_lookup:
            continue
        if source == "rag" and not settings.enable_rag:
            continue
        if source == "web" and not settings.enable_web_search:
            continue
        if source == "web" and not should_use_web_source(query=query, web_mode=web_mode):
            continue

        result = _context_from_source(
            source=source,
            query=query,
            web_mode=web_mode,
            invoke_tool=invoke_tool,
            config=config,
        )
        if result.context_text:
            return result

    return EMPTY_CONTEXT_RESULT


def _context_from_source(
    source: ToolName,
    query: str,
    web_mode: str,
    invoke_tool: InvokeToolFn,
    config: RunnableConfig | None = None,
) -> ContextBuildResult:
    """Строит контекст из указанного источника данных."""
    if source == "lookup":
        return _context_from_lookup(query, invoke_tool=invoke_tool, config=config)
    if source == "rag":
        return _context_from_rag(query, invoke_tool=invoke_tool, config=config)
    return _context_from_web(query, mode=web_mode, invoke_tool=invoke_tool, config=config)


def _context_from_lookup(
    query: str,
    invoke_tool: InvokeToolFn,
    config: RunnableConfig | None = None,
) -> ContextBuildResult:
    """Пытается получить контекст из базы товаров (LOOKUP)."""
    payload = invoke_tool(
        product_lookup.invoke,
        {"query": query, "limit": 5},
        "tool_lookup",
        config,
    )
    items = _extract_results(payload)
    if not items:
        return EMPTY_CONTEXT_RESULT

    return ContextBuildResult(
        context_text=_format_lookup_context(payload, items),
        web_urls=[],
        used_web=False,
        used_source="lookup",
    )


def _context_from_rag(
    query: str,
    invoke_tool: InvokeToolFn,
    config: RunnableConfig | None = None,
) -> ContextBuildResult:
    """Пытается получить контекст из RAG-базы знаний."""
    payload = invoke_tool(
        rag_search.invoke,
        {"query": query},
        "tool_rag",
        config,
    )
    items = _extract_results(payload)
    if not _is_rag_useful(items):
        return EMPTY_CONTEXT_RESULT

    return ContextBuildResult(
        context_text=_format_rag_context(items),
        web_urls=[],
        used_web=False,
        used_source="rag",
    )


def _context_from_web(
    query: str,
    mode: str,
    invoke_tool: InvokeToolFn,
    config: RunnableConfig | None = None,
) -> ContextBuildResult:
    """Пытается получить контекст из web-поиска."""
    enhanced_query = enhance_search_query(query, mode)
    payload = invoke_tool(
        web_search.invoke,
        {"query": enhanced_query, "max_results": settings.web_search_max_results},
        "tool_web",
        config,
    )
    items = _extract_results(payload)
    if not (_is_web_useful(items) and _is_sanitary_relevant(items)):
        return EMPTY_CONTEXT_RESULT

    return ContextBuildResult(
        context_text=_format_web_context(items),
        web_urls=_extract_web_urls(items),
        used_web=True,
        used_source="web",
    )


def enhance_search_query(original_query: str, search_type: str = "general") -> str:
    """Улучшает формулировку запроса для внешнего web-поиска."""
    lowered_query = original_query.lower()

    if any(marker in lowered_query for marker in ("сантехник", "унитаз", "смеситель")):
        return original_query

    if "новинк" in lowered_query or "новые" in lowered_query:
        year_match = YEAR_PATTERN.search(lowered_query)
        target_year = year_match.group(1) if year_match else str(datetime.now().year)
        return f"новинки сантехники {target_year} каталог"

    if any(marker in lowered_query for marker in ("бюджет", "цен", "стоит", "сколько")):
        return f"{original_query} сантехника"

    if "купить" in lowered_query or "где" in lowered_query:
        return f"{original_query} сантехника интернет магазин"

    if search_type == "fallback":
        return f"{original_query} сантехника"

    return f"{original_query} сантехника товары"


def extract_ai_text(message: Any) -> str:
    """Извлекает текст из ответа модели с учетом разных форматов content."""
    from langchain_core.messages import AIMessage

    if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
        return message.content

    if isinstance(message, AIMessage) and isinstance(message.content, list):
        parts = [str(part.get("text", "")) for part in message.content if isinstance(part, dict)]
        text = "\n".join(part for part in parts if part.strip()).strip()
        if text:
            return text

    return "Не удалось получить ответ."


def ensure_sources_block(answer: str, urls: list[str], max_urls: int = 5) -> str:
    """Добавляет блок `Источники`, если использовался WEB и блока еще нет."""
    if "Источники:" in answer:
        return answer

    lines = ["", "Источники:"]
    if urls:
        for url in urls[:max_urls]:
            lines.append(f"- {url}")
    else:
        lines.append("- внешние ссылки не найдены")

    return answer.rstrip() + "\n" + "\n".join(lines)


def build_final_prompt(user_text: str, context_block: str) -> str:
    """Собирает финальный prompt для LLM из вопроса и контекста."""
    return (
        f"Вопрос пользователя:\n{user_text}\n\n"
        f"Контекст для ответа:\n{context_block}\n\n"
        "Ответь строго по вопросу пользователя. "
        "Если вопрос только про сантехнику — отвечай только про сантехнику. "
        "Не добавляй информацию про ремонт, стройматериалы, мебель и другие темы, "
        "если пользователь о них не спрашивал. Будь краток и точен."
    )


def clarifying_question() -> str:
    """Возвращает вопрос-уточнение, если контекст не найден."""
    return (
        "Пока не нашел достаточно надежных данных по вашему запросу. "
        "Уточните, пожалуйста, бренд, артикул (если есть) или ключевой параметр "
        "(например, диаметр/тип подключения/назначение)."
    )


def domain_redirect_response() -> str:
    """Мягко возвращает диалог в домен бота, не уводя в посторонние темы."""
    return (
        "Я помогаю по сантехническим товарам и отоплению. "
        "Напишите, пожалуйста, что именно нужно: бренд, артикул или задачу "
        "(например, подобрать насос, коллектор, редуктор, трубу или сервопривод)."
    )


def assistant_scope_response() -> str:
    """Отвечает на вопросы о роли ассистента и возвращает в целевой домен."""
    return (
        "Я чат-бот по сантехническим товарам и отоплению. "
        "Помогаю подобрать оборудование, объяснить характеристики и совместимость, "
        "а также найти варианты по артикулу или задаче."
    )


def smalltalk_response() -> str:
    """Возвращает короткий ответ для бытовых реплик без запуска поиска."""
    return (
        "Привет! Я в порядке и готов помочь по сантехническим товарам. "
        "Напишите, пожалуйста, бренд, модель, артикул или технический вопрос."
    )


def _extract_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Возвращает список объектов результата из payload."""
    items = payload.get("results")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _is_rag_useful(items: list[dict[str, Any]]) -> bool:
    """Проверяет, что RAG-результаты содержат полезный текст с приемлемым score."""
    for item in items:
        text = str(item.get("text", "")).strip()
        score = float(item.get("score", 0.0) or 0.0)
        if text and score >= MIN_RAG_SCORE:
            return True
    return False


def _is_web_useful(items: list[dict[str, Any]]) -> bool:
    """Проверяет, что WEB-результаты содержат валидные ссылки."""
    for item in items:
        url = str(item.get("url", "")).strip().lower()
        if url.startswith("https://") or url.startswith("http://"):
            return True
    return False


def _is_sanitary_relevant(items: list[dict[str, Any]]) -> bool:
    """Проверяет, что WEB-результаты действительно относятся к сантехнике."""
    for item in items:
        title = str(item.get("title", "")).lower()
        snippet = str(item.get("snippet", "")).lower()
        combined = f"{title} {snippet}"
        if any(keyword in combined for keyword in SANITARY_KEYWORDS):
            return True
    return False


def _format_rag_context(items: list[dict[str, Any]]) -> str:
    """Форматирует контекстный блок из результатов RAG."""
    lines: list[str] = []
    for index, item in enumerate(items[:MAX_RAG_CONTEXT_ITEMS], start=1):
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


def _format_lookup_context(payload: dict[str, Any], items: list[dict[str, Any]]) -> str:
    """Форматирует контекстный блок из результатов LOOKUP."""
    mode = str(payload.get("mode", "lookup"))
    lines = [f"[LOOKUP] mode={mode} count={len(items)}"]
    for index, item in enumerate(items[:MAX_LOOKUP_CONTEXT_ITEMS], start=1):
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


def _format_web_context(items: list[dict[str, Any]]) -> str:
    """Форматирует контекстный блок из результатов WEB-поиска."""
    lines: list[str] = []
    for index, item in enumerate(items[:MAX_WEB_CONTEXT_ITEMS], start=1):
        title = str(item.get("title", "")).strip()
        snippet = str(item.get("snippet", "")).strip().replace("\n", " ")
        url = str(item.get("url", "")).strip()
        lines.append(f"[WEB {index}] {title} | {snippet} | {url}")
    return "\n".join(lines).strip()


def _extract_web_urls(items: list[dict[str, Any]]) -> list[str]:
    """Собирает уникальные URL из WEB-результатов."""
    urls: list[str] = []
    seen: set[str] = set()
    for item in items:
        url = str(item.get("url", "")).strip()
        if not (url.startswith("https://") or url.startswith("http://")):
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def parse_tool_payload(raw: Any) -> dict[str, Any]:
    """Парсит JSON-ответ инструмента в dict."""
    if not isinstance(raw, str):
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
