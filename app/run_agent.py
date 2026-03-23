from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import settings
from app.graph import create_chat_model
from app.history_store import load_messages, save_turn
from app.prompts import SYSTEM_PROMPT
from app.tools.product_lookup import product_lookup
from app.tools.rag_search import rag_search
from app.tools.web_search import web_search

logger = logging.getLogger(__name__)

ToolName = Literal["lookup", "rag", "web"]

SKU_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{4,}\b")
TOOL_TIMEOUT_SEC = 20
MODEL_TIMEOUT_SEC = 45
MIN_RAG_SCORE = 0.2
MAX_RAG_CONTEXT_ITEMS = 4
MAX_LOOKUP_CONTEXT_ITEMS = 5
MAX_WEB_CONTEXT_ITEMS = 5
MAX_SOURCE_URLS = 5
QUOTA_ERROR_MARKERS: tuple[str, ...] = (
    "insufficient",
    "quota",
    "rate limit",
    "credit",
    "payment",
    "402",
    "free",
)

WEB_PRIORITY_MARKERS: tuple[str, ...] = (
    "сейчас",
    "сегодня",
    "в 2026",
    "в 2025",
    "в 2024",
    "новые требования",
    "что изменилось",
    "по отзывам",
    "отзывы",
    "где купить",
    "в москве",
    "в россии",
    "средняя цена",
    "цена",
    "аналоги",
    "сравни",
    "новости",
    "что такое",
    "как работает",
    "для чего",
    "зачем",
    "лучший",
    "рейтинг",
    "топ",
    "популярный",
    "актуальный",
    "последний",
    "новинки",
    "тренды",
)

LOOKUP_PRIORITY_MARKERS: tuple[str, ...] = (
    "артикул",
    "sku",
    "модель",
    "код",
    "товар",
    "бренд",
    "серия",
    "позици",
)

SANITARY_KEYWORDS: tuple[str, ...] = (
    "унитаз",
    "ванна",
    "смеситель",
    "душ",
    "сантехник",
    "раковина",
    "труба",
    "фитинг",
    "кран",
    "бойлер",
    "сантехника",
    "санфаянс",
    "кранбукс",
    "картридж",
    "гидробокс",
    "инсталляция",
    "поддон",
    "лейка",
)


@dataclass(frozen=True)
class ContextBuildResult:
    """Результат получения контекста из инструментов."""

    context_text: str
    web_urls: list[str]
    used_web: bool


EMPTY_CONTEXT_RESULT = ContextBuildResult(context_text="", web_urls=[], used_web=False)
PRIMARY_CHAT_MODEL = create_chat_model(settings.resolved_model_name)
FALLBACK_MODEL_NAME = settings.resolved_fallback_model_name
FALLBACK_CHAT_MODEL = create_chat_model(FALLBACK_MODEL_NAME) if FALLBACK_MODEL_NAME else None


def run_agent(user_text: str, user_id: str = "unknown") -> str:
    """Запускает full-pipeline ответа: tool routing -> LLM -> сохранение в историю."""
    session_id = user_id or "unknown"
    query = user_text.strip()

    history_messages = _to_langchain_messages(load_messages(session_id=session_id))
    context = _build_context(query)

    if not context.context_text:
        assistant_text = _clarifying_question()
        save_turn(session_id=session_id, user_text=user_text, assistant_text=assistant_text)
        return assistant_text

    final_prompt = _build_final_prompt(user_text=user_text, context_block=context.context_text)
    model_input = [SystemMessage(content=SYSTEM_PROMPT), *history_messages, HumanMessage(content=final_prompt)]
    response = _invoke_model_with_fallback(model_input)

    assistant_text = _extract_ai_text(response)
    if context.used_web:
        assistant_text = _ensure_sources_block(assistant_text, context.web_urls)

    save_turn(session_id=session_id, user_text=user_text, assistant_text=assistant_text)
    return assistant_text


def _to_langchain_messages(history: list[tuple[str, str]]) -> list[BaseMessage]:
    """Преобразует сохраненную историю в объекты сообщений LangChain."""
    messages: list[BaseMessage] = []

    for role, content in history:
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))

    return messages


def _build_context(query: str) -> ContextBuildResult:
    """Подбирает лучший доступный контекст согласно приоритету источников."""
    source_order = _resolve_source_order(query)

    for index, source in enumerate(source_order):
        if source == "lookup" and not settings.enable_product_lookup:
            continue
        if source == "rag" and not settings.enable_rag:
            continue
        if source == "web" and not settings.enable_web_search:
            continue

        web_mode = "primary" if index == 0 else "fallback"
        result = _context_from_source(source=source, query=query, web_mode=web_mode)
        if result.context_text:
            return result

    return EMPTY_CONTEXT_RESULT


def _resolve_source_order(query: str) -> list[ToolName]:
    """Возвращает порядок источников в зависимости от типа пользовательского запроса."""
    if _should_prefer_web(query):
        return ["web", "rag", "lookup"]
    if _should_prefer_lookup(query):
        return ["lookup", "rag", "web"]
    return ["rag", "lookup", "web"]


def _context_from_source(source: ToolName, query: str, web_mode: str) -> ContextBuildResult:
    """Строит контекст из указанного источника данных."""
    if source == "lookup":
        return _context_from_lookup(query)
    if source == "rag":
        return _context_from_rag(query)
    return _context_from_web(query, mode=web_mode)


def _context_from_lookup(query: str) -> ContextBuildResult:
    """Пытается получить контекст из базы товаров (LOOKUP)."""
    payload = _invoke_tool(product_lookup.invoke, {"query": query, "limit": 5}, "product_lookup")
    items = _extract_results(payload)

    if not items:
        return EMPTY_CONTEXT_RESULT

    return ContextBuildResult(
        context_text=_format_lookup_context(payload, items),
        web_urls=[],
        used_web=False,
    )


def _context_from_rag(query: str) -> ContextBuildResult:
    """Пытается получить контекст из RAG-базы знаний."""
    payload = _invoke_tool(rag_search.invoke, {"query": query}, "rag_search")
    items = _extract_results(payload)

    if not _is_rag_useful(items):
        return EMPTY_CONTEXT_RESULT

    return ContextBuildResult(
        context_text=_format_rag_context(items),
        web_urls=[],
        used_web=False,
    )


def _context_from_web(query: str, mode: str) -> ContextBuildResult:
    """Пытается получить контекст из web-поиска."""
    enhanced_query = enhance_search_query(query, mode)
    payload = _invoke_tool(
        web_search.invoke,
        {"query": enhanced_query, "max_results": settings.web_search_max_results},
        "web_search",
    )
    items = _extract_results(payload)

    if not (_is_web_useful(items) and _is_sanitary_relevant(items)):
        return EMPTY_CONTEXT_RESULT

    return ContextBuildResult(
        context_text=_format_web_context(items),
        web_urls=_extract_web_urls(items),
        used_web=True,
    )


def _invoke_tool(func: Callable[[dict[str, Any]], Any], payload: dict[str, Any], op_name: str) -> dict[str, Any]:
    """Вызывает tool с таймаутом и возвращает JSON-object payload."""
    raw = _invoke_with_timeout(func, payload, timeout_sec=TOOL_TIMEOUT_SEC, op_name=op_name)
    return _parse_object_json(raw)


def _invoke_model_with_fallback(messages: list[BaseMessage]) -> Any:
    """
    Вызывает primary-модель и, при ошибке квоты free-тарифа, переключается на fallback-модель.

    Поведение:
    1. Пытаемся вызвать `MODEL_NAME`.
    2. Если ответ пустой/ошибочный и есть `MODEL_FALLBACK_NAME`, пробуем fallback.
    """
    primary_result, primary_error = _invoke_with_timeout_capture_error(
        PRIMARY_CHAT_MODEL.invoke,
        messages,
        timeout_sec=MODEL_TIMEOUT_SEC,
        op_name=f"model_invoke:{settings.resolved_model_name}",
    )
    if _is_non_empty_ai_response(primary_result):
        return primary_result

    if FALLBACK_CHAT_MODEL is None:
        return primary_result

    if not _should_switch_to_fallback(primary_error, settings.resolved_model_name):
        return primary_result

    logger.warning(
        "Switching model fallback: primary='%s' -> fallback='%s' due to: %s",
        settings.resolved_model_name,
        FALLBACK_MODEL_NAME,
        primary_error or "empty response",
    )
    fallback_result, _ = _invoke_with_timeout_capture_error(
        FALLBACK_CHAT_MODEL.invoke,
        messages,
        timeout_sec=MODEL_TIMEOUT_SEC,
        op_name=f"model_invoke:{FALLBACK_MODEL_NAME}",
    )
    return fallback_result


def _invoke_with_timeout(func: Callable[[Any], Any], arg: Any, timeout_sec: int, op_name: str) -> Any:
    """Безопасно вызывает функцию в отдельном потоке с ограничением времени."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, arg)
        try:
            return future.result(timeout=max(1, timeout_sec))
        except FuturesTimeoutError:
            logger.warning("%s timed out after %s sec", op_name, timeout_sec)
            return ""
        except Exception:
            logger.exception("%s failed", op_name)
            return ""


def _invoke_with_timeout_capture_error(
    func: Callable[[Any], Any], arg: Any, timeout_sec: int, op_name: str
) -> tuple[Any, str]:
    """Аналог `_invoke_with_timeout`, но дополнительно возвращает текст ошибки."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, arg)
        try:
            return future.result(timeout=max(1, timeout_sec)), ""
        except FuturesTimeoutError:
            message = f"timeout after {timeout_sec} sec"
            logger.warning("%s %s", op_name, message)
            return "", message
        except Exception as exc:
            logger.exception("%s failed", op_name)
            return "", str(exc)


def _is_non_empty_ai_response(response: Any) -> bool:
    """Проверяет, что response содержит непустой ответ модели."""
    return _extract_ai_text(response) != "Не удалось получить ответ."


def _should_switch_to_fallback(error_message: str, primary_model_name: str) -> bool:
    """Определяет, нужно ли переключаться на fallback-модель."""
    if not error_message.strip():
        # Если free-модель вернула пустой/битый ответ без явной ошибки,
        # тоже переключаемся на fallback.
        return primary_model_name.endswith(":free")

    lowered_error = error_message.lower()
    if any(marker in lowered_error for marker in QUOTA_ERROR_MARKERS):
        return True

    return primary_model_name.endswith(":free")


def _parse_object_json(raw: Any) -> dict[str, Any]:
    """Парсит JSON-строку в dict. Некорректные данные приводятся к пустому dict."""
    if not isinstance(raw, str):
        return {}

    try:
        payload = json.loads(raw)
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


def _extract_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Возвращает список объектов результата из payload."""
    items = payload.get("results")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _should_prefer_web(query: str) -> bool:
    """Определяет, когда web должен быть первым источником."""
    lowered_query = query.lower()
    words = lowered_query.split()

    if len(words) < 3:
        return SKU_PATTERN.search(query.upper()) is None

    return any(marker in lowered_query for marker in WEB_PRIORITY_MARKERS)


def _should_prefer_lookup(query: str) -> bool:
    """Определяет, когда lookup должен быть первым источником."""
    if SKU_PATTERN.search(query.upper()):
        return True

    lowered_query = query.lower()
    return any(marker in lowered_query for marker in LOOKUP_PRIORITY_MARKERS)


def enhance_search_query(original_query: str, search_type: str = "general") -> str:
    """Улучшает формулировку запроса для внешнего web-поиска."""
    lowered_query = original_query.lower()

    if any(marker in lowered_query for marker in ("сантехник", "унитаз", "смеситель")):
        return original_query

    if "новинк" in lowered_query or "новые" in lowered_query:
        if "2026" in lowered_query:
            return "новинки сантехники 2026 каталог"
        if "2025" in lowered_query:
            return "новинки сантехники 2025 каталог"
        return "новые сантехнические товары 2026 каталог"

    if any(marker in lowered_query for marker in ("бюджет", "цен", "стоит", "сколько")):
        return f"{original_query} сантехника"

    if "купить" in lowered_query or "где" in lowered_query:
        return f"{original_query} сантехника интернет магазин"

    if search_type == "fallback":
        return f"{original_query} сантехника"

    return f"{original_query} сантехника товары"


def _is_rag_useful(items: list[dict[str, Any]]) -> bool:
    """Проверяет, что RAG-результаты содержат полезный текст с приемлемым score."""
    for item in items:
        text = str(item.get("text", "")).strip()
        score = float(item.get("score", 0.0) or 0.0)
        if text and score >= MIN_RAG_SCORE:
            return True
    return False


def _is_web_useful(items: list[dict[str, Any]]) -> bool:
    """Проверяет, что web-результаты содержат валидные ссылки."""
    for item in items:
        url = str(item.get("url", "")).strip().lower()
        if url.startswith("https://") or url.startswith("http://"):
            return True
    return False


def _is_sanitary_relevant(items: list[dict[str, Any]]) -> bool:
    """Проверяет, что web-результаты действительно относятся к сантехнике."""
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
        score = float(item.get("score", 0.0) or 0.0)
        text = str(item.get("text", "")).strip().replace("\n", " ")
        lines.append(f"[RAG {index}] source={source} score={score:.3f} text={text[:800]}")

    return "\n".join(lines).strip()


def _format_lookup_context(payload: dict[str, Any], items: list[dict[str, Any]]) -> str:
    """Форматирует контекстный блок из результатов lookup."""
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
    """Форматирует контекстный блок из результатов web-поиска."""
    lines: list[str] = []

    for index, item in enumerate(items[:MAX_WEB_CONTEXT_ITEMS], start=1):
        title = str(item.get("title", "")).strip()
        snippet = str(item.get("snippet", "")).strip().replace("\n", " ")
        url = str(item.get("url", "")).strip()
        lines.append(f"[WEB {index}] {title} | {snippet} | {url}")

    return "\n".join(lines).strip()


def _extract_web_urls(items: list[dict[str, Any]]) -> list[str]:
    """Собирает уникальные URL из web-результатов."""
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


def _build_final_prompt(user_text: str, context_block: str) -> str:
    """Собирает финальный prompt для LLM на основе вопроса и контекста."""
    return (
        f"Вопрос пользователя:\n{user_text}\n\n"
        f"Контекст для ответа:\n{context_block}\n\n"
        "Ответь строго по вопросу пользователя. "
        "Если вопрос только про сантехнику — отвечай только про сантехнику. "
        "Не добавляй информацию про ремонт, стройматериалы, мебель и другие темы, "
        "если пользователь о них не спрашивал. Будь краток и точен."
    )


def _extract_ai_text(message: Any) -> str:
    """Извлекает текст из ответа модели с учетом разных форматов content."""
    if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
        return message.content

    if isinstance(message, AIMessage) and isinstance(message.content, list):
        parts = [str(part.get("text", "")) for part in message.content if isinstance(part, dict)]
        text = "\n".join(part for part in parts if part.strip()).strip()
        if text:
            return text

    return "Не удалось получить ответ."


def _ensure_sources_block(answer: str, urls: list[str]) -> str:
    """Добавляет блок `Источники`, если web использовался и блока еще нет."""
    if "Источники:" in answer:
        return answer

    lines = ["", "Источники:"]
    if urls:
        for url in urls[:MAX_SOURCE_URLS]:
            lines.append(f"- {url}")
    else:
        lines.append("- внешние ссылки не найдены")

    return answer.rstrip() + "\n" + "\n".join(lines)


def _clarifying_question() -> str:
    """Возвращает вопрос-уточнение, если контекст не найден."""
    return (
        "Пока не нашел достаточно надежных данных по вашему запросу. "
        "Уточните, пожалуйста, бренд, артикул (если есть) или ключевой параметр "
        "(например, диаметр/тип подключения/назначение)."
    )
