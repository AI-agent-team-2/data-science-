from __future__ import annotations

import json
import logging
import re
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from app.config import settings
from app.graph import model
from app.history_store import load_messages, save_turn
from app.observability import (
    get_langchain_callback_handler,
    hash_user_id,
    score_response_trace,
    sanitize_text,
)
from app.prompts import SYSTEM_PROMPT
from app.tools.product_lookup import product_lookup
from app.tools.rag_search import rag_search
from app.tools.web_search import web_search

logger = logging.getLogger(__name__)

ToolName = Literal["lookup", "rag", "web"]

SKU_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{4,}\b")
WEB_YEAR_PATTERN = re.compile(r"\bв\s+20\d{2}\b")
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
TOOL_TIMEOUT_SEC = 20
MODEL_TIMEOUT_SEC = 45
MIN_RAG_SCORE = 0.2
MAX_RAG_CONTEXT_ITEMS = 4
MAX_LOOKUP_CONTEXT_ITEMS = 5
MAX_WEB_CONTEXT_ITEMS = 5
MAX_SOURCE_URLS = 5

SMALLTALK_MARKERS: tuple[str, ...] = (
    "привет",
    "здравствуйте",
    "здравствуй",
    "добрый день",
    "добрый вечер",
    "доброе утро",
    "как дела",
    "что делаешь",
    "кто ты",
    "ты кто",
    "как настроение",
    "че как",
    "норм работаешь",
    "ты вообще шаришь",
    "спасибо",
    "ок",
    "понял",
)

IDENTITY_OR_CAPABILITY_MARKERS: tuple[str, ...] = (
    "кто ты",
    "ты кто",
    "ты человек",
    "ты чат-бот",
    "что умеешь",
    "чем ты можешь помочь",
    "ты разбираешься в сантехнике",
    "ты можешь подобрать оборудование",
    "что ты знаешь про отопление",
)

OFFTOPIC_OR_RUDE_MARKERS: tuple[str, ...] = (
    "ты тупой",
    "ты бесполезный",
    "ты ничего не понимаешь",
    "дай нормальный ответ",
    "кто выиграет чемпионат мира",
    "напиши код на python",
    "кто такой наполеон",
    "расскажи анекдот",
)

WEB_PRIORITY_MARKERS: tuple[str, ...] = (
    "сейчас",
    "сегодня",
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
    "лучший",
    "рейтинг",
    "топ",
    "популярный",
    "актуальный",
    "последний",
    "новинки",
    "тренды",
    "тенденции",
    "современные решения",
    "чаще используют",
    "стандарты",
    "бренды",
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
    "насос",
    "отоплен",
    "коллектор",
    "редуктор",
    "сервопривод",
    "мультифлекс",
    "радиатор",
    "теплый пол",
    "теплого пола",
    "давлен",
)

DOMAIN_MARKERS: tuple[str, ...] = SANITARY_KEYWORDS + (
    "ondo",
    "stm",
    "optima",
    "roegen",
    "rispa",
    "atlas",
    "акс",
    "мультифлекс",
    "редуктор",
    "коллектор",
    "сервопривод",
    "воздухоотвод",
    "тепл",
)


@dataclass(frozen=True)
class ContextBuildResult:
    """Результат получения контекста из инструментов."""

    context_text: str
    web_urls: list[str]
    used_web: bool


EMPTY_CONTEXT_RESULT = ContextBuildResult(context_text="", web_urls=[], used_web=False)


def run_agent(user_text: str, user_id: str = "unknown") -> str:
    """
    Выполняет полный цикл ответа пользователю.

    Parameters
    ----------
    user_text : str
        Текст запроса пользователя.
    user_id : str, default="unknown"
        Внешний идентификатор пользователя/сессии.

    Returns
    -------
    str
        Финальный ответ ассистента.
    """
    session_id = user_id or "unknown"
    query = user_text.strip()
    hashed_user = hash_user_id(session_id)
    trace_id = uuid4().hex
    source_order = _resolve_source_order(query)

    callback_handler = get_langchain_callback_handler(
        trace_id=trace_id,
        session_id=hashed_user,
        user_id=hashed_user,
        tags=["telegram", "san-bot", "run_agent"],
    )

    root_config: RunnableConfig = {
        "callbacks": [callback_handler] if callback_handler is not None else [],
        "run_name": "run_agent",
        "metadata": {
            "query": sanitize_text(query),
            "model": settings.resolved_model_name,
            "provider": settings.resolved_model_provider,
            "source_order": source_order,
            "enable_web_search": settings.enable_web_search,
            "enable_rag": settings.enable_rag,
            "enable_product_lookup": settings.enable_product_lookup,
            "langfuse_trace_id": trace_id,
            "langfuse_session_id": hashed_user,
            "langfuse_user_id": hashed_user,
            "langfuse_tags": ["telegram", "san-bot", "run_agent"],
        },
    }

    pipeline = RunnableLambda(_run_agent_pipeline)
    return pipeline.invoke(
        {
            "user_text": user_text,
            "session_id": session_id,
            "hashed_user": hashed_user,
            "trace_id": trace_id,
            "source_order": source_order,
        },
        config=root_config,
    )


def _run_agent_pipeline(payload: dict[str, Any], config: RunnableConfig | None = None) -> str:
    """Выполняет основной pipeline ответа внутри callback-aware runnable."""
    user_text = str(payload.get("user_text", ""))
    session_id = str(payload.get("session_id", "unknown"))
    hashed_user = str(payload.get("hashed_user", hash_user_id(session_id)))
    trace_id = str(payload.get("trace_id", uuid4().hex))
    source_order = payload.get("source_order")
    if not isinstance(source_order, list):
        source_order = _resolve_source_order(user_text.strip())

    query = user_text.strip()
    assistant_text = ""

    if _is_identity_or_capability_query(query):
        assistant_text = _assistant_scope_response()
        return _save_and_return(
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            trace_id=trace_id,
        )

    if _is_smalltalk(query):
        assistant_text = _smalltalk_response()
        return _save_and_return(
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            trace_id=trace_id,
        )

    if _is_noise_query(query) or _is_offtopic_or_rude_query(query):
        assistant_text = _domain_redirect_response()
        return _save_and_return(
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            trace_id=trace_id,
            blocked=True,
        )

    history_messages = _to_langchain_messages(load_messages(session_id=session_id))

    context = _build_context(query, source_order=source_order, config=config)
    if not context.context_text:
        assistant_text = _clarifying_question()
        return _save_and_return(
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            trace_id=trace_id,
        )

    final_prompt = _build_final_prompt(user_text=user_text, context_block=context.context_text)
    model_input = [SystemMessage(content=SYSTEM_PROMPT), *history_messages, HumanMessage(content=final_prompt)]

    effective_model_config = _child_config(config, "model_invoke") or {}
    model_metadata = {
        "provider": settings.resolved_model_provider,
        "model": settings.resolved_model_name,
        "source_order": source_order,
        "langfuse_trace_id": trace_id,
        "langfuse_session_id": hashed_user,
        "langfuse_user_id": hashed_user,
        "langfuse_tags": ["telegram", "san-bot", "run_agent"],
    }
    existing_meta = effective_model_config.get("metadata")
    if isinstance(existing_meta, dict):
        effective_model_config["metadata"] = {**existing_meta, **model_metadata}
    else:
        effective_model_config["metadata"] = model_metadata
    response = _invoke_with_timeout(
        lambda payload_input: model.invoke(payload_input, config=effective_model_config),
        model_input,
        timeout_sec=MODEL_TIMEOUT_SEC,
    )

    assistant_text = _extract_ai_text(response)
    if context.used_web:
        assistant_text = _ensure_sources_block(assistant_text, context.web_urls)

    return _save_and_return(
        session_id=session_id,
        user_text=user_text,
        assistant_text=assistant_text,
        trace_id=trace_id,
    )


def _to_langchain_messages(history: list[tuple[str, str]]) -> list[BaseMessage]:
    """Преобразует сохраненную историю в объекты сообщений LangChain."""
    messages: list[BaseMessage] = []

    for role, content in history:
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))

    return messages


def _save_and_return(
    session_id: str,
    user_text: str,
    assistant_text: str,
    trace_id: str | None = None,
    blocked: bool = False,
) -> str:
    """Единая точка сохранения ответа и возврата результата из pipeline."""
    if trace_id:
        score_response_trace(
            trace_id=trace_id,
            user_text=user_text,
            assistant_text=assistant_text,
            blocked=blocked,
        )
    save_turn(session_id=session_id, user_text=user_text, assistant_text=assistant_text)
    return assistant_text


def _build_context(
    query: str,
    source_order: list[ToolName] | None = None,
    config: RunnableConfig | None = None,
) -> ContextBuildResult:
    """
    Подбирает контекст из доступных источников по приоритету.

    Parameters
    ----------
    query : str
        Текст запроса.
    source_order : list[ToolName] | None
        Явный порядок источников.

    Returns
    -------
    ContextBuildResult
        Итоговый контекст, ссылки и признак использования web-источника.
    """
    source_order = source_order or _resolve_source_order(query)

    for index, source in enumerate(source_order):
        web_mode = "primary" if index == 0 else "fallback"

        if source == "lookup" and not settings.enable_product_lookup:
            continue
        if source == "rag" and not settings.enable_rag:
            continue
        if source == "web" and not settings.enable_web_search:
            continue
        if source == "web" and not _should_use_web_source(query=query, web_mode=web_mode):
            continue

        result = _context_from_source(
            source=source,
            query=query,
            web_mode=web_mode,
            config=config,
        )
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


def _context_from_source(
    source: ToolName,
    query: str,
    web_mode: str,
    config: RunnableConfig | None = None,
) -> ContextBuildResult:
    """Строит контекст из указанного источника данных."""
    if source == "lookup":
        return _context_from_lookup(query, config=config)
    if source == "rag":
        return _context_from_rag(query, config=config)
    return _context_from_web(query, mode=web_mode, config=config)


def _context_from_lookup(query: str, config: RunnableConfig | None = None) -> ContextBuildResult:
    """Пытается получить контекст из базы товаров (LOOKUP)."""
    payload = _invoke_tool(
        product_lookup.invoke,
        {"query": query, "limit": 5},
        "tool_lookup",
        config=config,
    )
    items = _extract_results(payload)

    if not items:
        return EMPTY_CONTEXT_RESULT

    return ContextBuildResult(
        context_text=_format_lookup_context(payload, items),
        web_urls=[],
        used_web=False,
    )


def _context_from_rag(query: str, config: RunnableConfig | None = None) -> ContextBuildResult:
    """Пытается получить контекст из RAG-базы знаний."""
    payload = _invoke_tool(
        rag_search.invoke,
        {"query": query},
        "tool_rag",
        config=config,
    )
    items = _extract_results(payload)

    if not _is_rag_useful(items):
        return EMPTY_CONTEXT_RESULT

    return ContextBuildResult(
        context_text=_format_rag_context(items),
        web_urls=[],
        used_web=False,
    )


def _context_from_web(
    query: str,
    mode: str,
    config: RunnableConfig | None = None,
) -> ContextBuildResult:
    """Пытается получить контекст из web-поиска."""
    enhanced_query = enhance_search_query(query, mode)
    payload = _invoke_tool(
        web_search.invoke,
        {"query": enhanced_query, "max_results": settings.web_search_max_results},
        "tool_web",
        config=config,
    )
    items = _extract_results(payload)

    if not (_is_web_useful(items) and _is_sanitary_relevant(items)):
        return EMPTY_CONTEXT_RESULT

    return ContextBuildResult(
        context_text=_format_web_context(items),
        web_urls=_extract_web_urls(items),
        used_web=True,
    )


def _invoke_tool(
    func: Callable[[dict[str, Any]], Any],
    payload: dict[str, Any],
    op_name: str,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Вызывает tool с таймаутом и возвращает JSON-объект."""
    tool_config = _child_config(config, op_name)
    raw = _invoke_with_timeout(
        lambda tool_payload: func(tool_payload, config=tool_config),
        payload,
        timeout_sec=TOOL_TIMEOUT_SEC,
    )
    return _parse_object_json(raw)


def _invoke_with_timeout(
    func: Callable[[Any], Any],
    arg: Any,
    timeout_sec: int,
) -> Any:
    """
    Вызывает функцию в отдельном потоке с ограничением времени.

    Parameters
    ----------
    func : Callable[[Any], Any]
        Целевая функция.
    arg : Any
        Аргумент для вызова.
    timeout_sec : int
        Лимит времени в секундах.

    Returns
    -------
    Any
        Результат `func(arg)` либо пустая строка при ошибке/таймауте.
    """
    def _runner() -> Any:
        return func(arg)

    with ThreadPoolExecutor(max_workers=1) as executor:
        # Переносим contextvars в worker-поток, чтобы callback сохранял parent-child связь trace.
        ctx = copy_context()
        future = executor.submit(ctx.run, _runner)
        try:
            return future.result(timeout=max(1, timeout_sec))
        except FuturesTimeoutError:
            logger.warning("Операция превысила таймаут %s сек", timeout_sec)
            return ""
        except Exception:
            logger.exception("Операция завершилась с ошибкой")
            return ""


def _child_config(config: RunnableConfig | None, run_name: str) -> RunnableConfig | None:
    """Создает дочерний runnable-config с тем же callback-контекстом."""
    if config is None:
        return None

    child: RunnableConfig = {}
    callbacks = config.get("callbacks")
    if callbacks is not None:
        child["callbacks"] = callbacks
    tags = config.get("tags")
    if tags is not None:
        child["tags"] = tags

    metadata = config.get("metadata")
    if isinstance(metadata, dict):
        child["metadata"] = dict(metadata)
    child["run_name"] = run_name
    return child

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
    """Определяет, когда WEB должен быть первым источником."""
    lowered_query = query.lower()
    words = lowered_query.split()
    has_web_marker = any(marker in lowered_query for marker in WEB_PRIORITY_MARKERS)
    has_web_year_marker = WEB_YEAR_PATTERN.search(lowered_query) is not None
    has_lookup_marker = any(marker in lowered_query for marker in LOOKUP_PRIORITY_MARKERS)
    has_sku = SKU_PATTERN.search(query.upper()) is not None

    # Явно внешние и динамичные запросы отправляем в web в первую очередь.
    if (has_web_marker or has_web_year_marker) and not has_lookup_marker and not has_sku:
        return True

    # Короткие общие фразы можно отправлять в web, если это не SKU/артикул.
    if len(words) < 3:
        return (has_web_marker or has_web_year_marker) and not has_sku

    # Для остальных доменных вопросов web не должен быть первичным источником.
    if _is_domain_query(lowered_query):
        return False

    return has_web_marker or has_web_year_marker


def _should_prefer_lookup(query: str) -> bool:
    """Определяет, когда LOOKUP должен быть первым источником."""
    if SKU_PATTERN.search(query.upper()):
        return True

    lowered_query = query.lower()
    return any(marker in lowered_query for marker in LOOKUP_PRIORITY_MARKERS)


def _is_smalltalk(query: str) -> bool:
    """Проверяет, что сообщение похоже на короткую бытовую реплику."""
    lowered_query = query.lower().strip()
    if not lowered_query:
        return False
    if _is_domain_query(lowered_query):
        return False

    normalized = re.sub(r"[^\w\s]", " ", lowered_query)
    normalized = " ".join(normalized.split())
    if not normalized:
        return False

    # Реагируем только на короткие бытовые реплики.
    if len(normalized.split()) > 5:
        return False

    for marker in SMALLTALK_MARKERS:
        if " " in marker:
            if normalized == marker:
                return True
            continue

        if re.search(rf"\b{re.escape(marker)}\b", normalized):
            return True

    return False


def _is_identity_or_capability_query(query: str) -> bool:
    """Проверяет вопросы о роли бота и его возможностях."""
    lowered_query = query.lower().strip()
    if not lowered_query:
        return False
    return any(marker in lowered_query for marker in IDENTITY_OR_CAPABILITY_MARKERS)


def _is_noise_query(query: str) -> bool:
    """Определяет шумовые/неинформативные сообщения, которые не нужно слать в WEB."""
    lowered_query = query.lower().strip()
    if not lowered_query:
        return True
    if _is_domain_query(lowered_query):
        return False

    alnum = re.sub(r"[\W_]+", "", lowered_query, flags=re.UNICODE)
    if not alnum:
        return True
    if lowered_query in {"???", "...", "...."}:
        return True
    if alnum.isdigit():
        return True
    if len(alnum) <= 4 and not re.search(r"[а-яa-z]", alnum):
        return True
    if len(set(alnum)) <= 2 and len(alnum) >= 6:
        return True
    return False


def _is_offtopic_or_rude_query(query: str) -> bool:
    """Возвращает True для оффтопа/резких фраз без доменного контекста."""
    lowered_query = query.lower()
    if _is_domain_query(lowered_query):
        return False
    return any(marker in lowered_query for marker in OFFTOPIC_OR_RUDE_MARKERS)


def _should_use_web_source(query: str, web_mode: str) -> bool:
    """Ограничивает WEB: как fallback используем только для доменных запросов."""
    lowered_query = query.lower()
    if _is_noise_query(query):
        return False
    if web_mode == "fallback":
        return _is_domain_query(lowered_query)
    return _should_prefer_web(query) or _is_domain_query(lowered_query)


def _is_domain_query(lowered_query: str) -> bool:
    """Проверяет, что запрос относится к товарам/техтематике проекта."""
    if SKU_PATTERN.search(lowered_query.upper()):
        return True
    return any(marker in lowered_query for marker in DOMAIN_MARKERS)


def _smalltalk_response() -> str:
    """Возвращает короткий ответ для бытовых реплик без запуска поиска."""
    return (
        "Привет! Я в порядке и готов помочь по сантехническим товарам. "
        "Напишите, пожалуйста, бренд, модель, артикул или технический вопрос."
    )


def enhance_search_query(original_query: str, search_type: str = "general") -> str:
    """
    Улучшает формулировку запроса для внешнего web-поиска.

    Parameters
    ----------
    original_query : str
        Исходный запрос пользователя.
    search_type : str, default="general"
        Режим поиска (`general` или `fallback`).

    Returns
    -------
    str
        Нормализованный поисковый запрос.
    """
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


def _build_final_prompt(user_text: str, context_block: str) -> str:
    """Собирает финальный prompt для LLM из вопроса и контекста."""
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
    """Добавляет блок `Источники`, если использовался WEB и блока еще нет."""
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


def _domain_redirect_response() -> str:
    """Мягко возвращает диалог в домен бота, не уводя в посторонние темы."""
    return (
        "Я помогаю по сантехническим товарам и отоплению. "
        "Напишите, пожалуйста, что именно нужно: бренд, артикул или задачу "
        "(например, подобрать насос, коллектор, редуктор, трубу или сервопривод)."
    )


def _assistant_scope_response() -> str:
    """Отвечает на вопросы о роли ассистента и возвращает в целевой домен."""
    return (
        "Я чат-бот по сантехническим товарам и отоплению. "
        "Помогаю подобрать оборудование, объяснить характеристики и совместимость, "
        "а также найти варианты по артикулу или задаче."
    )
