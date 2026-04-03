from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from app.config import settings
from app.context import (
    assistant_scope_response,
    build_context,
    build_final_prompt,
    clarifying_question,
    domain_redirect_response,
    ensure_sources_block,
    extract_ai_text,
    parse_tool_payload,
    smalltalk_response,
    tool_failure_response,
    ToolExecutionResult,
)
from app.graph import model
from app.history_store import load_messages, save_turn
from app.observability import (
    get_langchain_callback_handler,
    hash_user_id,
    sanitize_text,
)
from app.prompts import SYSTEM_PROMPT
from app.routing import (
    ToolName,
    is_domain_query,
    is_identity_or_capability_query,
    is_noise_query,
    is_offtopic_or_rude_query,
    is_smalltalk,
    resolve_source_order,
)
from app.utils.sku import extract_sku_candidates

logger = logging.getLogger(__name__)


INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)ignore (all|previous|prior) instructions"),
    re.compile(r"(?i)system prompt"),
    re.compile(r"(?i)developer message"),
    re.compile(r"(?i)игнорируй\s+предыдущ"),
    re.compile(r"(?i)раскрой\s+системн"),
)
TRUNCATED_MARKER_PATTERN = re.compile(r"[\[\(\{]?\s*\.{0,3}\s*truncated\s*[\]\)\}]?", re.IGNORECASE)
MODEL_PATTERN = re.compile(r"\b[A-Z]{2,}\s*\d{1,2}\s*[-/]\s*\d{1,2}\s*[A-Z]{1,3}\b")


@dataclass(frozen=True)
class InvocationResult:
    """Результат выполнения модели/инструмента с явным статусом."""

    status: str
    value: Any = None
    error_type: str = ""
    error_message: str = ""


def run_agent(user_text: str, user_id: str = "unknown") -> str:
    """Выполняет полный цикл ответа пользователю."""
    session_id = user_id or "unknown"
    query = user_text.strip()
    hashed_user = hash_user_id(session_id)
    source_order = resolve_source_order(query)
    intent = _detect_intent(query)
    _, guard_action, risk_flags = _apply_guard(query)

    callback_handler = get_langchain_callback_handler()
    trace_metadata = _build_trace_metadata(
        query=query,
        source_order=source_order,
        intent=intent,
        risk_flags=risk_flags,
        guard_action=guard_action,
        hashed_user=hashed_user,
    )
    root_config: RunnableConfig = {
        "callbacks": [callback_handler] if callback_handler is not None else [],
        "run_name": "agent_request",
        "metadata": trace_metadata,
    }

    pipeline = RunnableLambda(_run_agent_pipeline).with_config({"run_name": "agent_pipeline"})
    return pipeline.invoke(
        {
            "user_text": user_text,
            "session_id": session_id,
            "hashed_user": hashed_user,
            "source_order": source_order,
        },
        config=root_config,
    )


def _run_agent_pipeline(payload: dict[str, Any], config: RunnableConfig | None = None) -> str:
    """Выполняет основной pipeline ответа внутри callback-aware runnable."""
    user_text = str(payload.get("user_text", ""))
    session_id = str(payload.get("session_id", "unknown"))
    hashed_user = str(payload.get("hashed_user", hash_user_id(session_id)))
    source_order = payload.get("source_order")
    if not isinstance(source_order, list):
        source_order = resolve_source_order(user_text.strip())

    query = user_text.strip()
    safe_query, guard_action, risk_flags = _apply_guard(query)

    if guard_action == "block":
        return _finalize_response(
            session_id=session_id,
            user_text=user_text,
            raw_assistant_text=(
                "Не могу обработать этот запрос в таком виде. "
                "Сформулируйте, пожалуйста, вопрос по сантехническим товарам."
            ),
        )

    if is_identity_or_capability_query(safe_query):
        return _finalize_response(
            session_id=session_id,
            user_text=user_text,
            raw_assistant_text=assistant_scope_response(),
        )

    if is_smalltalk(safe_query):
        return _finalize_response(
            session_id=session_id,
            user_text=user_text,
            raw_assistant_text=smalltalk_response(),
        )

    if is_noise_query(safe_query) or is_offtopic_or_rude_query(safe_query):
        return _finalize_response(
            session_id=session_id,
            user_text=user_text,
            raw_assistant_text=domain_redirect_response(),
        )

    constraint_response = _known_domain_constraint_response(safe_query)
    if constraint_response:
        return _finalize_response(
            session_id=session_id,
            user_text=user_text,
            raw_assistant_text=constraint_response,
        )

    history_messages = _to_langchain_messages(load_messages(session_id=session_id))
    dialogue_context = _build_dialogue_memory_summary(history_messages)
    context = build_context(safe_query, source_order, invoke_tool=_invoke_tool, config=config)
    if not context.context_text:
        if context.terminal_response:
            return _finalize_response(
                session_id=session_id,
                user_text=user_text,
                raw_assistant_text=context.terminal_response,
            )
        return _finalize_response(
            session_id=session_id,
            user_text=user_text,
            raw_assistant_text=clarifying_question(),
        )

    final_prompt = build_final_prompt(
        user_text=safe_query,
        context_block=context.context_text,
        dialogue_context=dialogue_context,
    )
    model_input = [SystemMessage(content=SYSTEM_PROMPT), *history_messages, HumanMessage(content=final_prompt)]

    effective_model_config = _child_config(config, "model_invoke") or {}
    model_metadata = {
        "provider": settings.resolved_model_provider,
        "model": settings.resolved_model_name,
        "source_order": source_order,
        "used_source": context.used_source,
        "attempted_sources": context.attempted_sources,
        "source_status_map": context.source_status_map,
        "failed_sources": context.failed_sources,
        "fallback_reason": context.fallback_reason,
        "risk_flags": risk_flags,
        "guard_action": guard_action,
        "trace_session_id": hashed_user,
        "trace_user_id": hashed_user,
        "trace_tags": ["telegram", "san-bot", "run_agent"],
    }
    existing_meta = effective_model_config.get("metadata")
    if isinstance(existing_meta, dict):
        effective_model_config["metadata"] = {**existing_meta, **model_metadata}
    else:
        effective_model_config["metadata"] = model_metadata

    response = _invoke_with_timeout(
        lambda payload_input: model.invoke(payload_input, config=effective_model_config),
        model_input,
        timeout_sec=settings.model_timeout_sec,
    )
    if response.status != "ok":
        logger.warning("model_invoke failed: %s %s", response.error_type, response.error_message)
        return _finalize_response(
            session_id=session_id,
            user_text=user_text,
            raw_assistant_text=tool_failure_response(),
        )

    raw_assistant_text = extract_ai_text(response.value)
    assistant_text = _prepare_user_answer(raw_assistant_text)
    if context.used_web:
        assistant_text = ensure_sources_block(assistant_text, context.web_urls)

    return _save_and_return(session_id=session_id, user_text=user_text, assistant_text=assistant_text)


def _to_langchain_messages(history: list[tuple[str, str]]) -> list[BaseMessage]:
    """Преобразует сохраненную историю в объекты сообщений LangChain."""
    messages: list[BaseMessage] = []
    for role, content in history:
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages


def _build_dialogue_memory_summary(messages: list[BaseMessage], max_messages: int = 8) -> str:
    """Извлекает компактный контекст последних сущностей диалога (SKU/модели)."""
    if not messages:
        return ""

    tail = messages[-max_messages:]
    recent_text = "\n".join(str(message.content) for message in tail if getattr(message, "content", ""))
    sku_candidates = sorted(extract_sku_candidates(recent_text, require_digit=True))
    model_candidates = sorted(
        {
            re.sub(r"\s+", " ", match.group(0)).strip()
            for match in MODEL_PATTERN.finditer(recent_text.upper())
        }
    )
    if not sku_candidates and not model_candidates:
        return ""

    parts: list[str] = []
    if sku_candidates:
        parts.append(f"артикулы: {', '.join(sku_candidates[:6])}")
    if model_candidates:
        parts.append(f"модели: {', '.join(model_candidates[:4])}")
    preview = "; ".join(parts)
    return (
        f"Последние упомянутые сущности: {preview}. "
        "Используй их для корректной кореференции (например: 'он', 'этот', 'второй вариант')."
    )


def _save_and_return(session_id: str, user_text: str, assistant_text: str) -> str:
    """Сохраняет шаг диалога и возвращает итоговый ответ."""
    save_turn(session_id=session_id, user_text=user_text, assistant_text=assistant_text)
    return assistant_text


def _finalize_response(
    session_id: str,
    user_text: str,
    raw_assistant_text: str,
) -> str:
    """Подготавливает и сохраняет итоговый ответ пользователю."""
    assistant_text = _prepare_user_answer(raw_assistant_text)
    return _save_and_return(session_id=session_id, user_text=user_text, assistant_text=assistant_text)


def _prepare_user_answer(raw_assistant_text: str) -> str:
    """Очищает только технический мусор и сохраняет валидные пользовательские ссылки."""
    value = str(raw_assistant_text or "").strip()
    if not value:
        return "Не удалось получить ответ."
    cleaned = TRUNCATED_MARKER_PATTERN.sub(" ", value)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _build_trace_metadata(
    query: str,
    source_order: list[ToolName],
    intent: str,
    risk_flags: list[str],
    guard_action: str,
    hashed_user: str,
) -> dict[str, Any]:
    """Формирует единый metadata-формат для run/model/tool шагов."""
    return {
        "query": sanitize_text(query),
        "provider": settings.resolved_model_provider,
        "model": settings.resolved_model_name,
        "intent": intent,
        "source_order": list(source_order),
        "attempted_sources": [],
        "source_status_map": {},
        "failed_sources": [],
        "fallback_reason": "",
        "risk_flags": list(risk_flags),
        "guard_action": guard_action,
        "trace_session_id": hashed_user,
        "trace_user_id": hashed_user,
        "trace_tags": ["telegram", "san-bot", "run_agent"],
        "features": {
            "enable_web_search": settings.enable_web_search,
            "enable_rag": settings.enable_rag,
            "enable_product_lookup": settings.enable_product_lookup,
        },
    }


def _invoke_tool(
    func: Callable[[dict[str, Any]], Any],
    payload: dict[str, Any],
    op_name: str,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Вызывает инструмент с таймаутом и возвращает JSON-объект."""
    tool_config = _child_config(config, op_name)
    if tool_config is not None:
        metadata = tool_config.get("metadata")
        if isinstance(metadata, dict):
            tool_name = op_name.replace("tool_", "", 1)
            tool_config["metadata"] = {
                **metadata,
                "tool_name": tool_name,
                "tool_operation": op_name,
            }
    raw = _invoke_with_timeout(
        lambda tool_payload: func(tool_payload, config=tool_config),
        payload,
        timeout_sec=settings.tool_timeout_sec,
    )
    if raw.status != "ok":
        return ToolExecutionResult(
            status="failed",
            payload={},
            error_type=raw.error_type,
            error_message=raw.error_message,
        )

    parsed = parse_tool_payload(raw.value)
    if parsed:
        return ToolExecutionResult(status="ok", payload=parsed)

    logger.warning("Tool %s returned unparseable payload", op_name)
    return ToolExecutionResult(
        status="failed",
        payload={},
        error_type="parse_error",
        error_message=f"{op_name} returned an unparseable payload",
    )


def _invoke_with_timeout(
    func: Callable[[Any], Any],
    arg: Any,
    timeout_sec: int,
) -> Any:
    """Вызывает функцию в отдельном потоке с ограничением времени."""

    def _runner() -> Any:
        return func(arg)

    with ThreadPoolExecutor(max_workers=1) as executor:
        ctx = copy_context()
        future = executor.submit(ctx.run, _runner)
        try:
            return InvocationResult(status="ok", value=future.result(timeout=max(1, timeout_sec)))
        except FuturesTimeoutError:
            logger.warning("Операция превысила таймаут %s сек", timeout_sec)
            return InvocationResult(status="failed", error_type="timeout", error_message=f"timeout>{timeout_sec}s")
        except Exception as exc:
            logger.exception("Операция завершилась с ошибкой")
            return InvocationResult(
                status="failed",
                error_type="exception",
                error_message=exc.__class__.__name__,
            )


def _child_config(config: RunnableConfig | None, run_name: str) -> RunnableConfig | None:
    """Создает дочерний runnable-config с тем же callback-контекстом."""
    if config is None:
        return None

    child: RunnableConfig = {"run_name": run_name}
    callbacks = config.get("callbacks")
    if callbacks is not None:
        child["callbacks"] = callbacks
    tags = config.get("tags")
    if tags is not None:
        child["tags"] = tags

    metadata = config.get("metadata")
    if isinstance(metadata, dict):
        child["metadata"] = dict(metadata)
    return child


def _detect_prompt_injection(query: str) -> tuple[bool, list[str]]:
    """Определяет признаки prompt injection в пользовательском запросе."""
    matched: list[str] = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(query):
            matched.append(pattern.pattern)
    return bool(matched), matched


def _rewrite_suspicious_query(query: str) -> str:
    """Убирает из подозрительного запроса управляющие инъекционные фрагменты."""
    rewritten = query
    for pattern in INJECTION_PATTERNS:
        rewritten = pattern.sub(" ", rewritten)
    rewritten = re.sub(r"\s+", " ", rewritten).strip(" ,.;:")
    return rewritten


def _apply_guard(query: str) -> tuple[str, str, list[str]]:
    """
    Применяет легкий guard и возвращает:
    safe_query, action (allow|rewrite|block), risk_flags.
    """
    is_injection, matched = _detect_prompt_injection(query)
    if not is_injection:
        return query, "allow", []

    risk_flags = ["prompt_injection"] + [f"pattern:{value}" for value in matched[:3]]
    rewritten = _rewrite_suspicious_query(query)
    if rewritten and is_domain_query(rewritten.lower()):
        return rewritten, "rewrite", risk_flags
    return query, "block", risk_flags


def _detect_intent(query: str) -> str:
    """Возвращает короткую intent-метку для observability."""
    if is_identity_or_capability_query(query):
        return "identity"
    if is_smalltalk(query):
        return "smalltalk"
    if is_noise_query(query) or is_offtopic_or_rude_query(query):
        return "offtopic"
    return "domain"


def _known_domain_constraint_response(query: str) -> str:
    """Возвращает детерминированный ответ для критичных доменных ограничений."""
    lowered = str(query or "").lower()
    if "мультифлекс" in lowered and any(marker in lowered for marker in ("душить поток", "регулировать поток", "регулиров", "придушить")):
        return (
            "Нет, шаровыми кранами мультифлекса нельзя регулировать (душить) поток. "
            "Они предназначены для полного открытия/закрытия. "
            "Для регулирования используйте штатные регулирующие элементы контура."
        )
    return ""
