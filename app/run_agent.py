from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from app.agent.guards import apply_guard, known_domain_constraint_response
from app.agent.invoke import (
    child_config,
    invoke_tool,
    invoke_with_timeout,
)
from app.agent.memory import build_dialogue_memory_summary, to_langchain_messages
from app.agent.response import prepare_user_answer
from app.agent.trace import build_trace_metadata, detect_intent
from app.config import settings
from app.context_engine import (
    assistant_scope_response,
    build_context,
    build_final_prompt,
    clarifying_question,
    domain_redirect_response,
    ensure_sources_block,
    extract_ai_text,
    smalltalk_response,
    tool_failure_response,
)
from app.graph import model
from app.history_store import load_messages, save_turn
from app.observability import (
    get_langchain_callback_handler,
    hash_user_id,
)
from app.prompts import SYSTEM_PROMPT
from app.routing import (
    is_identity_or_capability_query,
    is_noise_query,
    is_offtopic_or_rude_query,
    is_smalltalk,
    resolve_source_order,
)

logger = logging.getLogger(__name__)


def run_agent(user_text: str, user_id: str = "unknown") -> str:
    """Выполняет полный цикл ответа пользователю."""
    session_id = user_id or "unknown"
    query = user_text.strip()
    hashed_user = hash_user_id(session_id)
    source_order = resolve_source_order(query)
    intent = detect_intent(query)
    _, guard_action, risk_flags = apply_guard(query)

    callback_handler = get_langchain_callback_handler()
    trace_metadata = build_trace_metadata(
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
    safe_query, guard_action, risk_flags = apply_guard(query)

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

    constraint_response = known_domain_constraint_response(safe_query)
    if constraint_response:
        return _finalize_response(
            session_id=session_id,
            user_text=user_text,
            raw_assistant_text=constraint_response,
        )

    history_messages = to_langchain_messages(load_messages(session_id=session_id))
    dialogue_context = build_dialogue_memory_summary(history_messages)
    context = build_context(safe_query, source_order, invoke_tool=invoke_tool, config=config)
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

    effective_model_config = child_config(config, "model_invoke") or {}
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

    response = invoke_with_timeout(
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
    assistant_text = prepare_user_answer(raw_assistant_text)
    if context.used_web:
        assistant_text = ensure_sources_block(assistant_text, context.web_urls)

    return _save_and_return(session_id=session_id, user_text=user_text, assistant_text=assistant_text)


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
    assistant_text = prepare_user_answer(raw_assistant_text)
    return _save_and_return(session_id=session_id, user_text=user_text, assistant_text=assistant_text)
