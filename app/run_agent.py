from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from app.guards.prompt_injection import apply_guard, known_domain_constraint_response
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
from app.graph import get_model, model_circuit_breaker
from app.history_store import load_messages, save_turn
from app.observability.rate_limiter import rate_limiter
from app.observability.token_usage import token_manager
from app.observability import (
    build_log_fields,
    format_log_fields,
    get_langchain_callback_handler,
    hash_user_id,
)
from app.prompts import SYSTEM_PROMPT
from app.routing import (
    is_identity_or_capability_query,
    is_domain_query,
    is_noise_query,
    is_offtopic_or_rude_query,
    is_smalltalk,
    resolve_source_order,
)
from app.guards import ai_domain_check, ai_input_policy_check, ai_output_policy_check, redact_pii

logger = logging.getLogger(__name__)


HARD_POLICY_CATEGORIES: set[str] = {
    "self_harm",
    "privacy",
    "doxxing",
    "weapons",
    "violence",
    "illegal",
    "sexual",
    "fraud",
}


def _should_enforce_policy(decision_categories: list[str], confidence: float) -> bool:
    if any(cat in HARD_POLICY_CATEGORIES for cat in decision_categories):
        return True
    return float(confidence) >= float(settings.ai_guard_enforce_min_confidence)


def _policy_refusal_text() -> str:
    return (
        "Не могу помочь с этим запросом. "
        "Я могу помочь по сантехническим товарам и отоплению: подбор, характеристики, совместимость, артикулы."
    )


def _self_harm_safe_reply() -> str:
    return (
        "Мне жаль, что вам сейчас тяжело. Я не могу помогать с причинением вреда себе.\n\n"
        "Если вы в опасности прямо сейчас или есть риск, пожалуйста, обратитесь за срочной помощью "
        "в вашем регионе (экстренные службы) или к близкому человеку рядом.\n\n"
        "Если хотите, вы можете рассказать, что происходит, и я постараюсь поддержать и помочь найти безопасные шаги."
    )


def run_agent(user_text: str, user_id: str = "unknown", *, source_order_override: list[str] | None = None) -> str:
    """Выполняет полный цикл ответа пользователю."""
    session_id = user_id or "unknown"
    query = user_text.strip()
    hashed_user = hash_user_id(session_id)
    if source_order_override is not None:
        source_order = [str(value) for value in source_order_override if str(value)]
    else:
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
    intent = detect_intent(safe_query)
    log_fields = build_log_fields(session_hash=hashed_user, intent=intent)

    # 1. Rate Limit Check
    allowed, wait_time = rate_limiter.is_allowed(session_id)
    if not allowed:
        return _finalize_response(
            session_id=session_id,
            user_text=user_text,
            raw_assistant_text=f"Слишком много запросов. Пожалуйста, подождите {wait_time} сек.",
        )

    # 2. Token Budget Check
    if not token_manager.has_budget(session_id):
        return _finalize_response(
            session_id=session_id,
            user_text=user_text,
            raw_assistant_text="Исчерпан лимит токенов для вашей сессии. Пожалуйста, обратитесь в поддержку.",
        )

    if guard_action == "block":
        return _finalize_response(
            session_id=session_id,
            user_text=user_text,
            raw_assistant_text=(
                "Не могу обработать этот запрос в таком виде. "
                "Сформулируйте, пожалуйста, вопрос по сантехническим товарам."
            ),
        )

    # 3. AI Guard (input policy)
    input_guard = ai_input_policy_check(safe_query, user_id=session_id)
    if input_guard is not None:
        logger.info(
            "%s ai_guard input: decision=%s categories=%s conf=%.2f reason=%s",
            format_log_fields(log_fields),
            input_guard.decision,
            input_guard.categories,
            input_guard.confidence,
            sanitize_text(input_guard.reason_short),
            extra=log_fields,
        )
        if settings.resolved_ai_guard_mode == "enforce" and _should_enforce_policy(
            input_guard.categories, input_guard.confidence
        ):
            if input_guard.decision == "safe_reply":
                return _finalize_response(
                    session_id=session_id,
                    user_text=user_text,
                    raw_assistant_text=input_guard.safe_reply or _self_harm_safe_reply(),
                )
            if input_guard.decision == "rewrite" and input_guard.rewrite_query:
                safe_query = input_guard.rewrite_query.strip()
            if input_guard.decision == "block":
                return _finalize_response(
                    session_id=session_id,
                    user_text=user_text,
                    raw_assistant_text=_policy_refusal_text(),
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

    if not is_domain_query(safe_query.lower()):
        domain_guard = ai_domain_check(safe_query, user_id=session_id)
        if domain_guard is not None:
            logger.info(
                "%s ai_guard domain: decision=%s categories=%s conf=%.2f reason=%s",
                format_log_fields(log_fields),
                domain_guard.decision,
                domain_guard.categories,
                domain_guard.confidence,
                sanitize_text(domain_guard.reason_short),
                extra=log_fields,
            )
            if settings.resolved_ai_guard_mode == "enforce" and domain_guard.decision == "allow":
                # Proceed as domain query
                pass
            else:
                return _finalize_response(
                    session_id=session_id,
                    user_text=user_text,
                    raw_assistant_text=domain_redirect_response(),
                )
        else:
            return _finalize_response(
                session_id=session_id,
                user_text=user_text,
                raw_assistant_text=domain_redirect_response(),
            )

    history_messages = to_langchain_messages(load_messages(session_id=session_id))
    dialogue_context = build_dialogue_memory_summary(history_messages)
    context = build_context(safe_query, source_order, invoke_tool=invoke_tool, config=config, log_fields=log_fields)
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
    
    # итог. корректировка запроса, чтобы избежать переполнения контекстного окна или чрезмерных затрат
    # допускается большее значение для последнего запроса (context + history), e.g. 4x input limit
    max_prompt_len = settings.max_input_text_len * 10 
    if len(final_prompt) > max_prompt_len:
        final_prompt = f"{final_prompt[:max_prompt_len]}... [prompt truncated]"

    model_input = [SystemMessage(content=SYSTEM_PROMPT), *history_messages, HumanMessage(content=final_prompt)]

    effective_model_config = child_config(config, "model_invoke") or {}
    model_metadata = {
        "user_id": session_id,
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
        lambda payload_input: get_model(session_id).invoke(payload_input, config=effective_model_config),
        model_input,
        timeout_sec=settings.model_timeout_sec,
        breaker=model_circuit_breaker,
        pool="model",
    )
    if response.status != "ok":
        failed_fields = build_log_fields(
            session_hash=hashed_user,
            intent=intent,
            used_source=context.used_source,
            fallback_reason=context.fallback_reason,
        )
        logger.error(
            "%s model_invoke failed: %s %s",
            format_log_fields(failed_fields),
            response.error_type,
            response.error_message,
            extra=failed_fields,
        )
        return _finalize_response(
            session_id=session_id,
            user_text=user_text,
            raw_assistant_text=tool_failure_response(),
        )

    raw_assistant_text = extract_ai_text(response.value)
    assistant_text = prepare_user_answer(raw_assistant_text)
    if context.used_web:
        assistant_text = ensure_sources_block(assistant_text, context.web_urls)

    # Output PII redaction (public bot)
    pii = redact_pii(assistant_text)
    if pii.redacted:
        logger.info("%s output_pii_redacted=true", format_log_fields(log_fields), extra=log_fields)
        assistant_text = pii.text

    # 4. AI Guard (output policy) — triggered for risk/web/policy paths
    should_check_output = bool(context.used_web or risk_flags or (input_guard is not None and input_guard.decision != "allow"))
    if should_check_output:
        output_guard = ai_output_policy_check(safe_query, assistant_text, user_id=session_id)
        if output_guard is not None:
            logger.info(
                "%s ai_guard output: decision=%s categories=%s conf=%.2f reason=%s",
                format_log_fields(log_fields),
                output_guard.decision,
                output_guard.categories,
                output_guard.confidence,
                sanitize_text(output_guard.reason_short),
                extra=log_fields,
            )
            if settings.resolved_ai_guard_mode == "enforce" and _should_enforce_policy(
                output_guard.categories, output_guard.confidence
            ):
                if output_guard.decision == "redact" and output_guard.redacted_text:
                    assistant_text = output_guard.redacted_text
                elif output_guard.decision in {"rephrase_safe", "safe_reply"} and output_guard.safe_reply:
                    assistant_text = output_guard.safe_reply
                elif output_guard.decision == "block":
                    assistant_text = _policy_refusal_text()

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
