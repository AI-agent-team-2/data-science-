from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any

from app.config import settings
from app.observability.sanitize import sanitize_payload

logger = logging.getLogger(__name__)

_langfuse_client: Any | None = None
_client_init_attempted = False
_callback_init_attempted = False
_callback_available = True
_thread_ctx = threading.local()


def _is_enabled() -> bool:
    return bool(
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    )


def get_langfuse_client() -> Any | None:
    """Возвращает singleton Langfuse client или None при отключении/ошибке."""
    global _langfuse_client, _client_init_attempted
    if _client_init_attempted:
        return _langfuse_client

    _client_init_attempted = True
    if not _is_enabled():
        logger.info("Langfuse отключен или не сконфигурирован.")
        return None

    try:
        from langfuse import Langfuse  # type: ignore

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        logger.exception("Не удалось инициализировать Langfuse, observability отключен.")
        _langfuse_client = None

    return _langfuse_client


def get_langchain_callback_handler(
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Any | None:
    """Создает callback handler для LangChain/OpenAI usage metrics."""
    global _callback_init_attempted, _callback_available
    if not _is_enabled():
        return None
    if _callback_init_attempted and not _callback_available:
        return None

    kwargs: dict[str, Any] = {
        "public_key": settings.langfuse_public_key,
        "secret_key": settings.langfuse_secret_key,
        "host": settings.langfuse_host,
    }
    if trace_id:
        kwargs["trace_id"] = trace_id
    if session_id:
        kwargs["session_id"] = session_id
    if user_id:
        kwargs["user_id"] = user_id

    try:
        from langfuse.callback import CallbackHandler as LangfuseCallbackHandler  # type: ignore
        _callback_init_attempted = True

        try:
            return LangfuseCallbackHandler(**kwargs)
        except TypeError:
            kwargs.pop("trace_id", None)
            kwargs.pop("session_id", None)
            kwargs.pop("user_id", None)
            return LangfuseCallbackHandler(**kwargs)
    except Exception:
        try:
            from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler  # type: ignore
            _callback_init_attempted = True

            try:
                return LangfuseCallbackHandler(**kwargs)
            except TypeError:
                kwargs.pop("trace_id", None)
                kwargs.pop("session_id", None)
                kwargs.pop("user_id", None)
                return LangfuseCallbackHandler(**kwargs)
        except Exception:
            _callback_init_attempted = True
            _callback_available = False
            logger.warning(
                "Langfuse callback handler недоступен для текущих версий LangChain/Langfuse. "
                "Используется ручная отправка generation для model_invoke."
            )
            return None


def _extract_usage_payload(response: Any) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    metadata = getattr(response, "response_metadata", None)
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(metadata, dict):
        token_usage = metadata.get("token_usage")
        if isinstance(token_usage, dict):
            usage["input"] = token_usage.get("prompt_tokens")
            usage["output"] = token_usage.get("completion_tokens")
            usage["total"] = token_usage.get("total_tokens")
            usage["cost"] = token_usage.get("cost")
    if isinstance(usage_metadata, dict):
        usage["input"] = usage.get("input") or usage_metadata.get("input_tokens")
        usage["output"] = usage.get("output") or usage_metadata.get("output_tokens")
        usage["total"] = usage.get("total") or usage_metadata.get("total_tokens")
    return {k: v for k, v in usage.items() if isinstance(v, (int, float))}


def capture_model_generation(
    parent: Any | None,
    model_name: str,
    input_payload: Any,
    output_payload: Any,
    response: Any,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Пишет generation для model_invoke (usage/tokens/cost) без callback-зависимости."""
    client = get_langfuse_client()
    if client is None:
        return

    usage = _extract_usage_payload(response)
    safe_input = sanitize_payload(input_payload)
    safe_output = sanitize_payload(output_payload)
    safe_metadata = sanitize_payload(metadata or {})
    safe_metadata["model"] = sanitize_payload(model_name)

    generation_kwargs: dict[str, Any] = {
        "name": "model_invoke_generation",
        "model": sanitize_payload(model_name),
        "input": safe_input,
        "output": safe_output,
        "metadata": safe_metadata,
    }
    if "input" in usage:
        generation_kwargs["usage_details"] = {
            "input": usage.get("input", 0),
            "output": usage.get("output", 0),
            "total": usage.get("total", 0),
        }
    if "cost" in usage:
        generation_kwargs["cost_details"] = {"total": usage["cost"]}

    try:
        if parent is not None:
            generated = _safe_call(parent, "generation", **generation_kwargs)
            if generated is not None:
                return

        trace_id = str(getattr(parent, "id", "") or "")
        if trace_id:
            _safe_call(client, "generation", trace_id=trace_id, **generation_kwargs)
            return
    except Exception:
        logger.exception("Не удалось записать model generation в Langfuse.")


def _safe_call(obj: Any, method_name: str, **kwargs: Any) -> Any | None:
    method = getattr(obj, method_name, None)
    if callable(method):
        return method(**kwargs)
    return None


def create_trace(
    name: str,
    session_id: str,
    input_payload: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any | None:
    """Создает trace, при недоступности Langfuse возвращает None."""
    client = get_langfuse_client()
    if client is None:
        return None

    try:
        return _safe_call(
            client,
            "trace",
            name=name,
            session_id=session_id,
            input=sanitize_payload(input_payload),
            metadata=sanitize_payload(metadata or {}),
        )
    except Exception:
        logger.exception("Не удалось создать Langfuse trace: %s", name)
        return None


def create_span(
    parent: Any | None,
    name: str,
    input_payload: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any | None:
    """Создает span как дочерний для trace/span; при ошибке возвращает None."""
    if parent is None:
        parent = get_observability_parent() or get_observability_trace()
    if parent is None:
        return None

    try:
        return _safe_call(
            parent,
            "span",
            name=name,
            input=sanitize_payload(input_payload),
            metadata=sanitize_payload(metadata or {}),
        )
    except Exception:
        logger.exception("Не удалось создать Langfuse span: %s", name)
        return None


def end_observation(
    target: Any | None,
    output_payload: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Завершает span/trace с безопасным payload."""
    if target is None:
        return

    sanitized_output = sanitize_payload(output_payload)
    sanitized_meta = sanitize_payload(metadata or {})

    try:
        ended = _safe_call(target, "end", output=sanitized_output, metadata=sanitized_meta)
        if ended is None:
            _safe_call(target, "update", output=sanitized_output, metadata=sanitized_meta)
    except Exception:
        logger.exception("Не удалось завершить observation.")


def capture_error(
    target: Any | None,
    error: Exception | str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Отправляет ошибку в Langfuse без влияния на бизнес-логику."""
    if target is None:
        return

    error_text = str(error)
    sanitized_meta = sanitize_payload(metadata or {})
    try:
        emitted = _safe_call(
            target,
            "event",
            name="error",
            level="ERROR",
            input={"message": sanitize_payload(error_text)},
            metadata=sanitized_meta,
        )
        if emitted is None:
            _safe_call(
                target,
                "update",
                status_message=sanitize_payload(error_text),
                metadata=sanitized_meta,
            )
    except Exception:
        logger.exception("Не удалось отправить ошибку в Langfuse.")


def flush_if_available() -> None:
    """Принудительно отправляет буфер observability, если клиент доступен."""
    client = get_langfuse_client()
    if client is None:
        return

    try:
        _safe_call(client, "flush")
    except Exception:
        logger.exception("Не удалось выполнить Langfuse.flush().")


def set_observability_context(trace: Any | None = None, parent: Any | None = None) -> None:
    _thread_ctx.trace = trace
    _thread_ctx.parent = parent


def get_observability_trace() -> Any | None:
    return getattr(_thread_ctx, "trace", None)


def get_observability_parent() -> Any | None:
    return getattr(_thread_ctx, "parent", None)


@contextmanager
def bind_observability_context(trace: Any | None = None, parent: Any | None = None):
    """Контекстный биндинг trace/span для текущего потока (включая ThreadPool worker)."""
    prev_trace = get_observability_trace()
    prev_parent = get_observability_parent()
    set_observability_context(trace=trace, parent=parent)
    try:
        yield
    finally:
        set_observability_context(trace=prev_trace, parent=prev_parent)
