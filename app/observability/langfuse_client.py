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
_callback_handler: Any | None = None
_callback_init_error: str | None = None
_thread_ctx = threading.local()


def _is_enabled() -> bool:
    """Проверяет, включена ли интеграция Langfuse и заданы ли ключи."""
    return bool(
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    )


def get_langfuse_client() -> Any | None:
    """
    Возвращает singleton-клиент Langfuse.

    Returns
    -------
    Any | None
        Клиент Langfuse или `None`, если интеграция отключена/недоступна.
    """
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
        logger.exception("Не удалось инициализировать Langfuse, observability отключена.")
        _langfuse_client = None

    return _langfuse_client


def get_langchain_callback_handler(
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Any | None:
    """
    Возвращает singleton `CallbackHandler` для LangChain.

    Parameters
    ----------
    trace_id : str | None
        Аргумент сохранен для совместимости сигнатуры.
    session_id : str | None
        Аргумент сохранен для совместимости сигнатуры.
    user_id : str | None
        Аргумент сохранен для совместимости сигнатуры.

    Returns
    -------
    Any | None
        Экземпляр callback handler или `None`, если инициализация не удалась.
    """
    _ = (trace_id, session_id, user_id)
    global _callback_handler, _callback_init_attempted, _callback_init_error
    if not _is_enabled():
        return None
    if _callback_init_attempted:
        return _callback_handler

    _callback_init_attempted = True
    _ = get_langfuse_client()
    try:
        from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler  # type: ignore

        _callback_handler = LangfuseCallbackHandler()
        _callback_init_error = None
        logger.debug("Langfuse CallbackHandler успешно инициализирован.")
        return _callback_handler
    except Exception as exc:
        _callback_handler = None
        _callback_init_error = str(exc)
        logger.error(
            "Не удалось инициализировать Langfuse CallbackHandler: %s. "
            "Проверьте совместимость версий langfuse/langchain.",
            _callback_init_error,
        )
        return None


def get_callback_init_error() -> str | None:
    """
    Возвращает последнюю ошибку инициализации callback.

    Returns
    -------
    str | None
        Текст ошибки или `None`, если ошибок не было.
    """
    return _callback_init_error


def _safe_call(obj: Any, method_name: str, **kwargs: Any) -> Any | None:
    """Безопасно вызывает метод объекта по имени."""
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
    """
    Создает trace в Langfuse.

    Returns
    -------
    Any | None
        Объект trace или `None`, если создать trace не удалось.
    """
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
    """
    Создает дочерний span для текущего trace/span.

    Returns
    -------
    Any | None
        Объект span или `None` при ошибке.
    """
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
    """Завершает span или trace с очищенным payload."""
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
    """Отправляет информацию об ошибке в Langfuse без влияния на бизнес-логику."""
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
    """Сохраняет текущий trace и родительский span в thread-local контексте."""
    _thread_ctx.trace = trace
    _thread_ctx.parent = parent


def get_observability_trace() -> Any | None:
    """Возвращает текущий trace из thread-local контекста."""
    return getattr(_thread_ctx, "trace", None)


def get_observability_parent() -> Any | None:
    """Возвращает текущий родительский span из thread-local контекста."""
    return getattr(_thread_ctx, "parent", None)


@contextmanager
def bind_observability_context(trace: Any | None = None, parent: Any | None = None):
    """Временно привязывает trace/span к текущему потоку выполнения."""
    prev_trace = get_observability_trace()
    prev_parent = get_observability_parent()
    set_observability_context(trace=trace, parent=parent)
    try:
        yield
    finally:
        set_observability_context(trace=prev_trace, parent=prev_parent)
