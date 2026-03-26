from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_langfuse_client: Any | None = None
_client_init_attempted = False
_callback_handler_class: Any | None = None
_callback_init_error: str | None = None


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
    parent_observation_id: str | None = None,
    tags: list[str] | None = None,
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
    parent_observation_id : str | None
        Идентификатор родительского observation для явной привязки callback.
    tags : list[str] | None
        Теги trace/наблюдения для callback handler.

    Returns
    -------
    Any | None
        Экземпляр callback handler или `None`, если инициализация не удалась.
    """
    global _callback_handler_class, _callback_init_error
    if not _is_enabled():
        return None

    _ = get_langfuse_client()
    try:
        if _callback_handler_class is None:
            from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler  # type: ignore

            _callback_handler_class = LangfuseCallbackHandler

        constructor_kwargs: dict[str, Any] = {}
        if trace_id:
            constructor_kwargs["trace_id"] = str(trace_id).strip()
        if session_id:
            constructor_kwargs["session_id"] = str(session_id).strip()
        if user_id:
            constructor_kwargs["user_id"] = str(user_id).strip()
        if parent_observation_id:
            constructor_kwargs["parent_observation_id"] = str(parent_observation_id).strip()
        if tags:
            constructor_kwargs["tags"] = tags

        callback_handler: Any
        if constructor_kwargs:
            try:
                callback_handler = _callback_handler_class(**constructor_kwargs)
            except TypeError:
                # Фолбэк на старые/ограниченные сигнатуры CallbackHandler.
                callback_handler = _callback_handler_class()
        else:
            callback_handler = _callback_handler_class()
        _callback_init_error = None
        logger.debug("Langfuse CallbackHandler успешно создан для запроса.")
        return callback_handler
    except Exception as exc:
        _callback_handler_class = None
        _callback_init_error = str(exc)
        logger.error(
            "Не удалось инициализировать Langfuse CallbackHandler: %s. "
            "Проверьте совместимость версий langfuse/langchain.",
            _callback_init_error,
        )
        return None


