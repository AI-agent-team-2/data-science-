from __future__ import annotations

import logging
from inspect import signature
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_langfuse_client: Any | None = None
_client_init_attempted = False
_callback_handler_class: Any | None = None
_callback_init_failed = False


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

        init_kwargs: dict[str, Any] = {
            "public_key": settings.langfuse_public_key,
            "secret_key": settings.langfuse_secret_key,
        }
        params = signature(Langfuse).parameters
        if "base_url" in params:
            init_kwargs["base_url"] = settings.langfuse_host
        elif "host" in params:
            init_kwargs["host"] = settings.langfuse_host
        _langfuse_client = Langfuse(**init_kwargs)
    except Exception:
        logger.exception("Не удалось инициализировать Langfuse, observability отключена.")
        _langfuse_client = None

    return _langfuse_client


def get_langchain_callback_handler() -> Any | None:
    """
    Возвращает singleton `CallbackHandler` для LangChain.

    Returns
    -------
    Any | None
        Экземпляр callback handler или `None`, если инициализация не удалась.
    """
    global _callback_handler_class
    if not _is_enabled():
        return None
    global _callback_init_failed
    if _callback_init_failed:
        return None

    _ = get_langfuse_client()
    try:
        if _callback_handler_class is None:
            from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler  # type: ignore

            _callback_handler_class = LangfuseCallbackHandler

        callback_handler: Any = _callback_handler_class()
        logger.debug("Langfuse CallbackHandler успешно создан для запроса.")
        return callback_handler
    except Exception as exc:
        _callback_handler_class = None
        _callback_init_failed = True
        logger.error(
            "Не удалось инициализировать Langfuse CallbackHandler: %s. "
            "Проверьте совместимость версий langfuse/langchain.",
            exc,
        )
        return None


