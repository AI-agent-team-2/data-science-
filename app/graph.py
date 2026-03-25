from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from langchain_openai import ChatOpenAI

from app.config import settings
from app.observability.langfuse_client import get_langchain_callback_handler

logger = logging.getLogger(__name__)


def _normalize_run_id(trace_id: str | None) -> str | None:
    """
    Нормализует trace_id для передачи в `config.run_id`.

    Parameters
    ----------
    trace_id : str | None
        Идентификатор trace, полученный из manual tracing.

    Returns
    -------
    str | None
        Валидный UUID-идентификатор или `None`.
    """
    if not trace_id:
        return None

    raw = str(trace_id).strip()
    if not raw:
        return None

    try:
        return str(UUID(raw))
    except ValueError:
        logger.debug("trace_id не является UUID и не будет передан в run_id: %s", raw)
        return None


def create_chat_model() -> ChatOpenAI:
    """
    Создает основной LLM-клиент приложения.

    Returns
    -------
    ChatOpenAI
        Инициализированный клиент чата.
    """
    return ChatOpenAI(
        model=settings.resolved_model_name,
        temperature=0,
        api_key=settings.resolved_openai_api_key,
        base_url=settings.resolved_openai_base_url,
    )


def build_model_invoke_config(
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    run_name: str | None = None,
) -> dict[str, Any] | None:
    """
    Формирует config для `model.invoke` с callback Langfuse.

    Parameters
    ----------
    trace_id : str | None
        Идентификатор trace для совместимости сигнатуры.
    session_id : str | None
        Идентификатор сессии для Langfuse.
    user_id : str | None
        Идентификатор пользователя для Langfuse.
    tags : list[str] | None
        Теги trace.
    metadata : dict[str, Any] | None
        Дополнительные метаданные вызова модели.
    run_name : str | None
        Имя запуска в LangChain.

    Returns
    -------
    dict[str, Any] | None
        Словарь config для LangChain или `None`, если callback недоступен.
    """
    callback_handler = get_langchain_callback_handler(trace_id=trace_id, session_id=session_id, user_id=user_id)
    if callback_handler is None:
        return None

    config: dict[str, Any] = {"callbacks": [callback_handler]}
    run_id = _normalize_run_id(trace_id)
    if run_id:
        config["run_id"] = run_id

    langfuse_metadata: dict[str, Any] = dict(metadata or {})
    if session_id:
        langfuse_metadata["langfuse_session_id"] = session_id
    if user_id:
        langfuse_metadata["langfuse_user_id"] = user_id
    if tags:
        langfuse_metadata["langfuse_tags"] = tags
    if trace_id:
        langfuse_metadata["langfuse_trace_id"] = str(trace_id).strip()

    if langfuse_metadata:
        config["metadata"] = langfuse_metadata
    if run_name:
        config["run_name"] = run_name
    return config


model = create_chat_model()
