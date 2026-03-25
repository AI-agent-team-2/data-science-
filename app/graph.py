from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from app.config import settings
from app.observability.langfuse_client import get_langchain_callback_handler


def create_chat_model() -> ChatOpenAI:
    """Создает и возвращает основной LLM-клиент приложения."""
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
    """Возвращает config для model.invoke с Langfuse callbacks (если включено)."""
    callback_handler = get_langchain_callback_handler(trace_id=trace_id, session_id=session_id, user_id=user_id)
    if callback_handler is None:
        return None

    config: dict[str, Any] = {"callbacks": [callback_handler]}
    langfuse_metadata: dict[str, Any] = dict(metadata or {})
    if session_id:
        langfuse_metadata["langfuse_session_id"] = session_id
    if user_id:
        langfuse_metadata["langfuse_user_id"] = user_id
    if tags:
        langfuse_metadata["langfuse_tags"] = tags

    if langfuse_metadata:
        config["metadata"] = langfuse_metadata
    if run_name:
        config["run_name"] = run_name
    return config


model = create_chat_model()
