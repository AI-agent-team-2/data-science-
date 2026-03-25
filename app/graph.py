from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


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


model = create_chat_model()
