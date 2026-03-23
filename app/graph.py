from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import settings


def create_chat_model() -> ChatOpenAI:
    """Создает и возвращает основной LLM-клиент приложения."""
    return ChatOpenAI(
        model=settings.resolved_model_name,
        temperature=0,
        api_key=settings.resolved_openai_api_key,
        base_url=settings.resolved_openai_base_url,
    )


model = create_chat_model()
