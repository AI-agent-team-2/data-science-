from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import settings


def create_chat_model(model_name: str | None = None) -> ChatOpenAI:
    """Создает LLM-клиент для указанной модели (или основной модели из настроек)."""
    resolved_model_name = model_name or settings.resolved_model_name
    return ChatOpenAI(
        model=resolved_model_name,
        temperature=0,
        api_key=settings.resolved_openai_api_key,
        base_url=settings.resolved_openai_base_url,
    )


model = create_chat_model()
