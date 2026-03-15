from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import settings

# Единая точка инициализации LLM-клиента.
# Основной pipeline приложения реализован в app/run_agent.py.
model = ChatOpenAI(
    model=settings.resolved_model_name,
    temperature=0,
    api_key=settings.resolved_openai_api_key,
    base_url=settings.resolved_openai_base_url,
)
