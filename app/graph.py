from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from app.config import settings
from app.resilience.circuit_breaker import CircuitBreaker

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
        timeout=max(1, settings.model_timeout_sec),
        max_retries=max(0, settings.model_max_retries),
    )


model = create_chat_model()


def create_model_circuit_breaker() -> CircuitBreaker:
    return CircuitBreaker(
        name="llm_api",
        failure_threshold=settings.model_circuit_breaker_failure_threshold,
        cooldown_sec=settings.model_circuit_breaker_cooldown_sec,
        half_open_success_threshold=settings.model_circuit_breaker_half_open_success_threshold,
        half_open_max_calls=1,
    )


model_circuit_breaker = create_model_circuit_breaker() if settings.model_circuit_breaker_enabled else None
