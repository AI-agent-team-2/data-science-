from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

from app.config import settings
from app.observability.token_usage import token_manager
from app.resilience.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class TokenTrackingCallbackHandler(BaseCallbackHandler):
    """Обработчик для отслеживания использования токенов."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Вызывается при завершении работы LLM."""
        for generation in response.generations:
            for chunk in generation:
                info = chunk.generation_info
                if info and "token_usage" in info:
                    usage = info["token_usage"]
                    prompt = usage.get("prompt_tokens", 0)
                    completion = usage.get("completion_tokens", 0)
                    token_manager.update_usage(prompt, completion)
                    logger.debug(f"Tokens updated: +{prompt} prompt, +{completion} completion")


def create_chat_model() -> ChatOpenAI:
    """
    Создает основной LLM-клиент приложения.

    Returns
    -------
    ChatOpenAI
        Инициализированный клиент чата.
    """
    if not token_manager.has_budget():
        logger.error("Token budget exceeded before LLM call")
        raise RuntimeError("Token budget exceeded. Please contact support.")

    return ChatOpenAI(
        model=settings.resolved_model_name,
        temperature=0,
        api_key=settings.resolved_openai_api_key,
        base_url=settings.resolved_openai_base_url,
        timeout=max(1, settings.model_timeout_sec),
        max_retries=max(0, settings.model_max_retries),
        callbacks=[TokenTrackingCallbackHandler()],
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
