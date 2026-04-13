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
        run_metadata = kwargs.get("metadata", {})
        user_id = run_metadata.get("user_id", "unknown")

        # 1. Пытаемся найти usage в общем выводе (стандарт для многих провайдеров)
        usage = None
        if response.llm_output and isinstance(response.llm_output, dict):
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")

        # 2. Если в корне нет, ищем в чанках (для некоторых провайдеров/стриминга)
        if not usage:
            for generation in response.generations:
                for chunk in generation:
                    info = chunk.generation_info
                    if info and ("token_usage" in info or "usage" in info):
                        usage = info.get("token_usage") or info.get("usage")
                        break
                if usage:
                    break

        if usage and isinstance(usage, dict):
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            token_manager.update_usage(user_id, prompt, completion)
            logger.debug(f"Tokens updated for {user_id}: +{prompt} prompt, +{completion} completion")
        else:
            logger.warning(f"Could not find token usage in LLM response for {user_id}")


def create_chat_model(user_id: str = "unknown") -> ChatOpenAI:
    """
    Создает основной LLM-клиент приложения.

    Returns
    -------
    ChatOpenAI
        Инициализированный клиент чата.
    """
    if not token_manager.has_budget(user_id):
        logger.error(f"Token budget exceeded for {user_id} before LLM call")
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
