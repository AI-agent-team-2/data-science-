from __future__ import annotations

import logging
from typing import Any
from threading import Lock
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

from app.config import settings
from app.observability.token_usage import token_manager
from app.resilience.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class TokenTrackingCallbackHandler(BaseCallbackHandler):
    """Обработчик для отслеживания использования токенов."""

    def __init__(self) -> None:
        super().__init__()
        self._prompt_token_estimates: dict[str, int] = {}
        self._lock: Lock = Lock()
        self._max_tracked_runs: int = 2000

    def _estimate_tokens(self, text: str) -> int:
        """
        Оценивает количество токенов в тексте.
        
        Используется эвристика: 1 токен ≈ 4 символа для английского языка.
        Для русского языка погрешность выше (1 токен ≈ 1.5-2 символа).
        
        Условия, когда провайдер может не вернуть usage:
        1. Использование Streaming (потоковая передача) без поддержки usage в финальном чанке.
        2. Ошибки на стороне провайдера или прокси (например, OpenRouter).
        3. Использование локальных моделей через несовместимые API.
        """
        cleaned = str(text or "")
        if not cleaned:
            return 0
        # Базовая эвристика len/4. Для кириллицы это даст занижение в 2-3 раза,
        # что является допустимым компромиссом для "безопасной" оценки бюджета.
        return max(1, len(cleaned) // 4)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            parts: list[str] = []
            for turn in messages or []:
                for msg in turn or []:
                    content = getattr(msg, "content", "")
                    if isinstance(content, str):
                        parts.append(content)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and isinstance(item.get("text"), str):
                                parts.append(item["text"])
            estimate = self._estimate_tokens("\n".join(parts))
        except Exception:
            return

        key = str(run_id)
        with self._lock:
            self._prompt_token_estimates[key] = estimate
            if len(self._prompt_token_estimates) > self._max_tracked_runs:
                self._prompt_token_estimates.pop(next(iter(self._prompt_token_estimates)), None)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Вызывается при завершении работы LLM."""
        run_metadata = kwargs.get("metadata", {})
        user_id = run_metadata.get("user_id", "unknown")

        # 1. Пытаемся найти usage в общем выводе (стандарт для многих провайдеров)
        usage = None
        if response.llm_output and isinstance(response.llm_output, dict):
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")

        # 2. Если в корне нет, ищем в чанках/сообщениях (для разных провайдеров/версий)
        if not usage:
            for generation in response.generations:
                for chunk in generation:
                    message = getattr(chunk, "message", None)
                    if message is not None:
                        meta = getattr(message, "response_metadata", None)
                        if isinstance(meta, dict):
                            usage = meta.get("token_usage") or meta.get("usage")
                            if usage:
                                break
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
            run_id = kwargs.get("run_id")
            run_id_key = str(run_id) if run_id is not None else ""
            prompt_estimate = 0
            if run_id_key:
                with self._lock:
                    prompt_estimate = int(self._prompt_token_estimates.pop(run_id_key, 0) or 0)

            completion_text = ""
            try:
                for generation in response.generations:
                    for chunk in generation:
                        message = getattr(chunk, "message", None)
                        content = getattr(message, "content", "") if message is not None else ""
                        if isinstance(content, str) and content.strip():
                            completion_text = content
                            break
                    if completion_text:
                        break
            except Exception:
                completion_text = ""

            completion_estimate = self._estimate_tokens(completion_text)
            if prompt_estimate or completion_estimate:
                token_manager.update_usage(user_id, prompt_estimate, completion_estimate)
                logger.warning(
                    "Token usage missing for %s; used estimates prompt=%s completion=%s",
                    user_id,
                    prompt_estimate,
                    completion_estimate,
                )
            else:
                logger.warning("Could not find token usage in LLM response for %s", user_id)


def create_chat_model(
    user_id: str = "unknown",
    *,
    model_name: str | None = None,
    timeout_sec: int | None = None,
    max_retries: int | None = None,
) -> ChatOpenAI:
    """
    Создает основной LLM-клиент приложения.

    Returns
    -------
    ChatOpenAI
        Инициализированный клиент чата.
    """
    return ChatOpenAI(
        model=model_name or settings.resolved_model_name,
        temperature=0,
        api_key=settings.resolved_openai_api_key,
        base_url=settings.resolved_openai_base_url,
        timeout=max(1, int(timeout_sec or settings.model_timeout_sec)),
        max_retries=max(0, int(max_retries if max_retries is not None else settings.model_max_retries)),
        callbacks=[TokenTrackingCallbackHandler()],
    )


_model: ChatOpenAI | None = None
_model_lock: Lock = Lock()


def get_model(user_id: str = "unknown") -> ChatOpenAI:
    """Возвращает singleton LLM-клиент без сайд-эффектов на импорт."""
    if not token_manager.has_budget(user_id):
        logger.error("Token budget exceeded for %s before LLM call", user_id)
        raise RuntimeError("Token budget exceeded. Please contact support.")

    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            _model = create_chat_model(user_id=user_id)
        return _model


_guard_model: ChatOpenAI | None = None
_guard_model_lock: Lock = Lock()


def get_guard_model(user_id: str = "unknown") -> ChatOpenAI:
    """Возвращает singleton LLM-клиент для AI guard."""
    global _guard_model
    if _guard_model is not None:
        return _guard_model

    with _guard_model_lock:
        if _guard_model is None:
            _guard_model = create_chat_model(
                user_id=user_id,
                model_name=settings.resolved_ai_guard_model_name,
                timeout_sec=max(1, int(settings.ai_guard_timeout_sec)),
                max_retries=0,
            )
        return _guard_model


def create_model_circuit_breaker() -> CircuitBreaker:
    return CircuitBreaker(
        name="llm_api",
        failure_threshold=settings.model_circuit_breaker_failure_threshold,
        cooldown_sec=settings.model_circuit_breaker_cooldown_sec,
        half_open_success_threshold=settings.model_circuit_breaker_half_open_success_threshold,
        half_open_max_calls=1,
    )


model_circuit_breaker = create_model_circuit_breaker() if settings.model_circuit_breaker_enabled else None
