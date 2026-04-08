from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

from app.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    """Статистика использования токенов."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += (prompt + completion)

class TokenBudgetManager:
    """Менеджер бюджета токенов."""
    
    def __init__(self, settings: Settings):
        self.limit: Final[int] = settings.max_total_token_budget
        self.warning_threshold: Final[float] = settings.token_budget_warning_threshold
        self.usage = TokenUsage()

    def update_usage(self, prompt: int, completion: int) -> None:
        """Обновляет текущее использование и логирует превышение порогов."""
        self.usage.add(prompt, completion)
        
        if self.usage.total_tokens >= self.limit:
            logger.error(f"TOKEN BUDGET EXCEEDED: {self.usage.total_tokens}/{self.limit}")
        elif self.usage.total_tokens >= self.limit * self.warning_threshold:
            logger.warning(f"TOKEN BUDGET WARNING: {self.usage.total_tokens}/{self.limit}")

    def has_budget(self, estimated_needed: int = 500) -> bool:
        """Проверяет, достаточно ли бюджета для следующего запроса."""
        return (self.usage.total_tokens + estimated_needed) < self.limit

    @property
    def remaining(self) -> int:
        """Возвращает остаток бюджета."""
        return max(0, self.limit - self.usage.total_tokens)

# Инициализируем глобальный менеджер (в реальном приложении может быть привязан к сессии)
from app.config import settings
token_manager = TokenBudgetManager(settings)
