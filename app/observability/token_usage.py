from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Final

from app.config import Settings, settings

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
        self.global_limit: Final[int] = settings.max_total_token_budget
        self.user_limit: Final[int] = settings.max_user_token_budget
        self.warning_threshold: Final[float] = settings.token_budget_warning_threshold
        self.global_usage = TokenUsage()
        self.user_usage: dict[str, TokenUsage] = {}
        self._lock = Lock()

    def update_usage(self, user_id: str, prompt: int, completion: int) -> None:
        """Обновляет использование для пользователя и глобально."""
        with self._lock:
            # Глобальный учет
            self.global_usage.add(prompt, completion)
            if self.global_usage.total_tokens >= self.global_limit:
                logger.error(f"GLOBAL TOKEN BUDGET EXCEEDED: {self.global_usage.total_tokens}/{self.global_limit}")
            elif self.global_usage.total_tokens >= self.global_limit * self.warning_threshold:
                logger.warning(f"GLOBAL TOKEN BUDGET WARNING: {self.global_usage.total_tokens}/{self.global_limit}")

            # Пользовательский учет
            if user_id not in self.user_usage:
                self.user_usage[user_id] = TokenUsage()
            
            u_usage = self.user_usage[user_id]
            u_usage.add(prompt, completion)
            
            if u_usage.total_tokens >= self.user_limit:
                logger.error(f"USER TOKEN BUDGET EXCEEDED for {user_id}: {u_usage.total_tokens}/{self.user_limit}")
            elif u_usage.total_tokens >= self.user_limit * self.warning_threshold:
                logger.warning(f"USER TOKEN BUDGET WARNING for {user_id}: {u_usage.total_tokens}/{self.user_limit}")

    def has_budget(self, user_id: str, estimated_needed: int = 500) -> bool:
        """Проверяет, достаточно ли бюджета (глобального и пользовательского)."""
        with self._lock:
            global_ok = (self.global_usage.total_tokens + estimated_needed) < self.global_limit
            
            u_usage = self.user_usage.get(user_id)
            user_ok = True
            if u_usage:
                user_ok = (u_usage.total_tokens + estimated_needed) < self.user_limit
            else:
                user_ok = estimated_needed < self.user_limit

            return global_ok and user_ok

    def get_user_remaining(self, user_id: str) -> int:
        """Возвращает остаток бюджета пользователя."""
        with self._lock:
            u_usage = self.user_usage.get(user_id)
            if not u_usage:
                return self.user_limit
            return max(0, self.user_limit - u_usage.total_tokens)

# Инициализируем глобальный менеджер (в реальном приложении может быть привязан к сессии)
from app.config import settings
token_manager = TokenBudgetManager(settings)
