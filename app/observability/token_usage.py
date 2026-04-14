from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Final
from collections import OrderedDict

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


@dataclass
class _UserUsageRecord:
    usage: TokenUsage
    last_seen: float


class TokenBudgetManager:
    """Менеджер бюджета токенов."""
    
    def __init__(self, settings: Settings):
        self.global_limit: Final[int] = settings.max_total_token_budget
        self.user_limit: Final[int] = settings.max_user_token_budget
        self.warning_threshold: Final[float] = settings.token_budget_warning_threshold
        self.max_users: Final[int] = max(1, int(settings.token_budget_max_users))
        self.user_ttl_sec: Final[int] = max(1, int(settings.token_budget_user_ttl_sec))
        self.prune_every: Final[int] = max(1, int(settings.token_budget_prune_every))
        self.global_usage = TokenUsage()
        self._records: "OrderedDict[str, _UserUsageRecord]" = OrderedDict()
        self._lock: Lock = Lock()
        self._ops: int = 0

    def _now(self) -> float:
        return time.monotonic()

    def _prune_locked(self, now: float) -> None:
        if not self._records:
            return

        # TTL prune (oldest first)
        ttl = float(self.user_ttl_sec)
        while self._records:
            user_id, record = next(iter(self._records.items()))
            if now - record.last_seen <= ttl:
                break
            self._records.popitem(last=False)

        # Size bound (LRU)
        while len(self._records) > self.max_users:
            self._records.popitem(last=False)

    def update_usage(self, user_id: str, prompt: int, completion: int) -> None:
        """Обновляет использование для пользователя и глобально."""
        normalized_user = str(user_id or "unknown")
        prompt = max(0, int(prompt))
        completion = max(0, int(completion))
        now = self._now()

        with self._lock:
            self._ops += 1
            if self._ops % self.prune_every == 0:
                self._prune_locked(now)

            # Глобальный учет
            self.global_usage.add(prompt, completion)
            global_total = self.global_usage.total_tokens

            # Пользовательский учет (LRU)
            record = self._records.get(normalized_user)
            if record is None:
                record = _UserUsageRecord(usage=TokenUsage(), last_seen=now)
                self._records[normalized_user] = record
            record.last_seen = now
            self._records.move_to_end(normalized_user)

            record.usage.add(prompt, completion)
            user_total = record.usage.total_tokens

        # Logging outside lock
        if global_total >= self.global_limit:
            logger.error("GLOBAL TOKEN BUDGET EXCEEDED: %s/%s", global_total, self.global_limit)
        elif global_total >= int(self.global_limit * self.warning_threshold):
            logger.warning("GLOBAL TOKEN BUDGET WARNING: %s/%s", global_total, self.global_limit)

        if user_total >= self.user_limit:
            logger.error("USER TOKEN BUDGET EXCEEDED for %s: %s/%s", normalized_user, user_total, self.user_limit)
        elif user_total >= int(self.user_limit * self.warning_threshold):
            logger.warning("USER TOKEN BUDGET WARNING for %s: %s/%s", normalized_user, user_total, self.user_limit)

    def has_budget(self, user_id: str, estimated_needed: int = 500) -> bool:
        """Проверяет, достаточно ли бюджета (глобального и пользовательского)."""
        normalized_user = str(user_id or "unknown")
        estimated_needed = max(0, int(estimated_needed))
        now = self._now()

        with self._lock:
            self._ops += 1
            if self._ops % self.prune_every == 0:
                self._prune_locked(now)

            global_ok = (self.global_usage.total_tokens + estimated_needed) < self.global_limit
            record = self._records.get(normalized_user)
            if record is None:
                user_ok = estimated_needed < self.user_limit
            else:
                record.last_seen = now
                self._records.move_to_end(normalized_user)
                user_ok = (record.usage.total_tokens + estimated_needed) < self.user_limit

        return bool(global_ok and user_ok)

    def get_user_remaining(self, user_id: str) -> int:
        """Возвращает остаток бюджета пользователя."""
        normalized_user = str(user_id or "unknown")
        now = self._now()
        with self._lock:
            self._ops += 1
            if self._ops % self.prune_every == 0:
                self._prune_locked(now)

            record = self._records.get(normalized_user)
            if record is None:
                return self.user_limit
            record.last_seen = now
            self._records.move_to_end(normalized_user)
            return max(0, self.user_limit - record.usage.total_tokens)

# Инициализируем глобальный менеджер (в реальном приложении может быть привязан к сессии)
from app.config import settings
token_manager = TokenBudgetManager(settings)
