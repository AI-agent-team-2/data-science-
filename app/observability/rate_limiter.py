from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Final

from app.config import settings

DEFAULT_RATE_LIMIT_REQUESTS: Final[int] = 10
DEFAULT_RATE_LIMIT_WINDOW_SEC: Final[int] = 60


@dataclass
class RateLimiter:
    """Простой in-memory rate limiter для защиты от слишком частых запросов."""

    limit: int = DEFAULT_RATE_LIMIT_REQUESTS
    window: int = DEFAULT_RATE_LIMIT_WINDOW_SEC

    _requests: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: Lock = field(default_factory=Lock, repr=False)

    def is_allowed(self, user_id: str) -> tuple[bool, int]:
        """
        Проверяет, может ли пользователь сделать запрос.

        Returns:
            tuple[bool, int]: (разрешено, время ожидания в секундах если запрещено)
        """
        now = time.time()
        with self._lock:
            user_requests = self._requests[user_id]

            # Удалить старые записи
            user_requests[:] = [t for t in user_requests if now - t < self.window]

            if len(user_requests) >= self.limit:
                wait_time = int(self.window - (now - user_requests[0]))
                return False, max(1, wait_time)

            user_requests.append(now)
            return True, 0

    def reset(self, user_id: str) -> None:
        """Сбрасывает лимиты для пользователя."""
        with self._lock:
            if user_id in self._requests:
                del self._requests[user_id]


# Единый экземпляр для всего приложения
rate_limiter = RateLimiter(
    limit=settings.rate_limit_requests,
    window=settings.rate_limit_window_sec,
)
