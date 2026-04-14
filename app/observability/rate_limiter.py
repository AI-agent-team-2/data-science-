from __future__ import annotations

import time
from collections import OrderedDict, defaultdict
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
    max_users: int = 50000
    user_ttl_sec: int = 24 * 60 * 60
    prune_every: int = 200

    _requests: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _last_seen: "OrderedDict[str, float]" = field(default_factory=OrderedDict)
    _lock: Lock = field(default_factory=Lock, repr=False)
    _ops: int = 0

    def _prune_locked(self, now: float) -> None:
        if not self._last_seen:
            return

        ttl = float(max(1, int(self.user_ttl_sec)))
        # TTL prune (oldest first)
        while self._last_seen:
            user_id, last_seen = next(iter(self._last_seen.items()))
            if now - last_seen <= ttl:
                break
            self._last_seen.popitem(last=False)
            if user_id in self._requests:
                del self._requests[user_id]

        # Size bound (LRU-ish)
        max_users = max(1, int(self.max_users))
        while len(self._last_seen) > max_users:
            user_id, _ = self._last_seen.popitem(last=False)
            if user_id in self._requests:
                del self._requests[user_id]

    def is_allowed(self, user_id: str) -> tuple[bool, int]:
        """
        Проверяет, может ли пользователь сделать запрос.

        Returns:
            tuple[bool, int]: (разрешено, время ожидания в секундах если запрещено)
        """
        now = time.time()
        normalized_user = str(user_id or "unknown")
        with self._lock:
            self._ops += 1
            if self._ops % max(1, int(self.prune_every)) == 0:
                self._prune_locked(now)

            self._last_seen[normalized_user] = now
            self._last_seen.move_to_end(normalized_user)

            user_requests = self._requests[normalized_user]

            # Удалить старые записи
            user_requests[:] = [t for t in user_requests if now - t < self.window]

            if len(user_requests) >= self.limit:
                wait_time = int(self.window - (now - user_requests[0]))
                return False, max(1, wait_time)

            user_requests.append(now)
            return True, 0

    def reset(self, user_id: str) -> None:
        """Сбрасывает лимиты для пользователя."""
        normalized_user = str(user_id or "unknown")
        with self._lock:
            if normalized_user in self._requests:
                del self._requests[normalized_user]
            if normalized_user in self._last_seen:
                del self._last_seen[normalized_user]


# Единый экземпляр для всего приложения
rate_limiter = RateLimiter(
    limit=settings.rate_limit_requests,
    window=settings.rate_limit_window_sec,
    max_users=settings.rate_limit_max_users,
    user_ttl_sec=settings.rate_limit_user_ttl_sec,
    prune_every=settings.rate_limit_prune_every,
)
