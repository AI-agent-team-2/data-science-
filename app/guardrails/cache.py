from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    value: T
    expires_at: float


class TtlCache(Generic[T]):
    """Простой in-memory TTL cache с ограничением по размеру."""

    def __init__(self, *, ttl_sec: int, max_items: int = 5000) -> None:
        self._ttl_sec = max(1, int(ttl_sec))
        self._max_items = max(1, int(max_items))
        self._lock = Lock()
        self._data: dict[str, _Entry[T]] = {}
        self._ops = 0

    def _now(self) -> float:
        return time.monotonic()

    def _prune_locked(self, now: float) -> None:
        expired = [k for k, v in self._data.items() if v.expires_at <= now]
        for k in expired:
            self._data.pop(k, None)

        if len(self._data) <= self._max_items:
            return

        # Best-effort size prune: drop arbitrary oldest-ish keys.
        for key in list(self._data.keys())[: max(1, len(self._data) - self._max_items)]:
            self._data.pop(key, None)

    def get(self, key: str) -> T | None:
        now = self._now()
        with self._lock:
            self._ops += 1
            if self._ops % 200 == 0:
                self._prune_locked(now)
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._data.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: T) -> None:
        now = self._now()
        with self._lock:
            self._ops += 1
            if self._ops % 200 == 0:
                self._prune_locked(now)
            self._data[key] = _Entry(value=value, expires_at=now + self._ttl_sec)

