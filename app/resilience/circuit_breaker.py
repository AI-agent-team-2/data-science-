from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class CircuitBreakerOpenError(RuntimeError):
    def __init__(self, message: str, retry_after_sec: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_sec = retry_after_sec


@dataclass(frozen=True)
class _CallToken:
    half_open: bool


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker with CLOSED/OPEN/HALF_OPEN states.

    - CLOSED: calls pass through, failures are counted.
    - OPEN: calls are rejected until cooldown passes.
    - HALF_OPEN: allows limited probe calls; closes on enough successes, opens on failure.
    """

    def __init__(
        self,
        *,
        name: str,
        failure_threshold: int = 5,
        cooldown_sec: int = 30,
        half_open_success_threshold: int = 1,
        half_open_max_calls: int = 1,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._name = name
        self._failure_threshold = max(1, int(failure_threshold))
        self._cooldown_sec = max(1, int(cooldown_sec))
        self._half_open_success_threshold = max(1, int(half_open_success_threshold))
        self._half_open_max_calls = max(1, int(half_open_max_calls))
        self._clock = clock or time.monotonic

        self._lock = threading.Lock()
        self._state: str = "closed"
        self._opened_at: float = 0.0
        self._failure_count: int = 0
        self._half_open_successes: int = 0
        self._half_open_in_flight: int = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def begin_call(self) -> _CallToken:
        now = self._clock()
        with self._lock:
            if self._state == "open":
                elapsed = now - self._opened_at
                if elapsed < self._cooldown_sec:
                    retry_after = max(0.0, float(self._cooldown_sec) - elapsed)
                    raise CircuitBreakerOpenError(
                        f"circuit '{self._name}' is open (retry_after_sec={retry_after:.1f})",
                        retry_after_sec=retry_after,
                    )
                self._state = "half_open"
                self._half_open_successes = 0
                self._half_open_in_flight = 0

            if self._state == "half_open":
                if self._half_open_in_flight >= self._half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"circuit '{self._name}' is half_open (probe_in_flight)",
                        retry_after_sec=0.0,
                    )
                self._half_open_in_flight += 1
                return _CallToken(half_open=True)

            return _CallToken(half_open=False)

    def record_success(self, token: _CallToken) -> None:
        with self._lock:
            if token.half_open:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._half_open_successes += 1
                if self._half_open_successes >= self._half_open_success_threshold:
                    self._state = "closed"
                    self._failure_count = 0
                    self._half_open_successes = 0
                    self._half_open_in_flight = 0
                return

            if self._state == "closed":
                self._failure_count = 0

    def record_failure(self, token: _CallToken) -> None:
        now = self._clock()
        with self._lock:
            if token.half_open:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._state = "open"
                self._opened_at = now
                self._failure_count = 0
                self._half_open_successes = 0
                self._half_open_in_flight = 0
                return

            if self._state != "closed":
                self._state = "open"
                self._opened_at = now
                self._failure_count = 0
                self._half_open_successes = 0
                self._half_open_in_flight = 0
                return

            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                self._state = "open"
                self._opened_at = now
                self._failure_count = 0

