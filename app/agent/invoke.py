from __future__ import annotations

import atexit
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextvars import copy_context
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable
from functools import lru_cache

from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.context_engine import ToolExecutionResult, parse_tool_payload
from app.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

logger = logging.getLogger(__name__)

DEFAULT_INVOKE_MAX_WORKERS = 8
DEFAULT_INVOKE_MAX_QUEUE = 64


@dataclass
class _PoolCounters:
    max_workers: int
    max_queue: int

    submitted: int = 0
    rejected: int = 0
    completed: int = 0
    exceptions: int = 0
    cancelled: int = 0
    timed_out: int = 0

    pending: int = 0  # running + queued
    running: int = 0


class _InvokePool:
    def __init__(self, *, name: str, max_workers: int, max_queue: int) -> None:
        self.name = name
        self.max_workers = max(1, int(max_workers))
        self.max_queue = max(0, int(max_queue))

        self._pending_limit = self.max_workers + self.max_queue
        self._pending_sem = threading.BoundedSemaphore(self._pending_limit)
        self._lock = threading.Lock()
        self._counters = _PoolCounters(max_workers=self.max_workers, max_queue=self.max_queue)

        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=f"sanbot-{name}",
        )

        def _shutdown() -> None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                return

        atexit.register(_shutdown)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            pending = self._counters.pending
            running = self._counters.running
            queued = max(0, pending - running)
            return {
                "max_workers": self._counters.max_workers,
                "max_queue": self._counters.max_queue,
                "pending": pending,
                "running": running,
                "queued": queued,
                "submitted": self._counters.submitted,
                "rejected": self._counters.rejected,
                "completed": self._counters.completed,
                "exceptions": self._counters.exceptions,
                "cancelled": self._counters.cancelled,
                "timed_out": self._counters.timed_out,
            }

    def record_timeout(self) -> None:
        with self._lock:
            self._counters.timed_out += 1

    def try_submit(self, fn: Callable[[], Any]) -> tuple[bool, Any]:
        if not self._pending_sem.acquire(blocking=False):
            with self._lock:
                self._counters.rejected += 1
            return (False, None)

        with self._lock:
            self._counters.submitted += 1
            self._counters.pending += 1

        def _wrapped() -> Any:
            with self._lock:
                self._counters.running += 1
            try:
                return fn()
            finally:
                with self._lock:
                    self._counters.running -= 1

        future = self._executor.submit(_wrapped)

        def _done_callback(fut) -> None:
            try:
                if fut.cancelled():
                    with self._lock:
                        self._counters.cancelled += 1
                else:
                    exc = fut.exception()
                    if exc is not None:
                        with self._lock:
                            self._counters.exceptions += 1
            except Exception:
                # Don't let metrics bookkeeping crash the worker.
                pass
            finally:
                with self._lock:
                    self._counters.completed += 1
                    self._counters.pending -= 1
                try:
                    self._pending_sem.release()
                except Exception:
                    pass

        future.add_done_callback(_done_callback)
        return (True, future)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_workers(value: int, default: int) -> int:
    value = _to_int(value, default)
    return max(1, min(64, value))


def _normalize_queue(value: int, default: int) -> int:
    try:
        value = int(value)
    except Exception:
        value = default
    return max(0, min(4096, value))


def _override_workers(raw_value: Any) -> int:
    value = _to_int(raw_value, 0)
    return value if value > 0 else 0


def _override_queue(raw_value: Any) -> int:
    value = _to_int(raw_value, 0)
    return value


def _resolve_pool_params(kind: str) -> tuple[str, int, int]:
    total_workers = _normalize_workers(getattr(settings, "invoke_max_workers", DEFAULT_INVOKE_MAX_WORKERS), DEFAULT_INVOKE_MAX_WORKERS)
    default_queue = _normalize_queue(getattr(settings, "invoke_max_queue", DEFAULT_INVOKE_MAX_QUEUE), DEFAULT_INVOKE_MAX_QUEUE)

    model_override = _override_workers(getattr(settings, "invoke_model_max_workers", 0))
    tool_override = _override_workers(getattr(settings, "invoke_tool_max_workers", 0))

    model_queue_override = _override_queue(getattr(settings, "invoke_model_max_queue", -1))
    tool_queue_override = _override_queue(getattr(settings, "invoke_tool_max_queue", -1))

    has_overrides = any(v > 0 for v in (model_override, tool_override)) or any(
        v >= 0 for v in (model_queue_override, tool_queue_override)
    )

    # Если всего 1 поток и override'ов нет — не плодим лишние пулы.
    if not has_overrides and total_workers < 2:
        return ("shared", total_workers, default_queue)

    if kind == "model":
        max_workers = model_override if model_override > 0 else max(1, int(total_workers * 0.25))
        max_queue = model_queue_override if model_queue_override >= 0 else default_queue
        return ("model", max_workers, max_queue)

    if kind == "tool":
        if tool_override > 0:
            max_workers = tool_override
        else:
            model_workers = model_override if model_override > 0 else max(1, int(total_workers * 0.25))
            max_workers = max(1, total_workers - model_workers)
        max_queue = tool_queue_override if tool_queue_override >= 0 else default_queue
        return ("tool", max_workers, max_queue)

    # Fallback: shared
    return ("shared", total_workers, default_queue)


@lru_cache(maxsize=3)
def _get_pool_by_name(name: str) -> _InvokePool:
    _, max_workers, max_queue = _resolve_pool_params(name)
    max_workers = _normalize_workers(max_workers, DEFAULT_INVOKE_MAX_WORKERS)
    max_queue = _normalize_queue(max_queue, DEFAULT_INVOKE_MAX_QUEUE)
    return _InvokePool(name=name, max_workers=max_workers, max_queue=max_queue)


def _get_pool(kind: str) -> _InvokePool:
    name, _, _ = _resolve_pool_params(kind)
    return _get_pool_by_name(name)


def invoke_pool_metrics() -> dict[str, dict[str, int]]:
    """Снимок метрик пулов внешних вызовов (model/tool/shared)."""
    pools: dict[str, dict[str, int]] = {}
    for kind in ("model", "tool", "shared"):
        pool = _get_pool(kind)
        pools[pool.name] = pool.snapshot()
    return pools


@dataclass(frozen=True)
class InvocationResult:
    """Результат выполнения модели/инструмента с явным статусом."""

    status: str
    value: Any = None
    error_type: str = ""
    error_message: str = ""


def child_config(config: RunnableConfig | None, run_name: str) -> RunnableConfig | None:
    """Создает дочерний runnable-config с тем же callback-контекстом."""
    if config is None:
        return None

    child: RunnableConfig = {"run_name": run_name}
    callbacks = config.get("callbacks")
    if callbacks is not None:
        child["callbacks"] = callbacks
    tags = config.get("tags")
    if tags is not None:
        child["tags"] = tags

    metadata = config.get("metadata")
    if isinstance(metadata, dict):
        child["metadata"] = dict(metadata)
    return child


def invoke_with_timeout(
    func: Callable[[Any], Any],
    arg: Any,
    timeout_sec: int,
    *,
    breaker: CircuitBreaker | None = None,
    pool: str = "tool",
) -> InvocationResult:
    """
    Вызывает функцию в отдельном пуле потоков с ограничением времени.

    Важно: таймаут НЕ убивает поток. Если внешний вызов завис, поток останется занят
    до завершения функции. Поэтому отправка задач в пул ограничена по емкости (workers+queue),
    а при насыщении возвращается быстрый отказ (overloaded).
    """

    def _runner() -> Any:
        return func(arg)

    token = None
    if breaker is not None:
        try:
            token = breaker.begin_call()
        except CircuitBreakerOpenError as exc:
            logger.warning("Circuit breaker rejected call (%s): %s", breaker.name, exc)
            return InvocationResult(
                status="failed",
                error_type="circuit_open",
                error_message=str(exc),
            )

    pool_obj = _get_pool(pool)
    ctx = copy_context()

    start = time.perf_counter()
    accepted, future = pool_obj.try_submit(lambda: ctx.run(_runner))
    if not accepted:
        snapshot = pool_obj.snapshot()
        logger.warning(
            "Invoke pool is saturated (%s): pending=%s running=%s queued=%s max_workers=%s max_queue=%s",
            pool_obj.name,
            snapshot.get("pending"),
            snapshot.get("running"),
            snapshot.get("queued"),
            snapshot.get("max_workers"),
            snapshot.get("max_queue"),
        )
        if breaker is not None and token is not None:
            breaker.record_failure(token)
        return InvocationResult(
            status="failed",
            error_type="overloaded",
            error_message=f"invoke_pool_saturated:{pool_obj.name}",
        )

    try:
        value = future.result(timeout=max(1, timeout_sec))
        if breaker is not None and token is not None:
            breaker.record_success(token)
        return InvocationResult(status="ok", value=value)
    except FuturesTimeoutError:
        # NB: ThreadPoolExecutor cannot forcibly terminate a running thread.
        # We avoid blocking the caller by not waiting for executor shutdown here.
        future.cancel()
        pool_obj.record_timeout()
        elapsed = time.perf_counter() - start
        logger.warning(
            "Операция превысила таймаут %s сек (elapsed=%.3fs, pool=%s). Поток может продолжать выполнение.",
            timeout_sec,
            elapsed,
            pool_obj.name,
        )
        if breaker is not None and token is not None:
            breaker.record_failure(token)
        return InvocationResult(
            status="failed",
            error_type="timeout",
            error_message=f"timeout>{timeout_sec}s:thread_not_killed",
        )
    except Exception as exc:
        logger.exception("Операция завершилась с ошибкой")
        if breaker is not None and token is not None:
            breaker.record_failure(token)
        return InvocationResult(
            status="failed",
            error_type="exception",
            error_message=exc.__class__.__name__,
        )


def invoke_tool(
    func: Callable[[dict[str, Any]], Any],
    payload: dict[str, Any],
    op_name: str,
    config: RunnableConfig | None = None,
) -> ToolExecutionResult:
    """Вызывает инструмент с таймаутом и возвращает JSON-объект."""
    tool_config = child_config(config, op_name)
    if tool_config is not None:
        metadata = tool_config.get("metadata")
        if isinstance(metadata, dict):
            tool_name = op_name.replace("tool_", "", 1)
            tool_config["metadata"] = {
                **metadata,
                "tool_name": tool_name,
                "tool_operation": op_name,
            }
    raw = invoke_with_timeout(
        lambda tool_payload: func(tool_payload, config=tool_config),
        payload,
        timeout_sec=settings.tool_timeout_sec,
        pool="tool",
    )
    if raw.status != "ok":
        return ToolExecutionResult(
            status="failed",
            payload={},
            error_type=raw.error_type,
            error_message=raw.error_message,
        )

    parsed = parse_tool_payload(raw.value)
    if parsed:
        return ToolExecutionResult(status="ok", payload=parsed)

    logger.warning("Tool %s returned unparseable payload", op_name)
    return ToolExecutionResult(
        status="failed",
        payload={},
        error_type="parse_error",
        error_message=f"{op_name} returned an unparseable payload",
    )
