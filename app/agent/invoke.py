from __future__ import annotations

import atexit
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any, Callable
from functools import lru_cache

from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.context_engine import ToolExecutionResult, parse_tool_payload
from app.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

logger = logging.getLogger(__name__)

DEFAULT_INVOKE_MAX_WORKERS = 8


@lru_cache(maxsize=1)
def _get_executor() -> ThreadPoolExecutor:
    max_workers = int(getattr(settings, "invoke_max_workers", DEFAULT_INVOKE_MAX_WORKERS) or DEFAULT_INVOKE_MAX_WORKERS)
    max_workers = max(1, min(64, max_workers))
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sanbot-invoke")

    def _shutdown() -> None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            return

    atexit.register(_shutdown)
    return executor


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
) -> InvocationResult:
    """Вызывает функцию в отдельном потоке с ограничением времени."""

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

    executor = _get_executor()
    ctx = copy_context()
    future = executor.submit(ctx.run, _runner)
    try:
        value = future.result(timeout=max(1, timeout_sec))
        if breaker is not None and token is not None:
            breaker.record_success(token)
        return InvocationResult(status="ok", value=value)
    except FuturesTimeoutError:
        # NB: ThreadPoolExecutor cannot forcibly terminate a running thread.
        # We avoid blocking the caller by not waiting for executor shutdown here.
        future.cancel()
        logger.warning("Операция превысила таймаут %s сек", timeout_sec)
        if breaker is not None and token is not None:
            breaker.record_failure(token)
        return InvocationResult(status="failed", error_type="timeout", error_message=f"timeout>{timeout_sec}s")
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
