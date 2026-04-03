from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypeAlias

from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.context_engine.helpers import (
    extract_results as _extract_results,
    format_lookup_context as _format_lookup_context,
    format_rag_context as _format_rag_context,
    format_web_context as _format_web_context,
    is_rag_useful as _is_rag_useful,
    is_strict_sku_existence_query as _is_strict_sku_existence_query,
    needs_technical_context as _needs_technical_context,
)
from app.context_engine.response import (
    assistant_scope_response,
    build_final_prompt,
    clarifying_question,
    domain_redirect_response,
    ensure_sources_block,
    extract_ai_text,
    smalltalk_response,
    tool_failure_response,
)
from app.context_engine.web import (
    clean_web_text as _clean_web_text,
    contains_instruction_like_text as _contains_instruction_like_text,
    enhance_search_query,
    extract_web_urls as _extract_web_urls,
    filter_safe_web_items as _filter_safe_web_items,
    filter_trusted_web_items as _filter_trusted_web_items,
    has_minimum_web_evidence as _has_minimum_web_evidence,
    is_sanitary_relevant as _is_sanitary_relevant,
    is_web_useful as _is_web_useful,
)
from app.routing.sources import ToolName, should_use_web_source
from app.tools.product_lookup import product_lookup
from app.tools.rag_search import rag_search
from app.tools.web_search import web_search

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class ContextBuildResult:
    """Результат получения контекста из инструментов."""

    context_text: str
    web_urls: list[str]
    used_web: bool
    used_source: str
    terminal_response: str = ""
    failed_sources: list[str] = field(default_factory=list)
    attempted_sources: list[str] = field(default_factory=list)
    source_status_map: dict[str, str] = field(default_factory=dict)
    fallback_reason: str = ""


@dataclass(frozen=True)
class ToolExecutionResult:
    """Результат вызова инструмента с явным статусом выполнения."""

    status: Literal["ok", "failed"]
    payload: dict[str, Any]
    error_type: str = ""
    error_message: str = ""


EMPTY_CONTEXT_RESULT = ContextBuildResult(
    context_text="",
    web_urls=[],
    used_web=False,
    used_source="none",
    terminal_response="",
    failed_sources=[],
    attempted_sources=[],
    source_status_map={},
    fallback_reason="",
)

ToolCallable: TypeAlias = Callable[[dict[str, Any]], Any]
ToolPayload: TypeAlias = dict[str, Any]
InvokeToolFn: TypeAlias = Callable[[ToolCallable, ToolPayload, str, RunnableConfig | None], ToolExecutionResult]


def build_context(
    query: str,
    source_order: list[ToolName],
    invoke_tool: InvokeToolFn,
    config: RunnableConfig | None = None,
) -> ContextBuildResult:
    """Подбирает контекст из доступных источников по приоритету."""
    logger.debug("build_context start: query=%r source_order=%s", query, source_order)
    failed_sources: list[str] = []
    attempted_sources: list[str] = []
    source_status_map: dict[str, str] = {}
    for index, source in enumerate(source_order):
        web_mode = "primary" if index == 0 else "fallback"

        if source == "lookup" and not settings.enable_product_lookup:
            logger.debug("build_context skip source=%s: disabled", source)
            source_status_map[source] = "disabled"
            continue
        if source == "rag" and not settings.enable_rag:
            logger.debug("build_context skip source=%s: disabled", source)
            source_status_map[source] = "disabled"
            continue
        if source == "web" and not settings.enable_web_search:
            logger.debug("build_context skip source=%s: disabled", source)
            source_status_map[source] = "disabled"
            continue
        if source == "web" and not should_use_web_source(query=query, web_mode=web_mode):
            logger.debug("build_context skip source=%s: not allowed for query", source)
            source_status_map[source] = "skipped"
            continue

        attempted_sources.append(source)
        result = _context_from_source(
            source=source,
            query=query,
            web_mode=web_mode,
            invoke_tool=invoke_tool,
            config=config,
        )
        logger.debug(
            "build_context source=%s used_source=%s has_context=%s terminal=%s used_web=%s",
            source,
            result.used_source,
            bool(result.context_text),
            bool(result.terminal_response),
            result.used_web,
        )
        if result.failed_sources:
            failed_sources.extend(result.failed_sources)
            source_status_map[source] = "failed"
        elif result.terminal_response:
            source_status_map[source] = "terminal"
        elif result.context_text:
            source_status_map[source] = "used"
        else:
            source_status_map[source] = "empty"
        if result.context_text or result.terminal_response:
            fallback_reason = _compute_fallback_reason(
                source=source,
                attempted_sources=attempted_sources,
                source_status_map=source_status_map,
            )
            if failed_sources and not result.failed_sources:
                result = ContextBuildResult(
                    context_text=result.context_text,
                    web_urls=result.web_urls,
                    used_web=result.used_web,
                    used_source=result.used_source,
                    terminal_response=result.terminal_response,
                    failed_sources=failed_sources,
                    attempted_sources=attempted_sources,
                    source_status_map=source_status_map,
                    fallback_reason=fallback_reason,
                )
            elif not result.attempted_sources and not result.source_status_map and not result.fallback_reason:
                result = ContextBuildResult(
                    context_text=result.context_text,
                    web_urls=result.web_urls,
                    used_web=result.used_web,
                    used_source=result.used_source,
                    terminal_response=result.terminal_response,
                    failed_sources=result.failed_sources,
                    attempted_sources=attempted_sources,
                    source_status_map=source_status_map,
                    fallback_reason=fallback_reason,
                )
            return result

    if failed_sources:
        logger.warning("build_context finished with source failures: %s", failed_sources)
        return ContextBuildResult(
            context_text="",
            web_urls=[],
            used_web=False,
            used_source="none",
            terminal_response=tool_failure_response(),
            failed_sources=failed_sources,
            attempted_sources=attempted_sources,
            source_status_map=source_status_map,
            fallback_reason="all_attempted_sources_failed",
        )
    if attempted_sources or source_status_map:
        return ContextBuildResult(
            context_text="",
            web_urls=[],
            used_web=False,
            used_source="none",
            terminal_response="",
            failed_sources=[],
            attempted_sources=attempted_sources,
            source_status_map=source_status_map,
            fallback_reason="no_source_produced_context" if attempted_sources else "no_source_attempted",
        )
    return EMPTY_CONTEXT_RESULT


def _compute_fallback_reason(source: str, attempted_sources: list[str], source_status_map: dict[str, str]) -> str:
    """Кратко объясняет, почему был выбран текущий источник."""
    if not attempted_sources or attempted_sources[0] == source:
        return "primary_source_succeeded"

    previous = attempted_sources[:-1]
    previous_statuses = [source_status_map.get(item, "unknown") for item in previous]
    if any(status == "failed" for status in previous_statuses):
        return "fallback_after_source_failure"
    if any(status == "empty" for status in previous_statuses):
        return "fallback_after_empty_result"
    return "fallback_after_skipped_source"


def _context_from_source(
    source: ToolName,
    query: str,
    web_mode: str,
    invoke_tool: InvokeToolFn,
    config: RunnableConfig | None = None,
) -> ContextBuildResult:
    """Строит контекст из указанного источника данных."""
    if source == "lookup":
        return _context_from_lookup(query, invoke_tool=invoke_tool, config=config)
    if source == "rag":
        return _context_from_rag(query, invoke_tool=invoke_tool, config=config)
    return _context_from_web(query, mode=web_mode, invoke_tool=invoke_tool, config=config)


def _context_from_lookup(
    query: str,
    invoke_tool: InvokeToolFn,
    config: RunnableConfig | None = None,
) -> ContextBuildResult:
    """Пытается получить контекст из базы товаров (LOOKUP)."""
    execution = invoke_tool(
        product_lookup.invoke,
        {"query": query, "limit": 5},
        "tool_lookup",
        config,
    )
    if execution.status == "failed":
        logger.warning("LOOKUP failed: %s %s", execution.error_type, execution.error_message)
        return ContextBuildResult("", [], False, "lookup", failed_sources=["lookup"])
    payload = execution.payload
    mode = str(payload.get("mode", "")).strip()
    items = _extract_results(payload)
    if mode == "sku_not_found":
        if not _is_strict_sku_existence_query(query):
            return EMPTY_CONTEXT_RESULT
        note = str(payload.get("note", "")).strip() or "Точный артикул не найден в базе товаров."
        return ContextBuildResult(
            context_text="",
            web_urls=[],
            used_web=False,
            used_source="lookup",
            terminal_response=note,
            failed_sources=[],
        )
    if not items:
        return EMPTY_CONTEXT_RESULT

    context_text = _format_lookup_context(payload, items)
    if _needs_technical_context(query):
        rag_execution = invoke_tool(
            rag_search.invoke,
            {"query": query},
            "tool_rag_from_lookup",
            config,
        )
        if rag_execution.status == "ok":
            rag_items = _extract_results(rag_execution.payload)
            if _is_rag_useful(rag_items):
                context_text = f"{context_text}\n{_format_rag_context(rag_items)}"

    return ContextBuildResult(
        context_text=context_text,
        web_urls=[],
        used_web=False,
        used_source="lookup",
    )


def _context_from_rag(
    query: str,
    invoke_tool: InvokeToolFn,
    config: RunnableConfig | None = None,
) -> ContextBuildResult:
    """Пытается получить контекст из RAG-базы знаний."""
    execution = invoke_tool(
        rag_search.invoke,
        {"query": query},
        "tool_rag",
        config,
    )
    if execution.status == "failed":
        logger.warning("RAG failed: %s %s", execution.error_type, execution.error_message)
        return ContextBuildResult("", [], False, "rag", failed_sources=["rag"])
    payload = execution.payload
    items = _extract_results(payload)
    if not _is_rag_useful(items):
        return EMPTY_CONTEXT_RESULT

    return ContextBuildResult(
        context_text=_format_rag_context(items),
        web_urls=[],
        used_web=False,
        used_source="rag",
    )


def _context_from_web(
    query: str,
    mode: str,
    invoke_tool: InvokeToolFn,
    config: RunnableConfig | None = None,
) -> ContextBuildResult:
    """Пытается получить контекст из web-поиска."""
    enhanced_query = enhance_search_query(query, mode)
    execution = invoke_tool(
        web_search.invoke,
        {"query": enhanced_query, "max_results": settings.web_search_max_results},
        "tool_web",
        config,
    )
    if execution.status == "failed":
        logger.warning("WEB failed: %s %s", execution.error_type, execution.error_message)
        return ContextBuildResult("", [], False, "web", failed_sources=["web"])
    payload = execution.payload
    items = _filter_safe_web_items(_extract_results(payload))
    items = _filter_trusted_web_items(items)
    if not (_is_web_useful(items) and _is_sanitary_relevant(items) and _has_minimum_web_evidence(items)):
        return EMPTY_CONTEXT_RESULT

    return ContextBuildResult(
        context_text=_format_web_context(items, _clean_web_text),
        web_urls=_extract_web_urls(items),
        used_web=True,
        used_source="web",
    )


def parse_tool_payload(raw: Any) -> dict[str, Any]:
    """Парсит ответ инструмента в dict (str JSON / dict / ToolMessage.content)."""
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "content"):
        raw = getattr(raw, "content")
        if isinstance(raw, list):
            parts: list[str] = []
            for item in raw:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            raw = "".join(parts)
    if not isinstance(raw, str):
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
