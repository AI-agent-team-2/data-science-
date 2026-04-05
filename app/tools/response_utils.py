from __future__ import annotations

from typing import Any


def build_tool_payload(
    query: str,
    results: list[dict[str, Any]] | None = None,
    note: str = "",
    meta: dict[str, Any] | None = None,
    status: str = "ok",
    error: str = "",
    **extra: Any,
) -> dict[str, Any]:
    """Строит единый payload для runtime-инструментов."""
    normalized_results = list(results or [])
    payload: dict[str, Any] = {
        "status": status,
        "query": query,
        "count": len(normalized_results),
        "results": normalized_results,
        "note": note,
        "error": error,
        "meta": dict(meta or {}),
    }
    payload.update(extra)
    return payload


def empty_results_payload(query: str, note: str = "", **extra: Any) -> dict[str, Any]:
    """Строит типовой payload для пустой выдачи инструмента."""
    base = build_tool_payload(query=query, results=[], note=note)
    base.update(extra)
    return base


def error_payload(
    query: str,
    *,
    note: str,
    error: str,
    meta: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Строит типовой payload для ошибки инструмента (отличать от 'пусто, но ок')."""
    payload = build_tool_payload(
        query=query,
        results=[],
        note=note,
        meta=meta,
        status="failed",
        error=error,
    )
    payload.update(extra)
    return payload
