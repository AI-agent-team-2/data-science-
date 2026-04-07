from __future__ import annotations

from typing import Any

from app.config import settings
from app.observability import sanitize_text
from app.routing import (
    ToolName,
    is_identity_or_capability_query,
    is_noise_query,
    is_offtopic_or_rude_query,
    is_smalltalk,
)


def build_trace_metadata(
    query: str,
    source_order: list[ToolName],
    intent: str,
    risk_flags: list[str],
    guard_action: str,
    hashed_user: str,
) -> dict[str, Any]:
    """Формирует единый metadata-формат для run/model/tool шагов."""
    return {
        "query": sanitize_text(query),
        "provider": settings.resolved_model_provider,
        "model": settings.resolved_model_name,
        "intent": intent,
        "source_order": list(source_order),
        "attempted_sources": [],
        "source_status_map": {},
        "failed_sources": [],
        "fallback_reason": "",
        "risk_flags": list(risk_flags),
        "guard_action": guard_action,
        "trace_session_id": hashed_user,
        "trace_user_id": hashed_user,
        "trace_tags": ["telegram", "san-bot", "run_agent"],
        "features": {
            "enable_web_search": settings.enable_web_search,
            "enable_rag": settings.enable_rag,
            "enable_product_lookup": settings.enable_product_lookup,
        },
    }


def detect_intent(query: str) -> str:
    """Возвращает короткую intent-метку для observability."""
    if is_identity_or_capability_query(query):
        return "identity"
    if is_smalltalk(query):
        return "smalltalk"
    if is_noise_query(query) or is_offtopic_or_rude_query(query):
        return "offtopic"
    return "domain"
