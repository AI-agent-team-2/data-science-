from __future__ import annotations

import logging
from typing import Any

from app.observability.logging_fields import build_log_fields, merge_log_fields
from app.observability.postgres_event_store import write_event_to_postgres

logger = logging.getLogger(__name__)


def build_event_payload(
    event_name: str,
    *,
    session_hash: str = "",
    intent: str = "",
    used_source: str = "none",
    fallback_reason: str = "",
    **metadata: Any,
) -> dict[str, Any]:
    fields = build_log_fields(
        session_hash=session_hash,
        intent=intent,
        used_source=used_source,
        fallback_reason=fallback_reason,
    )
    payload: dict[str, Any] = {"event_name": event_name, **fields}
    for key, value in metadata.items():
        if value is None:
            continue
        payload[key] = value
    return payload


def emit_event(event_name: str, *, log_fields: dict[str, str] | None = None, **metadata: Any) -> dict[str, Any]:
    merged_fields = merge_log_fields(log_fields)
    payload = build_event_payload(
        event_name,
        session_hash=merged_fields.get("session_hash", ""),
        intent=merged_fields.get("intent", ""),
        used_source=merged_fields.get("used_source", "none"),
        fallback_reason=merged_fields.get("fallback_reason", ""),
        **metadata,
    )
    logger.info("event=%s payload=%s", event_name, payload, extra=merged_fields)
    write_event_to_postgres(payload)
    return payload
