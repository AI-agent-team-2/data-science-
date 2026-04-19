from __future__ import annotations

from typing import Any


def build_log_fields(
    *,
    session_hash: str = "",
    intent: str = "",
    used_source: str = "none",
    fallback_reason: str = "",
) -> dict[str, str]:
    """Единый набор полей для логов runtime-пайплайна."""
    return {
        "session_hash": session_hash or "unknown",
        "intent": intent or "unknown",
        "used_source": used_source or "none",
        "fallback_reason": fallback_reason or "",
    }


def merge_log_fields(base: dict[str, str] | None, **overrides: Any) -> dict[str, str]:
    merged: dict[str, str] = dict(base or build_log_fields())
    for key, value in overrides.items():
        if value is None:
            continue
        merged[key] = str(value)
    # гарантируем наличие ключей для унификации формата
    for required_key, default_value in build_log_fields().items():
        merged.setdefault(required_key, default_value)
    return merged


def format_log_fields(fields: dict[str, str] | None) -> str:
    """Возвращает человекочитаемый префикс key=value для логов."""
    effective = merge_log_fields(fields)
    return (
        f"session_hash={effective.get('session_hash','unknown')} "
        f"intent={effective.get('intent','unknown')} "
        f"used_source={effective.get('used_source','none')} "
        f"fallback_reason={effective.get('fallback_reason','')}"
    )

