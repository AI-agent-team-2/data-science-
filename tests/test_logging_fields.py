from __future__ import annotations

from app.observability.logging_fields import build_log_fields, format_log_fields, merge_log_fields


def test_build_log_fields_has_required_keys() -> None:
    fields = build_log_fields(session_hash="abc", intent="domain", used_source="rag", fallback_reason="x")
    assert fields["session_hash"] == "abc"
    assert fields["intent"] == "domain"
    assert fields["used_source"] == "rag"
    assert fields["fallback_reason"] == "x"


def test_merge_log_fields_keeps_required_keys() -> None:
    merged = merge_log_fields({"session_hash": "abc"}, used_source="web")
    assert set(merged.keys()) >= {"session_hash", "intent", "used_source", "fallback_reason"}
    assert merged["session_hash"] == "abc"
    assert merged["used_source"] == "web"


def test_format_log_fields_is_stable() -> None:
    rendered = format_log_fields({"session_hash": "abc", "intent": "domain", "used_source": "lookup", "fallback_reason": "r"})
    assert "session_hash=abc" in rendered
    assert "intent=domain" in rendered
    assert "used_source=lookup" in rendered
    assert "fallback_reason=r" in rendered

