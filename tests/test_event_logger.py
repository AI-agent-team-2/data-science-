from __future__ import annotations

from unittest.mock import patch

from app.observability.event_logger import build_event_payload, emit_event


def test_build_event_payload_contains_base_fields() -> None:
    payload = build_event_payload(
        "agent_pipeline_started",
        session_hash="u_hash",
        intent="domain",
        used_source="rag",
        fallback_reason="primary_source_succeeded",
        source_order=["rag", "web"],
    )
    assert payload["event_name"] == "agent_pipeline_started"
    assert payload["session_hash"] == "u_hash"
    assert payload["intent"] == "domain"
    assert payload["used_source"] == "rag"
    assert payload["fallback_reason"] == "primary_source_succeeded"
    assert payload["source_order"] == ["rag", "web"]


@patch("app.observability.event_logger.logger")
@patch("app.observability.event_logger.write_event_to_postgres")
def test_emit_event_logs_and_returns_payload(mock_pg_sink, mock_logger) -> None:
    payload = emit_event(
        "agent_pipeline_completed",
        log_fields={"session_hash": "u_hash", "intent": "domain"},
        used_web=False,
    )
    assert payload["event_name"] == "agent_pipeline_completed"
    assert payload["session_hash"] == "u_hash"
    assert payload["intent"] == "domain"
    assert payload["used_source"] == "none"
    assert payload["used_web"] is False
    mock_logger.info.assert_called_once()
    mock_pg_sink.assert_called_once_with(payload)
