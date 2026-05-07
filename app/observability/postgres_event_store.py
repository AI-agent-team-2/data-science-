from __future__ import annotations

import importlib
import json
import logging
import re
from threading import Lock
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_TABLE_PART_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_init_lock: Lock = Lock()
_schema_ready = False


def _quote_table_name(raw_table: str) -> str:
    parts = [part.strip() for part in raw_table.split(".") if part.strip()]
    if not parts:
        parts = ["san_bot_events"]
    if not all(_TABLE_PART_RE.fullmatch(part) for part in parts):
        logger.warning("Invalid EVENT_LOG_POSTGRES_TABLE=%r; fallback to san_bot_events", raw_table)
        parts = ["san_bot_events"]
    return ".".join(f'"{part}"' for part in parts)


def _get_psycopg_module() -> Any | None:
    try:
        return importlib.import_module("psycopg")
    except Exception:
        return None


def _ensure_table(psycopg: Any) -> bool:
    global _schema_ready
    if _schema_ready:
        return True

    table_name = _quote_table_name(settings.event_log_postgres_table)
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id BIGSERIAL PRIMARY KEY,
        event_name TEXT NOT NULL,
        session_hash TEXT NOT NULL,
        intent TEXT NOT NULL,
        used_source TEXT NOT NULL,
        fallback_reason TEXT NOT NULL,
        payload JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """
    table_for_index = table_name.replace(".", "_").replace('"', "")
    index_sql = f"CREATE INDEX IF NOT EXISTS idx_{table_for_index}_created_at ON {table_name}(created_at DESC)"

    with _init_lock:
        if _schema_ready:
            return True
        try:
            with psycopg.connect(settings.event_log_postgres_dsn) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(create_sql)
                    cursor.execute(index_sql)
                connection.commit()
            _schema_ready = True
            return True
        except Exception:
            logger.exception("Failed to initialize postgres event table")
            return False


def write_event_to_postgres(payload: dict[str, Any]) -> bool:
    if not settings.event_log_postgres_enabled:
        return False
    if not settings.event_log_postgres_dsn:
        return False

    psycopg = _get_psycopg_module()
    if psycopg is None:
        logger.warning("EVENT_LOG_POSTGRES_ENABLED=true but psycopg is not installed")
        return False
    if not _ensure_table(psycopg):
        return False

    table_name = _quote_table_name(settings.event_log_postgres_table)
    insert_sql = f"""
    INSERT INTO {table_name} (
        event_name, session_hash, intent, used_source, fallback_reason, payload
    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
    """
    try:
        with psycopg.connect(settings.event_log_postgres_dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    insert_sql,
                    (
                        str(payload.get("event_name", "")),
                        str(payload.get("session_hash", "unknown")),
                        str(payload.get("intent", "unknown")),
                        str(payload.get("used_source", "none")),
                        str(payload.get("fallback_reason", "")),
                        json.dumps(payload, ensure_ascii=False),
                    ),
                )
            connection.commit()
        return True
    except Exception:
        logger.exception("Failed to write event to postgres")
        return False
