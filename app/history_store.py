from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings


def _connect() -> sqlite3.Connection:
    db_path = Path(settings.history_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_history_store() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at)"
        )


def prune_expired_history(ttl_days: int | None = None) -> None:
    days = ttl_days if ttl_days is not None else settings.history_ttl_days
    if days <= 0:
        return
    cutoff_ts = int(time.time()) - days * 24 * 60 * 60
    with _connect() as conn:
        conn.execute("DELETE FROM chat_history WHERE created_at < ?", (cutoff_ts,))


def load_messages(session_id: str, limit: int | None = None) -> list:
    max_messages = limit if limit is not None else settings.history_max_messages
    max_messages = max(0, max_messages)
    if max_messages == 0:
        return []

    prune_expired_history()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, max_messages),
        ).fetchall()

    rows.reverse()
    messages = []
    for role, content in rows:
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def save_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    now = int(time.time())
    prune_expired_history()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_history(session_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
            (session_id, user_text, now),
        )
        conn.execute(
            "INSERT INTO chat_history(session_id, role, content, created_at) VALUES (?, 'assistant', ?, ?)",
            (session_id, assistant_text, now),
        )


# Инициализируем таблицу при импорте, чтобы рантайм не зависел от ручной миграции.
init_history_store()
