from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, TypeAlias

from app.config import settings

logger = logging.getLogger(__name__)

HistoryMessage: TypeAlias = tuple[str, str]
HistoryStats: TypeAlias = dict[str, Any]

CREATE_HISTORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_text TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_SESSION_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_session_id ON history(session_id)"


def get_connection() -> sqlite3.Connection:
    """Создает соединение с SQLite и включает словарный доступ к строкам."""
    db_path = Path(settings.history_db_path)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    """Инициализирует таблицы и индексы для хранения истории диалогов."""
    try:
        with get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(CREATE_HISTORY_TABLE_SQL)
            cursor.execute(CREATE_SESSION_INDEX_SQL)
            connection.commit()
    except Exception:
        logger.exception("Failed to initialize history database")


def save_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    """Сохраняет один шаг диалога и запускает очистку устаревших записей."""
    try:
        with get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO history (session_id, user_text, assistant_text) VALUES (?, ?, ?)",
                (session_id, user_text, assistant_text),
            )
            connection.commit()
    except Exception:
        logger.exception("Failed to save history turn for session_id=%s", session_id)
        return

    _cleanup_old(session_id=session_id)


def load_messages(session_id: str, limit: int | None = None) -> list[HistoryMessage]:
    """
    Загружает историю диалога в формате LangChain tuples.

    Возвращает список вида: [("human", text), ("ai", text), ...].
    """
    effective_limit = max(1, int(limit or settings.history_max_messages))
    ttl_days = max(1, settings.history_ttl_days)

    try:
        with get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT user_text, assistant_text
                FROM history
                WHERE session_id = ?
                  AND created_at > datetime('now', ?)
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, f"-{ttl_days} days", effective_limit * 2),
            )
            rows = cursor.fetchall()
    except Exception:
        logger.exception("Failed to load history for session_id=%s", session_id)
        return []

    messages: list[HistoryMessage] = []
    for row in reversed(rows):
        messages.append(("human", str(row["user_text"])))
        messages.append(("ai", str(row["assistant_text"])))

    return messages


def clear_history(session_id: str) -> None:
    """Очищает историю диалога конкретной сессии."""
    try:
        with get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
            connection.commit()
    except Exception:
        logger.exception("Failed to clear history for session_id=%s", session_id)


def get_history_stats(session_id: str) -> HistoryStats:
    """Возвращает статистику по истории пользователя."""
    default_stats: HistoryStats = {
        "count": 0,
        "first_message": None,
        "last_message": None,
    }

    try:
        with get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS count,
                    MIN(created_at) AS first,
                    MAX(created_at) AS last
                FROM history
                WHERE session_id = ?
                """,
                (session_id,),
            )
            row = cursor.fetchone()
    except Exception:
        logger.exception("Failed to read history stats for session_id=%s", session_id)
        return default_stats

    if row is None:
        return default_stats

    return {
        "count": int(row["count"]),
        "first_message": row["first"],
        "last_message": row["last"],
    }


def _cleanup_old(session_id: str) -> None:
    """Удаляет сообщения старше `history_ttl_days` для указанной сессии."""
    ttl_days = max(1, settings.history_ttl_days)

    try:
        with get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "DELETE FROM history WHERE session_id = ? AND created_at < datetime('now', ?)",
                (session_id, f"-{ttl_days} days"),
            )
            connection.commit()
    except Exception:
        logger.exception("Failed to cleanup old history for session_id=%s", session_id)


init_db()
