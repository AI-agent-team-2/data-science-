from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from threading import Lock
from typing import TypeAlias

from app.config import settings

logger = logging.getLogger(__name__)

HistoryMessage: TypeAlias = tuple[str, str]

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

_db_init_lock: Lock = Lock()
_db_initialized = False


def get_connection() -> sqlite3.Connection:
    """Создает соединение с SQLite и включает словарный доступ к строкам."""
    db_path = Path(settings.history_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=5.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> bool:
    """Инициализирует таблицы и индексы для хранения истории диалогов."""
    try:
        connection = get_connection()
        try:
            cursor = connection.cursor()
            cursor.execute(CREATE_HISTORY_TABLE_SQL)
            cursor.execute(CREATE_SESSION_INDEX_SQL)
            connection.commit()
            return True
        finally:
            connection.close()
    except Exception:
        logger.exception("Не удалось инициализировать базу истории")
        return False


def ensure_db_initialized() -> None:
    """Гарантирует инициализацию схемы SQLite без сайд-эффектов на import."""
    global _db_initialized
    if _db_initialized:
        return

    with _db_init_lock:
        if _db_initialized:
            return
        _db_initialized = init_db()


def save_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    """Сохраняет один шаг диалога и запускает очистку устаревших записей."""
    ensure_db_initialized()
    try:
        connection = get_connection()
        try:
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO history (session_id, user_text, assistant_text) VALUES (?, ?, ?)",
                (session_id, user_text, assistant_text),
            )
            connection.commit()
        finally:
            connection.close()
    except Exception:
        logger.exception("Не удалось сохранить шаг истории для session_id=%s", session_id)
        return

    _cleanup_old(session_id=session_id)


def load_messages(session_id: str, limit: int | None = None) -> list[HistoryMessage]:
    """
    Загружает историю диалога в формате пар сообщений.

    Возвращает список вида: [("human", text), ("ai", text), ...].
    """
    ensure_db_initialized()
    effective_limit = max(1, int(limit or settings.history_max_messages))
    ttl_days = max(1, settings.history_ttl_days)

    try:
        connection = get_connection()
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT user_text, assistant_text
                FROM history
                WHERE session_id = ?
                  AND created_at > datetime('now', ?)
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, f"-{ttl_days} days", effective_limit),
            )
            rows = cursor.fetchall()
        finally:
            connection.close()
    except Exception:
        logger.exception("Не удалось загрузить историю для session_id=%s", session_id)
        return []

    messages: list[HistoryMessage] = []
    for row in reversed(rows):
        messages.append(("human", str(row["user_text"])))
        messages.append(("ai", str(row["assistant_text"])))

    return messages


def load_turns(session_id: str, limit: int | None = None) -> list[dict[str, str]]:
    """
    Загружает историю диалога в формате "ходов" (user+assistant) с timestamp.

    Returns:
        list[dict[str, str]]: [{"user": "...", "assistant": "...", "timestamp": "..."}]
    """
    ensure_db_initialized()
    effective_limit = max(1, int(limit or settings.history_max_messages))
    ttl_days = max(1, settings.history_ttl_days)

    try:
        connection = get_connection()
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT user_text, assistant_text, created_at
                FROM history
                WHERE session_id = ?
                  AND created_at > datetime('now', ?)
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, f"-{ttl_days} days", effective_limit),
            )
            rows = cursor.fetchall()
        finally:
            connection.close()
    except Exception:
        logger.exception("Не удалось загрузить историю turns для session_id=%s", session_id)
        return []

    turns: list[dict[str, str]] = []
    for row in reversed(rows):
        turns.append(
            {
                "user": str(row["user_text"]),
                "assistant": str(row["assistant_text"]),
                "timestamp": str(row["created_at"]),
            }
        )
    return turns


def clear_history(session_id: str) -> None:
    """Очищает историю диалога конкретной сессии."""
    ensure_db_initialized()
    try:
        connection = get_connection()
        try:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
            connection.commit()
        finally:
            connection.close()
    except Exception:
        logger.exception("Не удалось очистить историю для session_id=%s", session_id)


def _cleanup_old(session_id: str) -> None:
    """Удаляет сообщения старше `history_ttl_days` для указанной сессии."""
    ensure_db_initialized()
    ttl_days = max(1, settings.history_ttl_days)

    try:
        connection = get_connection()
        try:
            cursor = connection.cursor()
            cursor.execute(
                "DELETE FROM history WHERE session_id = ? AND created_at < datetime('now', ?)",
                (session_id, f"-{ttl_days} days"),
            )
            connection.commit()
        finally:
            connection.close()
    except Exception:
        logger.exception("Не удалось удалить устаревшую историю для session_id=%s", session_id)

