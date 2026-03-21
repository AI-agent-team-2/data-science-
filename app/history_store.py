from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any

from app.config import settings


def get_connection():
    """Получить соединение с БД"""
    conn = sqlite3.connect(settings.history_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Инициализировать базу данных"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_text TEXT NOT NULL,
            assistant_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON history(session_id)')
    conn.commit()
    conn.close()


def save_turn(session_id: str, user_text: str, assistant_text: str):
    """Сохранить диалог"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (session_id, user_text, assistant_text) VALUES (?, ?, ?)",
        (session_id, user_text, assistant_text)
    )
    conn.commit()
    conn.close()
    
    # Очистка старых записей
    _cleanup_old(session_id)


def load_messages(session_id: str, limit: int = None) -> List[Dict[str, Any]]:
    """Загрузить историю диалога в формате LangChain"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if limit is None:
        limit = settings.history_max_messages
    
    ttl_days = settings.history_ttl_days
    
    # Только сообщения не старше TTL, используем SQLite datetime функции
    cursor.execute(
        """
        SELECT user_text, assistant_text FROM history 
        WHERE session_id = ? AND created_at > datetime('now', ?)
        ORDER BY created_at DESC LIMIT ?
        """,
        (session_id, f'-{ttl_days} days', limit * 2)
    )
    
    rows = cursor.fetchall()
    conn.close()
    
    # Преобразуем в формат LangChain (в хронологическом порядке)
    messages = []
    for row in reversed(rows):
        messages.append(("human", row["user_text"]))
        messages.append(("ai", row["assistant_text"]))
    
    return messages


def clear_history(session_id: str):
    """Очистить историю для пользователя"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def get_history_stats(session_id: str) -> Dict[str, Any]:
    """Получить статистику по истории пользователя"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) as count, MIN(created_at) as first, MAX(created_at) as last FROM history WHERE session_id = ?",
        (session_id,)
    )
    row = cursor.fetchone()
    conn.close()
    
    return {
        'count': row['count'] if row else 0,
        'first_message': row['first'] if row else None,
        'last_message': row['last'] if row else None
    }


def _cleanup_old(session_id: str):
    """Очистить старые записи по TTL"""
    ttl_days = settings.history_ttl_days
    conn = get_connection()
    cursor = conn.cursor()
    
    # Используем SQLite datetime функции
    cursor.execute(
        "DELETE FROM history WHERE session_id = ? AND created_at < datetime('now', ?)",
        (session_id, f'-{ttl_days} days')
    )
    conn.commit()
    conn.close()


# Инициализация при импорте
init_db()