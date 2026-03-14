from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.graph import graph
from app.history_store import load_messages, save_turn


def run_agent(user_text: str, user_id: str = "unknown") -> str:
    # Стабильный идентификатор сессии: используем user_id Telegram.
    session_id = user_id or "unknown"
    # Загружаем ограниченную историю диалога из SQLite (с учетом TTL).
    history = load_messages(session_id=session_id)
    input_messages = history + [HumanMessage(content=user_text)]

    result = graph.invoke(
        {
            "messages": input_messages,
            "user_id": session_id,
            "session_id": session_id,
        }
    )

    # Достаем итоговую историю и берем последний текстовый ответ ассистента.
    messages = result["messages"]
    assistant_text = _extract_last_assistant_text(messages)
    # Сохраняем текущий turn в persistent history.
    save_turn(session_id=session_id, user_text=user_text, assistant_text=assistant_text)
    return assistant_text


def _extract_last_assistant_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    return "Не удалось получить ответ."
