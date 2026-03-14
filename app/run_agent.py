from __future__ import annotations

from langchain_core.messages import HumanMessage
from app.graph import graph


def run_agent(user_text: str, user_id: str = "unknown") -> str:
    # Передаем в граф стартовое сообщение пользователя и базовый контекст сессии.
    session_id = user_id or "unknown"
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_text)],
            "user_id": session_id,
            "session_id": session_id,
        },
        config={"configurable": {"thread_id": session_id}},
    )

    # Достаем итоговую историю и берем последний ответ ассистента.
    messages = result["messages"]
    # Fallback на случай пустого результата.
    return messages[-1].content if messages else "Не удалось получить ответ."
