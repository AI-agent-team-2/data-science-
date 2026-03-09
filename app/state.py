from __future__ import annotations

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage


class AgentState(TypedDict, total=False):
    # Основная история сообщений в графе.
    # operator.add позволяет LangGraph автоматически аппендить новые элементы.
    messages: Annotated[list[AnyMessage], operator.add]

    # Технический контекст пользователя/сессии.
    user_id: str
    session_id: str

    # Поля для промежуточных результатов инструментов (под расширение пайплайна).
    retrieved_docs: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    product_results: list[dict[str, Any]]
    compatibility_results: list[dict[str, Any]]

    # Финальный текст ответа (опционально, если решим хранить отдельно от messages).
    final_answer: str
