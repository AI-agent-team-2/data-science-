from __future__ import annotations

# Архивная структура состояния под LangGraph.
# Файл сохранен для справки и не используется в текущем runtime.

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_id: str
    session_id: str
    retrieved_docs: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    product_results: list[dict[str, Any]]
    final_answer: str
