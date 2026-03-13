from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import settings
from app.prompts import SYSTEM_PROMPT
from app.state import AgentState
from app.tools.rag_search import rag_search
from app.tools.web_search import web_search
from app.tools.product_lookup import product_lookup


# Регистрируем инструменты, к которым LLM может обратиться через tool-calls.
tools = [rag_search, web_search, product_lookup]

# Инициализируем чат-модель через OpenAI-compatible API.
# По умолчанию это локальный Ollama на http://localhost:11434/v1.
model = ChatOpenAI(
    model=settings.resolved_model_name,
    temperature=0,
    api_key=settings.resolved_openai_api_key,
    base_url=settings.resolved_openai_base_url,
)

# Оборачиваем модель в режим tool-calling.
model_with_tools = model.bind_tools(tools)


def agent_node(state: AgentState) -> AgentState:
    # Каждый шаг добавляем системную инструкцию и текущую историю сообщений.
    response = model_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    # Возвращаем только новый ответ; LangGraph сам склеит его с историей через operator.add.
    return {"messages": [response]}


# Узел, который исполняет вызванные инструменты.
tool_node = ToolNode(tools)


def build_graph():
    # Создаем state machine поверх AgentState.
    builder = StateGraph(AgentState)

    # Добавляем узел генерации ответа и узел исполнения инструментов.
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    # Запуск графа всегда начинается с ответа модели.
    builder.add_edge(START, "agent")

    # Если модель запросила tool-call, идем в tools; иначе завершаем диалоговый шаг.
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    # После инструментов повторно возвращаемся к модели для финализации ответа.
    builder.add_edge("tools", "agent")

    # Компилируем граф в исполняемый объект.
    return builder.compile()


# Глобальный граф: используется run_agent и telegram-обработчиком.
graph = build_graph()
