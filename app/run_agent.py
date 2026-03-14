from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph import model
from app.history_store import load_messages, save_turn
from app.prompts import SYSTEM_PROMPT
from app.tools.rag_search import rag_search
from app.tools.web_search import web_search


def run_agent(user_text: str, user_id: str = "unknown") -> str:
    # Стабильный идентификатор сессии: используем user_id Telegram.
    session_id = user_id or "unknown"
    # Загружаем ограниченную историю диалога из SQLite (с учетом TTL).
    history = load_messages(session_id=session_id)

    # Жесткая fallback-цепочка:
    # 1) сначала внутренний RAG;
    # 2) если пусто/слабо — web_search;
    # 3) если снова пусто — уточняющий вопрос пользователю.
    rag_raw = rag_search.invoke({"query": user_text})
    rag_results = _parse_list_json(rag_raw)

    context_block = ""
    if _is_rag_useful(rag_results):
        context_block = _format_rag_context(rag_results)
    else:
        web_raw = web_search.invoke({"query": user_text, "max_results": 5})
        web_results = _parse_list_json(web_raw)
        if _is_web_useful(web_results):
            context_block = _format_web_context(web_results)
        else:
            assistant_text = _clarifying_question()
            save_turn(session_id=session_id, user_text=user_text, assistant_text=assistant_text)
            return assistant_text

    final_prompt = (
        f"Вопрос пользователя:\n{user_text}\n\n"
        f"Контекст для ответа:\n{context_block}\n\n"
        "Ответь кратко и по делу, не выдумывай данные и не ссылайся на скрытые служебные поля."
    )
    response = model.invoke([SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=final_prompt)])
    assistant_text = _extract_ai_text(response)

    # Сохраняем текущий turn в persistent history.
    save_turn(session_id=session_id, user_text=user_text, assistant_text=assistant_text)
    return assistant_text


def _extract_ai_text(message: Any) -> str:
    if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
        return message.content
    if isinstance(message, AIMessage) and isinstance(message.content, list):
        # На случай мультимодального формата контента.
        parts = [str(p.get("text", "")) for p in message.content if isinstance(p, dict)]
        text = "\n".join([p for p in parts if p.strip()]).strip()
        if text:
            return text
    return "Не удалось получить ответ."


def _parse_list_json(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, str):
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    return []


def _is_rag_useful(items: list[dict[str, Any]]) -> bool:
    if not items:
        return False
    for item in items:
        text = str(item.get("text", "")).strip()
        score = float(item.get("score", 0.0) or 0.0)
        if text and score >= 0.2:
            return True
    return False


def _is_web_useful(items: list[dict[str, Any]]) -> bool:
    if not items:
        return False
    for item in items:
        title = str(item.get("title", "")).strip().lower()
        url = str(item.get("url", "")).strip().lower()
        if title == "ошибка web_search":
            continue
        if url.startswith("http://") or url.startswith("https://"):
            return True
    return False


def _format_rag_context(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, item in enumerate(items[:4], start=1):
        meta = item.get("metadata") or {}
        source = str(meta.get("source", "unknown"))
        score = float(item.get("score", 0.0) or 0.0)
        text = str(item.get("text", "")).strip().replace("\n", " ")
        snippet = text[:800]
        lines.append(f"[RAG {idx}] source={source} score={score:.3f} text={snippet}")
    return "\n".join(lines).strip()


def _format_web_context(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, item in enumerate(items[:5], start=1):
        title = str(item.get("title", "")).strip()
        snippet = str(item.get("snippet", "")).strip().replace("\n", " ")
        url = str(item.get("url", "")).strip()
        lines.append(f"[WEB {idx}] {title} | {snippet} | {url}")
    return "\n".join(lines).strip()


def _clarifying_question() -> str:
    return (
        "Пока не нашел достаточно надежных данных по вашему запросу. "
        "Уточните, пожалуйста, бренд, артикул (если есть) или ключевой параметр "
        "(например, диаметр/тип подключения/назначение)."
    )
