from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.utils.sku import extract_sku_candidates

MODEL_PATTERN = re.compile(r"\b[A-Z]{2,}\s*\d{1,2}\s*[-/]\s*\d{1,2}\s*[A-Z]{1,3}\b")


def to_langchain_messages(history: list[tuple[str, str]]) -> list[BaseMessage]:
    """Преобразует сохраненную историю в объекты сообщений LangChain."""
    messages: list[BaseMessage] = []
    for role, content in history:
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages


def build_dialogue_memory_summary(messages: list[BaseMessage], max_messages: int = 8) -> str:
    """Извлекает компактный контекст последних сущностей диалога (SKU/модели)."""
    if not messages:
        return ""

    tail = messages[-max_messages:]
    recent_text = "\n".join(str(message.content) for message in tail if getattr(message, "content", ""))
    sku_candidates = sorted(extract_sku_candidates(recent_text, require_digit=True))
    model_candidates = sorted(
        {
            re.sub(r"\s+", " ", match.group(0)).strip()
            for match in MODEL_PATTERN.finditer(recent_text.upper())
        }
    )
    if not sku_candidates and not model_candidates:
        return ""

    parts: list[str] = []
    if sku_candidates:
        parts.append(f"артикулы: {', '.join(sku_candidates[:6])}")
    if model_candidates:
        parts.append(f"модели: {', '.join(model_candidates[:4])}")
    preview = "; ".join(parts)
    return (
        f"Последние упомянутые сущности: {preview}. "
        "Используй их для корректной кореференции (например: 'он', 'этот', 'второй вариант')."
    )
