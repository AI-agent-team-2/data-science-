from __future__ import annotations

import json
from langchain_core.tools import tool

from app.rag.retriever import ChromaRetriever

# Создаем retriever один раз на модуль, чтобы не переинициализировать клиент на каждый вызов.
retriever = ChromaRetriever()


@tool
def rag_search(query: str) -> str:
    """
    Поиск по внутренней базе знаний о сантехнических товарах.
    Используй для вопросов о характеристиках, применении, типах товаров,
    материалах, размерах, брендах и совместимости, если эта информация
    должна браться из внутренней базы.
    """
    # Выполняем семантический поиск по Chroma.
    results = retriever.search(query=query)

    # Возвращаем честный ответ, если контекст не найден.
    if not results:
        return "Ничего не найдено во внутренней базе знаний."

    # Инструменты возвращают строку; сериализуем структурированные данные в JSON.
    return json.dumps(results, ensure_ascii=False, indent=2)
