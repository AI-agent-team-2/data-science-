from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph import model
from app.history_store import load_messages, save_turn
from app.prompts import SYSTEM_PROMPT
from app.tools.product_lookup import product_lookup
from app.tools.rag_search import rag_search
from app.tools.web_search import web_search

logger = logging.getLogger(__name__)

SKU_RE = re.compile(r"\b[A-Z][A-Z0-9]{4,}\b")
TOOL_TIMEOUT_SEC = 20
MODEL_TIMEOUT_SEC = 45


def _should_prefer_web(query: str) -> bool:
    lowered = query.lower()
    
    # Всегда ищем в интернете, если запрос:
    # 1. Содержит маркеры времени/цены/места
    # 2. Спрашивает про новинки, тренды
    # 3. Содержит вопрос про "что такое", "как работает"
    # 4. Или просто всегда ищем для непонятных запросов
    
    markers = (
        "сейчас", "сегодня", "в 2026", "в 2025", "в 2024",
        "новые требования", "что изменилось", "по отзывам",
        "отзывы", "где купить", "в москве", "в россии",
        "средняя цена", "цена", "аналоги", "сравни", "новости",
        "что такое", "как работает", "для чего", "зачем",
        "лучший", "рейтинг", "топ", "популярный",
        "актуальный", "последний", "новинки", "тренды"
    )
    
    # Всегда ищем для коротких запросов (меньше 3 слов)
    words = lowered.split()
    if len(words) < 3:
        return True
        
    return any(marker in lowered for marker in markers)


def _should_prefer_lookup(query: str) -> bool:
    if SKU_RE.search(query.upper()):
        return True

    lowered = query.lower()
    markers = ("артикул", "sku", "модель", "код", "товар", "бренд", "серия", "позици")
    return any(marker in lowered for marker in markers)


def enhance_search_query(original_query: str, search_type: str = "general") -> str:
    """Улучшает поисковый запрос для веб-поиска"""
    lowered = original_query.lower()
    
    # Если запрос уже про сантехнику — оставляем как есть
    if "сантехник" in lowered or "унитаз" in lowered or "смеситель" in lowered:
        return original_query
    
    # Для запросов о новинках
    if "новинк" in lowered or "новые" in lowered:
        if "2026" in lowered:
            return f"новинки сантехники 2026 каталог"
        elif "2025" in lowered:
            return f"новинки сантехники 2025 каталог"
        else:
            return f"новые сантехнические товары 2026 каталог"
    
    # Для запросов о ценах и бюджете
    elif "бюджет" in lowered or "цен" in lowered or "стоит" in lowered or "сколько" in lowered:
        return f"{original_query} сантехника"
    
    # Для запросов о покупке
    elif "купить" in lowered or "где" in lowered:
        return f"{original_query} сантехника интернет магазин"
    
    # Для общих запросов
    elif search_type == "fallback":
        return f"{original_query} сантехника"
    else:
        return f"{original_query} сантехника товары"


def run_agent(user_text: str, user_id: str = "unknown") -> str:
    # Активный runtime-пайплайн:
    # актуальные/рыночные запросы -> web_search;
    # SKU/товарный запрос -> product_lookup, иначе -> rag_search;
    # затем fallback на оставшиеся источники.
    session_id = user_id or "unknown"
    history = load_messages(session_id=session_id)
    tool_query = user_text.strip()

    context_block = ""
    prefer_web = _should_prefer_web(tool_query)
    prefer_lookup = _should_prefer_lookup(tool_query)
    web_urls: list[str] = []

    # 1) Первый источник зависит от типа запроса.
    if prefer_web:
        # Улучшаем запрос для веб-поиска
        enhanced_query = enhance_search_query(tool_query, "primary")
        
        web_raw = _invoke_with_timeout(
            web_search.invoke,
            {"query": enhanced_query, "max_results": 5},
            TOOL_TIMEOUT_SEC,
            op_name="web_search",
        )
        web_data = _parse_object_json(web_raw)
        web_results = _extract_results(web_data)
        web_urls = _extract_web_urls(web_results)
        # Добавляем проверку на релевантность сантехнике
        if _is_web_useful(web_results) and _is_sanitary_relevant(web_results):
            context_block = _format_web_context(web_results)
        else:
            # Если веб-поиск не дал результатов или не про сантехнику, пробуем RAG
            rag_raw = _invoke_with_timeout(
                rag_search.invoke,
                {"query": tool_query},
                TOOL_TIMEOUT_SEC,
                op_name="rag_search",
            )
            rag_data = _parse_object_json(rag_raw)
            rag_results = _extract_results(rag_data)
            if _is_rag_useful(rag_results):
                context_block = _format_rag_context(rag_results)
    elif prefer_lookup:
        lookup_raw = _invoke_with_timeout(
            product_lookup.invoke,
            {"query": tool_query, "limit": 5},
            TOOL_TIMEOUT_SEC,
            op_name="product_lookup",
        )
        lookup_data = _parse_object_json(lookup_raw)
        lookup_results = _extract_results(lookup_data)
        if _is_lookup_useful(lookup_results):
            context_block = _format_lookup_context(lookup_data, lookup_results)
    else:
        rag_raw = _invoke_with_timeout(
            rag_search.invoke,
            {"query": tool_query},
            TOOL_TIMEOUT_SEC,
            op_name="rag_search",
        )
        rag_data = _parse_object_json(rag_raw)
        rag_results = _extract_results(rag_data)
        if _is_rag_useful(rag_results):
            context_block = _format_rag_context(rag_results)

    # 2) Fallback на второй/третий источник.
    if not context_block:
        if prefer_web:
            rag_raw = _invoke_with_timeout(
                rag_search.invoke,
                {"query": tool_query},
                TOOL_TIMEOUT_SEC,
                op_name="rag_search",
            )
            rag_data = _parse_object_json(rag_raw)
            rag_results = _extract_results(rag_data)
            if _is_rag_useful(rag_results):
                context_block = _format_rag_context(rag_results)
            if not context_block:
                lookup_raw = _invoke_with_timeout(
                    product_lookup.invoke,
                    {"query": tool_query, "limit": 5},
                    TOOL_TIMEOUT_SEC,
                    op_name="product_lookup",
                )
                lookup_data = _parse_object_json(lookup_raw)
                lookup_results = _extract_results(lookup_data)
                if _is_lookup_useful(lookup_results):
                    context_block = _format_lookup_context(lookup_data, lookup_results)
        elif prefer_lookup:
            rag_raw = _invoke_with_timeout(
                rag_search.invoke,
                {"query": tool_query},
                TOOL_TIMEOUT_SEC,
                op_name="rag_search",
            )
            rag_data = _parse_object_json(rag_raw)
            rag_results = _extract_results(rag_data)
            if _is_rag_useful(rag_results):
                context_block = _format_rag_context(rag_results)
        else:
            lookup_raw = _invoke_with_timeout(
                product_lookup.invoke,
                {"query": tool_query, "limit": 5},
                TOOL_TIMEOUT_SEC,
                op_name="product_lookup",
            )
            lookup_data = _parse_object_json(lookup_raw)
            lookup_results = _extract_results(lookup_data)
            if _is_lookup_useful(lookup_results):
                context_block = _format_lookup_context(lookup_data, lookup_results)

    # 3) Последний fallback — внешний поиск (если не ходили в него первым).
    if not context_block:
        if not prefer_web:
            # Для fallback используем улучшенный запрос
            enhanced_query = enhance_search_query(tool_query, "fallback")
            
            web_raw = _invoke_with_timeout(
                web_search.invoke,
                {"query": enhanced_query, "max_results": 5},
                TOOL_TIMEOUT_SEC,
                op_name="web_search",
            )
            web_data = _parse_object_json(web_raw)
            web_results = _extract_results(web_data)
            web_urls = _extract_web_urls(web_results)
            # Добавляем проверку на релевантность сантехнике
            if _is_web_useful(web_results) and _is_sanitary_relevant(web_results):
                context_block = _format_web_context(web_results)

    if not context_block:
        assistant_text = _clarifying_question()
        save_turn(session_id=session_id, user_text=user_text, assistant_text=assistant_text)
        return assistant_text

    final_prompt = (
        f"Вопрос пользователя:\n{user_text}\n\n"
        f"Контекст для ответа:\n{context_block}\n\n"
        "Ответь строго по вопросу пользователя. "
        "Если вопрос только про сантехнику — отвечай только про сантехнику. "
        "Не добавляй информацию про ремонт, стройматериалы, мебель и другие темы, "
        "если пользователь о них не спрашивал. Будь краток и точен."
    )
    response = _invoke_with_timeout(
        model.invoke,
        [SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=final_prompt)],
        MODEL_TIMEOUT_SEC,
        op_name="model_invoke",
    )
    assistant_text = _extract_ai_text(response)
    if prefer_web:
        assistant_text = _ensure_sources_block(assistant_text, web_urls)

    save_turn(session_id=session_id, user_text=user_text, assistant_text=assistant_text)
    return assistant_text


def _invoke_with_timeout(func, arg, timeout_sec: int, op_name: str = "operation"):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(func, arg)
        try:
            return fut.result(timeout=max(1, timeout_sec))
        except FuturesTimeoutError:
            logger.warning("%s timed out after %s sec", op_name, timeout_sec)
            return ""
        except Exception as e:
            logger.exception("%s failed: %s", op_name, e)
            return ""


def _extract_ai_text(message: Any) -> str:
    if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
        return message.content
    if isinstance(message, AIMessage) and isinstance(message.content, list):
        parts = [str(p.get("text", "")) for p in message.content if isinstance(p, dict)]
        text = "\n".join([p for p in parts if p.strip()]).strip()
        if text:
            return text
    return "Не удалось получить ответ."


def _parse_object_json(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("results")
    if isinstance(items, list):
        return [p for p in items if isinstance(p, dict)]
    return []


def _is_lookup_useful(items: list[dict[str, Any]]) -> bool:
    return len(items) > 0


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
        url = str(item.get("url", "")).strip().lower()
        if url.startswith("http://") or url.startswith("https://"):
            return True
    return False


def _is_sanitary_relevant(items: list[dict[str, Any]]) -> bool:
    """Проверяет, что результаты поиска относятся к сантехнике"""
    sanitary_keywords = ["унитаз", "ванна", "смеситель", "душ", "сантехник", 
                         "раковина", "труба", "фитинг", "кран", "бойлер",
                         "сантехника", "санфаянс", "кранбукс", "картридж",
                         "гидробокс", "инсталляция", "поддон", "лейка"]
    
    for item in items:
        title = str(item.get("title", "")).lower()
        snippet = str(item.get("snippet", "")).lower()
        combined = title + " " + snippet
        
        # Если есть хотя бы одно слово из сантехники — считаем релевантным
        if any(keyword in combined for keyword in sanitary_keywords):
            return True
    
    return False


def _format_rag_context(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, item in enumerate(items[:4], start=1):
        meta = item.get("metadata") or {}
        source = str(meta.get("source", "unknown"))
        score = float(item.get("score", 0.0) or 0.0)
        text = str(item.get("text", "")).strip().replace("\n", " ")
        lines.append(f"[RAG {idx}] source={source} score={score:.3f} text={text[:800]}")
    return "\n".join(lines).strip()


def _format_web_context(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, item in enumerate(items[:5], start=1):
        title = str(item.get("title", "")).strip()
        snippet = str(item.get("snippet", "")).strip().replace("\n", " ")
        url = str(item.get("url", "")).strip()
        lines.append(f"[WEB {idx}] {title} | {snippet} | {url}")
    return "\n".join(lines).strip()


def _extract_web_urls(items: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for item in items:
        url = str(item.get("url", "")).strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _ensure_sources_block(answer: str, urls: list[str]) -> str:
    if "Источники:" in answer:
        return answer

    block_lines = ["", "Источники:"]
    if urls:
        for url in urls[:5]:
            block_lines.append(f"- {url}")
    else:
        block_lines.append("- внешние ссылки не найдены")

    return answer.rstrip() + "\n" + "\n".join(block_lines)


def _format_lookup_context(payload: dict[str, Any], items: list[dict[str, Any]]) -> str:
    mode = str(payload.get("mode", "lookup"))
    lines: list[str] = [f"[LOOKUP] mode={mode} count={len(items)}"]
    for idx, item in enumerate(items[:5], start=1):
        name = str(item.get("name", "")).strip()
        brand = str(item.get("brand", "")).strip()
        category = str(item.get("category", "")).strip()
        source = str(item.get("source", "")).strip()
        score = item.get("score", "")
        skus = item.get("sku_list") or []
        sku_preview = ", ".join([str(s) for s in skus[:5]])
        lines.append(
            f"[LOOKUP {idx}] {name} | brand={brand} | category={category} | sku={sku_preview} | source={source} | score={score}"
        )
    return "\n".join(lines).strip()


def _clarifying_question() -> str:
    return (
        "Пока не нашел достаточно надежных данных по вашему запросу. "
        "Уточните, пожалуйста, бренд, артикул (если есть) или ключевой параметр "
        "(например, диаметр/тип подключения/назначение)."
    )