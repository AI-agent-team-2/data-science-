from __future__ import annotations

from app.utils.sku import extract_sku_candidates


def is_multiflex_flow_control_query(query: str) -> bool:
    """Определяет вопросы о регулировке потока кранами мультифлекса."""
    lowered = query.lower()
    if "мультифлекс" not in lowered:
        return False
    markers = ("регулир", "душ", "дроссел", "прикры", "поток")
    return any(marker in lowered for marker in markers)


def resolve_followup_reference(query: str, history: list[tuple[str, str]]) -> str:
    """
    Добавляет в двусмысленный follow-up явные SKU из недавнего контекста.
    Это снижает потерю связи в репликах вида "второй вариант"/"какой из них".
    """
    lowered = query.lower()
    followup_markers = ("второй вариант", "второго варианта", "какой из них", "кто из них", "из них")
    if not any(marker in lowered for marker in followup_markers):
        return query

    if extract_sku_candidates(query, require_digit=True):
        return query

    recent_texts = [text for role, text in history[-12:] if role in {"human", "ai"}]
    skus: list[str] = []
    for text in reversed(recent_texts):
        for sku in extract_sku_candidates(text, require_digit=True):
            if sku not in skus:
                skus.append(sku)
        if len(skus) >= 2:
            break

    if len(skus) >= 2:
        return f"{query}\nУчитывай контекст диалога: сравни SKU {skus[0]} и SKU {skus[1]}."

    if len(skus) == 1:
        inferred_pair = infer_nc_no_pair(skus[0])
        if inferred_pair:
            return (
                f"{query}\nУчитывай контекст диалога: речь о паре SKU {skus[0]} "
                f"и SKU {inferred_pair} (если второй вариант не был назван явно)."
            )
        return f"{query}\nУчитывай контекст диалога: речь о SKU {skus[0]}."

    return query


def infer_nc_no_pair(sku: str) -> str | None:
    """Пробует вывести парный SKU для NC/NO-вариантов сервопривода."""
    value = str(sku or "").upper()
    if "NC" in value:
        return value.replace("NC", "NO", 1)
    if "NO" in value:
        return value.replace("NO", "NC", 1)
    return None


def direct_nc_no_answer_if_possible(query: str) -> str | None:
    """Возвращает прямой ответ для вопросов о NC/NO, если пара SKU уже определена."""
    lowered = query.lower()
    asks_closed = "нормально закрыт" in lowered
    asks_open = "нормально открыт" in lowered
    if not (asks_closed or asks_open):
        return None

    skus = extract_sku_candidates(query, require_digit=True)
    if not skus:
        return None

    nc_sku = next((sku for sku in skus if "NC" in sku.upper()), None)
    no_sku = next((sku for sku in skus if "NO" in sku.upper()), None)
    if nc_sku is None and no_sku is not None:
        nc_sku = infer_nc_no_pair(no_sku)
    if no_sku is None and nc_sku is not None:
        no_sku = infer_nc_no_pair(nc_sku)

    if nc_sku is None and no_sku is None:
        return None

    if asks_closed and nc_sku:
        if no_sku:
            return f"Нормально закрытый вариант — {nc_sku}, а нормально открытый — {no_sku}."
        return f"Нормально закрытый вариант — {nc_sku}."

    if asks_open and no_sku:
        if nc_sku:
            return f"Нормально открытый вариант — {no_sku}, а нормально закрытый — {nc_sku}."
        return f"Нормально открытый вариант — {no_sku}."

    return None
