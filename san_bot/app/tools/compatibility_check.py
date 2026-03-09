from __future__ import annotations

import json
from langchain_core.tools import tool


@tool
def compatibility_check(item_a: str, item_b: str) -> str:
    """
    Проверка совместимости двух товаров или компонентов.
    """
    # TODO: заменить на реальную бизнес-логику проверки (диаметры, резьба, давление, материал).
    # Заглушка оставлена для отладки взаимодействия LLM с инструментом.
    result = {
        "item_a": item_a,
        "item_b": item_b,
        "compatible": True,
        "reason": "Заглушка: размеры и тип подключения считаются совместимыми.",
    }
    # Возвращаем JSON-ответ, чтобы LLM мог прозрачно объяснить решение пользователю.
    return json.dumps(result, ensure_ascii=False, indent=2)
