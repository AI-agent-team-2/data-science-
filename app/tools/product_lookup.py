from __future__ import annotations

import json
from langchain_core.tools import tool


@tool
def product_lookup(query: str) -> str:
    """
    Поиск товара по названию, бренду, артикулу или ключевым параметрам.
    """
    # TODO: заменить на поиск в реальном каталоге/БД (SQL/Elastic/API ERP).
    # Сейчас заглушка нужна для проверки маршрутизации tool-calls в графе.
    mock_results = [
        {
            "name": "Шаровой кран 1/2",
            "brand": "DemoBrand",
            "sku": "SKU-001",
            "category": "краны",
            "attrs": {
                "diameter": "1/2",
                "material": "латунь",
                "use_case": "горячая/холодная вода",
            },
        }
    ]
    # На выходе всегда строка JSON со структурой найденных позиций.
    return json.dumps(mock_results, ensure_ascii=False, indent=2)
