from __future__ import annotations

import json
from typing import Any


def to_json(payload: dict[str, Any]) -> str:
    """Сериализует payload в единый JSON-формат для инструментов."""
    return json.dumps(payload, ensure_ascii=False, indent=2)


def empty_results_payload(query: str, note: str = "", **extra: Any) -> dict[str, Any]:
    """Строит типовой payload для пустой выдачи инструмента."""
    base: dict[str, Any] = {
        "query": query,
        "count": 0,
        "results": [],
    }
    if note:
        base["note"] = note
    base.update(extra)
    return base
