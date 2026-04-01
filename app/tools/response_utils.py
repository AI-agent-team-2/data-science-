from __future__ import annotations

from typing import Any


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
