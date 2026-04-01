from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings

SKU_INDEX_FILENAME = "sku_index.json"


def sku_index_path() -> Path:
    return Path(settings.chroma_path) / SKU_INDEX_FILENAME


def load_sku_index() -> dict[str, list[dict[str, Any]]]:
    path = sku_index_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    normalized: dict[str, list[dict[str, Any]]] = {}
    for sku, items in payload.items():
        if not isinstance(sku, str) or not isinstance(items, list):
            continue
        normalized[sku] = [item for item in items if isinstance(item, dict)]
    return normalized


def save_sku_index(index: dict[str, list[dict[str, Any]]]) -> None:
    path = sku_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
