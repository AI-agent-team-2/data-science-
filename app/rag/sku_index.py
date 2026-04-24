from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings

SKU_INDEX_FILENAME = "sku_index.json"
logger = logging.getLogger(__name__)

def sku_index_path() -> Path:
    """Возвращает путь к файлу индекса SKU."""
    return Path(settings.chroma_path) / SKU_INDEX_FILENAME


def load_sku_index() -> Dict[str, List[Dict[str, Any]]]:
    """Загружает SKU индекс из файла."""
    path = sku_index_path()
    if not path.exists():
        logger.warning("SKU index file does not exist: %s", path)
        return {} # Возвращаем пустой словарь, если файл не найден 
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON from SKU index: %s", e)
        return {} # Возвращаем пустой словарь в случае ошибки декодирования 
    except Exception as e:
        logger.error("Unexpected error reading SKU index: %s", e)
        return {} # Возвращаем пустой словарь в случае других ошибок 
    if not isinstance(payload, dict):
        logger.warning("SKU index payload is not a dictionary.")
        return {} # Возвращаем пустой словарь, если загрузка не удалась

    normalized: Dict[str, List[Dict[str, Any]]] = {
        sku: [item for item in items if isinstance(item, dict)]
        for sku, items in payload.items()
        if isinstance(sku, str) and isinstance(items, list)
    }
    logger.info("Loaded SKU index with %d entries.", len(normalized))
    return normalized # Возвращаем нормализованный индекс 

def save_sku_index(index: Dict[str, List[Dict[str, Any]]]) -> None:
    """Сохраняет индекс SKU в файл."""
    path = sku_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error("Failed to save SKU index: %s", e)

