from __future__ import annotations

import re
from pathlib import Path

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    # Убираем HTML-теги и нормализуем пробелы, сохраняя знаки препинания.
    without_html = _HTML_TAG_RE.sub(" ", text)
    return _MULTISPACE_RE.sub(" ", without_html).strip()


def struct_data(text: str, source_path: str) -> str:
    # Добавляем минимальную структуру по имени файла как в legacy-реализации.
    title = Path(source_path).stem
    return (
        f"Название:\n{title}\n\n"
        f"Описание:\n{text}\n\n"
        "Характеристики:\n-\n\n"
        "Применение:\n-"
    )


def preprocess_for_rag(text: str, source_path: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    return struct_data(cleaned, source_path)
