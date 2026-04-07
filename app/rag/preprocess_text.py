from __future__ import annotations

import re

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_TRAILING_SPACES_PATTERN = re.compile(r"[ \t]+$")
_INLINE_SPACES_PATTERN = re.compile(r"[ \t]{2,}")
_EXCESSIVE_BLANKS_PATTERN = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """
    Выполняет бережную очистку текста для RAG.

    Что делает:
    1. Нормализует переносы строк.
    2. Удаляет HTML-теги.
    3. Убирает лишние пробелы и избыточные пустые строки.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    without_html = _HTML_TAG_PATTERN.sub(" ", normalized)

    cleaned_lines: list[str] = []
    for raw_line in without_html.split("\n"):
        line = _TRAILING_SPACES_PATTERN.sub("", raw_line)
        line = _INLINE_SPACES_PATTERN.sub(" ", line)
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = _EXCESSIVE_BLANKS_PATTERN.sub("\n\n", cleaned)
    return cleaned.strip()


def preprocess_for_rag(text: str, source_path: str) -> str:
    """
    Подготавливает документ к индексации в RAG.

    Args:
        text: Сырой текст документа.
        source_path: Относительный путь источника (параметр сохранен для расширения логики).

    Returns:
        Очищенный текст или пустую строку.
    """
    _ = source_path
    cleaned = clean_text(text)
    return cleaned if cleaned else ""
