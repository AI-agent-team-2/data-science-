from __future__ import annotations

import re

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TRAILING_SPACES_RE = re.compile(r"[ \t]+$")
_INLINE_SPACES_RE = re.compile(r"[ \t]{2,}")
_EXCESSIVE_BLANKS_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    # Бережная очистка для RAG-каталога:
    # - сохраняем переносы строк и блоки таблиц;
    # - убираем HTML-теги;
    # - убираем хвостовые пробелы и избыточные пустые строки.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    without_html = _HTML_TAG_RE.sub(" ", normalized)

    cleaned_lines = []
    for line in without_html.split("\n"):
        line = _TRAILING_SPACES_RE.sub("", line)
        line = _INLINE_SPACES_RE.sub(" ", line)
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = _EXCESSIVE_BLANKS_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def preprocess_for_rag(text: str, source_path: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    return cleaned
