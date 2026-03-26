from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.rag.embeddings import create_embedding_function
from app.rag.preprocess_text import preprocess_for_rag

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/knowledge_base")
MIN_CHUNK_SIZE = 200
MISSING_DATA_MARKER = "[НЕТ ДАННЫХ В ИСХОДНОМ ДОКУМЕНТЕ]"

HEADER_LINE_PATTERN = re.compile(r"^\s*([A-Z][A-Z0-9_ ]{1,50})\s*:\s*(.+?)\s*$")
SECTION_HEADER_PATTERN = re.compile(r"^[A-Z][A-Z0-9 ()/&_-]{2,}$")
NON_ID_CHARS_PATTERN = re.compile(r"[^a-z0-9]+")
NON_SKU_CHARS_PATTERN = re.compile(r"[^A-Z0-9]+")
MULTI_VALUE_SPLIT_PATTERN = re.compile(r"[,\n;|]+")
PRODUCT_COLLECTION_SUFFIX = "_products"


@dataclass(frozen=True)
class SourceDocument:
    """Документ, подготовленный к чанкованию."""

    source: str
    text: str


@dataclass(frozen=True)
class TextChunk:
    """Чанк документа для индексации в Chroma."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StructuredDocument:
    """Структурированное представление документа с секциями."""

    source: str
    text: str
    fields: dict[str, str]
    sections: list[tuple[str, str]]


@dataclass(frozen=True)
class ProductRecord:
    """Карточка товара для product-level индекса в Chroma."""

    product_id: str
    lookup_text: str
    metadata: dict[str, Any]


def load_documents(data_dir: str) -> list[SourceDocument]:
    """Рекурсивно загружает `.txt` документы и очищает их для RAG."""
    data_path = Path(data_dir)
    documents: list[SourceDocument] = []

    for path in sorted(data_path.rglob("*.txt")):
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError:
            logger.exception("Не удалось прочитать документ: %s", path)
            continue

        source = str(path.relative_to(data_path)).replace("\\", "/")
        cleaned_text = preprocess_for_rag(raw_text, source)
        if not cleaned_text:
            continue

        documents.append(SourceDocument(source=source, text=cleaned_text))

    logger.info("Загружено %d исходных документов из %s", len(documents), data_path)
    return documents


def _create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Создает splitter с безопасной валидацией параметров из конфигурации."""
    chunk_size = max(MIN_CHUNK_SIZE, settings.chunk_size)
    chunk_overlap = max(0, min(settings.chunk_overlap, chunk_size - 1))

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", ";", " ", ""],
    )


def _normalize_value(value: str) -> str:
    """Нормализует текстовое значение поля."""
    normalized = " ".join(value.strip().split())
    if not normalized or normalized == MISSING_DATA_MARKER:
        return ""
    return normalized


def _slugify(value: str) -> str:
    """Преобразует строку в безопасный идентификатор для chunk-id."""
    lowered = value.strip().lower()
    lowered = NON_ID_CHARS_PATTERN.sub("_", lowered)
    lowered = lowered.strip("_")
    return lowered or "section"


def _extract_fields(text: str) -> dict[str, str]:
    """Извлекает поля формата `FIELD: value`."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = HEADER_LINE_PATTERN.match(line.strip())
        if not match:
            continue
        key = match.group(1).strip().upper()
        value = _normalize_value(match.group(2))
        fields[key] = value
    return fields


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Разбивает документ на именованные секции по верхнеуровневым заголовкам."""
    sections: list[tuple[str, str]] = []
    current_header = "OVERVIEW"
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and ":" not in line and SECTION_HEADER_PATTERN.fullmatch(line):
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_header, body))
            current_header = line
            current_lines = []
            continue

        current_lines.append(raw_line)

    tail = "\n".join(current_lines).strip()
    if tail:
        sections.append((current_header, tail))

    return sections


def _extract_list_from_section(sections: list[tuple[str, str]], section_name: str) -> str:
    """Извлекает элементы списочной секции и возвращает их строкой через запятую."""
    for header, body in sections:
        if header != section_name:
            continue
        values: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                normalized = _normalize_value(stripped[2:])
            else:
                normalized = _normalize_value(stripped)
            if normalized:
                values.append(normalized)
        return ", ".join(values)
    return ""


def _extract_field_value(text: str, field_name: str) -> str:
    """Извлекает значение поля с поддержкой имен, не покрытых _extract_fields()."""
    pattern = re.compile(rf"^\s*{re.escape(field_name)}\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
    match = pattern.search(text)
    return _normalize_value(match.group(1)) if match else ""


def _split_multi_values(raw: str) -> list[str]:
    """Разбирает поле/секцию со списком значений в нормализованный список."""
    if not raw:
        return []
    items: list[str] = []
    for part in MULTI_VALUE_SPLIT_PATTERN.split(raw):
        normalized = _normalize_value(part.lstrip("-").strip())
        if normalized:
            items.append(normalized)
    return items


def _canonical_sku(value: str) -> str:
    """Нормализует SKU для стабильного сравнения и dedupe."""
    return NON_SKU_CHARS_PATTERN.sub("", value.upper())


def _extract_articles_values(structured: StructuredDocument) -> list[str]:
    """
    Извлекает артикулы в порядке приоритета:
    1) секция ARTICLES
    2) секция/поле VARIANTS (АРТИКУЛЫ)
    3) поле SKU
    """
    values: list[str] = []
    values.extend(_split_multi_values(_extract_list_from_section(structured.sections, "ARTICLES")))
    values.extend(_split_multi_values(_extract_list_from_section(structured.sections, "VARIANTS (АРТИКУЛЫ)")))
    values.extend(_split_multi_values(_extract_field_value(structured.text, "VARIANTS (АРТИКУЛЫ)")))
    values.extend(_split_multi_values(_extract_field_value(structured.text, "SKU")))

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _canonical_sku(value)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(value)
    return deduped


def _extract_aliases_values(structured: StructuredDocument) -> list[str]:
    """Извлекает aliases из поля/секции ALIASES."""
    values: list[str] = []
    values.extend(_split_multi_values(structured.fields.get("ALIASES", "")))
    values.extend(_split_multi_values(_extract_list_from_section(structured.sections, "ALIASES")))
    values.extend(_split_multi_values(_extract_field_value(structured.text, "ALIASES")))

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(value)
    return deduped


def _parse_structured_document(document: SourceDocument) -> StructuredDocument:
    """Строит структурированное представление документа для секционного чанкования."""
    fields = _extract_fields(document.text)
    sections = _split_sections(document.text)
    return StructuredDocument(source=document.source, text=document.text, fields=fields, sections=sections)


def _build_chunk_text(structured: StructuredDocument, section: str, section_part: str) -> str:
    """Собирает чанк с кратким контекстом карточки и текущей секции."""
    lead_lines = [
        f"DOCUMENT: {structured.fields.get('DOCUMENT', '')}",
        f"DOC_ID: {structured.fields.get('DOC_ID', '')}",
        f"PRODUCT: {structured.fields.get('PRODUCT', '')}",
        f"BRAND: {structured.fields.get('BRAND', '')}",
        f"CATEGORY: {structured.fields.get('CATEGORY', '')}",
        f"SECTION: {section}",
    ]
    lead = "\n".join(line for line in lead_lines if not line.endswith(": "))
    return f"{lead}\n\n{section_part.strip()}".strip()


def _build_chunk_metadata(
    structured: StructuredDocument,
    section: str,
    content_hash: str,
    section_index: int,
    chunk_index: int,
) -> dict[str, Any]:
    """Формирует расширенную metadata для Chroma."""
    fields = structured.fields
    articles = _extract_articles_values(structured)
    return {
        "source": structured.source,
        "chunk_id": chunk_index,
        "content_hash": content_hash,
        "doc_id": fields.get("DOC_ID", ""),
        "document": fields.get("DOCUMENT", ""),
        "product": fields.get("PRODUCT", ""),
        "brand": fields.get("BRAND", ""),
        "category": fields.get("CATEGORY", ""),
        "model": fields.get("MODEL", ""),
        "manufacturer": fields.get("MANUFACTURER", ""),
        "country": fields.get("COUNTRY", ""),
        "section": section,
        "section_index": section_index,
        "articles": ", ".join(articles),
    }


def chunk_documents(documents: list[SourceDocument]) -> list[TextChunk]:
    """Разбивает документы на секционные чанки и генерирует стабильные идентификаторы."""
    splitter = _create_text_splitter()
    chunks: list[TextChunk] = []

    for document in documents:
        structured = _parse_structured_document(document)
        seen_hashes: dict[str, int] = {}
        global_index = 0
        sections = structured.sections or [("OVERVIEW", structured.text)]

        for section_index, (section_name, section_body) in enumerate(sections):
            parts = splitter.split_text(section_body)
            section_slug = _slugify(section_name)

            for part in parts:
                chunk_text = _build_chunk_text(structured, section_name, part)
                content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                dedupe_key = f"{section_slug}:{content_hash}"
                occurrence = seen_hashes.get(dedupe_key, 0)
                seen_hashes[dedupe_key] = occurrence + 1
                stable_chunk_id = f"{document.source}:{section_slug}:{content_hash[:16]}:{occurrence}"

                chunks.append(
                    TextChunk(
                        chunk_id=stable_chunk_id,
                        text=chunk_text,
                        metadata=_build_chunk_metadata(
                            structured=structured,
                            section=section_name,
                            content_hash=content_hash,
                            section_index=section_index,
                            chunk_index=global_index,
                        ),
                    )
                )
                global_index += 1

    logger.info("Подготовлено %d чанков", len(chunks))
    return chunks


def _build_product_lookup_text(structured: StructuredDocument, aliases: list[str], articles: list[str]) -> str:
    """Собирает компактный lookup_text карточки товара без длинного эксплуатационного контента."""
    fields = structured.fields
    lines = [
        f"PRODUCT: {fields.get('PRODUCT', '')}",
        f"BRAND: {fields.get('BRAND', '')}",
        f"CATEGORY: {fields.get('CATEGORY', '')}",
        f"MODEL: {fields.get('MODEL', '')}",
        f"ALIASES: {', '.join(aliases)}",
        f"ARTICLES: {', '.join(articles)}",
        f"DOCUMENT: {fields.get('DOCUMENT', '')}",
        f"DOC_ID: {fields.get('DOC_ID', '')}",
    ]
    return "\n".join(line for line in lines if not line.endswith(": ")).strip()


def build_product_records(documents: list[SourceDocument]) -> list[ProductRecord]:
    """Строит product-level записи: одна карточка на один исходный документ."""
    records: list[ProductRecord] = []

    for document in documents:
        structured = _parse_structured_document(document)
        fields = structured.fields
        aliases = _extract_aliases_values(structured)
        articles = _extract_articles_values(structured)
        lookup_text = _build_product_lookup_text(structured, aliases=aliases, articles=articles)
        if not lookup_text:
            continue

        product_id = f"product:{structured.source}"
        metadata = {
            "source": structured.source,
            "doc_id": fields.get("DOC_ID", ""),
            "document": fields.get("DOCUMENT", ""),
            "product": fields.get("PRODUCT", ""),
            "brand": fields.get("BRAND", ""),
            "category": fields.get("CATEGORY", ""),
            "model": fields.get("MODEL", ""),
            "manufacturer": fields.get("MANUFACTURER", ""),
            "country": fields.get("COUNTRY", ""),
            "aliases": ", ".join(aliases),
            "articles": ", ".join(articles),
            "articles_norm": ", ".join(_canonical_sku(article) for article in articles if _canonical_sku(article)),
            "content_hash": hashlib.sha256(lookup_text.encode("utf-8")).hexdigest(),
        }

        records.append(ProductRecord(product_id=product_id, lookup_text=lookup_text, metadata=metadata))

    logger.info("Подготовлено %d product-records", len(records))
    return records


def _get_or_create_collection(client: chromadb.PersistentClient, collection_name: str):
    """Возвращает коллекцию Chroma по имени и привязывает embedding-функцию."""
    embedding_function = create_embedding_function()

    try:
        collection = client.get_collection(name=collection_name)
        collection._embedding_function = embedding_function
        return collection
    except Exception:
        logger.info("Коллекция '%s' не найдена. Создаю новую.", collection_name)
        return client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )


def _build_upsert_batches(
    chunk_ids: list[str],
    metadatas: list[dict[str, Any]],
    existing_meta_by_id: dict[str, dict[str, Any]],
) -> list[int]:
    """Возвращает индексы чанков, которые нужно вставить/обновить."""
    to_upsert_indices: list[int] = []

    for index, chunk_id in enumerate(chunk_ids):
        new_hash = metadatas[index].get("content_hash")
        old_hash = existing_meta_by_id.get(chunk_id, {}).get("content_hash")
        if old_hash != new_hash:
            to_upsert_indices.append(index)

    return to_upsert_indices


def store_in_chroma(chunks: list[TextChunk]) -> None:
    """Синхронизирует чанки в persistent-коллекцию Chroma."""
    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = _get_or_create_collection(client, settings.collection_name)

    non_empty_chunks = [chunk for chunk in chunks if chunk.text.strip()]
    if not non_empty_chunks:
        logger.warning("Нет непустых чанков для индексации")
        return

    chunk_ids = [chunk.chunk_id for chunk in non_empty_chunks]
    documents = [chunk.text for chunk in non_empty_chunks]
    metadatas = [chunk.metadata for chunk in non_empty_chunks]

    existing = collection.get(include=["metadatas"])
    existing_ids = set(existing.get("ids") or [])
    new_ids = set(chunk_ids)

    existing_meta_by_id = {
        existing_id: metadata or {}
        for existing_id, metadata in zip(existing.get("ids") or [], existing.get("metadatas") or [])
    }

    to_upsert_indices = _build_upsert_batches(chunk_ids, metadatas, existing_meta_by_id)
    batch_size = max(1, settings.embedding_batch_size)

    for start in range(0, len(to_upsert_indices), batch_size):
        batch_indices = to_upsert_indices[start : start + batch_size]
        collection.upsert(
            ids=[chunk_ids[index] for index in batch_indices],
            documents=[documents[index] for index in batch_indices],
            metadatas=[metadatas[index] for index in batch_indices],
        )

    stale_ids = list(existing_ids - new_ids)
    for start in range(0, len(stale_ids), batch_size):
        collection.delete(ids=stale_ids[start : start + batch_size])

    logger.info(
        "Синхронизация Chroma завершена: upsert=%d, удалено=%d, всего=%d",
        len(to_upsert_indices),
        len(stale_ids),
        len(non_empty_chunks),
    )


def _product_collection_name() -> str:
    """Возвращает имя product-level коллекции на основе chunk коллекции."""
    return f"{settings.collection_name}{PRODUCT_COLLECTION_SUFFIX}"


def store_products_in_chroma(records: list[ProductRecord]) -> None:
    """Синхронизирует product-level карточки в отдельную Chroma-коллекцию."""
    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = _get_or_create_collection(client, _product_collection_name())

    if not records:
        logger.warning("Нет product-records для индексации")
        return

    record_ids = [record.product_id for record in records]
    documents = [record.lookup_text for record in records]
    metadatas = [record.metadata for record in records]

    existing = collection.get(include=["metadatas"])
    existing_ids = set(existing.get("ids") or [])
    new_ids = set(record_ids)

    existing_meta_by_id = {
        existing_id: metadata or {}
        for existing_id, metadata in zip(existing.get("ids") or [], existing.get("metadatas") or [])
    }

    to_upsert_indices = _build_upsert_batches(record_ids, metadatas, existing_meta_by_id)
    batch_size = max(1, settings.embedding_batch_size)

    for start in range(0, len(to_upsert_indices), batch_size):
        batch_indices = to_upsert_indices[start : start + batch_size]
        collection.upsert(
            ids=[record_ids[index] for index in batch_indices],
            documents=[documents[index] for index in batch_indices],
            metadatas=[metadatas[index] for index in batch_indices],
        )

    stale_ids = list(existing_ids - new_ids)
    for start in range(0, len(stale_ids), batch_size):
        collection.delete(ids=stale_ids[start : start + batch_size])

    logger.info(
        "Синхронизация product-коллекции завершена: upsert=%d, удалено=%d, всего=%d",
        len(to_upsert_indices),
        len(stale_ids),
        len(records),
    )


def main() -> int:
    """Точка входа CLI для индексации локальной базы знаний."""
    logging.basicConfig(level=logging.INFO)

    documents = load_documents(str(DEFAULT_DATA_DIR))
    chunks = chunk_documents(documents)
    product_records = build_product_records(documents)
    store_in_chroma(chunks)
    store_products_in_chroma(product_records)
    logger.info("Проиндексировано %d чанков", len(chunks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
