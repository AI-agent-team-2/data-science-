from __future__ import annotations

import hashlib
import logging
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


def load_documents(data_dir: str) -> list[SourceDocument]:
    """Рекурсивно загружает `.txt` документы и очищает их для RAG."""
    data_path = Path(data_dir)
    documents: list[SourceDocument] = []

    for path in sorted(data_path.rglob("*.txt")):
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError:
            logger.exception("Failed to read document: %s", path)
            continue

        source = str(path.relative_to(data_path)).replace("\\", "/")
        cleaned_text = preprocess_for_rag(raw_text, source)
        if not cleaned_text:
            continue

        documents.append(SourceDocument(source=source, text=cleaned_text))

    logger.info("Loaded %d source documents from %s", len(documents), data_path)
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


def chunk_documents(documents: list[SourceDocument]) -> list[TextChunk]:
    """Разбивает документы на чанки и генерирует стабильные идентификаторы."""
    splitter = _create_text_splitter()
    chunks: list[TextChunk] = []

    for document in documents:
        parts = splitter.split_text(document.text)
        seen_hashes: dict[str, int] = {}

        for index, part in enumerate(parts):
            content_hash = hashlib.sha256(part.encode("utf-8")).hexdigest()
            occurrence = seen_hashes.get(content_hash, 0)
            seen_hashes[content_hash] = occurrence + 1
            stable_chunk_id = f"{document.source}:{content_hash[:16]}:{occurrence}"

            chunks.append(
                TextChunk(
                    chunk_id=stable_chunk_id,
                    text=part,
                    metadata={
                        "source": document.source,
                        "chunk_id": index,
                        "content_hash": content_hash,
                    },
                )
            )

    logger.info("Prepared %d chunks", len(chunks))
    return chunks


def _get_or_create_collection(client: chromadb.PersistentClient):
    """Возвращает коллекцию Chroma и привязывает embedding-функцию."""
    embedding_function = create_embedding_function()

    try:
        collection = client.get_collection(name=settings.collection_name)
        collection._embedding_function = embedding_function
        return collection
    except Exception:
        logger.info("Collection '%s' not found. Creating a new one.", settings.collection_name)
        return client.create_collection(
            name=settings.collection_name,
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
    collection = _get_or_create_collection(client)

    non_empty_chunks = [chunk for chunk in chunks if chunk.text.strip()]
    if not non_empty_chunks:
        logger.warning("No non-empty chunks to index")
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
        "Chroma sync completed: upserted=%d, deleted=%d, total=%d",
        len(to_upsert_indices),
        len(stale_ids),
        len(non_empty_chunks),
    )


def main() -> int:
    """CLI-entrypoint для индексации локальной базы знаний."""
    logging.basicConfig(level=logging.INFO)

    documents = load_documents(str(DEFAULT_DATA_DIR))
    chunks = chunk_documents(documents)
    store_in_chroma(chunks)
    logger.info("Indexed %d chunks", len(chunks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
