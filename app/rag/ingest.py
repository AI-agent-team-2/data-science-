from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.rag.embeddings import create_embedding_function
from app.rag.preprocess_text import preprocess_for_rag


def load_documents(data_dir: str) -> list[dict]:
    # Загружаем все txt-файлы рекурсивно (включая вложенные каталоги cat/qe/tp).
    docs = []
    for path in Path(data_dir).rglob("*.txt"):
        raw_text = path.read_text(encoding="utf-8")
        text = preprocess_for_rag(raw_text, str(path.relative_to(data_dir)))
        if not text:
            continue
        docs.append(
            {
                # Храним относительный путь как источник для трассировки ответа.
                "source": str(path.relative_to(data_dir)),
                "text": text,
            }
        )
    return docs


def chunk_documents(documents: list[dict]) -> list[dict]:
    # Режем документы на пересекающиеся чанки для лучшего recall в векторном поиске.
    chunk_size = max(200, settings.chunk_size)
    chunk_overlap = max(0, min(settings.chunk_overlap, chunk_size - 1))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", ";", " ", ""],
    )

    chunks = []
    for doc in documents:
        parts = splitter.split_text(doc["text"])
        seen_hashes: dict[str, int] = {}
        for i, part in enumerate(parts):
            content_hash = hashlib.sha256(part.encode("utf-8")).hexdigest()
            occurrence = seen_hashes.get(content_hash, 0)
            seen_hashes[content_hash] = occurrence + 1
            # ID стабилен для одинакового source+content.
            # occurrence нужен, если в одном документе встречаются одинаковые чанки.
            chunk_id = f'{doc["source"]}:{content_hash[:16]}:{occurrence}'
            chunks.append(
                {
                    "id": chunk_id,
                    "text": part,
                    "metadata": {
                        "source": doc["source"],
                        "chunk_id": i,
                        "content_hash": content_hash,
                    },
                }
            )
    return chunks


def store_in_chroma(chunks: list[dict]) -> None:
    # Подключаемся к persistent-хранилищу Chroma.
    client = chromadb.PersistentClient(path=settings.chroma_path)
    embedding_fn = create_embedding_function()
    # Берем/создаем коллекцию и привязываем embedding-функцию.
    try:
        collection = client.get_collection(name=settings.collection_name)
        # Если коллекция существует, проверяем/обновляем функцию эмбеддингов
        collection._embedding_function = embedding_fn
    except Exception:
        collection = client.create_collection(
            name=settings.collection_name,
            embedding_function=embedding_fn,
        )
    # Подготавливаем структуры для массовой вставки (с фильтрацией пустых текстов).
    filtered_chunks = [c for c in chunks if c.get("text", "").strip()]
    ids = [c["id"] for c in filtered_chunks]
    documents = [c["text"] for c in filtered_chunks]
    metadatas = [c["metadata"] for c in filtered_chunks]

    # Получаем текущие id, чтобы сделать инкрементальный sync без полного удаления коллекции.
    existing = collection.get(include=["metadatas"])
    existing_ids = set(existing.get("ids") or [])
    new_ids = set(ids)

    # Сохраняем только новые/измененные чанки батчами.
    # Для существующих id с тем же content_hash перезапись не нужна.
    existing_meta_by_id = {
        ex_id: meta or {}
        for ex_id, meta in zip(existing.get("ids") or [], existing.get("metadatas") or [])
    }
    to_upsert_indices = []
    for idx, cid in enumerate(ids):
        new_hash = metadatas[idx].get("content_hash")
        old_hash = existing_meta_by_id.get(cid, {}).get("content_hash")
        if old_hash != new_hash:
            to_upsert_indices.append(idx)

    batch_size = max(1, settings.embedding_batch_size)
    for start in range(0, len(to_upsert_indices), batch_size):
        end = start + batch_size
        batch_idx = to_upsert_indices[start:end]
        collection.upsert(
            ids=[ids[i] for i in batch_idx],
            documents=[documents[i] for i in batch_idx],
            metadatas=[metadatas[i] for i in batch_idx],
        )

    # Удаляем только устаревшие чанки, которых больше нет в актуальном наборе.
    stale_ids = list(existing_ids - new_ids)
    for start in range(0, len(stale_ids), batch_size):
        end = start + batch_size
        collection.delete(ids=stale_ids[start:end])


if __name__ == "__main__":
    # Локальный CLI-режим: загрузка raw-текстов, чанкование, индексация в Chroma.
    docs = load_documents("data/knowledge_base")
    chunks = chunk_documents(docs)
    store_in_chroma(chunks)
    print(f"Indexed {len(chunks)} chunks")
