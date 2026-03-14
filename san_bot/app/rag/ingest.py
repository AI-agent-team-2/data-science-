from __future__ import annotations

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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", ",", ";", " ", ""],
    )

    chunks = []
    for doc in documents:
        parts = splitter.split_text(doc["text"])
        for i, part in enumerate(parts):
            chunks.append(
                {
                    # Детерминированный id нужен для повторных заливок и диагностики.
                    "id": f'{doc["source"]}_{i}',
                    "text": part,
                    "metadata": {
                        "source": doc["source"],
                        "chunk_id": i,
                    },
                }
            )
    return chunks


def store_in_chroma(chunks: list[dict]) -> None:
    # Подключаемся к persistent-хранилищу Chroma.
    client = chromadb.PersistentClient(path=settings.chroma_path)
    embedding_fn = create_embedding_function()
    # Берем/создаем коллекцию и привязываем embedding-функцию.
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=embedding_fn,
    )

    # Подготавливаем структуры для массовой вставки (с фильтрацией пустых текстов).
    filtered_chunks = [c for c in chunks if c.get("text", "").strip()]
    ids = [c["id"] for c in filtered_chunks]
    documents = [c["text"] for c in filtered_chunks]
    metadatas = [c["metadata"] for c in filtered_chunks]

    # Для простоты очищаем коллекцию и заливаем заново.
    # TODO: заменить на инкрементальную индексацию/upsert для больших объемов данных.
    existing = collection.get()
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    # Сохраняем чанки батчами, чтобы cloud embedding API не падал на больших payload.
    batch_size = max(1, settings.embedding_batch_size)
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


if __name__ == "__main__":
    # Локальный CLI-режим: загрузка raw-текстов, чанкование, индексация в Chroma.
    docs = load_documents("data/knowledge_base")
    chunks = chunk_documents(docs)
    store_in_chroma(chunks)
    print(f"Indexed {len(chunks)} chunks")
