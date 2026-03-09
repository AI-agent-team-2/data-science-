from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def load_documents(data_dir: str) -> list[dict]:
    # Загружаем все txt-файлы рекурсивно (включая вложенные каталоги cat/qe/tp).
    docs = []
    for path in Path(data_dir).rglob("*.txt"):
        text = path.read_text(encoding="utf-8")
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
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    # Берем/создаем коллекцию и привязываем embedding-функцию.
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=embedding_fn,
    )

    # Подготавливаем батч-структуры для массовой вставки.
    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Для простоты очищаем коллекцию и заливаем заново.
    # TODO: заменить на инкрементальную индексацию/upsert для больших объемов данных.
    existing = collection.get()
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    # Сохраняем новые чанки и метаданные в векторную БД.
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )


if __name__ == "__main__":
    # Локальный CLI-режим: загрузка raw-текстов, чанкование, индексация в Chroma.
    docs = load_documents("data/knowledge_base")
    chunks = chunk_documents(docs)
    store_in_chroma(chunks)
    print(f"Indexed {len(chunks)} chunks")
