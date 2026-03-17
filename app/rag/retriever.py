from __future__ import annotations

from typing import Any

import chromadb

from app.config import settings
from app.rag.embeddings import create_embedding_function


class ChromaRetriever:
    def __init__(self) -> None:
        # Инициализируем persistent-клиент Chroma в локальной директории.
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        # Модель эмбеддингов должна совпадать с моделью на этапе индексации.
        self.embedding_fn = create_embedding_function()
        # Получаем коллекцию, где лежат текстовые чанки и их векторы.
        try:
            self.collection = self.client.get_collection(name=settings.collection_name)
            # Если коллекция существует, проверяем/обновляем функцию эмбеддингов
            self.collection._embedding_function = self.embedding_fn
        except Exception:
            self.collection = self.client.create_collection(
                name=settings.collection_name,
                embedding_function=self.embedding_fn,
            )

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        # Приоритет у явно переданного top_k; иначе берем значение из настроек.
        n_results = top_k or settings.top_k

        # Выполняем векторный поиск по текстовому запросу.
        result = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        # Chroma возвращает массивы "по запросам", берем первый (у нас он один).
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        # Приводим формат результата к удобному виду для tool-ответа.
        output: list[dict[str, Any]] = []
        for doc, meta, distance in zip(docs, metadatas, distances):
            output.append(
                {
                    "text": doc,
                    "metadata": meta or {},
                    # Чем меньше distance, тем выше score (простая нормализация).
                    "score": float(1 / (1 + distance)) if distance is not None else 0.0,
                }
            )
        return output
