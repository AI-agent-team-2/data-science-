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

    def hybrid_search(self, query: str, top_k: int | None = None, use_keyword: bool = True) -> list[dict[str, Any]]:
        """Гибридный поиск: векторный + бустинг по ключевым словам"""
        n_results = top_k or settings.top_k
        
        # 1. Векторный поиск (берем в 2 раза больше для последующей фильтрации)
        vector_results = self.search(query, top_k=n_results * 2)
        
        if not use_keyword:
            return vector_results[:n_results]
        
        # 2. Извлекаем ключевые слова из запроса
        keywords = self._extract_keywords(query)
        
        # 3. Бустим результаты по ключевым словам
        for result in vector_results:
            text = result.get('text', '').lower()
            score = result.get('score', 0)
            
            # Повышаем релевантность за каждое ключевое слово
            boost = 0
            for kw in keywords:
                if kw in text:
                    boost += 0.15
            
            result['score'] = min(1.0, score + boost)
            result['score_original'] = score
            result['boost'] = boost
        
        # 4. Сортируем по новому скору
        vector_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return vector_results[:n_results]

    def _extract_keywords(self, query: str) -> list[str]:
        """Извлечь ключевые слова из запроса"""
        # Стоп-слова, которые не несет смысловой нагрузки
        stop_words = {'что', 'как', 'где', 'когда', 'почему', 'зачем', 'какой', 'сколько', 'это'}
        
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Добавляем слова из словаря сантехники, если они есть в запросе
        sanitary_terms = [
            'унитаз', 'смеситель', 'ванна', 'душ', 'кран', 'труба', 'фитинг',
            'инсталляция', 'раковина', 'бойлер', 'полотенцесушитель'
        ]
        for term in sanitary_terms:
            if term in query.lower() and term not in keywords:
                keywords.append(term)
        
        return keywords