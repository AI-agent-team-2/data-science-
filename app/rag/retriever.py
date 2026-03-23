from __future__ import annotations

import logging
from typing import Any

import chromadb

from app.config import settings
from app.rag.embeddings import create_embedding_function

logger = logging.getLogger(__name__)

KEYWORD_STOP_WORDS: set[str] = {
    "что",
    "как",
    "где",
    "когда",
    "почему",
    "зачем",
    "какой",
    "сколько",
    "это",
}

SANITARY_TERMS: tuple[str, ...] = (
    "унитаз",
    "смеситель",
    "ванна",
    "душ",
    "кран",
    "труба",
    "фитинг",
    "инсталляция",
    "раковина",
    "бойлер",
    "полотенцесушитель",
)


class ChromaRetriever:
    """Обертка над Chroma для векторного и гибридного поиска."""

    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.embedding_function = create_embedding_function()

        try:
            self.collection = self.client.get_collection(name=settings.collection_name)
            self.collection._embedding_function = self.embedding_function
        except Exception:
            logger.info("Collection '%s' not found. Creating a new one.", settings.collection_name)
            self.collection = self.client.create_collection(
                name=settings.collection_name,
                embedding_function=self.embedding_function,
            )

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Выполняет векторный поиск по текстовому запросу."""
        n_results = top_k if top_k is not None else settings.top_k
        n_results = max(1, int(n_results))

        try:
            result = self.collection.query(query_texts=[query], n_results=n_results)
        except Exception:
            logger.exception("Chroma query failed for query=%s", query)
            return []

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        output: list[dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            score = float(1 / (1 + distance)) if distance is not None else 0.0
            output.append(
                {
                    "text": str(document),
                    "metadata": metadata or {},
                    "score": score,
                }
            )

        return output

    def hybrid_search(self, query: str, top_k: int | None = None, use_keyword: bool = True) -> list[dict[str, Any]]:
        """Выполняет гибридный поиск: векторный скор + keyword boosting."""
        n_results = top_k if top_k is not None else settings.top_k
        n_results = max(1, int(n_results))

        vector_results = self.search(query=query, top_k=n_results * 2)
        if not use_keyword:
            return vector_results[:n_results]

        keywords = self._extract_keywords(query)
        for result in vector_results:
            text = str(result.get("text", "")).lower()
            original_score = float(result.get("score", 0.0) or 0.0)

            boost = 0.0
            for keyword in keywords:
                if keyword in text:
                    boost += 0.15

            result["score_original"] = original_score
            result["boost"] = boost
            result["score"] = min(1.0, original_score + boost)

        vector_results.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        return vector_results[:n_results]

    def _extract_keywords(self, query: str) -> list[str]:
        """Извлекает ключевые слова запроса для keyword boosting."""
        lowered_query = query.lower()
        raw_words = lowered_query.split()

        keywords = [word for word in raw_words if word not in KEYWORD_STOP_WORDS and len(word) > 2]
        for term in SANITARY_TERMS:
            if term in lowered_query and term not in keywords:
                keywords.append(term)

        return keywords
