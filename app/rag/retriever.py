from __future__ import annotations

import logging
import re
from typing import Any

import chromadb

from app.config import settings
from app.rag.embeddings import create_embedding_function
from app.rag.sku_index import load_sku_index
from app.utils.sku import canonical_sku, extract_sku_candidates

logger = logging.getLogger(__name__)


PRODUCT_COLLECTION_SUFFIX = "_products"
WORD_PATTERN = re.compile(r"[a-zA-Zа-яА-Я0-9]+")


def _split_csv_values(value: str) -> list[str]:
    """Делит CSV-подобную строку на аккуратный список значений."""
    if not value:
        return []
    raw_parts = re.split(r"[,\n;|]+", value)
    result = [" ".join(part.strip().split()) for part in raw_parts]
    return [part for part in result if part]


def _tokenize(text: str) -> set[str]:
    """Разбивает текст на набор токенов для легкого post-rerank."""
    return {token.lower() for token in WORD_PATTERN.findall(text)}


def _load_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    embedding_function: Any,
) -> Any:
    """Загружает существующую коллекцию и не маскирует отсутствие индекса."""
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
    except Exception as exc:
        raise RuntimeError(f"Chroma collection '{collection_name}' is missing or unavailable") from exc
    return collection


class ChromaRetriever:
    """Обертка над Chroma для векторного поиска."""

    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.embedding_function = create_embedding_function()
        self.collection = _load_collection(
            client=self.client,
            collection_name=settings.collection_name,
            embedding_function=self.embedding_function,
        )

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Выполняет векторный поиск по текстовому запросу."""
        n_results = top_k if top_k is not None else settings.top_k
        n_results = max(1, int(n_results))

        try:
            result = self.collection.query(query_texts=[query], n_results=n_results)
        except Exception:
            logger.exception("Ошибка запроса к Chroma для query=%s", query)
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


class ProductRetriever:
    """Ретривер карточек товаров из отдельной product-level коллекции Chroma."""

    def __init__(self, collection_name: str | None = None) -> None:
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.embedding_function = create_embedding_function()
        self.collection_name = collection_name or f"{settings.collection_name}{PRODUCT_COLLECTION_SUFFIX}"
        self._sku_index = load_sku_index()
        self.collection = _load_collection(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        self._exact_sku_base_score = float(settings.product_exact_sku_base_score)
        self._exact_sku_per_match_score = float(settings.product_exact_sku_per_match_score)

    def extract_query_skus(self, query: str) -> set[str]:
        """Извлекает candidate SKU из пользовательского запроса."""
        return extract_sku_candidates(query, require_digit=True)

    def find_exact_sku_matches(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Ищет карточки по точному совпадению SKU."""
        query_skus = self.extract_query_skus(query)
        if not query_skus:
            return []

        indexed_matches = self._find_exact_sku_matches_from_index(query_skus)
        if indexed_matches:
            return indexed_matches[: max(1, int(limit))]

        try:
            snapshot = self.collection.get(include=["metadatas"])
        except Exception:
            logger.exception("Ошибка чтения product-коллекции для exact SKU поиска")
            return []

        metadatas = snapshot.get("metadatas") or []
        matches: list[dict[str, Any]] = []

        for metadata in metadatas:
            if not isinstance(metadata, dict):
                continue

            item_skus = self._extract_item_skus(metadata)
            matched = sorted(query_skus.intersection(item_skus))
            if not matched:
                continue

            score = self._exact_sku_base_score + self._exact_sku_per_match_score * len(matched)
            item = self._serialize_item(metadata=metadata, score=score)
            item["matched_skus"] = matched
            matches.append(item)

        matches.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return matches[: max(1, int(limit))]

    def _find_exact_sku_matches_from_index(self, query_skus: set[str]) -> list[dict[str, Any]]:
        """Использует локальный SKU-индекс вместо полного scan product metadata."""
        matches: list[dict[str, Any]] = []
        for query_sku in sorted(query_skus):
            for metadata in self._sku_index.get(query_sku, []):
                item = self._serialize_item(
                    metadata=metadata,
                    score=self._exact_sku_base_score + self._exact_sku_per_match_score,
                )
                item["matched_skus"] = [query_sku]
                matches.append(item)

        deduped: dict[str, dict[str, Any]] = {}
        for item in matches:
            dedupe_key = str(item.get("source") or "") or str(item.get("doc_id") or "") or str(item.get("name") or "")
            existing = deduped.get(dedupe_key)
            if existing is None:
                deduped[dedupe_key] = item
                continue
            existing_skus = set(existing.get("matched_skus") or [])
            merged_skus = sorted(existing_skus.union(item.get("matched_skus") or []))
            existing["matched_skus"] = merged_skus
            existing["score"] = round(
                self._exact_sku_base_score + self._exact_sku_per_match_score * len(merged_skus),
                4,
            )

        return sorted(deduped.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)

    def semantic_search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Делает fallback semantic-поиск по `lookup_text` product-коллекции."""
        top_n = max(1, int(limit))
        query_tokens = _tokenize(query)

        try:
            result = self.collection.query(query_texts=[query], n_results=max(top_n * 3, top_n))
        except Exception:
            logger.exception("Ошибка semantic-поиска по product-коллекции для query=%s", query)
            return []

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        candidates: list[dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if not isinstance(metadata, dict):
                continue

            base_score = float(1 / (1 + distance)) if distance is not None else 0.0
            lookup_text = str(document or "")
            overlap = len(query_tokens.intersection(_tokenize(lookup_text)))
            rerank_bonus = 0.03 * overlap
            score = min(1.0, base_score + rerank_bonus)

            candidates.append(self._serialize_item(metadata=metadata, score=score))

        # Дедупликация по source/doc_id с сохранением лучшего score.
        deduped: dict[str, dict[str, Any]] = {}
        for item in candidates:
            dedupe_key = str(item.get("source") or "") or str(item.get("doc_id") or "")
            if not dedupe_key:
                dedupe_key = str(item.get("name", "unknown"))
            existing = deduped.get(dedupe_key)
            if existing is None or float(item.get("score", 0.0)) > float(existing.get("score", 0.0)):
                deduped[dedupe_key] = item

        ranked = sorted(deduped.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked[:top_n]

    def _extract_item_skus(self, metadata: dict[str, Any]) -> set[str]:
        """Извлекает normalized SKU из metadata карточки."""
        values: list[str] = []

        raw_articles_norm = metadata.get("articles_norm")
        if isinstance(raw_articles_norm, str):
            values.extend(_split_csv_values(raw_articles_norm))

        raw_articles = metadata.get("articles")
        if isinstance(raw_articles, str):
            values.extend(_split_csv_values(raw_articles))
        elif isinstance(raw_articles, list):
            values.extend(str(value) for value in raw_articles)

        normalized = {canonical_sku(value) for value in values}
        return {value for value in normalized if value}

    def _serialize_item(self, metadata: dict[str, Any], score: float) -> dict[str, Any]:
        """Приводит metadata карточки к единому runtime-формату product_lookup."""
        name = str(metadata.get("product") or metadata.get("document") or metadata.get("doc_id") or "").strip()
        brand = str(metadata.get("brand") or "").strip()
        category = str(metadata.get("category") or "").strip()
        source = str(metadata.get("source") or "").strip()
        doc_id = str(metadata.get("doc_id") or "").strip()

        raw_articles = metadata.get("articles")
        if isinstance(raw_articles, str):
            sku_list = _split_csv_values(raw_articles)
        elif isinstance(raw_articles, list):
            sku_list = [str(value).strip() for value in raw_articles if str(value).strip()]
        else:
            sku_list = []

        return {
            "name": name,
            "brand": brand,
            "category": category,
            "sku_list": sku_list[:20],
            "source": source,
            "doc_id": doc_id,
            "score": round(float(score), 4),
        }

