from __future__ import annotations

from dataclasses import dataclass

import chromadb

from app.config import settings

PRODUCT_COLLECTION_SUFFIX = "_products"


@dataclass(frozen=True)
class IndexHealth:
    chunk_collection: str
    product_collection: str
    chunk_count: int
    product_count: int

    @property
    def is_ready(self) -> bool:
        return self.chunk_count > 0 and self.product_count > 0


def product_collection_name() -> str:
    return f"{settings.collection_name}{PRODUCT_COLLECTION_SUFFIX}"


def _safe_collection_count(client: chromadb.PersistentClient, collection_name: str) -> int:
    try:
        return client.get_collection(collection_name).count()
    except Exception:
        return 0


def get_index_health() -> IndexHealth:
    client = chromadb.PersistentClient(path=settings.chroma_path)
    chunk_collection = settings.collection_name
    product_collection = product_collection_name()
    return IndexHealth(
        chunk_collection=chunk_collection,
        product_collection=product_collection,
        chunk_count=_safe_collection_count(client, chunk_collection),
        product_count=_safe_collection_count(client, product_collection),
    )


def require_ready_index() -> IndexHealth:
    health = get_index_health()
    if health.is_ready:
        return health
    raise RuntimeError(
        "Chroma index is not ready: "
        f"{health.chunk_collection}={health.chunk_count}, "
        f"{health.product_collection}={health.product_count}"
    )
