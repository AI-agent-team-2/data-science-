from __future__ import annotations

from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)

from app.config import settings


def create_embedding_function():
    # В auto-режиме backend выбирается на основе MODEL_PROVIDER.
    if settings.resolved_embedding_backend == "local":
        return SentenceTransformerEmbeddingFunction(
            model_name=settings.resolved_embedding_model_name
        )

    return OpenAIEmbeddingFunction(
        api_key=settings.resolved_embedding_api_key,
        api_base=settings.resolved_embedding_base_url,
        model_name=settings.resolved_embedding_model_name,
    )
