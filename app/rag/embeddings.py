from __future__ import annotations

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from app.config import settings


def create_embedding_function() -> OpenAIEmbeddingFunction:
    """Создает embedding-функцию с настройками OpenAI-compatible API."""
    return OpenAIEmbeddingFunction(
        api_key=settings.resolved_embedding_api_key,
        api_base=settings.resolved_embedding_base_url,
        model_name=settings.resolved_embedding_model_name,
    )
