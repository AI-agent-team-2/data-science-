from __future__ import annotations

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from app.config import settings


def create_embedding_function():
    # В текущем runtime поддерживаем один понятный вариант: OpenAI-compatible embeddings.
    return OpenAIEmbeddingFunction(
        api_key=settings.resolved_embedding_api_key,
        api_base=settings.resolved_embedding_base_url,
        model_name=settings.resolved_embedding_model_name,
    )
