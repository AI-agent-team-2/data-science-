from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROVIDER: Final[str] = "openrouter"
DEFAULT_OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_BASE_URL: Final[str] = "https://api.openai.com/v1"
DEFAULT_LANGFUSE_HOST: Final[str] = "https://cloud.langfuse.com"
DEFAULT_OPENROUTER_MODEL: Final[str] = "openai/gpt-4o-mini"
DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
DEFAULT_COLLECTION_NAME: Final[str] = "sanitary_goods"
DEFAULT_TOP_K: Final[int] = 6
DEFAULT_EMBEDDING_BATCH_SIZE: Final[int] = 64
DEFAULT_CHUNK_SIZE: Final[int] = 900
DEFAULT_CHUNK_OVERLAP: Final[int] = 140
DEFAULT_HISTORY_MAX_MESSAGES: Final[int] = 24
DEFAULT_HISTORY_TTL_DAYS: Final[int] = 30
DEFAULT_WEB_CACHE_ENABLED: Final[bool] = True
DEFAULT_WEB_CACHE_TTL_HOURS: Final[int] = 24
DEFAULT_WEB_SEARCH_MAX_RESULTS: Final[int] = 5
DEFAULT_ENABLE_WEB_SEARCH: Final[bool] = True
DEFAULT_ENABLE_RAG: Final[bool] = True
DEFAULT_ENABLE_PRODUCT_LOOKUP: Final[bool] = True
SUPPORTED_PROVIDERS: Final[set[str]] = {"openrouter", "openai"}


def _get_env_str(name: str, default: str = "") -> str:
    """Возвращает строковое значение переменной окружения без лишних пробелов."""
    return os.getenv(name, default).strip()


def _get_env_bool(name: str, default: bool) -> bool:
    """Преобразует переменную окружения в bool (`true/false`, `1/0`, `yes/no`)."""
    raw_value = _get_env_str(name, "true" if default else "false").lower()
    return raw_value in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    """Конфигурация приложения, загружаемая из `.env` и переменных окружения."""

    model_provider: str = _get_env_str("MODEL_PROVIDER", DEFAULT_PROVIDER).lower()
    openai_api_key: str = _get_env_str("OPENAI_API_KEY")
    model_name: str = _get_env_str("MODEL_NAME")
    openai_base_url: str = _get_env_str("OPENAI_BASE_URL")

    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    embedding_api_key: str = _get_env_str("EMBEDDING_API_KEY")
    embedding_base_url: str = _get_env_str("EMBEDDING_BASE_URL")

    telegram_token: str = _get_env_str("TELEGRAM_TOKEN")
    chroma_path: str = _get_env_str("CHROMA_PATH", "./chroma_db")
    collection_name: str = DEFAULT_COLLECTION_NAME

    top_k: int = DEFAULT_TOP_K
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    history_db_path: str = _get_env_str("HISTORY_DB_PATH", "./history.db")
    history_max_messages: int = DEFAULT_HISTORY_MAX_MESSAGES
    history_ttl_days: int = DEFAULT_HISTORY_TTL_DAYS

    web_cache_enabled: bool = DEFAULT_WEB_CACHE_ENABLED
    web_cache_ttl_hours: int = DEFAULT_WEB_CACHE_TTL_HOURS
    web_search_max_results: int = DEFAULT_WEB_SEARCH_MAX_RESULTS

    enable_web_search: bool = DEFAULT_ENABLE_WEB_SEARCH
    enable_rag: bool = DEFAULT_ENABLE_RAG
    enable_product_lookup: bool = DEFAULT_ENABLE_PRODUCT_LOOKUP

    langfuse_public_key: str = _get_env_str("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = _get_env_str("LANGFUSE_SECRET_KEY")
    langfuse_host: str = _get_env_str("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST)
    langfuse_enabled: bool = _get_env_bool("LANGFUSE_ENABLED", False)

    @property
    def resolved_model_provider(self) -> str:
        """Возвращает поддерживаемый провайдер LLM."""
        if self.model_provider in SUPPORTED_PROVIDERS:
            return self.model_provider
        return DEFAULT_PROVIDER

    @property
    def resolved_openai_base_url(self) -> str:
        """Возвращает итоговый базовый URL OpenAI-compatible API."""
        if self.openai_base_url:
            return self.openai_base_url

        if self.resolved_model_provider == "openai":
            return DEFAULT_OPENAI_BASE_URL
        return DEFAULT_OPENROUTER_BASE_URL

    @property
    def resolved_openai_api_key(self) -> str:
        """Возвращает ключ API для LLM."""
        return self.openai_api_key

    @property
    def resolved_model_name(self) -> str:
        """Возвращает итоговое имя модели для чата."""
        if self.model_name:
            return self.model_name

        if self.resolved_model_provider == "openai":
            return DEFAULT_OPENAI_MODEL
        return DEFAULT_OPENROUTER_MODEL

    @property
    def resolved_embedding_model_name(self) -> str:
        """Возвращает имя модели эмбеддингов."""
        return self.embedding_model_name or DEFAULT_EMBEDDING_MODEL

    @property
    def resolved_embedding_api_key(self) -> str:
        """Возвращает ключ embedding API (или общий API-ключ)."""
        return self.embedding_api_key or self.resolved_openai_api_key

    @property
    def resolved_embedding_base_url(self) -> str:
        """Возвращает базовый URL embedding API (или общий базовый URL)."""
        return self.embedding_base_url or self.resolved_openai_base_url


settings = Settings()
