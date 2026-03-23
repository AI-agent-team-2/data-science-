from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROVIDER: Final[str] = "openrouter"
DEFAULT_OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_BASE_URL: Final[str] = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_MODEL: Final[str] = "openai/gpt-4o-mini"
DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
SUPPORTED_PROVIDERS: Final[set[str]] = {"openrouter", "openai"}


def _get_env_str(name: str, default: str = "") -> str:
    """Возвращает строковое значение переменной окружения с trim."""
    return os.getenv(name, default).strip()


def _get_env_bool(name: str, default: bool) -> bool:
    """Преобразует переменную окружения в bool (`true/false`, `1/0`, `yes/no`)."""
    raw_value = _get_env_str(name, "true" if default else "false").lower()
    return raw_value in {"1", "true", "yes", "y", "on"}


def _get_env_int(name: str, default: int, min_value: int | None = None) -> int:
    """Возвращает целое число из окружения с безопасным fallback."""
    raw_value = _get_env_str(name, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        value = default

    if min_value is not None:
        return max(min_value, value)
    return value


@dataclass(frozen=True)
class Settings:
    """Конфигурация приложения, загружаемая из `.env` и переменных окружения."""

    model_provider: str = _get_env_str("MODEL_PROVIDER", DEFAULT_PROVIDER).lower()
    openai_api_key: str = _get_env_str("OPENAI_API_KEY")
    model_name: str = _get_env_str("MODEL_NAME")
    openai_base_url: str = _get_env_str("OPENAI_BASE_URL")

    embedding_model_name: str = _get_env_str("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
    embedding_api_key: str = _get_env_str("EMBEDDING_API_KEY")
    embedding_base_url: str = _get_env_str("EMBEDDING_BASE_URL")

    telegram_token: str = _get_env_str("TELEGRAM_TOKEN")
    chroma_path: str = _get_env_str("CHROMA_PATH", "./chroma_db")
    collection_name: str = _get_env_str("COLLECTION_NAME", "sanitary_goods")

    top_k: int = _get_env_int("TOP_K", 6, min_value=1)
    embedding_batch_size: int = _get_env_int("EMBEDDING_BATCH_SIZE", 64, min_value=1)
    chunk_size: int = _get_env_int("CHUNK_SIZE", 900, min_value=200)
    chunk_overlap: int = _get_env_int("CHUNK_OVERLAP", 140, min_value=0)

    history_db_path: str = _get_env_str("HISTORY_DB_PATH", "./history.db")
    history_max_messages: int = _get_env_int("HISTORY_MAX_MESSAGES", 24, min_value=1)
    history_ttl_days: int = _get_env_int("HISTORY_TTL_DAYS", 30, min_value=1)

    web_cache_enabled: bool = _get_env_bool("WEB_CACHE_ENABLED", True)
    web_cache_ttl_hours: int = _get_env_int("WEB_CACHE_TTL_HOURS", 24, min_value=0)
    web_search_max_results: int = _get_env_int("WEB_SEARCH_MAX_RESULTS", 5, min_value=1)

    enable_web_search: bool = _get_env_bool("ENABLE_WEB_SEARCH", True)
    enable_rag: bool = _get_env_bool("ENABLE_RAG", True)
    enable_product_lookup: bool = _get_env_bool("ENABLE_PRODUCT_LOOKUP", True)

    @property
    def resolved_model_provider(self) -> str:
        """Возвращает поддерживаемый провайдер LLM."""
        if self.model_provider in SUPPORTED_PROVIDERS:
            return self.model_provider
        return DEFAULT_PROVIDER

    @property
    def resolved_openai_base_url(self) -> str:
        """Возвращает итоговый base URL OpenAI-compatible API."""
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
        """Возвращает ключ для embedding API (или общий API key)."""
        return self.embedding_api_key or self.resolved_openai_api_key

    @property
    def resolved_embedding_base_url(self) -> str:
        """Возвращает base URL для embedding API (или общий base URL)."""
        return self.embedding_base_url or self.resolved_openai_base_url


settings = Settings()
