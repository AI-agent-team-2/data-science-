from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Загружаем переменные окружения из .env, чтобы настройки были доступны во всем проекте.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Провайдер LLM: openrouter | openai.
    model_provider: str = os.getenv("MODEL_PROVIDER", "openrouter").strip().lower()
    # Ключ для OpenAI-compatible API.
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    # Имя модели, которую будет вызывать агент.
    model_name: str = os.getenv("MODEL_NAME", "").strip()
    # Базовый URL OpenAI-compatible API.
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "").strip()
    # Бэкенд эмбеддингов: openai_compatible.
    embedding_backend: str = os.getenv(
        "EMBEDDING_BACKEND", "openai_compatible"
    ).strip().lower()
    # Имя модели эмбеддингов.
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2").strip()    # Опционально: отдельный ключ для embedding API (если не задан, используется OPENAI_API_KEY).
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "").strip()
    # Опционально: отдельный base URL для embedding API (если не задан, используется OPENAI_BASE_URL).
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "").strip()
    # Токен Telegram-бота для long polling.
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    # Путь к локальной persistent-базе Chroma.
    chroma_path: str = os.getenv("CHROMA_PATH", "./chroma_db")
    # Имя коллекции с векторами и текстовыми чанками.
    collection_name: str = os.getenv("COLLECTION_NAME", "sanitary_goods")
    # Количество документов, которое retriever вернет на запрос.
    top_k: int = int(os.getenv("TOP_K", "6"))
    # Размер батча на вставку в Chroma (актуально для cloud embeddings).
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    # Параметры разбиения документов на чанки для RAG.
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "140"))
    # Путь к SQLite-файлу с историей диалогов.
    history_db_path: str = os.getenv("HISTORY_DB_PATH", "./history.db")
    # Сколько последних сообщений хранить и передавать в модель на каждый запрос.
    history_max_messages: int = int(os.getenv("HISTORY_MAX_MESSAGES", "24"))
    # TTL истории диалогов в днях.
    history_ttl_days: int = int(os.getenv("HISTORY_TTL_DAYS", "30"))

    # Настройки прокси для Telegram (опционально).
    # Формат: http://user:password@host:port или http://host:port
    proxy_url: str = os.getenv("PROXY_URL", "").strip()

    @property
    def resolved_model_provider(self) -> str:
        if self.model_provider in {"openrouter", "openai"}:
            return self.model_provider
        return "openrouter"

    @property
    def resolved_openai_base_url(self) -> str:
        if self.openai_base_url:
            return self.openai_base_url
        if self.resolved_model_provider == "openrouter":
            return "https://openrouter.ai/api/v1"
        if self.resolved_model_provider == "openai":
            return "https://api.openai.com/v1"
        return "https://openrouter.ai/api/v1"

    @property
    def resolved_openai_api_key(self) -> str:
        return self.openai_api_key

    @property
    def resolved_model_name(self) -> str:
        if self.model_name:
            return self.model_name
        if self.resolved_model_provider == "openrouter":
            return "openai/gpt-4o-mini"
        if self.resolved_model_provider == "openai":
            return "gpt-4o-mini"
        return "openai/gpt-4o-mini"

    @property
    def resolved_embedding_backend(self) -> str:
        if self.embedding_backend == "openai_compatible":
            return self.embedding_backend
        return "openai_compatible"

    @property
    def resolved_embedding_model_name(self) -> str:
        if self.embedding_model_name:
            return self.embedding_model_name
        return "text-embedding-3-small"

    @property
    def resolved_embedding_api_key(self) -> str:
        if self.embedding_api_key:
            return self.embedding_api_key
        return self.resolved_openai_api_key

    @property
    def resolved_embedding_base_url(self) -> str:
        if self.embedding_base_url:
            return self.embedding_base_url
        return self.resolved_openai_base_url


# Единый объект настроек, который импортируют остальные модули.
settings = Settings()
