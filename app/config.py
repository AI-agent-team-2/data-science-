from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Загружаем переменные окружения из .env, чтобы настройки были доступны во всем проекте.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Ключ для OpenAI-compatible API (для Ollama можно оставить "ollama").
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "ollama")
    # Имя модели, которую будет вызывать агент (локально через Ollama по умолчанию).
    model_name: str = os.getenv("MODEL_NAME", "qwen3.5:4b")
    # Базовый URL OpenAI-compatible API (Ollama по умолчанию).
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    # Токен Telegram-бота для long polling.
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    # Путь к локальной persistent-базе Chroma.
    chroma_path: str = os.getenv("CHROMA_PATH", "./chroma_db")
    # Имя коллекции с векторами и текстовыми чанками.
    collection_name: str = os.getenv("COLLECTION_NAME", "sanitary_goods")
    # Количество документов, которое retriever вернет на запрос.
    top_k: int = int(os.getenv("TOP_K", "5"))


# Единый объект настроек, который импортируют остальные модули.
settings = Settings()
