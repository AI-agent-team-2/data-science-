# SAN Bot

Telegram-ассистент по сантехническим товарам на базе LangGraph + RAG (ChromaDB).

## Что делает проект

- принимает сообщения из Telegram;
- обрабатывает вопрос через агент-граф (`app/graph.py`);
- при необходимости вызывает инструменты `rag_search`, `web_search`, `product_lookup`;
- хранит векторный индекс в `chroma_db`.

Текущее состояние инструментов:
- `rag_search` - базовый поиск по локальной базе знаний (RAG еще в доработке);
- `web_search` - рабочий веб-поиск (Tavily при наличии `TAVILY_API_KEY`, иначе DuckDuckGo);
- `product_lookup` - заглушка;

Текущие ограничения:
- полноценный RAG-пайплайн пока не завершен (инкрементальная индексация, качество ретривера и eval еще в работе);
- роутинг инструментов и логика агентных переходов пока базовые и требуют доработки.

## План улучшений (из `rag_notebook.ipynb`)

1. Базовый ingestion и качество индекса:
- перейти с полной переиндексации на инкрементальный `upsert` (id/hash документа, обновление только измененных чанков);
- донастроить чанкование (`RecursiveCharacterTextSplitter`) и зафиксировать baseline: `chunk_size=300-800`, `chunk_overlap=10-20%`;
- сохранять и использовать метаданные чанков (источник, категория, страница/раздел) для фильтрации и цитирования.

2. Улучшение retrieval:
- внедрить двухэтапный поиск: быстрый retriever (Chroma top-k) + re-ranking (cross-encoder) для top-N кандидатов;
- протестировать MMR/обычный similarity и выбрать режим по метрикам;
- добавить fallback-режим, если локальный RAG не дал уверенного контекста.

3. Продвинутые RAG-техники:
- добавить HyDE для сложных/размытых запросов;
- добавить Query Expansion (синонимы и альтернативные формулировки);
- опционально протестировать RAG Fusion для объединения результатов нескольких поисковых запросов.

4. Роутинг инструментов и агентная логика:
- формализовать правила выбора `rag_search` / `web_search` / `product_lookup` (intent + уверенность + доменные триггеры);
- добавить явные guardrails: запрет выдумывания фактов, ответ "не знаю" при слабом контексте, обязательные ссылки на источники;
- сделать fallback-цепочку: `rag_search` -> `web_search` -> уточняющий вопрос пользователю.

## Требования

- Python 3.13+ (проект сейчас проверялся на 3.13.7)
- Windows PowerShell (команды ниже в формате PowerShell)
- Ollama (если используете локальную модель)

## Установка

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Настройка `.env`

Создайте/заполните файл `.env` в корне проекта:

```env
# Токен Telegram-бота.
# Где взять: в Telegram у @BotFather -> /newbot (или /token для существующего бота).
TELEGRAM_TOKEN=

# Провайдер модели: ollama | openrouter | openai.
MODEL_PROVIDER=openrouter

# Базовый URL (можно не задавать: подставится автоматически по MODEL_PROVIDER).
# OPENAI_BASE_URL=https://openrouter.ai/api/v1

# API-ключ (для ollama можно не задавать).
OPENAI_API_KEY=

# Имя модели (можно не задавать: есть дефолт по MODEL_PROVIDER).
# Для openrouter, например: openai/gpt-4o-mini
MODEL_NAME=

# Эмбеддинги:
# EMBEDDING_BACKEND=auto          # auto | local | openai_compatible
# EMBEDDING_MODEL_NAME=           # например text-embedding-3-small
# EMBEDDING_API_KEY=              # опционально, если отличается от OPENAI_API_KEY
# EMBEDDING_BASE_URL=             # опционально, если отличается от OPENAI_BASE_URL
# EMBEDDING_BATCH_SIZE=64         # размер батча при индексации в Chroma

# Путь к локальной базе Chroma.
CHROMA_PATH=./chroma_db

# Имя коллекции в Chroma.
COLLECTION_NAME=sanitary_goods

# Сколько чанков возвращать из RAG-поиска.
TOP_K=5

# (Опционально) Tavily Web Search API.
# Если не задано, web_search будет использовать DuckDuckGo (без токена).
TAVILY_API_KEY=
```

Быстрый рабочий профиль OpenRouter:

```env
MODEL_PROVIDER=openrouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=<ваш OpenRouter API key>
MODEL_NAME=openai/gpt-4o-mini

EMBEDDING_BACKEND=auto
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_BATCH_SIZE=64
```

Рекомендуемо для слабых ПК в команде:
- `MODEL_PROVIDER=openrouter`
- `OPENAI_API_KEY=<ваш ключ OpenRouter>`
- `MODEL_NAME=openai/gpt-4o-mini` (или другая доступная модель)

Для локального режима:
- `MODEL_PROVIDER=ollama`
- `OPENAI_BASE_URL=http://localhost:11434/v1` (опционально)
- `OPENAI_API_KEY=ollama` (опционально)
- `MODEL_NAME=qwen3.5:4b`

## Подготовка Ollama

Нужно только если `MODEL_PROVIDER=ollama`.

```powershell
ollama serve
ollama pull qwen3.5:4b
ollama list
```

## Подготовка RAG-базы

Перед запуском бота проиндексируйте документы из `data/knowledge_base`:

```powershell
.\myenv\Scripts\python.exe -m app.rag.ingest
```

После этого появится/обновится папка `chroma_db`.

Важно:
- при смене `EMBEDDING_BACKEND` или `EMBEDDING_MODEL_NAME` нужно заново выполнить индексацию;
- если ранее индекс был создан локальной `sentence-transformers` моделью, удалите старую коллекцию (или папку `chroma_db`) и переиндексируйте данные.

## Запуск бота

```powershell
.\myenv\Scripts\python.exe app\bot\telegram_bot.py
```

Альтернативный запуск как модуля:

```powershell
.\myenv\Scripts\python.exe -m app.bot.telegram_bot
```

## Быстрая проверка

Проверка, что модель поднимается и граф импортируется:

```powershell
.\myenv\Scripts\python.exe -c "import app.graph; print('graph_ok')"
```

## Проверка web_search

DuckDuckGo (без токена):

```powershell
.\myenv\Scripts\python.exe -c "from app.tools.web_search import web_search; print(web_search('что такое балансировочный клапан', 5))"
```

Tavily (нужен `TAVILY_API_KEY` в `.env`):

```powershell
.\myenv\Scripts\python.exe -c "import os; from app.tools.web_search import web_search; print(web_search('что такое балансировочный клапан', 5))"
```

## Структура проекта

- `app/config.py` - загрузка настроек из `.env`
- `app/prompts.py` - системный промпт агента
- `app/graph.py` - сборка LangGraph-агента
- `app/run_agent.py` - обертка запуска графа
- `app/tools/` - инструменты агента
- `app/rag/` - индексация и retrieval через Chroma
- `data/knowledge_base/` - исходные текстовые документы для RAG
