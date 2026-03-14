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
- `product_lookup` - рабочий локальный поиск по каталогу `data/knowledge_base/tp`;

Текущие ограничения:
- полноценный RAG-пайплайн пока не завершен (инкрементальная индексация, качество ретривера и eval еще в работе);
- роутинг инструментов и логика агентных переходов пока базовые и требуют доработки.

## План улучшений (приоритетный)

1. Must-have (делаем в первую очередь):
- заменить заглушку `product_lookup` на реальный поиск по каталогу (API/БД/файл);
- перейти с полной переиндексации на инкрементальный `upsert` в RAG (`id/hash`, обновление только измененных чанков);
- закрепить fallback-цепочку в коде: `rag_search` -> `web_search` -> уточняющий вопрос;
- добавить минимальные тесты и baseline-оценку качества retrieval (набор контрольных вопросов + ожидаемые факты).

2. Следующий этап (после стабилизации must-have):
- доработать метаданные чанков (категория, раздел/страница, источник) и использовать их в ответах;
- сравнить режимы retrieval (`similarity`/MMR) и выбрать по метрикам;
- при необходимости добавить re-ranking для top-N кандидатов.

3. Advanced (опционально):
- HyDE;
- Query Expansion;
- RAG Fusion.

Принцип принятия решений:
- сначала закрываем функциональные риски и воспроизводимость;
- затем повышаем качество поиска и маршрутизации;
- только после этого добавляем сложные RAG-техники.

## Требования

- Python 3.13+ (проект сейчас проверялся на 3.13.7)
- Любой терминал (`bash`, `zsh`, `cmd`, PowerShell, терминал IDE)

## Установка

```bash
python -m venv myenv
# Linux/macOS:
# source myenv/bin/activate
# Windows:
# myenv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Настройка `.env`

Создайте/заполните файл `.env` в корне проекта:

```env
# Токен Telegram-бота.
# Где взять: в Telegram у @BotFather -> /newbot (или /token для существующего бота).
TELEGRAM_TOKEN=

# Провайдер модели: openrouter | openai.
MODEL_PROVIDER=openrouter

# Базовый URL (можно не задавать: подставится автоматически по MODEL_PROVIDER).
# OPENAI_BASE_URL=https://openrouter.ai/api/v1

# API-ключ провайдера.
OPENAI_API_KEY=

# Имя модели (можно не задавать: есть дефолт по MODEL_PROVIDER).
# Для openrouter, например: openai/gpt-4o-mini
MODEL_NAME=

# Эмбеддинги:
# EMBEDDING_BACKEND=openai_compatible
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

EMBEDDING_BACKEND=openai_compatible
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_BATCH_SIZE=64
```

Рекомендуемо для VPS и слабых ПК в команде:
- `MODEL_PROVIDER=openrouter`
- `OPENAI_API_KEY=<ваш ключ OpenRouter>`
- `MODEL_NAME=openai/gpt-4o-mini` (или другая доступная модель)

## Подготовка RAG-базы

Перед запуском бота проиндексируйте документы из `data/knowledge_base`:

```bash
python -m app.rag.ingest
```

После этого появится/обновится папка `chroma_db`.

Важно:
- при смене `EMBEDDING_BACKEND` или `EMBEDDING_MODEL_NAME` нужно заново выполнить индексацию;
- если ранее индекс был создан локальной моделью, удалите старую коллекцию (или папку `chroma_db`) и переиндексируйте данные.

## Запуск бота

```bash
python app/bot/telegram_bot.py
```

Альтернативный запуск как модуля:

```bash
python -m app.bot.telegram_bot
```

## Быстрая проверка

Проверка, что модель поднимается и граф импортируется:

```bash
python -c "import app.graph; print('graph_ok')"
```

## Проверка web_search

DuckDuckGo (без токена):

```bash
python -c "from app.tools.web_search import web_search; print(web_search('что такое балансировочный клапан', 5))"
```

Tavily (нужен `TAVILY_API_KEY` в `.env`):

```bash
python -c "import os; from app.tools.web_search import web_search; print(web_search('что такое балансировочный клапан', 5))"
```

## Структура проекта

- `app/config.py` - загрузка настроек из `.env`
- `app/prompts.py` - системный промпт агента
- `app/graph.py` - сборка LangGraph-агента
- `app/run_agent.py` - обертка запуска графа
- `app/tools/` - инструменты агента
- `app/rag/` - индексация и retrieval через Chroma
- `data/knowledge_base/` - исходные текстовые документы для RAG
