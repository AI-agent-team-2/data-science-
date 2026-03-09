# SAN Bot

Telegram-ассистент по сантехническим товарам на базе LangGraph + RAG (ChromaDB).

## Что делает проект

- принимает сообщения из Telegram;
- обрабатывает вопрос через агент-граф (`app/graph.py`);
- при необходимости вызывает инструменты `rag_search`, `web_search`, `product_lookup`;
- хранит векторный индекс в `chroma_db`.

Текущее состояние инструментов:
- `rag_search` - рабочий поиск по локальной базе знаний;
- `web_search` - заглушка;
- `product_lookup` - заглушка;

## Требования

- Python 3.13+ (проект сейчас проверялся на 3.13.7)
- Windows PowerShell (команды ниже в формате PowerShell)
- Ollama (локальный OpenAI-compatible API)

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

# Локальный OpenAI-compatible endpoint (Ollama).
OPENAI_BASE_URL=http://localhost:11434/v1

# API-ключ для OpenAI-compatible клиента.
# Для Ollama можно оставить "ollama".
OPENAI_API_KEY=ollama

# Модель Ollama (должна быть предварительно скачана).
MODEL_NAME=qwen3.5:4b

# Путь к локальной базе Chroma.
CHROMA_PATH=./chroma_db

# Имя коллекции в Chroma.
COLLECTION_NAME=sanitary_goods

# Сколько чанков возвращать из RAG-поиска.
TOP_K=5
```

Если хотите использовать OpenAI вместо локальной модели:
- `OPENAI_BASE_URL=https://api.openai.com/v1`
- `OPENAI_API_KEY=<ваш ключ>`
- `MODEL_NAME=gpt-4o-mini` (или другая доступная модель)

## Подготовка Ollama

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

## Структура проекта

- `app/config.py` - загрузка настроек из `.env`
- `app/prompts.py` - системный промпт агента
- `app/graph.py` - сборка LangGraph-агента
- `app/run_agent.py` - обертка запуска графа
- `app/tools/` - инструменты агента
- `app/rag/` - индексация и retrieval через Chroma
- `data/knowledge_base/` - исходные текстовые документы для RAG
