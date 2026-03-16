# SAN Bot

Telegram-бот по сантехническим товарам на базе LLM + RAG (ChromaDB).

## Возможности

- отвечает на вопросы из Telegram;
- использует `product_lookup`, `rag_search`, `web_search`;
- простой роутинг: SKU/товарный запрос -> `product_lookup`, технический/общий запрос -> `rag_search`, если не найдено -> `web_search`;
- хранит векторный индекс в `chroma_db`;
- хранит историю диалога в `history.db` (SQLite).

## Требования

- Python 3.13+
- любой терминал (`bash`, `zsh`, `cmd`, PowerShell, IDE terminal)

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

Скопируйте шаблон:

```bash
cp .env.example .env
# Windows PowerShell:
# Copy-Item .env.example .env
```

Минимальный пример:

```env
TELEGRAM_TOKEN=

MODEL_PROVIDER=openrouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=
MODEL_NAME=openai/gpt-4o-mini

EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_BATCH_SIZE=64

CHROMA_PATH=./chroma_db
COLLECTION_NAME=sanitary_goods
TOP_K=6
CHUNK_SIZE=900
CHUNK_OVERLAP=140

HISTORY_DB_PATH=./history.db
HISTORY_MAX_MESSAGES=24
HISTORY_TTL_DAYS=30

TAVILY_API_KEY=
```

## Индексация RAG

Перед первым запуском:

```bash
python -m app.rag.ingest
```

Повторно запускайте индексацию после обновления файлов в `data/knowledge_base`.

## Запуск бота

```bash
python app/bot/telegram_bot.py
```

Альтернатива:

```bash
python -m app.bot.telegram_bot
```

## Ручная проверка

1. Запустите бота и отправьте `/start`.
2. Проверьте SKU-запрос (например артикул товара): должен сработать `product_lookup`.
3. Проверьте технический вопрос по базе знаний (например про совместимость): должен сработать `rag_search`.
4. Проверьте внешний вопрос, которого нет в базе: должен сработать `web_search`.

## Мини-оценка Retrieval

```bash
python scripts/retrieval_eval.py
# или с другим k:
# python scripts/retrieval_eval.py --top-k 8
```

Скрипт печатает `hit@k` по небольшому фиксированному набору запросов.

## Структура

- `app/config.py` - настройки проекта
- `app/run_agent.py` - основная логика ответа и линейный роутинг
- `app/graph.py` - инициализация LLM-клиента
- `app/tools/` - инструменты (`product_lookup`, `rag_search`, `web_search`)
- `app/rag/` - индексация и retrieval
- `data/knowledge_base/` - исходные документы
