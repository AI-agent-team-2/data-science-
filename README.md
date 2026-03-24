# SAN Bot

Telegram-бот по сантехническим товарам на базе LLM + RAG (ChromaDB).

## Возможности

- отвечает на вопросы из Telegram;
- использует `product_lookup`, `rag_search`, `web_search`;
- простой роутинг: SKU/товарный запрос -> `product_lookup`, технический/общий запрос -> `rag_search`, если не найдено -> `web_search`;
- хранит векторный индекс в `chroma_db`;
- хранит историю диалога в `history.db` (SQLite).

## Архитектура системы

```text
Telegram user
   |
   v
telegram_bot.py
   |
   v
run_agent.py (router)
   |-- product_lookup (локальный каталог из data/knowledge_base/*.txt)
   |-- rag_search (Chroma retriever по проиндексированным документам)
   |-- web_search (Tavily/DuckDuckGo для внешних данных)
   |
   v
LLM (app/graph.py + SYSTEM_PROMPT)
   |
   v
Ответ в Telegram
```

- История диалога хранится в `history.db` (SQLite).
- Векторный индекс хранится в `chroma_db`.
- Индексация документов выполняется через `python -m app.rag.ingest`.

## Data Pipeline

```text
data/knowledge_base/*.txt
   |
   v
preprocess_for_rag (clean_text)
   |
   v
chunk_documents (chunk_size/chunk_overlap)
   |
   v
embedding (OpenAI-compatible)
   |
   v
Chroma collection (chroma_db)
   |
   v
rag_search -> top_k chunks -> контекст для LLM
```

- Шаги индексации реализованы в `app/rag/ingest.py`.
- Чанкование выполняется по секциям документа (например, `INSTALLATION`, `LIMITATIONS`, `WARRANTY`).
- Поиск по векторной базе выполняет `app/rag/retriever.py`.
- Очистка текста перед индексацией выполняется в `app/rag/preprocess_text.py`.

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
### Дополнительные настройки (версия 2.0)

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `ENABLE_WEB_SEARCH` | `true` | Включение/отключение веб-поиска |
| `ENABLE_RAG` | `true` | Включение/отключение RAG (база знаний) |
| `ENABLE_PRODUCT_LOOKUP` | `true` | Включение/отключение поиска по SKU |
| `WEB_CACHE_ENABLED` | `true` | Включение кэша веб-поиска |
| `WEB_CACHE_TTL_HOURS` | `24` | Время жизни кэша (часы) |
| `WEB_SEARCH_MAX_RESULTS` | `5` | Максимальное количество результатов поиска |

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
python scripts/retrieval_eval.py --suite all --top-k 6
# отдельно по набору:
# python scripts/retrieval_eval.py --suite rag
# python scripts/retrieval_eval.py --suite lookup
# python scripts/retrieval_eval.py --suite web
# python scripts/retrieval_eval.py --suite owasp
```

Скрипт содержит 4 набора по 15 кейсов:
- `rag` (retrieval по базе знаний);
- `lookup` (поиск по товарному каталогу);
- `web` (внешний поиск);
- `owasp` (базовые security-проверки).

## Структура

- `app/config.py` - настройки проекта
- `app/run_agent.py` - основная логика ответа и линейный роутинг
- `app/graph.py` - инициализация LLM-клиента
- `app/tools/` - инструменты (`product_lookup`, `rag_search`, `web_search`)
- `app/rag/` - индексация и retrieval
- `data/knowledge_base/` - исходные документы

## Последние изменения

Актуальная версия: **2.0.2**

Основные улучшения:
- 🔍 **Веб-поиск** — фильтрация по сантехнике, кэширование, улучшение запросов
- 🤖 **Telegram бот** — новые команды `/help`, `/clear`, инлайн-кнопки
- 🧭 **Контроль домена ответов** — системный prompt ограничивает ответы сантехнической тематикой
- 🧱 **Секционное RAG-чанкование** — метаданные `doc_id/brand/category/section/articles/model`
- ✅ **Единые eval-наборы** — 60/60 (`rag`, `lookup`, `web`, `owasp`)
- 🧹 **Упрощение кода** — удалены неиспользуемые `app/monitoring.py` и `app/legacy`
- 🔄 **RAG** — гибридный поиск с бустингом по ключевым словам
- 🛠️ **Рефакторинг кода** — типизация, docstring, декомпозиция функций, единый стиль логирования

Подробнее: [CHANGELOG.md](CHANGELOG.md)
