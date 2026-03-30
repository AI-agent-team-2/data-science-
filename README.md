# SAN Bot

Telegram-бот для подбора и консультаций по сантехническим товарам.  
Проект использует LLM, product-level lookup и chunk-based RAG на ChromaDB, а также внешний web-поиск.

## Основные возможности

- ответы пользователю в Telegram с учетом истории диалога;
- маршрутизация запроса между источниками `lookup -> rag -> web` (или в другом порядке по типу запроса);
- поиск карточек товара (`product_lookup`) по отдельной product collection в ChromaDB (exact SKU + semantic fallback);
- поиск по базе знаний (`rag_search`) через chunk collection в ChromaDB;
- внешний поиск (`web_search`) через Tavily или DuckDuckGo с файловым кешем;
- observability через Langfuse (trace/span, ошибки, метаданные вызовов).
- auto-scoring в Langfuse (`relevance`, `safety`, `contains_pii`, `blocked`) с feature-flag.

## Архитектура

1. Telegram-сообщение приходит в `app/bot/telegram_bot.py`.
2. Оркестратор `app/run_agent.py`:
   - загружает историю,
   - выбирает источники контекста,
   - собирает prompt,
   - вызывает модель,
   - сохраняет историю.
3. Инструменты в `app/tools/` возвращают нормализованный JSON.
4. RAG-слой в `app/rag/` отвечает за индексацию и retrieval:
   - chunk collection для `rag_search`;
   - product collection для `product_lookup`.
5. Наблюдаемость и санитизация находятся в `app/observability/`.

## Технологический стек

- Python 3.10+
- LangChain (`langchain`, `langchain-core`, `langchain-openai`, `langchain-community`)
- OpenAI-compatible API (OpenAI/OpenRouter)
- ChromaDB
- pyTelegramBotAPI
- Langfuse

## Структура проекта

- `app/bot/` — Telegram-обработчики
- `app/run_agent.py` — orchestration и fallback-логика
- `app/tools/` — инструменты `product_lookup`, `rag_search`, `web_search`
- `app/rag/` — индексация документов и retrieval
- `app/observability/` — Langfuse-клиент и санитизация payload
- `data/knowledge_base/` — текстовые документы для каталога и RAG
- `chroma_db/` — локальное хранилище Chroma
- `history.db` — история диалогов (SQLite)

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

## Настройка окружения

```bash
cp .env.example .env
# Windows PowerShell:
# Copy-Item .env.example .env
```

Минимально обязательные переменные:

```env
TELEGRAM_TOKEN=
OPENAI_API_KEY=
```

Рекомендуемые (зависят от окружения):

```env
MODEL_PROVIDER=openrouter
MODEL_NAME=
OPENAI_BASE_URL=
CHROMA_PATH=./chroma_db
HISTORY_DB_PATH=./history.db
```

Опционально:

```env
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
TAVILY_API_KEY=
LANGFUSE_ENABLED=false
LANGFUSE_AUTO_SCORING_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Подготовка RAG (ингест)

После изменения файлов в `data/knowledge_base/` выполните переиндексацию:

```bash
python -m app.rag.ingest
```

Команда обновляет обе коллекции:
- chunk collection (для `rag_search`);
- product collection (для `product_lookup`).

## Запуск локально

```bash
python -m app.bot.telegram_bot
```

Альтернативный запуск:

```bash
python app/bot/telegram_bot.py
```

## Команды бота

- `/start` — приветствие
- `/help` — справка
- `/clear` — очистка истории диалога
- `/status` — статус подсистем
- `/id` — технические идентификаторы чата

## Запуск на VPS и systemd

Для production-обновления и безопасного rollout используйте:

- [DEPLOY_VPS.md](DEPLOY_VPS.md) — пошаговый deploy/rollback;
- `/etc/san-bot/san-bot.env` — переменные окружения для `san-bot.service`.

## Observability (Langfuse)

- Подробная схема trace/span: [OBSERVABILITY.md](OBSERVABILITY.md)
- Интеграция использует callback/native-first подход (LangChain + Langfuse CallbackHandler).
- Один пользовательский запрос = один handler instance = один root trace `run_agent`.
- Модель и инструменты (`tool_lookup`, `tool_rag`, `tool_web`) видны как нативные callback steps.
- `history_*` не управляют иерархией trace и не должны создавать отдельные root traces.
- Auto-scoring включается отдельно через `LANGFUSE_AUTO_SCORING_ENABLED=true`.
- Быстрое включение:
  1. заполните `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`;
  2. установите `LANGFUSE_ENABLED=true`;
  3. перезапустите сервис.

## Проверка и тестовые сценарии

- Базовые вопросы для ручной проверки: `TEST.txt`.
- Для локальной проверки импорта:

```bash
python -c "import app.graph; print('graph_ok')"
python -c "from app.run_agent import run_agent; print('run_agent_ok')"
```

## Типовые проблемы

1. Бот не отвечает в Telegram:
   - проверьте `TELEGRAM_TOKEN` и логи сервиса.
2. Пустые ответы по базе знаний:
   - убедитесь, что выполнен `python -m app.rag.ingest`.
3. `product_lookup` ничего не находит:
   - убедитесь, что после обновления `data/knowledge_base` выполнен `python -m app.rag.ingest` (строится product collection).
4. Нет trace в Langfuse:
   - проверьте `LANGFUSE_ENABLED`, ключи и доступность `LANGFUSE_HOST`.
5. Ошибка web-поиска:
   - при отсутствии `TAVILY_API_KEY` используется DuckDuckGo-бэкенд.

## История изменений

Смотрите [CHANGELOG.md](CHANGELOG.md).
