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

## Архитектура

1. Telegram-сообщение приходит в `app/bot/telegram_bot.py`.
2. Оркестратор `app/run_agent.py`:
   - загружает историю,
   - выбирает источники контекста,
   - собирает prompt,
   - вызывает модель,
   - сохраняет историю.
   - доменная логика разложена по пакетам:
     - `app/agent/` (guards/invoke/memory/response/trace),
     - `app/context_engine/` (core/helpers/web/response),
     - `app/routing/` (intents/sources).
3. Инструменты в `app/tools/` возвращают нормализованный `dict`-payload по единой схеме.
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
- `app/agent/` — guard-логика, invoke/timeout, memory-summary, trace metadata
- `app/context_engine/` — сборка и фильтрация контекста из lookup/rag/web
- `app/routing/` — intent-сигналы и выбор порядка источников
- `app/tools/` — инструменты `product_lookup`, `rag_search`, `web_search`
- `app/rag/` — индексация документов и retrieval
- `app/observability/` — Langfuse-клиент и санитизация payload
- `app/utils/sku.py` — единая SKU-нормализация и извлечение SKU-кандидатов
- `data/knowledge_base/` — текстовые документы для каталога и RAG

Локальные runtime-артефакты не являются частью исходного кода и не должны коммититься:
- `chroma_db/` — локальное хранилище Chroma
- `history.db` — локальная история диалогов (SQLite)
- `.web_cache/` — файловый кеш web-поиска
- `artifacts/` — временные отчеты и baseline-артефакты
- `myenv/`, `.venv/` — локальные виртуальные окружения
- `*.ipynb` в корне — локальные исследовательские ноутбуки

## Установка

```bash
python -m venv .venv
# Linux/macOS:
# source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
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
MODEL_MAX_RETRIES=2
WEB_API_KEY=
WEB_ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
WEB_TRUSTED_DOMAINS=
```

Опционально:

```env
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
TAVILY_API_KEY=
STARTUP_INDEX_MODE=if_empty
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

Для web API (`/api/chat`, `/api/history`, `/api/clear`) обязателен заголовок:

```http
X-API-Key: <WEB_API_KEY>
```

Встроенный web-интерфейс (`/`) при первом запросе попросит этот ключ и сохранит его локально в браузере.

## Подготовка RAG (ингест)

После изменения файлов в `data/knowledge_base/` выполните переиндексацию:

```bash
python -m app.rag.ingest
```

Команда обновляет обе коллекции:
- chunk collection (для `rag_search`);
- product collection (для `product_lookup`).

В Docker/runtime-контуре стартовая индексация управляется через `STARTUP_INDEX_MODE`:
- `never` — никогда не запускать ingest на старте;
- `if_empty` — запускать только если коллекции пустые;
- `always` — всегда запускать ingest перед стартом бота.

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

## Запуск на VPS (Docker Compose)

Для production-обновления и безопасного rollout используйте:

- [DEPLOY_VPS.md](DEPLOY_VPS.md) — пошаговый deploy/rollback;
- [OPERATIONS.md](OPERATIONS.md) — runbook для диагностики и инцидентов;
- `/etc/san-bot/san-bot.env` — переменные окружения для контейнера `san-bot`.
- `.github/workflows/deploy.yml` — CI/CD (checks -> Docker build/push -> SSH deploy -> health-check -> rollback).

Поддерживаемый production-контур только один: `GitHub Actions -> Docker Compose`.
Старый `systemd`/`.venv`-deploy больше не поддерживается и не должен использоваться.

## Docker (локально / VPS)

Сборка и запуск:

```bash
docker compose up -d --build
```

Проверка контейнера:

```bash
docker compose ps
docker compose logs --tail=100 san-bot
```

Остановка:

```bash
docker compose down
```

Состояние сохраняется в Docker volumes:
- `san_bot_chroma` (ChromaDB)
- `san_bot_history` (SQLite history)

## Observability (Langfuse)

- Подробная схема trace/span: [OBSERVABILITY.md](OBSERVABILITY.md)
- Интеграция использует callback/native-first подход (LangChain + Langfuse CallbackHandler).
- Один пользовательский запрос = один handler instance = один root trace `agent_request`.
- Модель и инструменты (`tool_lookup`, `tool_rag`, `tool_web`) видны как нативные callback steps.
- `history_*` не управляют иерархией trace и не должны создавать отдельные root traces.
- Быстрое включение:
  1. заполните `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`;
  2. установите `LANGFUSE_ENABLED=true`;
  3. перезапустите сервис.

## Проверка и тестовые сценарии

- Базовые вопросы для ручной проверки: `TEST.txt`.
- Структурированные eval-кейсы: `tests/evals/cases.jsonl`.
- Автоматизированный прогон eval-кейсов:

```bash
python tests/evals/run_eval.py
```

Отчет сохраняется в `artifacts/eval_report.json`.
- Для локальной проверки импорта:

```bash
python -c "import app.graph; print('graph_ok')"
python -c "from app.run_agent import run_agent; print('run_agent_ok')"
```

## Типовые проблемы

1. Бот не отвечает в Telegram:
   - проверьте `TELEGRAM_TOKEN` и логи контейнера (`docker compose logs san-bot`).
2. Пустые ответы по базе знаний:
   - проверьте `/status` и readiness контейнера;
   - при необходимости выполните `python -m app.rag.ingest` или установите подходящий `STARTUP_INDEX_MODE`.
3. `product_lookup` ничего не находит:
   - убедитесь, что после обновления `data/knowledge_base` выполнен `python -m app.rag.ingest` (строится product collection);
   - проверьте, что контейнер стартовал с готовым индексом, а не на пустых volumes.
4. Нет trace в Langfuse:
   - проверьте `LANGFUSE_ENABLED`, ключи и доступность `LANGFUSE_HOST`.
5. Ошибка web-поиска:
   - при отсутствии `TAVILY_API_KEY` используется DuckDuckGo-бэкенд.

## История изменений

Смотрите [CHANGELOG.md](CHANGELOG.md).
