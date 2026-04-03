# Журнал изменений

Формат файла основан на принципах [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/).

## [8] - 2026-04-03

### Архитектура
- Проведен структурный рефакторинг без изменения функционального поведения:
  - agent-слой вынесен в пакет `app/agent/`;
  - context-слой вынесен в пакет `app/context_engine/`;
  - routing-слой вынесен в пакет `app/routing/`.
- Удалены временные совместимые прокси и legacy-обертки после перевода импортов на пакетную структуру.

### Тесты и eval
- После рефакторинга подтверждена совместимость поведения:
  - `python -m unittest` (ключевые integration/failure/routing/cleanup тесты) — green;
  - `python tests/evals/run_eval.py` — baseline кейсы пройдены.

## [7] - 2026-04-01

### Крупные изменения
- Укреплен runtime pipeline:
  - стартовая индексация управляется через `STARTUP_INDEX_MODE=never|if_empty|always`;
  - readiness retrieval-данных вынесен в отдельную проверку;
  - runtime retrievers перестали молча создавать пустые коллекции.
- Оркестратор стал явно различать `success`, `empty` и `failed`:
  - timeout/exception инструментов больше не маскируются под пустой результат;
  - сбой model invoke отдает явный internal failure-response.
- Контракт инструментов упрощен:
  - убран JSON-roundtrip между tools и orchestration;
  - инструменты возвращают структурированные `dict`-payload;
  - payload schema унифицирована для `lookup`, `rag` и `web`.
- Ускорен exact SKU lookup:
  - добавлен прямой SKU-индекс;
  - exact match сначала ищется по индексу, а не только через полный scan product metadata.
- Улучшен routing и observability:
  - добавлены `attempted_sources`, `source_status_map`, `failed_sources`, `fallback_reason`;
  - trace структура переведена на `agent_request -> agent_pipeline -> tool_* / model_invoke`.
- Усилен security для внешнего web-контекста:
  - suspicious instruction-like snippets отбрасываются до prompt assembly;
  - финальный prompt явно маркирует внешний контекст как недоверенный.

### Docker / CI/CD / Ops
- Добавлен production Docker-контур:
  - `Dockerfile`;
  - `docker-compose.yml`;
  - health-check с проверкой readiness индекса.
- Настроен CI/CD workflow:
  - compile/tests/import smoke;
  - Docker build/push;
  - SSH deploy;
  - health-check и rollback.
- Legacy `systemd`/`.venv` deploy-path выведен из поддерживаемого контура.

### Тесты
- Добавлены integration/failure-mode тесты для:
  - startup lifecycle;
  - `run_agent`;
  - routing short-circuit;
  - security фильтрации web-контекста;
  - SKU-индекса.

### Документация
- Обновлены:
  - `README.md`;
  - `DEPLOY_VPS.md`;
  - `OPERATIONS.md`;
  - `OBSERVABILITY.md`.
- Документация синхронизирована с текущим runtime, tracing и Docker/CI-CD контуром.

## [6] - 2026-03-26

### Крупные изменения
- Переведен `product_lookup` на product-level индекс в ChromaDB:
  - runtime-чтение `.txt` файлов из `product_lookup` удалено;
  - добавлены exact SKU match + semantic fallback по `lookup_text`.
- Расширен ingestion pipeline:
  - сохраняется chunk collection для `rag_search`;
  - добавлена отдельная product collection для `product_lookup`.
- `rag_search` сохранен как thin-wrapper над chunk retrieval без изменения роли.

### Конфигурация
- Упрощена конфигурация проекта:
  - в `.env` оставлены только окруженческие параметры (ключи/токены/base_url/пути/observability);
  - внутренние tuning/feature-параметры переведены в кодовые defaults.

### Cleanup
- Удалены неиспользуемые manual tracing helpers из `app/observability/langfuse_client.py`.
- Удалены неиспользуемые `get_history_stats` и `hybrid_search`.

## [5] - 2026-03-25

### Крупные изменения
- Переход на callback/native-first observability.
- Один запрос пользователя = один `CallbackHandler` = один root trace `run_agent`.
- Инструменты (`tool_lookup`, `tool_rag`, `tool_web`) и `model_invoke` наблюдаются как нативные callback steps.

### Документация
- Обновлены `OBSERVABILITY.md` и `README.md`.

## [4] - 2026-03-25

### Крупные изменения
- Добавлена интеграция Langfuse observability.
- Введен trace `run_agent` для отслеживания обработки запроса.
- Добавлена передача metadata/контекста для привязки model вызовов к trace.

## [3] - 2026-03-24

### Крупные изменения
- Добавлен RAG-слой на ChromaDB (`rag_search`, ingestion/retrieval).
- Добавлен внешний web-поиск (`web_search`) с кешированием результатов.

## [2] - 2026-03-22

### Крупные изменения
- Добавлен поиск по каталогу товаров (`product_lookup`) с приоритизацией SKU.
- Добавлено хранение истории диалога в SQLite.

## [1] - 2026-03-21

### Крупные изменения
- Базовый релиз Telegram-бота для сантехнического домена.
- Добавлены базовые команды Telegram-бота и основной pipeline ответа.
