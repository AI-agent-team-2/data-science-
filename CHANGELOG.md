# Журнал изменений

Формат файла основан на принципах [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/).

## [2.0.6] - 2026-03-25

### Исправлено
- Устранен конфликт привязки callback к trace: убрана конкурирующая передача `run_id` в `model.invoke` config.
- Привязка model observation к `run_agent` выполняется через metadata `langfuse_trace_id` и `langfuse_parent_observation_id`.

### Изменено
- `CallbackHandler` Langfuse создается на каждый invoke, чтобы исключить межзапросное смешивание контекста.

## [2.0.5] - 2026-03-25

### Исправлено
- `app/observability/langfuse_client.py` адаптирован под Langfuse v3 API (`start_as_current_observation`, `trace_context`).
- Устранена несовместимость legacy `client.trace(...)`, из-за которой root trace `run_agent` не создавался.
- Восстановлена корректная parent-child иерархия между manual orchestration tracing и callback observations модели.

### Изменено
- Сохранен публичный интерфейс wrapper: `create_trace`, `create_span`, `end_observation`, `capture_error`, `flush_if_available`.
- В `run_agent` используется `trace_id` в формате `uuid4().hex` для корректной привязки в Langfuse v3.

## [2.0.4] - 2026-03-25

### Исправлено
- Починена иерархия Langfuse trace: `model_invoke` больше не должен создавать отдельный root trace при наличии `run_agent`.
- В `model.invoke` добавлена привязка `config.run_id` к `trace.id` root trace.
- В `run_agent` добавлена явная генерация UUID trace_id для стабильной parent-child привязки.

### Изменено
- Добавлена диагностическая debug-информация по созданию root trace и передаче trace context в вызов модели.
- Обновлена observability-документация с проверкой корректной parent-child структуры.

## [2.0.3] - 2026-03-25

### Изменено
- Приведен к единому стилю код в ключевых модулях orchestration, observability и инструментов.
- Нормализованы комментарии, docstring и служебные сообщения на русском языке.
- Обновлены и синхронизированы документы `README.md`, `OBSERVABILITY.md`, `DEPLOY_VPS.md`.

### Исправлено
- Унифицированы формулировки логов и сообщений об ошибках в Langfuse-интеграции.
- Устранены локальные style-несоответствия (форматирование и читаемость без изменения логики).

## [2.0.2] - 2026-03-24

### Добавлено
- Секционное чанкование в `app/rag/ingest.py`.
- Расширенная metadata чанков (`doc_id`, `brand`, `category`, `section`, `articles`, `model` и другие поля).
- Улучшенная observability-трассировка: orchestration spans, tool spans, обработка ошибок и таймаутов.

### Изменено
- `product_lookup` работает с каталогом `data/knowledge_base/*.txt`.
- Добавлена и стабилизирована интеграция Langfuse (manual tracing + callback для model invoke).

### Удалено
- Устаревший модуль `app/monitoring.py`.

## [2.0.1] - 2026-03-23

### Изменено
- Улучшены типизация, структура функций и обработка исключений.
- Синхронизированы базовые документы проекта и технические описания.

### Исправлено
- Устранены несоответствия между кодом и документацией.

## [2.0.0] - 2026-03-21

### Добавлено
- Веб-поиск (Tavily/DuckDuckGo) и кэширование результатов.
- Команды Telegram-бота: `/help`, `/clear`, `/status`, `/id`.
- Гибридный retrieval в `ChromaRetriever`.
- Хранение истории диалогов в SQLite.

### Исправлено
- Обработка неизвестных slash-команд.
- SKU-first роутинг для запросов по артикулам.

### Изменено
- Улучшен финальный prompt для ответа пользователю.
- Настроен fallback-роутинг источников `lookup/rag/web`.
