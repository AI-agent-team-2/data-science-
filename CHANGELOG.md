# Changelog

Все заметные изменения проекта фиксируются в этом файле.

## [2.0.2] - 2026-03-24

### Added
- Секционное чанкование в `app/rag/ingest.py`.
- Расширенная metadata чанков (`doc_id`, `brand`, `category`, `section`, `articles`, `model` и др.).
- Наборы проверки в `scripts/retrieval_eval.py`: `rag`, `lookup`, `web`, `owasp`, `all`.

### Changed
- `product_lookup` переведен на `data/knowledge_base/*.txt`.
- `README.md` и служебные тексты в коде синхронизированы и упрощены.

### Removed
- `app/monitoring.py`.

## [2.0.1] - 2026-03-23

### Changed
- Рефакторинг кода: стиль, типизация, декомпозиция, docstring.
- Улучшены обработка ошибок и логирование.
- Синхронизированы `README.md` и скрипты проверки.

### Fixed
- Несоответствия документации фактическому поведению проекта.

## [2.0.0] - 2026-03-21

### Added
- Веб-поиск (Tavily/DuckDuckGo) и кэширование.
- Команды Telegram-бота: `/help`, `/clear`, `/status`, `/id`.
- Гибридный retrieval в `ChromaRetriever`.
- История диалогов в SQLite.

### Fixed
- Обработка неизвестных slash-команд.
- SKU-first роутинг для коротких запросов.

### Changed
- Улучшен финальный prompt.
- Выстроен fallback-роутинг `lookup/rag/web`.
