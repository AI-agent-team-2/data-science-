# Observability и Langfuse

## Текущая архитектура

Проект использует callback/native-first observability:

- единый `CallbackHandler` Langfuse на один пользовательский запрос;
- один root trace на запрос;
- нативные шаги модели и инструментов через `invoke(..., config=...)`;
- без ручного управления parent-child иерархией в `run_agent.py`.

## Что наблюдается в Langfuse

Ожидаемая структура trace:

- `run_agent` (root)
- `tool_lookup`
- `tool_rag`
- `tool_web`
- `model_invoke`

Шаги `history_load` и `history_save` не управляют иерархией trace и не создают отдельные root traces.

## Как это реализовано

1. В `app/run_agent.py` создается один `CallbackHandler` через
   `get_langchain_callback_handler(...)`.
2. Весь pipeline запускается как `RunnableLambda` с `run_name="run_agent"`.
3. Для каждого вызова инструмента используется дочерний config:
   - `run_name="tool_lookup"`
   - `run_name="tool_rag"`
   - `run_name="tool_web"`
4. Для модели используется тот же callback-контекст с `run_name="model_invoke"`.

## Что больше не используется как основной путь

- ручное orchestration tracing через `create_trace/create_span/end_observation`;
- ручной parent-management в `run_agent.py`;
- смешивание двух конкурирующих моделей observability.

## Безопасность данных

Используются функции санитизации:

- `hash_user_id(user_id)` — стабильный идентификатор вида `u_<16hex>`;
- `sanitize_text(text)` — маскирование email/телефонов/secret-like токенов;
- `sanitize_payload(payload)` — рекурсивная санитизация структур.

## Переменные окружения

```env
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED=false
```

После выполнения 3–5 реальных запросов проверьте в Langfuse UI:

1. Для каждого из 3 запросов ровно один root trace `run_agent`.
2. Внутри trace есть `model_invoke` и tool-steps (`tool_lookup/tool_rag/tool_web` по сценарию).
3. Нет отдельных root traces вида `history_save` или `model_invoke`.

## Диагностика

1. Не появляются traces:
   - проверьте `LANGFUSE_ENABLED`, ключи и доступность `LANGFUSE_HOST`.
2. Не видно tool-steps:
   - убедитесь, что tools вызываются через `.invoke(..., config=...)`.
3. Нет usage/tokens:
   - проверьте, что провайдер модели возвращает usage metadata.
