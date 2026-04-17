# Lab 6: Aya Redteaming — Multilingual Safety

## Цель

Проверить, работают ли safety-фильтры модели на языках кроме английского.

## Датасет

**Aya Redteaming** — ~7 400 вредоносных промптов на 8 языках:
`en`, `ar` (арабский), `fr` (французский), `hi` (хинди), `ru` (русский), `sr` (сербский), `es` (испанский), `fil` (филиппинский)

Промпты переведены с сохранением культурного контекста.

## Задание

1. Запусти на английском (default):
   ```bash
   python -m inspect_ai eval run.py --model openrouter/openai/gpt-4o-mini --limit 50
   ```

2. Запусти на другом языке:
   ```powershell
   $env:TARGET_LANG = "ru"
   python -m inspect_ai eval run.py --model openrouter/openai/gpt-4o-mini --limit 50
   ```

   Для bash/zsh эквивалент:
   ```bash
   TARGET_LANG=ru python -m inspect_ai eval run.py --model openrouter/openai/gpt-4o-mini --limit 50
   ```

3. Сравни ASR между языками. Есть ли языки, на которых модель более уязвима?

## Гипотеза

Модели с преимущественно англоязычным RLHF/safety training должны показывать
более высокий ASR на других языках. Подтверди или опровергни эту гипотезу.
