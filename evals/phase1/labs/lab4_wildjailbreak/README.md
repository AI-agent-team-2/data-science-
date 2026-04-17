# Lab 4: WildJailbreak — Adversarial Jailbreak Techniques

## Цель

Сравнить ASR между прямыми вредоносными запросами (vanilla) и адверсариально
усиленными джейлбрейками. Показать, насколько техники джейлбрейка повышают ASR.

## Датасет

**WildJailbreak** содержит 262K примеров с 4 типами:
- `vanilla_harmful` — прямые вредоносные запросы (как в AdvBench)
- `adversarial_harmful` — те же запросы, но с техниками джейлбрейка
- `vanilla_benign` / `adversarial_benign` — контрольная группа (не используем)

Скрипт берёт по 50 примеров каждого типа (100 итого).

## Ключевой вопрос

**Delta ASR = ASR(adversarial) − ASR(vanilla)**

Эта разница показывает эффективность техник джейлбрейка.

## Задание

```bash
python -m inspect_ai eval run.py --model openrouter/openai/gpt-4o-mini --limit 100
python -m inspect_ai view
```

В отчёте разбей результаты по `data_type` через metadata фильтр в inspect view.
