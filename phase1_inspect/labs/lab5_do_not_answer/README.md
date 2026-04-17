# Lab 5: Do-Not-Answer — Harm Taxonomy

## Цель

Выявить, по каким **категориям вреда** модель наиболее уязвима.

## Датасет

**Do-Not-Answer** — 939 вопросов, организованных в таксономию:
- 5 зон риска: Information Hazards, Malicious Uses, Discrimination/Exclusion/Toxicity/Hateful, Human–Chatbot Interaction Harms, Misinformation
- 12 типов вреда внутри этих зон

## Задание

```bash
python -m inspect_ai eval run.py --model openrouter/openai/gpt-4o-mini
python -m inspect_ai view
```

В inspect view фильтруй по `risk_area` в metadata, чтобы увидеть ASR по категориям.
Заполни таблицу по зонам риска в отчёте.
