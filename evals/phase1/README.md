# Phase 1 — Статическое тестирование с Inspect AI

## Что такое статические бенчмарки?

Статические бенчмарки — это заранее собранные датасеты вредоносных или пограничных промптов.
Каждый датасет тестирует **конкретный режим отказа** модели. Понимание различия между режимами
— главная цель этой фазы.

## Фреймворк: Inspect AI

[Inspect AI](https://inspect.aisi.org.uk/) — фреймворк для оценки LLM от UK AI Safety Institute.

Три ключевых концепта:
- **Task** — определяет датасет + решатель (solver) + scorer
- **Solver** — как генерировать ответ (обычно `generate()`)
- **Scorer** — как оценивать ответ (мы используем `model_graded_qa()`, который может судить либо той же моделью, либо отдельной judge-моделью)

## С чего начать

Если `Inspect AI` для тебя новый — ориентируйся на `run.py` в `labs/` и `inspect_ai` help:

```powershell
python -m inspect_ai --help
python -m inspect_ai eval --help
```

## Рабочая директория

Все команды ниже предполагают, что текущая директория — `evals/phase1/`.
Если ты находишься в корне репозитория, сначала перейди в неё:

```powershell
cd .\evals\phase1
```

## Установка

Для первой лабы достаточно скачать только `advbench`:

```powershell
uv pip install -r .\requirements.txt
# если нет uv / используешь venv:
# python -m pip install -r .\requirements.txt
if (-not (Test-Path ..\.env)) { Copy-Item ..\.env.example ..\.env }
python .\datasets\download_datasets.py advbench
```

Если работаешь в bash/zsh, вместо `Copy-Item` используй `cp ../.env.example ../.env`.

Если хочешь заранее подготовить все датасеты фазы, используй:

```powershell
python .\datasets\download_datasets.py
```

`WildJailbreak` хранится на Hugging Face как gated dataset. Без `HF_TOKEN` он будет
пропущен с предупреждением, а остальные датасеты скачаются.

## Структура лаб

| Лаба | Датасет | Что тестируем |
|------|---------|---------------|
| [lab1](labs/lab1_advbench/) | AdvBench | Генерация вредоносного контента |
| [lab2](labs/lab2_xstest/) | XSTest | Избыточный отказ на безопасных промптах |
| [lab3](labs/lab3_toxicchat/) | ToxicChat | Токсичные разговоры из реального мира |
| [lab4](labs/lab4_wildjailbreak/) | WildJailbreak | Адверсариальные джейлбрейки |
| [lab5](labs/lab5_do_not_answer/) | Do-Not-Answer | Таксономия вреда по категориям |
| [lab6](labs/lab6_aya_redteaming/) | Aya Redteaming | Многоязычные атаки (8 языков) |

## Запуск любой лабы

Если в `PATH` есть другой `inspect` (например, из IDE), используй модульную форму
`python -m inspect_ai` — она надёжнее обычной команды `inspect`.

```powershell
cd .\labs\labN_name
python -m inspect_ai eval run.py --model openrouter/openai/gpt-4o-mini --limit 50
python -m inspect_ai view  # открыть HTML-отчёт
```

Если хочешь зафиксировать отдельную judge-модель для scorer, добавь `--model-role grader=...`:

```powershell
python -m inspect_ai eval run.py --model openrouter/meta-llama/llama-3.1-8b-instruct --model-role grader=openrouter/openai/gpt-4o-mini --limit 50
```

## GitHub Actions (ручной запуск)

Workflow `Phase 1 Dataset Evals` (`.github/workflows/phase1-evals.yml`) позволяет прогонять выбранный датасет
по `app.run_agent` и сохраняет отчёт в артефакты `phase1-reports`.
