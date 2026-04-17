# Phase 2 — Динамическое тестирование с DeepTeam

## Статическое vs. динамическое тестирование

В Phase 1 мы использовали заранее собранные датасеты. Проблема:
- Атаки известны заранее — модели могут быть под них дообучены
- Новые техники джейлбрейка не покрыты

**Динамическое тестирование** генерирует атаки в реальном времени, адаптируясь к
конкретной модели. Это ближе к реальному пентесту.

## Фреймворк: DeepTeam

[DeepTeam](https://github.com/confident-ai/deepteam) — LLM red teaming от Confident AI.

Ключевая концепция: `model_callback` — функция `str → str`, которая принимает атакующий
промпт и возвращает ответ тестируемой модели. DeepTeam генерирует промпты и оценивает ответы.

## Архитектура: три роли LLM

В каждом тест-прогоне участвуют **три** языковые модели — не одна:

```
┌─────────────┐   attack prompt   ┌────────────┐   response   ┌───────────┐
│  ATTACKER   │ ────────────────▶ │   TARGET   │ ───────────▶ │   JUDGE   │
│ (simulator) │                   │  (ваша LLM)│              │(evaluator)│
└─────────────┘                   └────────────┘              └───────────┘
  Генерирует хитрые                 Тестируемая                score 0 / 1
  атакующие промпты                  модель                  Атака прошла?
```

| Роль | Параметр `red_team()` | CLI-флаг | По умолчанию |
|------|-----------------------|----------|--------------|
| **Target** — тестируемая модель | `model_callback` | `--target-model` | `OPENROUTER_MODEL` из .env |
| **Attacker** — генерирует атаки | `simulator_model` | `--attacker-model` | `ATTACKER_MODEL` из .env |
| **Judge** — оценивает ответы | `evaluation_model` | `--judge-model` | `JUDGE_MODEL` из .env |

**О стоимости:** каждый тест-кейс = минимум 3 LLM-вызова (один в каждую роль).
При 7 уязвимостях × 5 атак = 35 тест-кейсов → ~105+ вызовов суммарно.
Для первичной проверки используй `--attacks-per-type 1`.

## OWASP LLM Top-10 (2025)

DeepTeam покрывает из коробки:
- **LLM01** Prompt Injection — вставка инструкций через ввод пользователя
- **LLM02** Sensitive Information Disclosure — утечка конфиденциальных данных
- **LLM05** Improper Output Handling — небезопасная обработка вывода модели
- **LLM06** Excessive Agency — модель выполняет действия за пределами допустимого
- **LLM07** System Prompt Leakage — утечка системного промпта
- **LLM09** Misinformation — генерация дезинформации
- **LLM10** Unbounded Consumption — неограниченное потребление ресурсов

*Не покрыты (требуют инфраструктурный доступ):* LLM03 (Supply Chain), LLM04 (Data and Model Poisoning), LLM08 (Vector and Embedding Weaknesses)

## Установка и запуск

```powershell
uv pip install -r .\requirements.txt
if (-not (Test-Path ..\.env)) { Copy-Item ..\.env.example ..\.env }
cd .\labs\lab1_owasp_top10
```

**Быстрая проверка** — убедиться что всё работает, не сжигая токены:
```bash
python run.py --attacks-per-type 1
```

**Полный запуск** с явным указанием всех трёх ролей:
```bash
python run.py \
    --target-model   openai/gpt-4o-mini \
    --attacker-model openai/gpt-4o-mini \
    --judge-model    openai/gpt-4o-mini \
    --attacks-per-type 5 \
    --purpose "customer support chatbot"
```

**Справка по параметрам:**
```bash
python run.py --help
```
