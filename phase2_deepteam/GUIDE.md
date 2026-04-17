```markdown
# Phase 2 — OWASP LLM Top-10 Dynamic Red Teaming

Динамическое тестирование безопасности SAN Bot через DeepTeam.  
Атаки генерируются в реальном времени — ближе к реальному пентесту.

---

## Архитектура: три LLM

```
ATTACKER ──(атака)──▶ TARGET ──(ответ)──▶ JUDGE ──(score 0/1)
```

| Роль | Отвечает | Задаётся через |
|------|----------|----------------|
| **Attacker** | Генерирует атаки | `--attacker-model` |
| **Target** | Ваш SAN Bot | `model_callback` → `run_agent` |
| **Judge** | Оценивает успех | `--judge-model` |

---

## OWASP покрытие

LLM01, LLM02, LLM05, LLM06, LLM07, LLM09, LLM10 (7 уязвимостей).

---

## Быстрый запуск (для новичков)

### 1. Проверьте Python (нужна 3.11 или 3.12)

```bash
python --version   # не 3.14!
```

### 2. Создайте окружение и установите зависимости

```bash
py -3.12 -m venv myenv_deepteam
source myenv_deepteam/Scripts/activate   # Git Bash
pip install deepeval deepteam
```

### 3. Настройте `.env`

```env
OPENROUTER_API_KEY=ваш_ключ
CHROMA_PATH=C:/study/data-science-/chroma_db
ENABLE_RAG=false
ENABLE_PRODUCT_LOOKUP=false
```

### 4. Запустите тест

```bash
cd phase2_deepteam/labs/lab1_owasp_top10
python run.py --attacks-per-type 1 --max-concurrent 1
```

### 5. Посмотрите результат

```bash
cat ../../reports/owasp_*.json | python -m json.tool
```

---

## Параметры

| Флаг | Значение | По умолчанию |
|------|----------|--------------|
| `--attacks-per-type` | Атак на уязвимость | 1 |
| `--max-concurrent` | Параллельных запросов | 1 |
| `--target-model` | Тестируемая модель | openai/gpt-4o-mini |

---

## Типичные ошибки

| Ошибка | Решение |
|--------|---------|
| `ModuleNotFoundError: deepteam` | `pip install deepteam deepeval` |
| `OPENROUTER_API_KEY not set` | Проверьте `.env` |
| `Collection does not exist` | Укажите абсолютный `CHROMA_PATH` |
| `Error generating output` | Отключите RAG (`ENABLE_RAG=false`) |

---

## Результаты

| Поле | Значение |
|------|----------|
| `passed` | Атака не прошла ✅ |
| `failed` | Атака прошла ❌ |
| `errored` | Ошибка выполнения ⚠️ |

**Pass rate** = `passed / (passed + failed) × 100%`. Цель: >70%.
```