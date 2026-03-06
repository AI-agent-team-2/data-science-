# Telegram RAG Bot: Гид Maker'а (Earth-6160)

Telegram-бот, который отвечает на вопросы по лору **Ultimate Universe Earth-6160** в роли *официального Гида Совета Maker'а*.

RAG реализован в `rag.py`: бот берёт фрагменты из локальной базы знаний `knowledge_base.txt` и использует LLM через OpenRouter.

## Возможности

- Ответы на вопросы по лору из `knowledge_base.txt`
- Команда `/event <запрос>` — генерация сюжетной сцены/события в рамках лора
- Работа через OpenRouter (`OPENROUTER_API_KEY`)

## Требования

- Python 3.10+ (рекомендуется)
- Токен Telegram-бота (`BOT_TOKEN`)
- Ключ OpenRouter (`OPENROUTER_API_KEY`)

## Быстрый старт (Windows / PowerShell)

Перейдите в папку проекта:

```powershell
cd C:\study\tg_bot
```

Создайте и активируйте виртуальное окружение:

```powershell
py -m venv myenv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\myenv\Scripts\Activate.ps1
```

Установите зависимости:

```powershell
pip install -r requirements.txt
```

Создайте файл `.env` в корне проекта:

```env
BOT_TOKEN=ваш_токен_телеграм_бота
OPENROUTER_API_KEY=ваш_ключ_openrouter
```

Убедитесь, что рядом с `rag.py` лежит база знаний:

- `knowledge_base.txt`

Запуск:

```powershell
python bot.py
```

## Команды бота

- `/start` — приветствие и краткая справка
- `/help` — помощь
- `/event <запрос>` — генерация события/сцены (например: `/event стычка в подземном городе свободы`)

## Файлы проекта

- `bot.py` — Telegram-бот (обработчики команд и сообщений)
- `rag.py` — RAG: разбиение на чанки, TF‑IDF поиск релевантных фрагментов, вызов LLM через OpenRouter
- `knowledge_base.txt` — локальная база знаний (не коммитится в git)

## Важно про git

Файл `knowledge_base.txt` **игнорируется** через `.gitignore` и не выгружается на GitHub.
