from __future__ import annotations

import json
import sys
from pathlib import Path

# Добавляем корень проекта в путь импорта, чтобы скрипт запускался из любой директории.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.tools.web_search import web_search  # noqa: E402


def main() -> None:
    # Чтобы печать не падала на Windows-консолях со старой кодировкой.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Тестовый запрос для быстрой smoke-проверки инструмента.
    query = "что такое балансировочный клапан"
    # Вызываем tool через стандартный интерфейс LangChain Tool.
    raw = web_search.invoke({"query": query, "max_results": 5})
    try:
        # Проверяем, что инструмент вернул валидный JSON.
        data = json.loads(raw)
    except Exception:
        raise SystemExit("web_search returned non-JSON output")

    if not isinstance(data, list):
        raise SystemExit(f"Expected list, got: {type(data)}")

    # Печатаем первые 3 результата и валидируем ключевые поля.
    for i, item in enumerate(data[:3], start=1):
        title = item.get("title")
        snippet = item.get("snippet")
        url = item.get("url")
        if not (title and url):
            raise SystemExit(f"Bad item #{i}: {item}")
        print(f"{i}. {title} — {url}")


if __name__ == "__main__":
    main()

