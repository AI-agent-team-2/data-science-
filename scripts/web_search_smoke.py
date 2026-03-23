from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.tools.web_search import web_search  # noqa: E402

logger = logging.getLogger(__name__)

TEST_QUERY = "что такое балансировочный клапан"
MAX_RESULTS = 5
PREVIEW_RESULTS = 3


def _configure_output() -> None:
    """Настраивает UTF-8 вывод в консоль, если это поддерживается окружением."""
    if not hasattr(sys.stdout, "reconfigure"):
        return

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        logger.exception("Failed to reconfigure stdout encoding")


def _validate_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Проверяет базовую структуру ответа инструмента веб-поиска."""
    results = data.get("results")
    if not isinstance(results, list):
        raise SystemExit(f"Expected 'results' list, got: {type(results)}")

    error = str(data.get("error", "")).strip()
    if error:
        raise SystemExit(f"web_search returned error: {error}")

    return [item for item in results if isinstance(item, dict)]


def main() -> int:
    """Запускает smoke-проверку web_search и печатает первые результаты."""
    logging.basicConfig(level=logging.INFO)
    _configure_output()

    raw = web_search.invoke({"query": TEST_QUERY, "max_results": MAX_RESULTS})
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise SystemExit(f"web_search returned non-JSON output: {exc}") from exc

    if not isinstance(parsed, dict):
        raise SystemExit(f"Expected dict, got: {type(parsed)}")

    results = _validate_response(parsed)
    for index, item in enumerate(results[:PREVIEW_RESULTS], start=1):
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        if not (title and url):
            raise SystemExit(f"Bad item #{index}: {item}")

        logger.info("%d. %s - %s", index, title, url)

    logger.info("Smoke check passed. Results: %d", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
