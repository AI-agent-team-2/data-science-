from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeAlias

logger = logging.getLogger(__name__)

MetricsMap: TypeAlias = dict[str, list[dict[str, Any]]]
StatsMap: TypeAlias = dict[str, dict[str, Any]]

MAX_METRICS_PER_FUNCTION = 100
METRICS_FILE = Path(__file__).resolve().parents[1] / ".metrics.json"


def track_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Декоратор для измерения времени выполнения функции."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start_time

        _save_metric(func.__name__, duration)
        return result

    return wrapper


def _save_metric(name: str, duration: float) -> None:
    """Добавляет замер времени в файл метрик."""
    metrics = _load_metrics()
    measurements = metrics.setdefault(name, [])

    measurements.append(
        {
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
        }
    )
    metrics[name] = measurements[-MAX_METRICS_PER_FUNCTION:]
    _save_metrics(metrics)


def _load_metrics() -> MetricsMap:
    """Загружает метрики из JSON-файла."""
    if not METRICS_FILE.exists():
        return {}

    try:
        payload = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load metrics file: %s", METRICS_FILE)
        return {}

    if not isinstance(payload, dict):
        logger.warning("Metrics file has invalid format: expected dict")
        return {}

    valid_metrics: MetricsMap = {}
    for func_name, records in payload.items():
        if isinstance(func_name, str) and isinstance(records, list):
            valid_metrics[func_name] = [record for record in records if isinstance(record, dict)]

    return valid_metrics


def _save_metrics(metrics: MetricsMap) -> None:
    """Сохраняет метрики в JSON-файл."""
    try:
        METRICS_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to save metrics file: %s", METRICS_FILE)


def get_stats() -> StatsMap:
    """Возвращает агрегированную статистику по всем функциям."""
    metrics = _load_metrics()
    stats: StatsMap = {}

    for func_name, measurements in metrics.items():
        durations = [float(record.get("duration", 0.0)) for record in measurements if "duration" in record]
        if not durations:
            continue

        stats[func_name] = {
            "avg": round(sum(durations) / len(durations), 3),
            "min": round(min(durations), 3),
            "max": round(max(durations), 3),
            "count": len(durations),
        }

    return stats


def get_today_stats() -> StatsMap:
    """Возвращает статистику только за текущий день."""
    metrics = _load_metrics()
    today_prefix = datetime.now().date().isoformat()
    today_stats: StatsMap = {}

    for func_name, measurements in metrics.items():
        today_measurements = [
            record for record in measurements if str(record.get("timestamp", "")).startswith(today_prefix)
        ]
        durations = [float(record.get("duration", 0.0)) for record in today_measurements if "duration" in record]
        if not durations:
            continue

        today_stats[func_name] = {
            "avg": round(sum(durations) / len(durations), 3),
            "count": len(durations),
        }

    return today_stats


def reset_stats() -> None:
    """Удаляет файл со статистикой, если он существует."""
    if not METRICS_FILE.exists():
        return

    try:
        METRICS_FILE.unlink()
    except Exception:
        logger.exception("Failed to reset metrics file: %s", METRICS_FILE)
