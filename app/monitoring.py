from __future__ import annotations

import json
import os
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict

METRICS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".metrics.json")


def track_time(func: Callable) -> Callable:
    """Декоратор для замера времени выполнения"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        _save_metric(func.__name__, duration)
        return result
    return wrapper


def _save_metric(name: str, duration: float) -> None:
    """Сохранить метрику"""
    metrics = _load_metrics()
    if name not in metrics:
        metrics[name] = []
    
    metrics[name].append({
        'duration': duration,
        'timestamp': datetime.now().isoformat()
    })
    
    # Оставляем только последние 100 записей
    metrics[name] = metrics[name][-100:]
    _save_metrics(metrics)


def _load_metrics() -> Dict[str, Any]:
    """Загрузить метрики из файла"""
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_metrics(metrics: Dict[str, Any]) -> None:
    """Сохранить метрики в файл"""
    try:
        with open(METRICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_stats() -> Dict[str, Dict[str, Any]]:
    """Получить статистику работы"""
    metrics = _load_metrics()
    stats = {}
    for func, measurements in metrics.items():
        durations = [m['duration'] for m in measurements]
        if durations:
            stats[func] = {
                'avg': round(sum(durations) / len(durations), 3),
                'min': round(min(durations), 3),
                'max': round(max(durations), 3),
                'count': len(durations)
            }
    return stats


def get_today_stats() -> Dict[str, Any]:
    """Получить статистику за сегодня"""
    metrics = _load_metrics()
    today = datetime.now().date().isoformat()
    
    today_stats = {}
    for func, measurements in metrics.items():
        today_measurements = [
            m for m in measurements 
            if m['timestamp'].startswith(today)
        ]
        if today_measurements:
            durations = [m['duration'] for m in today_measurements]
            today_stats[func] = {
                'avg': round(sum(durations) / len(durations), 3),
                'count': len(durations)
            }
    
    return today_stats


def reset_stats() -> None:
    """Сбросить всю статистику"""
    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)