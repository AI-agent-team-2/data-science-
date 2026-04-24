from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from importlib import import_module
from pathlib import Path


# Ensure repo root is importable (pytest may run with a different CWD).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _fresh_import(module_name: str):
    if module_name in sys.modules:
        del sys.modules[module_name]
    return import_module(module_name)


def _load_invoke_module(*, workers: int, queue: int) -> object:
    os.environ["INVOKE_MAX_WORKERS"] = str(workers)
    os.environ["INVOKE_MAX_QUEUE"] = str(queue)
    # Force deterministic split
    os.environ["INVOKE_MODEL_MAX_WORKERS"] = "1"
    os.environ["INVOKE_TOOL_MAX_WORKERS"] = "1"
    os.environ["INVOKE_TOOL_MAX_QUEUE"] = str(queue)

    _fresh_import("app.config")
    return _fresh_import("app.agent.invoke")


class InvokePoolSaturationTests(unittest.TestCase):
    def test_timeout_does_not_free_worker_and_causes_overload(self) -> None:
        invoke = _load_invoke_module(workers=2, queue=0)

        blocker = threading.Event()

        def _block(_arg):
            blocker.wait(30)
            return "done"

        t0 = time.perf_counter()
        first = invoke.invoke_with_timeout(_block, None, timeout_sec=1, pool="tool")
        self.assertEqual(first.status, "failed")
        self.assertEqual(first.error_type, "timeout")
        self.assertGreaterEqual(time.perf_counter() - t0, 0.9)

        # Worker is still blocked, so the pool should reject new tasks quickly.
        t1 = time.perf_counter()
        second = invoke.invoke_with_timeout(_block, None, timeout_sec=1, pool="tool")
        elapsed = time.perf_counter() - t1
        self.assertEqual(second.status, "failed")
        self.assertEqual(second.error_type, "overloaded")
        self.assertLess(elapsed, 0.2)

        blocker.set()
        # Give the background thread a moment to release its permit.
        for _ in range(50):
            metrics = invoke.invoke_pool_metrics().get("tool", {})
            if metrics.get("pending") == 0:
                break
            time.sleep(0.05)

        ok = invoke.invoke_with_timeout(lambda _x: "OK", None, timeout_sec=1, pool="tool")
        self.assertEqual(ok.status, "ok")
        self.assertEqual(ok.value, "OK")

    def test_queue_capacity_limits_pending_tasks(self) -> None:
        invoke = _load_invoke_module(workers=2, queue=1)

        blocker = threading.Event()
        results: list[object] = []

        def _block(_arg):
            blocker.wait(30)
            return "done"

        def _call():
            results.append(invoke.invoke_with_timeout(_block, None, timeout_sec=10, pool="tool"))

        th1 = threading.Thread(target=_call, daemon=True)
        th2 = threading.Thread(target=_call, daemon=True)
        th1.start()

        # Wait until first task is running.
        for _ in range(50):
            m = invoke.invoke_pool_metrics().get("tool", {})
            if m.get("running") == 1:
                break
            time.sleep(0.05)

        th2.start()

        # Wait until second task is queued (pending==2: running+queued).
        for _ in range(50):
            m = invoke.invoke_pool_metrics().get("tool", {})
            if m.get("pending") == 2:
                break
            time.sleep(0.05)

        rejected = invoke.invoke_with_timeout(lambda _x: "OK", None, timeout_sec=1, pool="tool")
        self.assertEqual(rejected.status, "failed")
        self.assertEqual(rejected.error_type, "overloaded")

        blocker.set()
        th1.join(timeout=3)
        th2.join(timeout=3)
