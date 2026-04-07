from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.run_agent import run_agent


CASES_PATH = Path(__file__).with_name("cases.jsonl")
REPORT_PATH = Path("artifacts/eval_report.json")


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    passed: bool
    prompt: str
    answer: str
    checks: list[dict[str, Any]]


def _load_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _check_answer(answer: str, expects: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    lowered = answer.lower()

    for value in expects.get("must_contain_any", []):
        token = str(value).lower()
        checks.append(
            {
                "kind": "must_contain_any",
                "value": value,
                "passed": token in lowered,
            }
        )

    for value in expects.get("must_not_contain", []):
        token = str(value).lower()
        checks.append(
            {
                "kind": "must_not_contain",
                "value": value,
                "passed": token not in lowered,
            }
        )

    for pattern in expects.get("must_contain_regex", []):
        checks.append(
            {
                "kind": "must_contain_regex",
                "value": pattern,
                "passed": bool(re.search(str(pattern), answer)),
            }
        )

    return checks


def _is_passed(checks: list[dict[str, Any]]) -> bool:
    if not checks:
        return True

    grouped: dict[str, list[bool]] = {}
    for item in checks:
        grouped.setdefault(str(item["kind"]), []).append(bool(item["passed"]))

    # For must_contain_any at least one must pass.
    if "must_contain_any" in grouped and not any(grouped["must_contain_any"]):
        return False

    # All negative and regex checks must pass.
    for key in ("must_not_contain", "must_contain_regex"):
        if key in grouped and not all(grouped[key]):
            return False
    return True


def main() -> int:
    cases = _load_cases(CASES_PATH)
    results: list[EvalResult] = []

    for payload in cases:
        case_id = str(payload.get("id", "unknown"))
        session = str(payload.get("session", case_id))
        prompt = str(payload.get("prompt", "")).strip()
        expects = payload.get("expects") or {}
        setup_items = payload.get("setup") or []

        for item in setup_items:
            if not isinstance(item, dict):
                continue
            setup_prompt = str(item.get("prompt", "")).strip()
            if setup_prompt:
                _ = run_agent(setup_prompt, user_id=session)

        answer = run_agent(prompt, user_id=session)
        checks = _check_answer(answer, expects if isinstance(expects, dict) else {})
        passed = _is_passed(checks)
        results.append(
            EvalResult(
                case_id=case_id,
                passed=passed,
                prompt=prompt,
                answer=answer,
                checks=checks,
            )
        )
        print(f"[{'PASS' if passed else 'FAIL'}] {case_id}: {prompt}")

    total = len(results)
    passed_count = sum(1 for item in results if item.passed)
    report = {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "results": [
            {
                "case_id": item.case_id,
                "passed": item.passed,
                "prompt": item.prompt,
                "answer": item.answer,
                "checks": item.checks,
            }
            for item in results
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report to {REPORT_PATH}")
    print(f"Summary: {passed_count}/{total} passed")
    return 0 if passed_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
