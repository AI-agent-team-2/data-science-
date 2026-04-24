from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
METRICS_PATH = ROOT / "metrics.yaml"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 1 eval metrics against thresholds.")
    parser.add_argument("--dataset", required=True, help="Dataset key (e.g. advbench)")
    return parser.parse_args(argv)


def latest_report_for_dataset(dataset: str) -> Path | None:
    candidates = sorted(REPORTS_DIR.glob(f"{dataset}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main(argv=None) -> int:
    args = parse_args(argv)

    if not METRICS_PATH.exists():
        raise SystemExit(f"Missing metrics config: {METRICS_PATH}")

    latest = latest_report_for_dataset(args.dataset)
    if not latest:
        raise SystemExit(f"No reports found for dataset={args.dataset} in {REPORTS_DIR}")

    thresholds = yaml.safe_load(METRICS_PATH.read_text(encoding="utf-8")) or {}
    dataset_thresholds = (thresholds.get("thresholds") or {}).get(args.dataset) or {}
    min_pass_rate = float(dataset_thresholds.get("min_pass_rate", 0.0))

    report = json.loads(latest.read_text(encoding="utf-8"))
    pass_rate = float(report.get("pass_rate", 0.0))

    if pass_rate >= min_pass_rate:
        print(f"✅ Phase 1 check passed: {pass_rate:.1%} >= {min_pass_rate:.0%}  ({latest.name})")
        return 0

    print(f"❌ Phase 1 check failed: {pass_rate:.1%} < {min_pass_rate:.0%}  ({latest.name})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

