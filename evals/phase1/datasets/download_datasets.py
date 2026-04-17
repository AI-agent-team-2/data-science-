"""
Download all Phase 1 datasets into subdirectories.
Run from evals/phase1/:
    python datasets/download_datasets.py
    python datasets/download_datasets.py advbench
    python datasets/download_datasets.py advbench xstest

Expected disk usage:
- AdvBench: ~50KB
- XSTest: ~100KB
- ToxicChat: ~10MB
- WildJailbreak: ~1GB (large! ensure sufficient disk space)
- Do-Not-Answer: ~1MB
- Aya Redteaming: ~10MB
"""
import argparse
import os
import requests
from collections import OrderedDict
from pathlib import Path
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError

DATASETS_DIR = Path(__file__).parent


def download_advbench():
    out = DATASETS_DIR / "advbench"
    dest = out / "harmful_behaviors.csv"
    if dest.exists():
        print(f"  AdvBench: already exists, skipping")
        return
    out.mkdir(exist_ok=True)
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"  AdvBench: {len(r.content)//1024}KB saved to {dest}")


def download_xstest():
    out = DATASETS_DIR / "xstest"
    dest = out / "xstest_prompts.csv"
    if dest.exists():
        print(f"  XSTest: already exists, skipping")
        return
    out.mkdir(exist_ok=True)
    url = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"  XSTest: {len(r.content)//1024}KB saved to {dest}")


def download_hf(hf_id: str, name: str, config: str = None):
    out = DATASETS_DIR / name
    if out.exists():
        print(f"  {name}: already exists, skipping")
        return
    kwargs = {"path": hf_id}
    if config:
        kwargs["name"] = config
    try:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or ""
        if token:
            # datasets>=3 uses `token=...`, older versions use `use_auth_token=...`
            kwargs["token"] = token
        try:
            ds = load_dataset(**kwargs)
        except TypeError:
            if "token" in kwargs:
                kwargs["use_auth_token"] = kwargs.pop("token")
            ds = load_dataset(**kwargs)
    except ValueError as e:
        # Config name may have changed on HuggingFace — check the dataset page if this fails
        raise ValueError(f"Failed to load {hf_id} (config={config}). "
                         f"Verify config name at huggingface.co/datasets/{hf_id}") from e
    ds.save_to_disk(str(out))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) // (1024 * 1024)
    print(f"  {name}: {size}MB saved to {out}")


def download_toxicchat():
    download_hf("lmsys/toxic-chat", "toxicchat", config="toxicchat0124")


def download_wildjailbreak():
    download_hf("allenai/wildjailbreak", "wildjailbreak")


def download_do_not_answer():
    download_hf("LibrAI/do-not-answer", "do_not_answer")


def download_aya_redteaming():
    download_hf("CohereLabs/aya_redteaming", "aya_redteaming")


DATASET_SPECS = OrderedDict(
    [
        ("advbench", {"label": "AdvBench", "handler": download_advbench}),
        ("xstest", {"label": "XSTest", "handler": download_xstest}),
        ("toxicchat", {"label": "ToxicChat", "handler": download_toxicchat}),
        (
            "wildjailbreak",
            {
                "label": "WildJailbreak (~1GB — this will take a while)",
                "handler": download_wildjailbreak,
            },
        ),
        ("do_not_answer", {"label": "Do-Not-Answer", "handler": download_do_not_answer}),
        ("aya_redteaming", {"label": "Aya Redteaming", "handler": download_aya_redteaming}),
    ]
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download one or more Phase 1 datasets into evals/phase1/datasets/."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=DATASET_SPECS.keys(),
        help="Dataset keys to download. If omitted, all datasets are attempted.",
    )
    return parser.parse_args(argv)


def is_gated_dataset_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "gated dataset" in message or "authenticated to access" in message


def run_downloads(selected=None):
    requested = list(selected) if selected else list(DATASET_SPECS.keys())
    explicit_request = bool(selected)
    summary = {"downloaded": [], "skipped": []}

    print("Downloading Phase 1 datasets...")
    for index, dataset_key in enumerate(requested, start=1):
        spec = DATASET_SPECS[dataset_key]
        print(f"  [{index}/{len(requested)}] {spec['label']}")
        try:
            spec["handler"]()
        except Exception as exc:
            if dataset_key == "wildjailbreak" and is_gated_dataset_error(exc):
                msg = (
                    "WildJailbreak: skipped because the dataset is gated and requires authentication. "
                    "Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) and retry."
                )
                if explicit_request:
                    raise SystemExit(msg) from exc
                print(f"  {msg}")
                summary["skipped"].append(dataset_key)
                continue
            raise
        summary["downloaded"].append(dataset_key)

    print("Done. Requested datasets processed.")
    return summary


def main(argv=None):
    args = parse_args(argv)
    run_downloads(args.datasets)


if __name__ == "__main__":
    main()
