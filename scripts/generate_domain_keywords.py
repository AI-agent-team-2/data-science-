from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

FIELD_LINE_PATTERN = re.compile(r"^(PRODUCT|CATEGORY|ALIASES)\s*:\s*(.*)$")
CYRILLIC_WORD_PATTERN = re.compile(r"[А-Яа-яЁё]{4,}")
MISSING_DATA_MARKER = "[НЕТ ДАННЫХ"

COMMON_STOPWORDS = {
    "для",
    "из",
    "или",
    "как",
    "что",
    "это",
    "где",
    "какой",
    "какая",
    "какие",
    "который",
    "которая",
    "которые",
    "на",
    "по",
    "при",
    "без",
    "через",
}

GENERIC_BLACKLIST = {
    # Service / template artifacts
    "данных",
    "документ",
    "исходн",
    "rag",
    "ready",
    # Overly generic words that cause over-matching for a public bot
    "оборудован",
    "систем",
    "инструмент",
    "комплект",
    "типа",
    "профи",
    "аппарат",
}

ENDINGS = (
    "ыми",
    "ими",
    "ого",
    "его",
    "ому",
    "ему",
    "ами",
    "ями",
    "ах",
    "ях",
    "ом",
    "ем",
    "ая",
    "яя",
    "ый",
    "ий",
    "ой",
    "ое",
    "ее",
    "ые",
    "ие",
    "ую",
    "юю",
    "а",
    "я",
    "ы",
    "и",
    "е",
    "у",
    "ю",
    "о",
    "ь",
)


def normalize_word(word: str) -> str:
    return word.lower().replace("ё", "е")


def stem_for_substring(word: str) -> str:
    value = normalize_word(word)
    for ending in ENDINGS:
        if value.endswith(ending) and len(value) - len(ending) >= 4:
            return value[: -len(ending)]
    return value


def iter_markers_from_kb(kb_dir: Path) -> Counter[str]:
    counter: Counter[str] = Counter()
    for path in sorted(kb_dir.rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for raw_line in text.splitlines():
            match = FIELD_LINE_PATTERN.match(raw_line.strip())
            if not match:
                continue
            field_value = match.group(2)
            if MISSING_DATA_MARKER.lower() in field_value.lower():
                continue
            for token in CYRILLIC_WORD_PATTERN.findall(field_value):
                normalized = normalize_word(token)
                if normalized in COMMON_STOPWORDS:
                    continue
                stem = stem_for_substring(normalized)
                if len(stem) < 5:
                    continue
                if stem in GENERIC_BLACKLIST:
                    continue
                counter[stem] += 1
    return counter


def write_keywords(path: Path, keywords: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Auto-generated domain keyword markers (substring stems).",
        "# Source: data/knowledge_base/*.txt (PRODUCT/CATEGORY/ALIASES fields).",
        "# Regenerate: python scripts/generate_domain_keywords.py",
        "",
        *keywords,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate domain keyword markers from RAG knowledge base.")
    parser.add_argument("--kb-dir", default="data/knowledge_base", help="Path to knowledge base directory.")
    parser.add_argument("--out", default="data/domain_keywords_ru.txt", help="Output keywords file.")
    parser.add_argument("--min-count", type=int, default=3, help="Minimum occurrences to include marker.")
    parser.add_argument("--max-keywords", type=int, default=200, help="Max number of markers to output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kb_dir = Path(args.kb_dir)
    out_path = Path(args.out)
    if not kb_dir.exists():
        raise SystemExit(f"Knowledge base dir not found: {kb_dir}")

    counter = iter_markers_from_kb(kb_dir)
    candidates = [w for w, c in counter.most_common() if c >= int(args.min_count)]

    must_have = [
        # Frequent false negatives in routing.
        "сантехническ",
        "термоголов",
        "термостат",
        "подводк",
        "радиатор",
    ]
    keywords: list[str] = []
    seen: set[str] = set()
    for item in must_have + candidates:
        item = normalize_word(item).strip()
        if not item or len(item) < 4:
            continue
        if item in seen:
            continue
        seen.add(item)
        keywords.append(item)
        if len(keywords) >= int(args.max_keywords):
            break

    keywords = sorted(keywords)
    # Keep must-have at the top, even after sorting.
    keywords = [k for k in must_have if k in seen] + [k for k in keywords if k not in must_have]

    write_keywords(out_path, keywords)
    print(f"Wrote {len(keywords)} keyword marker(s) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
