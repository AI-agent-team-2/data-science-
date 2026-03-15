from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

SKU_RE = re.compile(r"\b[A-Z0-9][A-Z0-9\-_]{4,}\b")
WORD_RE = re.compile(r"[a-zA-Zа-яА-Я0-9]+")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in WORD_RE.findall(text)}


def _extract_field(text: str, field_name: str) -> str:
    pattern = re.compile(rf"^\s*{field_name}\s*:\s*(.+?)\s*$", re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _extract_skus(text: str) -> list[str]:
    # Берем только латинские артикулы/коды, чтобы не тащить шум вроде ГОСТ.
    skus = set()
    for raw in (m.group(0) for m in SKU_RE.finditer(text.upper())):
        token = _canonical_sku(raw)
        if token and any(ch.isdigit() for ch in token):
            skus.add(token)
    return sorted(skus)


def _canonical_sku(value: str) -> str:
    # Приводим SKU к сопоставимому виду: убираем разделители вроде "-" и "_".
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def _build_title(doc_name: str, product: str, product_type: str, document: str) -> str:
    for candidate in (document, product, product_type):
        if candidate:
            return candidate
    return doc_name


@lru_cache(maxsize=1)
def _load_catalog() -> list[dict]:
    root = Path(__file__).resolve().parents[2]
    tp_dir = root / "data" / "knowledge_base" / "tp"
    items: list[dict] = []

    if not tp_dir.exists():
        return items

    for path in sorted(tp_dir.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        brand = _extract_field(text, "BRAND")
        product = _extract_field(text, "PRODUCT")
        product_type = _extract_field(text, "PRODUCT TYPE")
        document = _extract_field(text, "DOCUMENT")
        title = _build_title(path.stem, product, product_type, document)
        skus = _extract_skus(text)

        searchable = _normalize(
            " ".join([title, brand, product, product_type, document, " ".join(skus), text[:3500]])
        )
        items.append(
            {
                "name": title,
                "brand": brand,
                "category": product or product_type,
                "sku_list": skus,
                "source": str(path.relative_to(root)).replace("\\", "/"),
                "searchable": searchable,
                "tokens": _tokenize(searchable),
            }
        )

    return items


def _score_item(query: str, query_tokens: set[str], query_skus: set[str], item: dict) -> float:
    score = 0.0

    item_skus = {_canonical_sku(s) for s in item["sku_list"]}
    matched_skus = sorted(query_skus.intersection(item_skus))
    if matched_skus:
        score += 100 + 25 * len(matched_skus)

    overlap = len(query_tokens.intersection(item["tokens"]))
    if overlap:
        score += overlap * 5
        score += (overlap / max(1, len(query_tokens))) * 20

    if query in item["searchable"]:
        score += 15

    return score


@tool
def product_lookup(query: str, limit: int = 5) -> str:
    """
    Поиск товара по названию, бренду, артикулу или параметрам в локальном каталоге.
    """
    query = _normalize(query)
    if not query:
        return json.dumps(
            {"query": query, "count": 0, "results": [], "note": "Пустой поисковый запрос."},
            ensure_ascii=False,
            indent=2,
        )

    catalog = _load_catalog()
    if not catalog:
        return json.dumps(
            {
                "query": query,
                "count": 0,
                "results": [],
                "note": "Каталог не найден. Проверьте data/knowledge_base/tp.",
            },
            ensure_ascii=False,
            indent=2,
        )

    query_tokens = _tokenize(query)
    query_skus = {_canonical_sku(s) for s in SKU_RE.findall(query.upper())}
    top_n = max(1, min(limit, 20))

    # SKU-first: если пользователь прислал артикул, сначала отдаем точные SKU-совпадения.
    if query_skus:
        sku_ranked: list[tuple[float, dict]] = []
        for item in catalog:
            item_skus = {_canonical_sku(s) for s in item["sku_list"]}
            matched_count = len(query_skus.intersection(item_skus))
            if matched_count == 0:
                continue
            # Бонус за совпадение токенов помогает сортировать при нескольких SKU-кандидатах.
            token_overlap = len(query_tokens.intersection(item["tokens"]))
            score = matched_count * 100 + token_overlap * 3
            sku_ranked.append((score, item))

        if sku_ranked:
            sku_ranked.sort(key=lambda x: x[0], reverse=True)
            top = sku_ranked[:top_n]
            results = []
            for score, item in top:
                results.append(
                    {
                        "name": item["name"],
                        "brand": item["brand"],
                        "category": item["category"],
                        "sku_list": item["sku_list"][:20],
                        "source": item["source"],
                        "score": round(score, 2),
                    }
                )
            return json.dumps(
                {
                    "query": query,
                    "count": len(results),
                    "mode": "sku_first",
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )

    ranked: list[tuple[float, dict]] = []
    for item in catalog:
        score = _score_item(query, query_tokens, query_skus, item)
        if score > 0:
            ranked.append((score, item))

    ranked.sort(key=lambda x: x[0], reverse=True)
    top = ranked[:top_n]

    results = []
    for score, item in top:
        results.append(
            {
                "name": item["name"],
                "brand": item["brand"],
                "category": item["category"],
                "sku_list": item["sku_list"][:20],
                "source": item["source"],
                "score": round(score, 2),
            }
        )

    return json.dumps(
        {
            "query": query,
            "count": len(results),
            "mode": "text_ranked",
            "results": results,
        },
        ensure_ascii=False,
        indent=2,
    )
