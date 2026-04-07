from __future__ import annotations

import re

from app.utils.sku import extract_sku_candidates, is_russian_identifier

SMALLTALK_MARKERS: tuple[str, ...] = (
    "привет",
    "здравствуйте",
    "здравствуй",
    "добрый день",
    "добрый вечер",
    "доброе утро",
    "как дела",
    "что делаешь",
    "кто ты",
    "ты кто",
    "как настроение",
    "че как",
    "норм работаешь",
    "ты вообще шаришь",
    "спасибо",
    "ок",
    "понял",
)

IDENTITY_OR_CAPABILITY_MARKERS: tuple[str, ...] = (
    "кто ты",
    "ты кто",
    "ты человек",
    "ты чат-бот",
    "что умеешь",
    "чем ты можешь помочь",
    "ты разбираешься в сантехнике",
    "ты можешь подобрать оборудование",
    "что ты знаешь про отопление",
)

OFFTOPIC_OR_RUDE_MARKERS: tuple[str, ...] = (
    "ты тупой",
    "ты бесполезный",
    "ты ничего не понимаешь",
    "дай нормальный ответ",
    "кто выиграет чемпионат мира",
    "напиши код на python",
    "кто такой наполеон",
    "расскажи анекдот",
)

SANITARY_KEYWORDS: tuple[str, ...] = (
    "унитаз",
    "ванна",
    "смеситель",
    "душ",
    "сантехник",
    "раковина",
    "труба",
    "фитинг",
    "кран",
    "бойлер",
    "сантехника",
    "санфаянс",
    "кранбукс",
    "картридж",
    "гидробокс",
    "инсталляция",
    "поддон",
    "лейка",
    "насос",
    "отоплен",
    "коллектор",
    "редуктор",
    "сервопривод",
    "мультифлекс",
    "радиатор",
    "теплый пол",
    "теплого пола",
    "давлен",
)

DOMAIN_MARKERS: tuple[str, ...] = SANITARY_KEYWORDS + (
    "ondo",
    "stm",
    "optima",
    "roegen",
    "rispa",
    "atlas",
    "акс",
    "мультифлекс",
    "редуктор",
    "коллектор",
    "сервопривод",
    "воздухоотвод",
    "тепл",
)

SINGLE_TOKEN_PATTERN = re.compile(r"^[A-Z0-9\-_]{6,}$")


def is_smalltalk(query: str) -> bool:
    """Проверяет, что сообщение похоже на короткую бытовую реплику."""
    lowered_query = query.lower().strip()
    if not lowered_query:
        return False
    if is_domain_query(lowered_query):
        return False

    normalized = re.sub(r"[^\w\s]", " ", lowered_query)
    normalized = " ".join(normalized.split())
    if not normalized:
        return False

    if len(normalized.split()) > 5:
        return False

    for marker in SMALLTALK_MARKERS:
        if " " in marker:
            if normalized == marker:
                return True
            continue

        if re.search(rf"\b{re.escape(marker)}\b", normalized):
            return True

    return False


def is_identity_or_capability_query(query: str) -> bool:
    """Проверяет вопросы о роли бота и его возможностях."""
    lowered_query = query.lower().strip()
    if not lowered_query:
        return False
    return any(marker in lowered_query for marker in IDENTITY_OR_CAPABILITY_MARKERS)


def is_noise_query(query: str) -> bool:
    """Определяет шумовые/неинформативные сообщения."""
    lowered_query = query.lower().strip()
    if not lowered_query:
        return True
    if is_domain_query(lowered_query):
        return False

    alnum = re.sub(r"[\W_]+", "", lowered_query, flags=re.UNICODE)
    if not alnum:
        return True
    if lowered_query in {"???", "...", "...."}:
        return True
    if alnum.isdigit():
        return True
    if len(alnum) <= 4 and not re.search(r"[а-яa-z]", alnum):
        return True
    if len(set(alnum)) <= 2 and len(alnum) >= 6:
        return True
    return False


def is_offtopic_or_rude_query(query: str) -> bool:
    """Возвращает True для оффтопа/резких фраз без доменного контекста."""
    lowered_query = query.lower()
    if is_domain_query(lowered_query):
        return False
    return any(marker in lowered_query for marker in OFFTOPIC_OR_RUDE_MARKERS)


def is_domain_query(lowered_query: str) -> bool:
    """Проверяет, что запрос относится к товарам/техтематике проекта."""
    if has_sku_signal(lowered_query):
        return True
    return any(marker in lowered_query for marker in DOMAIN_MARKERS)


def has_sku_signal(query: str) -> bool:
    """
    Определяет SKU-сигнал без ложных срабатываний на российские идентификаторы и слова вроде EVOH.
    """
    if is_russian_identifier(query):
        return False

    if extract_sku_candidates(query, require_digit=True):
        return True

    compact = re.sub(r"[^\w\-]+", "", str(query or "").strip().upper())
    return bool(SINGLE_TOKEN_PATTERN.fullmatch(compact)) and any(char.isdigit() for char in compact)
