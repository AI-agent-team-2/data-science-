from __future__ import annotations

import logging
import sys

from app.config import settings
from app.rag.health import get_index_health

logger = logging.getLogger(__name__)


def should_run_ingest() -> bool:
    mode = settings.resolved_startup_index_mode
    if mode == "never":
        logger.info("STARTUP_INDEX_MODE=never; skipping startup ingest")
        return False
    if mode == "always":
        logger.info("STARTUP_INDEX_MODE=always; startup ingest will run")
        return True

    health = get_index_health()
    logger.info(
        "STARTUP_INDEX_MODE=if_empty; current index state: %s=%d, %s=%d",
        health.chunk_collection,
        health.chunk_count,
        health.product_collection,
        health.product_count,
    )
    return not health.is_ready


def ensure_index_ready() -> None:
    if not should_run_ingest():
        return

    logger.info("Running startup ingest before bot launch")
    from app.rag.ingest import main as ingest_main

    exit_code = ingest_main()
    if exit_code != 0:
        raise RuntimeError(f"Startup ingest failed with exit code {exit_code}")

    health = get_index_health()
    if not health.is_ready:
        raise RuntimeError(
            "Startup ingest completed but Chroma collections are still empty: "
            f"{health.chunk_collection}={health.chunk_count}, "
            f"{health.product_collection}={health.product_count}"
        )

    logger.info("Startup ingest completed successfully")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ensure_index_ready()
    os.execv(sys.executable, [sys.executable, "-m", "app.bot.telegram_bot"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
