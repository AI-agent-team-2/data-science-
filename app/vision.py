from __future__ import annotations

import logging
import time
from io import BytesIO

logger = logging.getLogger(__name__)

MAX_PHOTO_SIZE_MB = 10
MAX_PHOTO_SIZE_BYTES = MAX_PHOTO_SIZE_MB * 1024 * 1024


def prepare_image_for_vision(
    image_bytes: bytes,
    *,
    max_side_px: int = 1024,
    quality: int = 85,
    max_bytes: int = 1_500_000,
) -> bytes:
    """
    Готовит изображение для Vision LLM:
    - декодирует через Pillow
    - приводит к JPEG (RGB)
    - уменьшает сторону до max_side_px
    - старается уложиться в max_bytes
    - логирует размер и время обработки
    """
    start_time = time.time()
    original_size_mb = len(image_bytes) / (1024 * 1024)
    logger.info(f"📸 Image input size: {original_size_mb:.2f}MB")

    if not image_bytes:
        return image_bytes

    # Проверка на слишком большой файл
    if len(image_bytes) > MAX_PHOTO_SIZE_BYTES:
        logger.warning(f"⚠️ Image too large: {original_size_mb:.2f}MB, max allowed: {MAX_PHOTO_SIZE_MB}MB")
        # Вместо возврата оригинального байта, можно попробовать сжать
        # но в текущей логике сжатие происходит ниже

    try:
        from PIL import Image  # type: ignore
    except Exception:
        logger.warning("PIL not available, returning original image")
        return image_bytes

    def _encode(img: "Image.Image", *, side: int, q: int) -> bytes:
        resized = img.copy()
        resized.thumbnail((side, side))
        buff = BytesIO()
        resized.save(buff, format="JPEG", quality=q, optimize=True)
        return buff.getvalue()

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB")
            candidates = []
            for side in (max_side_px, 768, 512):
                if side > max_side_px:
                    continue
                for q in (quality, 75, 65):
                    candidates.append(_encode(rgb, side=side, q=q))

            best = min(candidates, key=len) if candidates else image_bytes
            for data in candidates:
                if len(data) <= max_bytes:
                    elapsed = time.time() - start_time
                    output_size_mb = len(data) / (1024 * 1024)
                    logger.info(f"📸 Image processed in {elapsed:.2f}s, output size: {output_size_mb:.2f}MB")
                    return data
            
            elapsed = time.time() - start_time
            output_size_mb = len(best) / (1024 * 1024)
            logger.info(f"📸 Image processed in {elapsed:.2f}s (fallback), output size: {output_size_mb:.2f}MB")
            return best
    except Exception as e:
        logger.exception(f"Failed to process image: {e}")
        return image_bytes

