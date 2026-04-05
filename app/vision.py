from __future__ import annotations

from io import BytesIO


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
    """
    if not image_bytes:
        return image_bytes

    try:
        from PIL import Image  # type: ignore
    except Exception:
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
                    return data
            return best
    except Exception:
        return image_bytes

