from __future__ import annotations

import unittest
from io import BytesIO


class VisionImagePrepTests(unittest.TestCase):
    def test_prepare_image_for_vision_returns_jpeg(self) -> None:
        try:
            from PIL import Image  # type: ignore
        except Exception:
            self.skipTest("Pillow is not available")

        from app.vision import prepare_image_for_vision

        img = Image.new("RGB", (2000, 1200), (255, 0, 0))
        buff = BytesIO()
        img.save(buff, format="PNG")
        raw = buff.getvalue()

        prepared = prepare_image_for_vision(raw, max_side_px=1024, max_bytes=500_000)
        self.assertTrue(prepared.startswith(b"\xFF\xD8\xFF"))  # JPEG SOI
        self.assertLessEqual(len(prepared), 500_000)


if __name__ == "__main__":
    unittest.main()

