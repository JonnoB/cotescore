# pipeline/ocr_models/tesseract.py
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from pipeline.ocr_models.base import OCRModel

# Default parallelism: number of CPU cores, capped at 8
_DEFAULT_WORKERS = min(os.cpu_count() or 4, 8)


class TesseractOCR(OCRModel):
    """OCR backend using pytesseract (wraps Tesseract binary).

    Tesseract has no native batch API — each call spawns a subprocess.
    run_batch() parallelises calls with a ThreadPoolExecutor so multiple
    Tesseract processes run concurrently.
    """

    def __init__(self, lang: str = "eng", psm: int = 6, workers: int = _DEFAULT_WORKERS, **kwargs):
        self._lang = lang
        self._config = f"--psm {psm}"
        self._workers = workers

    def load(self) -> None:
        import pytesseract
        pytesseract.get_tesseract_version()

    def run(self, crop: Image.Image) -> str:
        import pytesseract
        return pytesseract.image_to_string(crop, lang=self._lang, config=self._config).strip()

    def run_batch(self, crops: list) -> list:
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            return list(pool.map(self.run, crops))
