from __future__ import annotations
from abc import ABC, abstractmethod
from PIL import Image


class OCRModel(ABC):
    """Base class for all OCR model backends."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Called once before inference."""

    @abstractmethod
    def run(self, crop: Image.Image) -> str:
        """Run OCR on a PIL image crop. Returns extracted text string."""


class MockOCR(OCRModel):
    """Test stub — returns a fixed string regardless of input."""

    def __init__(self, return_text: str = "mock text"):
        self.return_text = return_text
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def run(self, crop: Image.Image) -> str:
        return self.return_text
