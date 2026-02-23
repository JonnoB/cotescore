"""
LayoutLMv3 model loader and inference.

This module provides a wrapper for the LayoutLMv3 model for document layout analysis.
"""

from typing import Any, Dict, List
from pathlib import Path
import logging
import torch
from PIL import Image
from transformers import LayoutLMv3ForObjectDetection, LayoutLMv3Processor
import pytesseract
from .loader import LayoutModel

logger = logging.getLogger(__name__)


class LayoutLMv3(LayoutModel):
    """LayoutLMv3 model for document layout analysis."""

    DEFAULT_MODEL = "microsoft/layoutlmv3-base-finetuned-publaynet"

    def __init__(self, model_name: str = DEFAULT_MODEL, threshold: float = 0.5, device: str = None):
        """
        Initialize the LayoutLMv3 model.

        Args:
            model_name: Name or path of the model (default: finetuned on PubLayNet)
            threshold: Confidence threshold for predictions (default: 0.5)
            device: Device to use (default: auto-detect)
        """
        super().__init__(model_name)
        self.threshold = threshold
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.processor = None

    def load(self):
        """Load the LayoutLMv3 model and processor."""
        logger.info(f"Loading LayoutLMv3 model: {self.model_name}")

        try:
            self.processor = LayoutLMv3Processor.from_pretrained(self.model_name)
            self.model = LayoutLMv3ForObjectDetection.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Run inference on an image.

        Args:
            image_path: Path to the input image

        Returns:
            List of predicted regions.
        """
        if self.model is None:
            self.load()

        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Prepare inputs with OCR
            # LayoutLMv3 requires 'images' but also benefits from OCR inputs if using processor
            # The processor typically handles OCR if not provided, but we can do it explicitly
            # to be safe or rely on processor default behavior (which uses pytesseract)

            # Using automatic OCR from processor
            inputs = self.processor(images=image, return_tensors="pt")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process
            target_sizes = torch.tensor([image.size[::-1]], device=self.device)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.threshold
            )[0]

            return self._extract_predictions(results)

        except Exception as e:
            logger.error(f"Error checking prediction for {image_path}: {e}")
            # If pyteeseract error (e.g. not installed), log it specifically
            if "tesseract" in str(e).lower():
                logger.error("Ensure tesseract-ocr system package is installed.")
            return []

    def _extract_predictions(self, result) -> List[Dict[str, Any]]:
        """Extract predictions from model output."""
        predictions = []

        scores = result["scores"]
        labels = result["labels"]
        boxes = result["boxes"]

        for score, label, box in zip(scores, labels, boxes):
            score_val = score.item()
            label_val = label.item()
            box_val = box.tolist()

            # Get label name from model config
            label_name = self.model.config.id2label.get(label_val, f"LABEL_{label_val}")

            x, y, x2, y2 = box_val
            width = x2 - x
            height = y2 - y

            prediction = {
                "x": float(x),
                "y": float(y),
                "width": float(width),
                "height": float(height),
                "class": label_name,
                "confidence": float(score_val),
            }
            predictions.append(prediction)

        return predictions
