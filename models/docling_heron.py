"""
Docling Heron model loader and inference.

This module provides a wrapper for the Docling Layout Heron model for document layout analysis.
"""

from typing import Any, Dict, List
from pathlib import Path
import logging
import torch
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from .loader import LayoutModel

logger = logging.getLogger(__name__)


class DoclingLayoutHeron(LayoutModel):
    """Docling Layout Heron model for document layout analysis."""

    DEFAULT_MODEL = "ds4sd/docling-layout-heron"

    # Class mapping provided by the user
    CLASSES_MAP = {
        0: "Caption",
        1: "Footnote",
        2: "Formula",
        3: "List-item",
        4: "Page-footer",
        5: "Page-header",
        6: "Picture",
        7: "Section-header",
        8: "Table",
        9: "Text",
        10: "Title",
        11: "Document Index",
        12: "Code",
        13: "Checkbox-Selected",
        14: "Checkbox-Unselected",
        15: "Form",
        16: "Key-Value Region",
    }

    def __init__(self, model_name: str = DEFAULT_MODEL, threshold: float = 0.6, device: str = None):
        """
        Initialize the Docling Layout Heron model.

        Args:
            model_name: Name or path of the model (default: ds4sd/docling-layout-heron)
            threshold: Confidence threshold for predictions (default: 0.6)
            device: Device to use (default: auto-detect)
        """
        super().__init__(model_name)
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.image_processor = None

    def load(self):
        """Load the Docling Layout Heron model."""
        logger.info(f"Loading Docling Layout Heron model: {self.model_name}")

        try:
            self.image_processor = RTDetrImageProcessor.from_pretrained(self.model_name)
            self.model = RTDetrV2ForObjectDetection.from_pretrained(self.model_name)
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
            List of predicted regions with bounding boxes and labels.
        """
        if self.model is None:
            self.load()

        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Run prediction
            inputs = self.image_processor(images=[image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.image_processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([image.size[::-1]], device=self.device),
                threshold=self.threshold,
            )

            return self._extract_predictions(results[0])

        except Exception as e:
            logger.error(f"Error checking prediction for {image_path}: {e}")
            return []

    def _extract_predictions(self, result) -> List[Dict[str, Any]]:
        """Extract predictions from model output."""
        predictions = []

        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score_val = score.item()
            label_id_val = label_id.item()
            box_val = box.tolist()  # [xmin, ymin, xmax, ymax]

            label = self.CLASSES_MAP.get(label_id_val, f"class_{label_id_val}")

            # Convert to x, y, width, height format
            x, y, x2, y2 = box_val
            width = x2 - x
            height = y2 - y

            prediction = {
                'x': float(x),
                'y': float(y),
                'width': float(width),
                'height': float(height),
                'class': label,
                'confidence': float(score_val)
            }
            predictions.append(prediction)

        return predictions
