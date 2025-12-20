"""
DocLayout-YOLO model loader and inference.

This module provides a wrapper for the DocLayout-YOLO model for document layout analysis.
"""

from typing import Any, Dict, List
from pathlib import Path
import logging
from .loader import LayoutModel

logger = logging.getLogger(__name__)


class DocLayoutYOLO(LayoutModel):
    """DocLayout-YOLO model for document layout analysis."""

    def __init__(self, model_name: str = "juliozhao/DocLayout-YOLO-DocStructBench",
                 conf_threshold: float = 0.2, imgsz: int = 1024, device: str = "cpu"):
        """
        Initialize the DocLayout-YOLO model.

        Args:
            model_name: Name or path of the model (default: pretrained from HuggingFace)
            conf_threshold: Confidence threshold for predictions (default: 0.2)
            imgsz: Image size for inference (default: 1024)
            device: Device to use for inference ('cpu' or 'cuda:0', etc.)
        """
        super().__init__(model_name)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device

    def load(self):
        """Load the DocLayout-YOLO model."""
        try:
            from doclayout_yolo import YOLOv10
        except ImportError:
            raise ImportError(
                "doclayout-yolo package not found. "
                "Please install it with: pip install doclayout-yolo"
            )

        logger.info(f"Loading DocLayout-YOLO model: {self.model_name}")

        if self.model_name == "juliozhao/DocLayout-YOLO-DocStructBench":
            self.model = self._load_docstructbench_model(YOLOv10)
        elif "/" in self.model_name:
            self.model = YOLOv10.from_pretrained(self.model_name)
        else:
            self.model = YOLOv10(self.model_name)

        logger.info(f"Model loaded successfully on device: {self.device}")

    def _load_docstructbench_model(self, YOLOv10):
        """Load the DocStructBench model from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
            logger.info("Downloading model from Hugging Face...")
            filepath = hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename="doclayout_yolo_docstructbench_imgsz1024.pt"
            )
            logger.info(f"Model downloaded to: {filepath}")
            return YOLOv10(filepath)
        except Exception as e:
            logger.error(f"Error downloading from HuggingFace: {e}")
            logger.info("Attempting alternative loading method...")
            return YOLOv10.from_pretrained(self.model_name)

    def predict(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Run inference on an image.

        Args:
            image_path: Path to the input image

        Returns:
            List of predicted regions with bounding boxes and labels in the format:
            [{'x': float, 'y': float, 'width': float, 'height': float, 'class': str, 'confidence': float}, ...]
        """
        if self.model is None:
            self.load()

        results = self.model.predict(
            str(image_path),
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=self.device
        )

        if not results or len(results) == 0:
            return []

        result = results[0]
        if not hasattr(result, 'boxes') or result.boxes is None:
            return []

        boxes = result.boxes
        if not hasattr(boxes, 'xyxy'):
            return []

        return self._extract_predictions(boxes, result)

    def _extract_predictions(self, boxes, result) -> List[Dict[str, Any]]:
        """Extract predictions from YOLO boxes."""
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
        cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
        names = result.names if hasattr(result, 'names') else {}

        predictions = []
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box
            class_id = int(cls_ids[i])
            confidence = float(confs[i])
            class_name = names.get(class_id, f"class_{class_id}")

            prediction = {
                'x': float(x1),
                'y': float(y1),
                'width': float(x2 - x1),
                'height': float(y2 - y1),
                'class': class_name,
                'confidence': confidence
            }
            predictions.append(prediction)

        return predictions
