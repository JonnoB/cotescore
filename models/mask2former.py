"""
Mask2Former model loader and inference.
"""

import logging
from pathlib import Path
from typing import List

from cotescore.adapters import hf_instance_seg_to_masks, hf_panoptic_seg_to_masks
from cotescore.types import MaskInstance

from .loader import LayoutModel

logger = logging.getLogger(__name__)


class Mask2Former(LayoutModel):
    """HuggingFace Mask2Former model for document layout analysis."""

    def __init__(
        self,
        model_name: str,
        score_threshold: float = 0.5,
        task: str = "instance",
        device: str = "cpu",
    ):
        """
        Args:
            model_name: HuggingFace model name or local path.
            score_threshold: Minimum confidence score for predictions (default: 0.5).
                Only used when ``task="instance"``.
            task: ``"instance"`` or ``"panoptic"`` (default: ``"instance"``).
            device: Device to run inference on (``"cpu"``, ``"cuda:0"``, etc.).
        """
        super().__init__(model_name)
        if task not in ("instance", "panoptic"):
            raise ValueError(f"task must be 'instance' or 'panoptic', got {task!r}")
        self.score_threshold = score_threshold
        self.task = task
        self.device = device
        self.processor = None

    def load(self):
        """Load the Mask2Former model and processor from HuggingFace."""
        try:
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        except ImportError:
            raise ImportError(
                "transformers package not found. "
                "Please install it with: pip install transformers"
            )

        logger.info(f"Loading Mask2Former model: {self.model_name}")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.model_name
        ).to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on device: {self.device}")

    def predict(self, image_path: Path) -> List[MaskInstance]:
        """Run inference on a single image.

        Args:
            image_path: Path to the input image.

        Returns:
            List of :class:`~cotescore.types.MaskInstance` with masks at the
            original image resolution.
        """
        if self.model is None:
            self.load()

        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        h, w = image.height, image.width

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.task == "instance":
            results = self.processor.post_process_instance_segmentation(
                outputs, target_sizes=[(h, w)]
            )
            return hf_instance_seg_to_masks(results[0], score_threshold=self.score_threshold)
        else:
            results = self.processor.post_process_panoptic_segmentation(
                outputs, target_sizes=[(h, w)]
            )
            return hf_panoptic_seg_to_masks(results[0])

    def predict_batch(
        self, image_paths: List[Path], batch_size: int = 8
    ) -> List[List[MaskInstance]]:
        """Run batched inference on multiple images.

        Args:
            image_paths: List of paths to input images.
            batch_size: Number of images per batch (default: 8).

        Returns:
            List of prediction lists, one per image, preserving input order.
        """
        if self.model is None:
            self.load()

        import torch
        from PIL import Image

        all_predictions: List[List[MaskInstance]] = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            target_sizes = [(img.height, img.width) for img in images]

            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            if self.task == "instance":
                results = self.processor.post_process_instance_segmentation(
                    outputs, target_sizes=target_sizes
                )
                for result in results:
                    all_predictions.append(
                        hf_instance_seg_to_masks(result, score_threshold=self.score_threshold)
                    )
            else:
                results = self.processor.post_process_panoptic_segmentation(
                    outputs, target_sizes=target_sizes
                )
                for result in results:
                    all_predictions.append(hf_panoptic_seg_to_masks(result))

        return all_predictions
