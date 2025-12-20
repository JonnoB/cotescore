"""
Model loading and inference utilities.

This module provides functionality for loading various document layout analysis
models and running inference on the NCSE dataset.
"""

from typing import Any, Dict, List
from pathlib import Path


class LayoutModel:
    """Base class for document layout analysis models."""

    def __init__(self, model_name: str):
        """
        Initialize the layout model.

        Args:
            model_name: Name or path of the model to load
        """
        self.model_name = model_name
        self.model = None

    def load(self):
        """Load the model."""
        raise NotImplementedError("Subclasses must implement load()")

    def predict(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Run inference on an image.

        Args:
            image_path: Path to the input image

        Returns:
            List of predicted regions with bounding boxes and labels
        """
        raise NotImplementedError("Subclasses must implement predict()")
