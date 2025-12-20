"""
Dataset loading utilities for the NCSE v2 dataset.

This module provides functionality for loading and preprocessing the
NCSE v2.0 Dataset of OCR-Processed 19th Century English Newspapers.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


class NCSEDataset:
    """Loader for the NCSE v2 dataset."""

    def __init__(self, dataset_path: Path, split: str = "test"):
        """
        Initialize the NCSE dataset loader.

        Args:
            dataset_path: Path to the NCSE dataset directory
            split: Dataset split to load ('test' is currently supported)
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.images = []
        self.annotations_by_image = {}
        self._loaded = False

    def load(self):
        """Load the dataset from disk."""
        if self._loaded:
            return

        if self.split == "test":
            csv_path = self.dataset_path / "ncse_testset_bboxes.csv"
            images_dir = self.dataset_path / "ncse_test_png_120"
        else:
            raise ValueError(f"Unsupported split: {
                             self.split}. Only 'test' is currently supported.")

        if not csv_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {csv_path}")
        if not images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}")

        df = pd.read_csv(csv_path)

        # Create filename mapping from CSV names to actual files
        actual_files = {f.name: f for f in images_dir.glob("*.png")}
        filename_mapping = self._create_filename_mapping(
            df['filename'].unique(), list(actual_files.keys())
        )

        # Group annotations by filename
        for csv_filename in df['filename'].unique():
            # Find corresponding actual file
            actual_filename = filename_mapping.get(csv_filename)
            if actual_filename:
                image_path = images_dir / actual_filename
                if image_path.exists():
                    self.images.append(str(image_path))

                    # Get all annotations for this image
                    image_annotations = df[df['filename'] == csv_filename]
                    annotations = []

                    for _, row in image_annotations.iterrows():
                        # Convert from (x1, y1, x2, y2) to (x, y, width, height) format
                        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                        annotation = {
                            'x': float(x1),
                            'y': float(y1),
                            'width': float(x2 - x1),
                            'height': float(y2 - y1),
                            'class': row['class'],
                            'confidence': row.get('confidence', 1.0),
                            'page_id': row.get('page_id', ''),
                        }
                        annotations.append(annotation)

                    self.annotations_by_image[str(image_path)] = annotations

        self._loaded = True

    def _create_filename_mapping(self, csv_filenames: list, actual_filenames: list) -> dict:
        """
        Create a mapping from CSV filenames to actual filenames in the directory.

        CSV format: {prefix}_{date}_page_{num}.png
        Actual format: {prefix}_pageid_{id}_pagenum_{num}_{date}_page_1.png

        Args:
            csv_filenames: List of filenames from CSV
            actual_filenames: List of actual filenames in directory

        Returns:
            Dictionary mapping CSV filenames to actual filenames
        """
        mapping = {}

        for csv_name in csv_filenames:
            # Parse CSV filename
            # Examples: EWJ_1858-08-01_page_5.png, EWJ_55_1858-06-01_page_55.png
            parts = csv_name.replace('.png', '').split('_')

            # Extract components
            prefix = parts[0]

            # Handle case where there's a number after prefix (e.g., EWJ_55_...)
            if len(parts) >= 4 and parts[1].isdigit():
                date = parts[2]
                page_num = parts[4] if len(parts) > 4 else parts[3].replace('page', '')
            else:
                date = parts[1]
                page_num = parts[3] if len(parts) > 3 else ''

            # Find matching actual filename
            for actual_name in actual_filenames:
                # Check if prefix, date, and page number match
                if (actual_name.startswith(prefix) and
                    date in actual_name and
                    f'pagenum_{page_num}_' in actual_name):
                    mapping[csv_name] = actual_name
                    break

        return mapping

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if not self._loaded:
            self.load()
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing image path and annotations
        """
        if not self._loaded:
            self.load()

        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {
                             len(self.images)}")

        image_path = self.images[idx]
        return {
            'image_path': image_path,
            'annotations': self.annotations_by_image[image_path],
            'filename': Path(image_path).name,
        }

    def get_annotations(self, idx: int) -> List[Dict[str, Any]]:
        """
        Get ground truth annotations for a sample.

        Args:
            idx: Index of the sample

        Returns:
            List of ground truth regions with bounding boxes and labels
        """
        if not self._loaded:
            self.load()

        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {
                             len(self.images)}")

        return self.annotations_by_image[self.images[idx]]
