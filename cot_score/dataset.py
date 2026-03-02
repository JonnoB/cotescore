"""
Dataset loading utilities for the NCSE v2 dataset.

This module provides functionality for loading and preprocessing the
NCSE v2.0 Dataset of OCR-Processed 19th Century English Newspapers.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import re
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class NCSEDataset:
    """Loader for the NCSE v2 dataset."""

    def __init__(
        self,
        dataset_path: Path,
        split: str = "test",
        csv_filename: Optional[str] = None,
        images_subdir: Optional[str] = None,
        image_ext: str = "png",
    ):
        """
        Initialize the NCSE dataset loader.

        Args:
            dataset_path: Path to the NCSE dataset directory
            split: Dataset split to load ('test' is currently supported)
            csv_filename: Name of the annotations CSV file (default: 'ncse_testset_bboxes.csv')
            images_subdir: Name of the images subdirectory (default: 'ncse_test_png_120')
            image_ext: Image file extension to glob for (default: 'png')
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.csv_filename = csv_filename or "ncse_testset_bboxes.csv"
        self.images_subdir = images_subdir or "ncse_test_png_120"
        self.image_ext = image_ext
        self.images = []
        self.annotations_by_image = {}
        self._loaded = False

    def load(self):
        """Load the dataset from disk."""
        if self._loaded:
            return

        if self.split != "test":
            raise ValueError("Only 'test' split is currently supported.")

        csv_path = self.dataset_path / self.csv_filename
        images_dir = self.dataset_path / self.images_subdir

        if not csv_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {csv_path}")

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        df = pd.read_csv(csv_path)

        actual_files = {f.name: f for f in images_dir.glob(f"*.{self.image_ext}")}

        # NOTE: Filenames in the ground-truth CSV are now expected to match the actual image
        # filenames in `images_dir` exactly (no translation/mapping required).
        #
        # The previous mapping logic is kept commented out for safety and to make it easy
        # to re-enable if a future dataset release reintroduces differing filename formats.
        # filename_mapping = self._create_filename_mapping(
        #     df["filename"].unique(), list(actual_files.keys())
        # )

        for csv_filename in df["filename"].unique():
            # Previous mapping-based resolution (kept for reference):
            # actual_filename = filename_mapping.get(csv_filename)
            # if not actual_filename:
            #     continue

            if csv_filename not in actual_files:
                continue

            image_path = images_dir / csv_filename
            if not image_path.exists():
                continue

            self.images.append(str(image_path))

            image_annotations = df[df["filename"] == csv_filename]
            annotations = self._build_annotations(image_annotations)
            self.annotations_by_image[str(image_path)] = annotations

        self._loaded = True

    def _create_filename_mapping(self, csv_filenames: list, actual_filenames: list) -> dict:
        """
        Create a mapping from CSV filenames to actual filenames in the directory.

        CSV format: {prefix}_{date}_page_{num}.png
            Example: EWJ_1858-08-01_page_5.png

        Actual format: {prefix}_pageid_{id}_pagenum_{num}_{date}_page_1.png
            Example: EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png

        The mapping matches on three components: prefix, date, and page number.

        Args:
            csv_filenames: List of filenames from CSV
            actual_filenames: List of actual filenames in directory

        Returns:
            Dictionary mapping CSV filenames to actual filenames
        """
        mapping = {}
        unmatched = []

        # Pattern to extract: PREFIX, DATE (YYYY-MM-DD), PAGE_NUM from CSV filename
        # Handles formats like: EWJ_1858-08-01_page_5.png, TEC_1884-03-15_page_23.png, NS2_1843-04-01_page_4.png
        # PREFIX can be letters and numbers (e.g., EWJ, TEC, NS2)
        csv_pattern = re.compile(r"([A-Z0-9]+)_.*?(\d{4}-\d{2}-\d{2})_page_(\d+)\.png")

        for csv_name in csv_filenames:
            match = csv_pattern.search(csv_name)
            if not match:
                logger.warning(f"Could not parse CSV filename: {csv_name}")
                unmatched.append(csv_name)
                continue

            prefix, date, page_num = match.groups()

            matching_file = self._find_matching_file(actual_filenames, prefix, page_num, date)

            if not matching_file:
                logger.warning(
                    f"No matching file found for {csv_name} "
                    f"(looking for: {prefix}_*_pagenum_{page_num}_*_{date}_*)"
                )
                unmatched.append(csv_name)
                continue

            mapping[csv_name] = matching_file

        logger.info(f"Mapped {len(mapping)}/{len(csv_filenames)}" f"CSV files to actual files")
        if unmatched:
            logger.warning(f"Failed to map {len(unmatched)}")

        return mapping

    @staticmethod
    def _find_matching_file(
        actual_filenames: list, prefix: str, page_num: str, date: str
    ) -> Optional[str]:
        """
        Find the actual filename that matches the given CSV components.

        Args:
            actual_filenames: List of actual filenames in the directory
            prefix: Publication prefix (e.g., 'EWJ', 'TEC')
            page_num: Page number as string
            date: Date in YYYY-MM-DD format

        Returns:
            Matching filename or None if not found
        """
        for actual_name in actual_filenames:
            if (
                actual_name.startswith(f"{prefix}_")
                and f"pagenum_{page_num}_" in actual_name
                and date in actual_name
            ):
                return actual_name
        return None

    @staticmethod
    def _build_annotations(image_annotations: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Build annotation dictionaries from DataFrame rows.

        Converts from (x1, y1, x2, y2) format to (x, y, width, height) format.

        Args:
            image_annotations: DataFrame rows for a single image

        Returns:
            List of annotation dictionaries
        """
        annotations = []
        for _, row in image_annotations.iterrows():
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            annotation = {
                "x": float(x1),
                "y": float(y1),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "class": row["class"],
                "confidence": row.get("confidence", 1.0),
                "page_id": row.get("page_id", ""),
            }
            annotations.append(annotation)
        return annotations

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
            raise IndexError(f"Index {idx} out of range for dataset of size " f"{len(self.images)}")

        image_path = self.images[idx]
        return {
            "image_path": image_path,
            "annotations": self.annotations_by_image[image_path],
            "filename": Path(image_path).name,
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
            raise IndexError(f"Index {idx} out of range for dataset of size " f"{len(self.images)}")

        return self.annotations_by_image[self.images[idx]]


class DocLayNetDataset:
    """Loader for the DocLayNet dataset using HuggingFace datasets."""

    def __init__(
        self,
        dataset_path: Path = Path("/teamspace/lightning_storage/doclayout"),
        split: str = "val",
    ):
        """
        Initialize the DocLayNet dataset loader.

        Args:
            dataset_path: Path to store local downloaded representations of images for benchmarking.
                Defaults to /teamspace/lightning_storage/doclayout.
            split: Dataset split to load ('train', 'val', or 'test'). Note: HF dataset uses 'validation' for 'val'.
        """
        self.dataset_path = Path(dataset_path)

        # DocLayNet HF uses 'validation' instead of 'val'
        if split == "val":
            self.split = "validation"
        else:
            self.split = split

        if self.split not in ("train", "validation", "test"):
            raise ValueError("Split must be one of 'train', 'val', or 'test'.")

        self.images_dir = self.dataset_path / "PNG"

        self.images = []
        self.annotations_by_image = {}
        self._loaded = False

        # DocLayNet-v1.1 HF category ID mapping
        self.category_names = {
            1: "Caption",
            2: "Footnote",
            3: "Formula",
            4: "List-item",
            5: "Page-footer",
            6: "Page-header",
            7: "Picture",
            8: "Section-header",
            9: "Table",
            10: "Text",
            11: "Title",
        }

    def load(self):
        """Load the dataset via HuggingFace datasets."""
        if self._loaded:
            return

        import datasets

        # We ensure the image directory exists
        self.images_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading docling-project/DocLayNet-v1.2 {self.split} split...")
        # Only download the specific split's parquet files via data_files.
        # We use split="train" as the virtual key and verification_mode="no_checks"
        # to prevent the library from downloading all splits.
        ds = datasets.load_dataset(
            "docling-project/DocLayNet-v1.2",
            data_files=f"data/{self.split}-*.parquet",
            split="train",
            verification_mode="no_checks",
        )

        for i, row in enumerate(ds):
            # Row has: image, bboxes, category_id, area, metadata
            metadata = row.get("metadata", {})
            original_filename = metadata.get("original_filename", f"doclaynet_{self.split}_{i}.png")

            # The name might be a .pdf in the metadata, ensure we save it as a .png
            if not original_filename.lower().endswith(".png"):
                original_filename += ".png"

            image_path = self.images_dir / original_filename

            # Save image if it doesn't exist
            if not image_path.exists():
                img = row["image"]
                img.save(image_path)

            self.images.append(str(image_path))

            bboxes = row.get("bboxes", [])
            categories = row.get("category_id", [])
            areas = row.get("area", [])

            annotations = []
            for j in range(len(bboxes)):
                # Bbox format in DocLayNet is COCO: [x, y, width, height]
                bbox = bboxes[j]
                if len(bbox) != 4:
                    continue

                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

                cat_id = categories[j] if j < len(categories) else 0
                class_name = self.category_names.get(cat_id, str(cat_id))
                area = areas[j] if j < len(areas) else (w * h)

                annotations.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "width": float(w),
                        "height": float(h),
                        "class": class_name,
                        "confidence": 1.0,
                        "area": float(area),
                    }
                )

            self.annotations_by_image[str(image_path)] = annotations

        self._loaded = True

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
            raise IndexError(f"Index {idx} out of range for dataset of size " f"{len(self.images)}")

        image_path = self.images[idx]
        return {
            "image_path": image_path,
            "annotations": self.annotations_by_image[image_path],
            "filename": Path(image_path).name,
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
            raise IndexError(f"Index {idx} out of range for dataset of size " f"{len(self.images)}")

        return self.annotations_by_image[self.images[idx]]
