"""Tests for the HierTextDataset loader."""

import json
from pathlib import Path

import pytest
from PIL import Image

from cotescore.dataset import HierTextDataset


def _make_dataset(tmp_path: Path, gt: dict, image_ids, ext: str = "jpg"):
    """Write a GT JSON + dummy images to tmp_path and return a loaded dataset."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for image_id in image_ids:
        Image.new("RGB", (100, 100), (255, 255, 255)).save(images_dir / f"{image_id}.{ext}")

    gt_path = tmp_path / "validation.jsonl"
    gt_path.write_text(json.dumps(gt))

    ds = HierTextDataset(images_path=images_dir, groundtruth_path=gt_path, image_ext=ext)
    ds.load()
    return ds


# Two paragraphs: first has 2 lines, second has 1 line. One line is illegible.
SAMPLE_GT = {
    "annotations": [
        {
            "image_id": "img_a",
            "paragraphs": [
                {
                    "legible": True,
                    "vertices": [[0, 0], [50, 0], [50, 40], [0, 40]],
                    "lines": [
                        {
                            "legible": True,
                            "text": "hello",
                            "vertices": [[10, 10], [30, 10], [30, 20], [10, 20]],
                            "words": [],
                        },
                        {
                            "legible": False,
                            "text": "",
                            "vertices": [[10, 25], [40, 25], [40, 35], [10, 35]],
                            "words": [],
                        },
                    ],
                },
                {
                    "legible": True,
                    "vertices": [[0, 50], [60, 50], [60, 80], [0, 80]],
                    "lines": [
                        {
                            "legible": True,
                            "text": "world",
                            "vertices": [[5, 55], [55, 55], [55, 75], [5, 75]],
                            "words": [],
                        }
                    ],
                },
            ],
        }
    ]
}


def test_line_is_region_count(tmp_path):
    ds = _make_dataset(tmp_path, SAMPLE_GT, ["img_a"])
    sample = ds[0]
    # 2 lines in paragraph 1 + 1 line in paragraph 2 = 3 regions
    assert len(sample["annotations"]) == 3


def test_paragraph_is_ssu_id(tmp_path):
    ds = _make_dataset(tmp_path, SAMPLE_GT, ["img_a"])
    anns = ds[0]["annotations"]
    ssu_ids = [a["ssu_id"] for a in anns]
    # First two lines share paragraph 1, third line is paragraph 2; 1-based.
    assert ssu_ids == [1, 1, 2]


def test_polygon_to_bbox(tmp_path):
    ds = _make_dataset(tmp_path, SAMPLE_GT, ["img_a"])
    first = ds[0]["annotations"][0]
    # vertices [[10,10],[30,10],[30,20],[10,20]] -> x=10 y=10 w=20 h=10
    assert first["x"] == 10.0
    assert first["y"] == 10.0
    assert first["width"] == 20.0
    assert first["height"] == 10.0


def test_illegible_lines_retained(tmp_path):
    ds = _make_dataset(tmp_path, SAMPLE_GT, ["img_a"])
    anns = ds[0]["annotations"]
    # The illegible second line (x=10,y=25,w=30,h=10) must be present.
    assert any(a["x"] == 10.0 and a["y"] == 25.0 and a["width"] == 30.0 for a in anns)


def test_missing_image_skipped(tmp_path):
    # GT references img_a and img_missing, but only img_a has an image file.
    gt = {
        "annotations": [
            SAMPLE_GT["annotations"][0],
            {"image_id": "img_missing", "paragraphs": []},
        ]
    }
    ds = _make_dataset(tmp_path, gt, ["img_a"])
    assert len(ds) == 1
    assert ds[0]["filename"] == "img_a.jpg"


def test_getitem_shape(tmp_path):
    ds = _make_dataset(tmp_path, SAMPLE_GT, ["img_a"])
    sample = ds[0]
    assert set(sample.keys()) == {"image_path", "annotations", "filename"}
    ann = sample["annotations"][0]
    for key in ("x", "y", "width", "height", "class", "ssu_id", "ssu_class", "confidence", "page_id"):
        assert key in ann
    assert ann["class"] == "text"
    assert ann["ssu_class"] == "object"
    assert ann["page_id"] == "img_a"


def test_index_out_of_range(tmp_path):
    ds = _make_dataset(tmp_path, SAMPLE_GT, ["img_a"])
    with pytest.raises(IndexError):
        _ = ds[5]
