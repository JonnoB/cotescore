import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.dofns.io import parse_alto_xml

ALTO_PATH = Path("data/the_spiritualist/ocr_gt_with_ssu/0001_p001.xml")
IMAGE_PATH = Path("data/the_spiritualist/spiritualist_images/0001_p001.jpg")


@pytest.fixture
def parsed():
    return parse_alto_xml(ALTO_PATH, IMAGE_PATH)


def test_parse_returns_box_records(parsed):
    gt_boxes, token_pos, tl_texts = parsed
    assert len(gt_boxes) > 0
    box = gt_boxes[0]
    assert "image_id" in box
    assert "box_id" in box
    assert box["source"] == "gt"
    for key in ("x", "y", "width", "height"):
        assert key in box
    assert "class" in box


def test_box_id_comes_from_ssu(parsed):
    gt_boxes, _, _ = parsed
    # First TextLine in 0001_p001.xml has SSU="ssu_1_span_tr_1743007782"
    assert gt_boxes[0]["box_id"] == "ssu_1_span_tr_1743007782"


def test_token_positions_are_numpy_arrays(parsed):
    _, tp, _ = parsed
    assert hasattr(tp, "tokens") and hasattr(tp, "xs") and hasattr(tp, "ys")
    assert len(tp.tokens) == len(tp.xs) == len(tp.ys)
    assert len(tp.tokens) > 0


def test_token_positions_have_valid_midpoints(parsed):
    _, tp, _ = parsed
    assert np.all(tp.xs >= 0)
    assert np.all(tp.ys >= 0)


def test_gt_textline_texts_nonempty(parsed):
    _, _, tl_texts = parsed
    assert len(tl_texts) > 0
    assert all(isinstance(t, str) for t in tl_texts)
    assert any(len(t) > 0 for t in tl_texts)


def test_textline_text_contains_content(parsed):
    _, _, tl_texts = parsed
    combined = " ".join(tl_texts)
    assert "Spiritualist" in combined
