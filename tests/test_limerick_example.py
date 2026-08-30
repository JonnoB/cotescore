"""Tests for the bundled limerick case study.

The assets are shipped inside the package, so these also guard against the
asset files being lost to a .gitignore rule — the loader raising
FileNotFoundError is the failure this suite exists to catch.
"""

import numpy as np
import pytest

from cotescore import (
    chars_to_region_chars,
    cote_score,
    extract_line_boxes,
    extract_ssu_boxes,
    extract_word_boxes,
    load_limerick_example,
    reconstruct_text,
)
from cotescore.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks


EXPECTED_COLUMNS = [
    "char", "x", "y", "width", "height",
    "word_id", "line_id", "ssu_id", "ssu_class", "semantic_unit",
]


@pytest.fixture(scope="module")
def example():
    return load_limerick_example()


class TestAssets:
    def test_loads(self, example):
        chars, image, predictions = example
        assert len(chars) > 0
        assert image.ndim in (2, 3)
        assert len(predictions) > 0

    def test_schema(self, example):
        chars, _, _ = example
        assert list(chars.columns) == EXPECTED_COLUMNS

    def test_no_space_characters(self, example):
        chars, _, _ = example
        assert not chars["char"].str.isspace().any()

    def test_ssu_ids_are_one_based(self, example):
        """0 is reserved for background; an SSU with id 0 would vanish."""
        chars, _, _ = example
        assert chars["ssu_id"].min() == 1

    def test_group_ids_are_contiguous(self, example):
        chars, _, _ = example
        for col, start in (("ssu_id", 1), ("line_id", 0), ("word_id", 0)):
            values = sorted(chars[col].unique().tolist())
            assert values == list(range(start, start + len(values))), col


class TestSemanticUnits:
    def test_a_semantic_unit_spans_several_ssus(self, example):
        """The point of the example: one poem continues across a column break."""
        chars, _, _ = example
        per_sem = chars.groupby("semantic_unit")["ssu_id"].nunique()
        assert (per_sem > 2).any(), "no semantic unit is split across columns"

    def test_every_ssu_has_exactly_one_semantic_unit(self, example):
        chars, _, _ = example
        assert (chars.groupby("ssu_id")["semantic_unit"].nunique() == 1).all()

    def test_every_ssu_has_exactly_one_class(self, example):
        chars, _, _ = example
        assert (chars.groupby("ssu_id")["ssu_class"].nunique() == 1).all()


class TestDerivedBoxes:
    def test_granularity_ordering(self, example):
        chars, _, _ = example
        assert (
            len(extract_ssu_boxes(chars))
            < len(extract_line_boxes(chars))
            < len(extract_word_boxes(chars))
        )

    def test_ssu_boxes_carry_what_the_adapters_need(self, example):
        chars, _, _ = example
        for box in extract_ssu_boxes(chars):
            assert {"x", "y", "width", "height", "ssu_id"} <= set(box)
            assert box["ssu_id"] >= 1
            assert box["width"] > 0 and box["height"] > 0

    def test_finer_boxes_carry_the_ssu_they_belong_to(self, example):
        """A line is part of its SSU whether or not it is drawn as its own box.

        Losing that identity makes an SSU-sized prediction covering several
        lines look like it misattributes all but one of them, manufacturing
        trespass that does not exist.
        """
        chars, _, _ = example
        expected = chars.groupby("line_id")["ssu_id"].first().to_dict()
        assert {b["line_id"]: b["ssu_id"] for b in extract_line_boxes(chars)} == expected

        expected = chars.groupby("word_id")["ssu_id"].first().to_dict()
        assert {b["word_id"]: b["ssu_id"] for b in extract_word_boxes(chars)} == expected

    def test_several_lines_share_one_ssu(self, example):
        """Otherwise the test above would hold trivially."""
        chars, _, _ = example
        lines_per_ssu = chars.groupby("ssu_id")["line_id"].nunique()
        assert (lines_per_ssu > 1).any()

    def test_ssu_predictions_do_not_trespass_on_line_ground_truth(self, example):
        """Lines and the SSU containing them are one unit: nothing is misplaced."""
        chars, image, _ = example
        height, width = image.shape[:2]
        gt_map = boxes_to_gt_ssu_map(
            extract_line_boxes(chars), width, height, width, height
        )
        masks = boxes_to_pred_masks(
            extract_ssu_boxes(chars), width, height, width, height
        )
        _, _, overlap, trespass, _ = cote_score(gt_map, masks)
        assert trespass == pytest.approx(0.0)
        assert overlap == pytest.approx(0.0)

    def test_boxes_lie_within_the_image(self, example):
        chars, image, _ = example
        height, width = image.shape[:2]
        for box in extract_ssu_boxes(chars):
            assert box["x"] >= 0 and box["y"] >= 0
            # A box may round a pixel past the edge; the adapters clamp.
            assert box["x"] + box["width"] <= width + 1
            assert box["y"] + box["height"] <= height + 1


class TestReconstruction:
    def test_text_round_trips(self, example):
        chars, _, _ = example
        text = reconstruct_text(chars)
        assert text.count("\n") + 1 == chars["line_id"].nunique()
        assert "There was a young man who read text," in text

    def test_region_chars_matches_the_table(self, example):
        chars, _, _ = example
        region_chars = chars_to_region_chars(chars)
        assert len(region_chars.tokens) == len(chars)
        assert np.array_equal(region_chars.region_ids, chars["ssu_id"].to_numpy())


class TestScoring:
    def test_every_ssu_survives_rasterisation(self, example):
        """A 0-based ssu_id would be painted as background and silently lost."""
        chars, image, _ = example
        height, width = image.shape[:2]
        gt_map = boxes_to_gt_ssu_map(
            extract_ssu_boxes(chars), width, height, width, height
        )
        painted = set(np.unique(gt_map).tolist()) - {0}
        assert painted == set(chars["ssu_id"].unique().tolist())

    def test_bundled_predictions_score_non_degenerately(self, example):
        chars, image, predictions = example
        height, width = image.shape[:2]
        gt_map = boxes_to_gt_ssu_map(
            extract_ssu_boxes(chars), width, height, width, height
        )
        masks = boxes_to_pred_masks(predictions, width, height, width, height)
        cote, coverage, overlap, trespass, excess = cote_score(gt_map, masks)

        assert 0.0 < coverage < 1.0
        for component in (overlap, trespass, excess):
            assert component > 0.0, "predictions no longer exercise every state"
        assert cote == pytest.approx(coverage - overlap - trespass)
