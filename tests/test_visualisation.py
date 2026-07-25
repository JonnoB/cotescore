"""Tests for cot_score.visualisation."""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.figure

from cotescore.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks
from cotescore.types import MaskInstance
from cotescore.visualisation import (
    MIXED_KEY,
    MIXED_LABEL,
    class_palette,
    compute_class_masks,
    compute_cote_masks,
    visualize_class_masks,
    visualize_cote_states,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

W, H = 100, 100  # default image size for tests


def _gt_map(boxes):
    """Rasterize GT boxes (auto-assigned ssu_id starting at 1)."""
    tagged = [{**b, "ssu_id": i + 1} for i, b in enumerate(boxes)]
    return boxes_to_gt_ssu_map(tagged, W, H, W, H)


def _pred_masks(boxes):
    return boxes_to_pred_masks(boxes, W, H, W, H)


def _sum_masks(masks):
    """Sum all state masks into a single pixel-count array."""
    return sum(masks.values())


# ---------------------------------------------------------------------------
# compute_cote_masks
# ---------------------------------------------------------------------------


class TestComputeCoteMasks:

    def test_perfect_coverage(self):
        """Single pred exactly matches single GT → all GT pixels are coverage."""
        gt = [{"x": 10, "y": 10, "width": 30, "height": 30}]
        pred = [{"x": 10, "y": 10, "width": 30, "height": 30}]

        masks = compute_cote_masks(_gt_map(gt), _pred_masks(pred))

        gt_pixels = 30 * 30
        assert np.sum(masks["coverage"]) == gt_pixels
        assert np.sum(masks["overlap"]) == 0
        assert np.sum(masks["trespass"]) == 0
        assert np.sum(masks["overlap_trespass"]) == 0
        assert np.sum(masks["excess"]) == 0

    def test_no_overlap_is_excess(self):
        """Pred entirely outside GT → all pred pixels are excess."""
        gt = [{"x": 0, "y": 0, "width": 20, "height": 20}]
        pred = [{"x": 50, "y": 50, "width": 20, "height": 20}]

        masks = compute_cote_masks(_gt_map(gt), _pred_masks(pred))

        assert np.sum(masks["coverage"]) == 0
        assert np.sum(masks["excess"]) == 20 * 20

    def test_two_preds_same_gt_creates_overlap(self):
        """Two preds covering the same GT region → overlap pixels."""
        gt = [{"x": 0, "y": 0, "width": 60, "height": 60}]
        pred = [
            {"x": 0, "y": 0, "width": 60, "height": 60},
            {"x": 0, "y": 0, "width": 60, "height": 60},
        ]

        masks = compute_cote_masks(_gt_map(gt), _pred_masks(pred))

        assert np.sum(masks["overlap"]) == 60 * 60
        assert np.sum(masks["coverage"]) == 0
        assert np.sum(masks["trespass"]) == 0

    def test_trespass(self):
        """Pred assigned to GT-A spills into GT-B → trespass pixels in GT-B."""
        # GT-A: left half, GT-B: right half
        gt = [
            {"x": 0, "y": 0, "width": 50, "height": 100},  # ssu_id=1
            {"x": 50, "y": 0, "width": 50, "height": 100},  # ssu_id=2
        ]
        # Pred mostly in GT-A but extends into GT-B
        pred = [{"x": 0, "y": 0, "width": 70, "height": 100}]

        masks = compute_cote_masks(_gt_map(gt), _pred_masks(pred))

        # The 20-pixel-wide column inside GT-B should be trespass
        assert np.sum(masks["trespass"]) == 20 * 100
        # The 50-pixel-wide column inside GT-A should be coverage
        assert np.sum(masks["coverage"]) == 50 * 100

    def test_excess_partial_overlap(self):
        """Pred partially overlaps GT → covered GT pixels + excess outside GT."""
        gt = [{"x": 0, "y": 0, "width": 50, "height": 50}]
        pred = [{"x": 25, "y": 0, "width": 50, "height": 50}]

        masks = compute_cote_masks(_gt_map(gt), _pred_masks(pred))

        assert np.sum(masks["coverage"]) == 25 * 50  # overlap with GT
        assert np.sum(masks["excess"]) == 25 * 50  # outside GT

    def test_empty_predictions(self):
        """No predictions → all GT pixels are 'missing', all other states zero."""
        gt = [{"x": 0, "y": 0, "width": 50, "height": 50}]

        masks = compute_cote_masks(_gt_map(gt), [])

        for state, mask in masks.items():
            if state == "missing":
                assert np.sum(mask) == 50 * 50
            else:
                assert np.sum(mask) == 0, f"Expected {state} to be zero with no predictions"

    def test_masks_are_mutually_exclusive(self):
        """No pixel should appear in more than one state mask."""
        gt = [
            {"x": 0, "y": 0, "width": 50, "height": 100},
            {"x": 50, "y": 0, "width": 50, "height": 100},
        ]
        pred = [
            {"x": 0, "y": 0, "width": 70, "height": 100},
            {"x": 30, "y": 0, "width": 70, "height": 100},
        ]

        masks = compute_cote_masks(_gt_map(gt), _pred_masks(pred))

        total = _sum_masks(masks)
        assert np.all(total <= 1), "Some pixels appear in multiple state masks"

    def test_output_keys(self):
        """compute_cote_masks always returns all six keys."""
        gt = [{"x": 0, "y": 0, "width": 50, "height": 50}]
        masks = compute_cote_masks(_gt_map(gt), _pred_masks(gt))

        assert set(masks.keys()) == {
            "coverage",
            "overlap",
            "trespass",
            "overlap_trespass",
            "missing",
            "excess",
        }

    def test_missing_marks_uncovered_gt(self):
        """GT region with no prediction anywhere → missing pixels, rest zero."""
        gt = [
            {"x": 0, "y": 0, "width": 50, "height": 100},  # ssu_id=1, uncovered
            {"x": 50, "y": 0, "width": 50, "height": 100},  # ssu_id=2, covered
        ]
        pred = [{"x": 50, "y": 0, "width": 50, "height": 100}]

        masks = compute_cote_masks(_gt_map(gt), _pred_masks(pred))

        assert np.sum(masks["missing"]) == 50 * 100
        assert np.sum(masks["coverage"]) == 50 * 100
        assert np.sum(masks["excess"]) == 0

    def test_output_shape_matches_gt_map(self):
        """Each mask has the same shape as the gt_ssu_map."""
        gt = [{"x": 0, "y": 0, "width": 50, "height": 50}]
        gt_map = _gt_map(gt)
        masks = compute_cote_masks(gt_map, _pred_masks(gt))

        for state, mask in masks.items():
            assert mask.shape == gt_map.shape, f"{state} mask has wrong shape"


# ---------------------------------------------------------------------------
# visualize_cote_states
# ---------------------------------------------------------------------------


class TestVisualizeCoteStates:

    def _simple_masks(self):
        gt = [{"x": 10, "y": 10, "width": 30, "height": 30}]
        pred = [{"x": 10, "y": 10, "width": 30, "height": 30}]
        return compute_cote_masks(_gt_map(gt), _pred_masks(pred))

    def test_returns_figure(self):
        image = np.ones((H, W), dtype=np.uint8) * 255
        fig = visualize_cote_states(image, self._simple_masks())
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_grayscale_image(self):
        image = np.zeros((H, W), dtype=np.uint8)
        fig = visualize_cote_states(image, self._simple_masks())
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_rgb_image(self):
        image = np.zeros((H, W, 3), dtype=np.uint8)
        fig = visualize_cote_states(image, self._simple_masks())
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_empty_masks(self):
        """Empty masks dict should not raise."""
        image = np.ones((H, W), dtype=np.uint8) * 200
        fig = visualize_cote_states(image, {})
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_all_zero_masks(self):
        """All-zero masks should produce a figure without error."""
        image = np.ones((H, W), dtype=np.uint8) * 128
        zero_masks = {
            k: np.zeros((H, W), dtype=np.int32)
            for k in (
                "coverage",
                "overlap",
                "trespass",
                "overlap_trespass",
                "missing",
                "excess",
            )
        }
        fig = visualize_cote_states(image, zero_masks)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def _masks_with_missing(self):
        """GT with no predictions at all → everything falls into 'missing'."""
        gt = [{"x": 10, "y": 10, "width": 30, "height": 30}]
        return compute_cote_masks(_gt_map(gt), [])

    def test_missing_drawn_by_default(self):
        image = np.ones((H, W), dtype=np.uint8) * 255
        masks = self._masks_with_missing()
        _, ax = matplotlib.pyplot.subplots()
        patches = visualize_cote_states(image, masks, ax=ax)
        matplotlib.pyplot.close(ax.figure)

        assert any(p.get_label() == "Missing (uncovered GT)" for p in patches)

    def test_missing_can_be_suppressed(self):
        image = np.ones((H, W), dtype=np.uint8) * 255
        masks = self._masks_with_missing()
        _, ax = matplotlib.pyplot.subplots()
        patches = visualize_cote_states(image, masks, ax=ax, show_missing=False)
        matplotlib.pyplot.close(ax.figure)

        assert not any(p.get_label() == "Missing (uncovered GT)" for p in patches)


# ---------------------------------------------------------------------------
# compute_class_masks
# ---------------------------------------------------------------------------


def _inst(box, label):
    """A MaskInstance covering `box` = (x, y, width, height), tagged `label`."""
    m = np.zeros((H, W), dtype=bool)
    x, y, w, h = box
    m[y : y + h, x : x + w] = True
    return MaskInstance(mask=m, label=label)


class TestComputeClassMasks:

    def test_single_class(self):
        """One prediction → its class carries every pixel, nothing is mixed."""
        masks = compute_class_masks([_inst((10, 10, 30, 30), "text")])

        assert np.sum(masks["text"]) == 30 * 30
        assert np.sum(masks[MIXED_KEY]) == 0

    def test_same_class_overlap_is_unioned_not_mixed(self):
        """Two overlapping preds of the SAME class union; overlap is not a conflict."""
        preds = [_inst((0, 0, 40, 40), "text"), _inst((20, 0, 40, 40), "text")]

        masks = compute_class_masks(preds)

        assert np.sum(masks["text"]) == 60 * 40  # union, not 2x the overlap
        assert np.sum(masks[MIXED_KEY]) == 0

    def test_different_class_overlap_becomes_mixed(self):
        """Overlap between DIFFERENT classes moves to MIXED_KEY, out of both classes."""
        preds = [_inst((0, 0, 40, 40), "text"), _inst((20, 0, 40, 40), "headline")]
        overlap = 20 * 40

        masks = compute_class_masks(preds)

        assert np.sum(masks[MIXED_KEY]) == overlap
        assert np.sum(masks["text"]) == 40 * 40 - overlap
        assert np.sum(masks["headline"]) == 40 * 40 - overlap

    def test_masks_are_mutually_exclusive(self):
        """No pixel appears in more than one mask — the invariant the renderer needs."""
        preds = [
            _inst((0, 0, 60, 100), "text"),
            _inst((40, 0, 60, 100), "headline"),
            _inst((30, 0, 40, 100), "table"),
        ]

        masks = compute_class_masks(preds)

        assert np.all(sum(masks.values()) <= 1)

    def test_classes_argument_fixes_the_key_set(self):
        """Requested classes all get a key, even with no predictions."""
        masks = compute_class_masks(
            [_inst((0, 0, 10, 10), "text")], classes=["text", "table", "image"]
        )

        assert set(masks) == {"text", "table", "image", MIXED_KEY}
        assert np.sum(masks["table"]) == 0
        assert np.sum(masks["image"]) == 0

    def test_predictions_outside_requested_classes_are_dropped(self):
        """A class not in `classes` contributes nothing — not even to mixed."""
        preds = [_inst((0, 0, 40, 40), "text"), _inst((0, 0, 40, 40), "footnote")]

        masks = compute_class_masks(preds, classes=["text"])

        assert set(masks) == {"text", MIXED_KEY}
        assert np.sum(masks["text"]) == 40 * 40
        assert np.sum(masks[MIXED_KEY]) == 0

    def test_empty_preds_with_shape(self):
        masks = compute_class_masks([], classes=["text"], shape=(H, W))

        assert masks["text"].shape == (H, W)
        assert np.sum(masks["text"]) == 0

    def test_empty_preds_without_shape_raises(self):
        with pytest.raises(ValueError, match="shape is required"):
            compute_class_masks([], classes=["text"])

    def test_unlabelled_prediction_raises(self):
        with pytest.raises(ValueError, match="label=None"):
            compute_class_masks([MaskInstance(mask=np.zeros((H, W), dtype=bool))])

    def test_reserved_mixed_key_as_class_raises(self):
        with pytest.raises(ValueError, match="reserved"):
            compute_class_masks([_inst((0, 0, 10, 10), "text")], classes=[MIXED_KEY])

    def test_output_shape(self):
        masks = compute_class_masks([_inst((0, 0, 10, 10), "text")])

        for key, mask in masks.items():
            assert mask.shape == (H, W), f"{key} mask has wrong shape"


# ---------------------------------------------------------------------------
# class_palette
# ---------------------------------------------------------------------------


class TestClassPalette:

    def test_colour_follows_position_not_presence(self):
        """The same class list yields the same colours regardless of what was predicted."""
        classes = ["text", "header", "headline", "table"]

        assert class_palette(classes)["table"] == class_palette(classes)["table"]

    def test_distinct_classes_get_distinct_colours(self):
        palette = class_palette(["a", "b", "c", "d", "e", "f", "g", "h"])
        rgb = [c[:3] for k, c in palette.items() if k != MIXED_KEY]

        assert len(set(rgb)) == 8

    def test_mixed_key_always_present(self):
        assert MIXED_KEY in class_palette(["text"])

    def test_duplicate_classes_raise(self):
        with pytest.raises(ValueError, match="duplicates"):
            class_palette(["text", "text"])

    def test_palette_cycles_past_eight(self):
        """A 9th class reuses the first colour rather than failing."""
        palette = class_palette([str(i) for i in range(9)])

        assert palette["8"][:3] == palette["0"][:3]

    def test_mixed_colour_is_not_a_class_colour(self):
        palette = class_palette([str(i) for i in range(8)])
        mixed_rgb = palette[MIXED_KEY][:3]

        assert all(palette[str(i)][:3] != mixed_rgb for i in range(8))


# ---------------------------------------------------------------------------
# visualize_class_masks
# ---------------------------------------------------------------------------


class TestVisualizeClassMasks:

    def test_returns_figure(self):
        image = np.ones((H, W), dtype=np.uint8) * 255
        masks = compute_class_masks([_inst((10, 10, 30, 30), "text")])

        fig = visualize_class_masks(image, masks)

        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_legend_names_the_classes(self):
        image = np.ones((H, W), dtype=np.uint8) * 255
        preds = [_inst((0, 0, 40, 40), "text"), _inst((60, 0, 30, 30), "headline")]
        masks = compute_class_masks(preds)

        _, ax = matplotlib.pyplot.subplots()
        patches = visualize_class_masks(image, masks, ax=ax)
        matplotlib.pyplot.close(ax.figure)

        assert {p.get_label() for p in patches} == {"text", "headline"}

    def test_mixed_appears_in_legend_only_when_present(self):
        image = np.ones((H, W), dtype=np.uint8) * 255
        clean = compute_class_masks([_inst((0, 0, 40, 40), "text")])
        conflicted = compute_class_masks(
            [_inst((0, 0, 40, 40), "text"), _inst((20, 0, 40, 40), "headline")]
        )

        labels = []
        for masks in (clean, conflicted):
            _, ax = matplotlib.pyplot.subplots()
            labels.append({p.get_label() for p in visualize_class_masks(image, masks, ax=ax)})
            matplotlib.pyplot.close(ax.figure)

        assert MIXED_LABEL not in labels[0]
        assert MIXED_LABEL in labels[1]

    def test_empty_masks(self):
        image = np.ones((H, W), dtype=np.uint8) * 200
        fig = visualize_class_masks(image, {})
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_explicit_palette_is_used(self):
        """Colours passed in override the default, so they can stay stable across images."""
        image = np.ones((H, W), dtype=np.uint8) * 255
        masks = compute_class_masks([_inst((0, 0, 40, 40), "text")])
        palette = {"text": (0.0, 0.0, 1.0, 0.5), MIXED_KEY: (0.5, 0.5, 0.5, 0.5)}

        _, ax = matplotlib.pyplot.subplots()
        patches = visualize_class_masks(image, masks, colors=palette, ax=ax)
        matplotlib.pyplot.close(ax.figure)

        assert patches[0].get_facecolor()[:3] == (0.0, 0.0, 1.0)
