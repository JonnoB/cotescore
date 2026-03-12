"""Tests for cot_score.visualisation."""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.figure

from cotescore.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks
from cotescore.visualisation import compute_cote_masks, visualize_cote_states


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

W, H = 100, 100  # default image size for tests


def _gt_map(boxes):
    """Rasterize GT boxes (auto-assigned ssu_id starting at 1)."""
    tagged = [{**b, "ssu_id": i + 1} for i, b in enumerate(boxes)]
    return boxes_to_gt_ssu_map(tagged, W, H, scale=1.0)


def _pred_masks(boxes):
    return boxes_to_pred_masks(boxes, W, H, scale=1.0)


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
        """No predictions → all masks are zero."""
        gt = [{"x": 0, "y": 0, "width": 50, "height": 50}]

        masks = compute_cote_masks(_gt_map(gt), [])

        for state, mask in masks.items():
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
        """compute_cote_masks always returns all five keys."""
        gt = [{"x": 0, "y": 0, "width": 50, "height": 50}]
        masks = compute_cote_masks(_gt_map(gt), _pred_masks(gt))

        assert set(masks.keys()) == {
            "coverage",
            "overlap",
            "trespass",
            "overlap_trespass",
            "excess",
        }

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
            for k in ("coverage", "overlap", "trespass", "overlap_trespass", "excess")
        }
        fig = visualize_cote_states(image, zero_masks)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)
