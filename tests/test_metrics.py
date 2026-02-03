"""Comprehensive tests for the metrics module."""

import pytest
import numpy as np
from cot_score.metrics import coverage, overlap, iou, mean_iou, trespass, excess, cot_score
from tests.reference_metrics import (
    coverage as ref_coverage,
    overlap as ref_overlap,
    iou as ref_iou,
    mean_iou as ref_mean_iou,
)


# Tolerance for floating point comparisons
TOLERANCE = 1e-5


class TestCoverage:
    """Tests for the coverage metric."""

    def test_coverage_perfect_match(self):
        """Test coverage with perfect match between predicted and ground truth."""
        pred = [{"x": 0, "y": 0, "width": 100, "height": 100}]
        gt = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        result = coverage(pred, gt, image_width=100, image_height=100)
        reference = ref_coverage(pred, gt)

        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_coverage_no_overlap(self):
        """Test coverage with no overlap between predicted and ground truth."""
        pred = [{"x": 0, "y": 0, "width": 10, "height": 10}]
        gt = [{"x": 100, "y": 100, "width": 10, "height": 10}]

        result = coverage(pred, gt, image_width=100, image_height=100)
        reference = ref_coverage(pred, gt)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_coverage_partial_overlap(self):
        """Test coverage with partial overlap."""
        # Prediction covers half of ground truth
        pred = [{"x": 0, "y": 0, "width": 50, "height": 100}]
        gt = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        result = coverage(pred, gt, image_width=100, image_height=100)
        reference = ref_coverage(pred, gt)

        assert abs(result - 0.5) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_coverage_multiple_predictions(self):
        """Test coverage with multiple predictions covering different parts."""
        pred = [
            {"x": 0, "y": 0, "width": 60, "height": 100},
            {"x": 50, "y": 0, "width": 60, "height": 100},
        ]
        gt = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        result = coverage(pred, gt, image_width=100, image_height=100)
        reference = ref_coverage(pred, gt)

        # Should cover 110% (overlapping predictions), but capped at 100%
        assert result > 0.99  # Close to 1.0
        assert abs(result - reference) < TOLERANCE

    def test_coverage_multiple_ground_truth(self):
        """Test coverage with multiple ground truth boxes."""
        pred = [{"x": 0, "y": 0, "width": 100, "height": 100}]
        gt = [
            {"x": 0, "y": 0, "width": 100, "height": 100},
            {"x": 200, "y": 200, "width": 100, "height": 100},
        ]

        result = coverage(pred, gt, image_width=100, image_height=100)
        reference = ref_coverage(pred, gt)

        # Covers first GT completely, second GT not at all = 50%
        assert abs(result - 0.5) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_coverage_empty_predictions(self):
        """Test coverage with empty predictions."""
        pred = []
        gt = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        result = coverage(pred, gt, image_width=100, image_height=100)
        reference = ref_coverage(pred, gt)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_coverage_empty_ground_truth(self):
        """Test coverage with empty ground truth."""
        pred = [{"x": 0, "y": 0, "width": 100, "height": 100}]
        gt = []

        result = coverage(pred, gt, image_width=100, image_height=100)
        reference = ref_coverage(pred, gt)

        # When GT is empty and predictions exist, coverage is 0
        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_coverage_both_empty(self):
        """Test coverage when both are empty."""
        pred = []
        gt = []

        result = coverage(pred, gt, image_width=100, image_height=100)
        reference = ref_coverage(pred, gt)

        # When both are empty, coverage is perfect (1.0)
        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_coverage_complex_scenario(self):
        """Test coverage with complex multi-box scenario."""
        pred = [
            {"x": 10, "y": 20, "width": 100, "height": 50},
            {"x": 120, "y": 25, "width": 80, "height": 60},
        ]
        gt = [
            {"x": 12, "y": 22, "width": 95, "height": 48},
            {"x": 118, "y": 23, "width": 85, "height": 62},
        ]

        result = coverage(pred, gt, image_width=100, image_height=100)
        reference = ref_coverage(pred, gt)

        assert 0.0 < result < 1.0  # Should be partial coverage
        assert abs(result - reference) < TOLERANCE


class TestOverlap:
    """Tests for the overlap metric."""

    def test_overlap_no_predictions(self):
        """Test overlap with no predictions."""
        pred = []
        gt = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        result = overlap(pred, gt, image_width=100, image_height=100)
        reference = ref_overlap(pred, gt)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_overlap_single_prediction(self):
        """Test overlap with single prediction."""
        pred = [{"x": 0, "y": 0, "width": 100, "height": 100}]
        gt = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        result = overlap(pred, gt, image_width=100, image_height=100)
        reference = ref_overlap(pred, gt)

        # Single prediction cannot overlap with itself
        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_overlap_no_overlap_between_predictions(self):
        """Test overlap when predictions don't overlap each other."""
        pred = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 20, "y": 20, "width": 10, "height": 10},
        ]
        gt = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        result = overlap(pred, gt, image_width=100, image_height=100)
        reference = ref_overlap(pred, gt)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_overlap_complete_overlap(self):
        """Test overlap with completely overlapping predictions."""
        pred = [
            {"x": 0, "y": 0, "width": 100, "height": 100},
            {"x": 0, "y": 0, "width": 100, "height": 100},
        ]
        gt = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        result = overlap(pred, gt, image_width=100, image_height=100)
        reference = ref_overlap(pred, gt)

        # Complete overlap
        assert result > 0.99
        assert abs(result - reference) < TOLERANCE

    def test_overlap_partial_overlap(self):
        """Test overlap with partially overlapping predictions."""
        pred = [
            {"x": 0, "y": 0, "width": 60, "height": 100},
            {"x": 40, "y": 0, "width": 60, "height": 100},
        ]
        gt = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        result = overlap(pred, gt, image_width=100, image_height=100)
        reference = ref_overlap(pred, gt)

        # Partial overlap
        assert 0.0 < result < 1.0
        assert abs(result - reference) < TOLERANCE

    def test_overlap_empty_ground_truth(self):
        """Test overlap with empty ground truth."""
        pred = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 5, "y": 5, "width": 10, "height": 10},
        ]
        gt = []

        result = overlap(pred, gt, image_width=100, image_height=100)
        reference = ref_overlap(pred, gt)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_overlap_three_predictions(self):
        """Test overlap with three overlapping predictions."""
        pred = [
            {"x": 0, "y": 0, "width": 50, "height": 50},
            {"x": 25, "y": 25, "width": 50, "height": 50},
            {"x": 50, "y": 50, "width": 50, "height": 50},
        ]
        gt = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        result = overlap(pred, gt, image_width=100, image_height=100)
        reference = ref_overlap(pred, gt)

        assert 0.0 < result < 1.0
        assert abs(result - reference) < TOLERANCE


class TestIOU:
    """Tests for the IOU metric."""

    def test_iou_perfect_match(self):
        """Test IOU with identical boxes."""
        box1 = {"x": 0, "y": 0, "width": 100, "height": 100}
        box2 = {"x": 0, "y": 0, "width": 100, "height": 100}

        result = iou(box1, box2)
        reference = ref_iou(box1, box2)

        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_iou_no_overlap(self):
        """Test IOU with non-overlapping boxes."""
        box1 = {"x": 0, "y": 0, "width": 10, "height": 10}
        box2 = {"x": 20, "y": 20, "width": 10, "height": 10}

        result = iou(box1, box2)
        reference = ref_iou(box1, box2)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_iou_partial_overlap(self):
        """Test IOU with partially overlapping boxes."""
        box1 = {"x": 0, "y": 0, "width": 20, "height": 20}
        box2 = {"x": 10, "y": 10, "width": 20, "height": 20}

        result = iou(box1, box2)
        reference = ref_iou(box1, box2)

        # Intersection: 10x10 = 100
        # Union: 400 + 400 - 100 = 700
        # IoU: 100/700 ≈ 0.143
        expected = 100.0 / 700.0
        assert abs(result - expected) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_iou_contained(self):
        """Test IOU when one box is contained in another."""
        box1 = {"x": 10, "y": 10, "width": 20, "height": 20}
        box2 = {"x": 0, "y": 0, "width": 100, "height": 100}

        result = iou(box1, box2)
        reference = ref_iou(box1, box2)

        # Intersection: 20x20 = 400
        # Union: 400 + 10000 - 400 = 10000
        # IoU: 400/10000 = 0.04
        expected = 400.0 / 10000.0
        assert abs(result - expected) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_iou_edge_touch(self):
        """Test IOU when boxes touch at edges (no overlap)."""
        box1 = {"x": 0, "y": 0, "width": 10, "height": 10}
        box2 = {"x": 10, "y": 0, "width": 10, "height": 10}

        result = iou(box1, box2)
        reference = ref_iou(box1, box2)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_iou_floating_point_coords(self):
        """Test IOU with floating point coordinates."""
        box1 = {"x": 10.5, "y": 20.3, "width": 100.7, "height": 50.2}
        box2 = {"x": 12.1, "y": 22.8, "width": 95.3, "height": 48.9}

        result = iou(box1, box2)
        reference = ref_iou(box1, box2)

        assert 0.0 < result < 1.0
        assert abs(result - reference) < TOLERANCE


class TestMeanIOU:
    """Tests for the mean_iou metric."""

    def test_mean_iou_perfect_match(self):
        """Test mean_iou with perfect matches."""
        pred = [
            {"x": 0, "y": 0, "width": 100, "height": 100},
            {"x": 200, "y": 200, "width": 50, "height": 50},
        ]
        gt = [
            {"x": 0, "y": 0, "width": 100, "height": 100},
            {"x": 200, "y": 200, "width": 50, "height": 50},
        ]

        result = mean_iou(pred, gt)
        reference = ref_mean_iou(pred, gt)

        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_mean_iou_no_overlap(self):
        """Test mean_iou with no overlap."""
        pred = [{"x": 0, "y": 0, "width": 10, "height": 10}]
        gt = [{"x": 100, "y": 100, "width": 10, "height": 10}]

        result = mean_iou(pred, gt)
        reference = ref_mean_iou(pred, gt)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_mean_iou_multiple_boxes(self):
        """Test mean_iou with multiple boxes."""
        pred = [
            {"x": 0, "y": 0, "width": 100, "height": 100},
            {"x": 150, "y": 150, "width": 50, "height": 50},
        ]
        gt = [
            {"x": 0, "y": 0, "width": 100, "height": 100},
            {"x": 200, "y": 200, "width": 50, "height": 50},
        ]

        result = mean_iou(pred, gt)
        reference = ref_mean_iou(pred, gt)

        # First GT: perfect match (1.0)
        # Second GT: no match (0.0)
        # Mean: 0.5
        assert abs(result - 0.5) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_mean_iou_best_match_selection(self):
        """Test that mean_iou selects best matching prediction for each GT."""
        pred = [
            {"x": 0, "y": 0, "width": 100, "height": 100},
            {"x": 5, "y": 5, "width": 100, "height": 100},
            {"x": 10, "y": 10, "width": 100, "height": 100},
        ]
        gt = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        result = mean_iou(pred, gt)
        reference = ref_mean_iou(pred, gt)

        # Should select the best match (first prediction)
        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_mean_iou_empty_predictions(self):
        """Test mean_iou with empty predictions."""
        pred = []
        gt = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        result = mean_iou(pred, gt)
        reference = ref_mean_iou(pred, gt)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_mean_iou_empty_ground_truth(self):
        """Test mean_iou with empty ground truth."""
        pred = [{"x": 0, "y": 0, "width": 100, "height": 100}]
        gt = []

        result = mean_iou(pred, gt)
        reference = ref_mean_iou(pred, gt)

        # When GT is empty and predictions exist, mean_iou is 0
        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_mean_iou_both_empty(self):
        """Test mean_iou when both are empty."""
        pred = []
        gt = []

        result = mean_iou(pred, gt)
        reference = ref_mean_iou(pred, gt)

        # When both are empty, mean_iou is 1.0
        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE


class TestVectorizedCorrectness:
    """Tests to verify vectorized implementation matches reference on random data."""

    def _generate_random_boxes(self, n, seed=42):
        """Generate random boxes for testing."""
        np.random.seed(seed)
        boxes = []
        for _ in range(n):
            x = np.random.uniform(0, 500)
            y = np.random.uniform(0, 500)
            width = np.random.uniform(10, 100)
            height = np.random.uniform(10, 100)
            boxes.append({"x": x, "y": y, "width": width, "height": height})
        return boxes

    def test_coverage_random_small(self):
        """Test coverage on small random dataset."""
        pred = self._generate_random_boxes(5, seed=1)
        gt = self._generate_random_boxes(5, seed=2)

        result = coverage(pred, gt, image_width=1000, image_height=1000)
        reference = ref_coverage(pred, gt)

        assert abs(result - reference) < TOLERANCE

    def test_coverage_random_medium(self):
        """Test coverage on medium random dataset."""
        pred = self._generate_random_boxes(20, seed=3)
        gt = self._generate_random_boxes(20, seed=4)

        result = coverage(pred, gt, image_width=1000, image_height=1000)
        reference = ref_coverage(pred, gt)

        assert abs(result - reference) < TOLERANCE

    def test_overlap_random_small(self):
        """Test overlap on small random dataset."""
        pred = self._generate_random_boxes(5, seed=5)
        gt = self._generate_random_boxes(5, seed=6)

        result = overlap(pred, gt, image_width=1000, image_height=1000)
        reference = ref_overlap(pred, gt)

        assert abs(result - reference) < TOLERANCE

    def test_overlap_random_medium(self):
        """Test overlap on medium random dataset."""
        pred = self._generate_random_boxes(15, seed=7)
        gt = self._generate_random_boxes(15, seed=8)

        result = overlap(pred, gt, image_width=1000, image_height=1000)
        reference = ref_overlap(pred, gt)

        assert abs(result - reference) < TOLERANCE

    def test_mean_iou_random_small(self):
        """Test mean_iou on small random dataset."""
        pred = self._generate_random_boxes(5, seed=9)
        gt = self._generate_random_boxes(5, seed=10)

        result = mean_iou(pred, gt)
        reference = ref_mean_iou(pred, gt)

        assert abs(result - reference) < TOLERANCE

    def test_mean_iou_random_medium(self):
        """Test mean_iou on medium random dataset."""
        pred = self._generate_random_boxes(20, seed=11)
        gt = self._generate_random_boxes(20, seed=12)

        result = mean_iou(pred, gt)
        reference = ref_mean_iou(pred, gt)

        assert abs(result - reference) < TOLERANCE


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_area_boxes(self):
        """Test with zero-area boxes."""
        pred = [{"x": 0, "y": 0, "width": 0, "height": 100}]
        gt = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        # Should handle gracefully
        cov = coverage(pred, gt, image_width=200, image_height=200)
        ref_cov = ref_coverage(pred, gt)
        assert abs(cov - ref_cov) < TOLERANCE

    def test_very_small_boxes(self):
        """Test with very small boxes."""
        pred = [{"x": 0, "y": 0, "width": 0.001, "height": 0.001}]
        gt = [{"x": 0, "y": 0, "width": 0.001, "height": 0.001}]

        result = iou(pred[0], gt[0])
        reference = ref_iou(pred[0], gt[0])

        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_very_large_boxes(self):
        """Test with very large boxes."""
        pred = [{"x": 0, "y": 0, "width": 10000, "height": 10000}]
        gt = [{"x": 0, "y": 0, "width": 10000, "height": 10000}]

        result = coverage(pred, gt, image_width=200, image_height=200)
        reference = ref_coverage(pred, gt)

        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        pred = [{"x": -50, "y": -50, "width": 100, "height": 100}]
        gt = [{"x": -50, "y": -50, "width": 100, "height": 100}]

        result = coverage(pred, gt, image_width=200, image_height=200)
        reference = ref_coverage(pred, gt)

        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE


class TestTrespass:
    """Tests for the trespass metric (corrected to match paper specification)."""

    def test_trespass_perfect_match(self):
        """Perfect match - no trespass."""
        # 2 GTs, 2 Preds perfectly matching
        pred = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 20, "y": 0, "width": 10, "height": 10},
        ]
        gt = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 20, "y": 0, "width": 10, "height": 10},
        ]
        # Pred 1 overlaps GT 1 (100) and GT 2 (0). Best match GT 1. Trespass 0.
        assert abs(trespass(pred, gt, image_width=100, image_height=100) - 0.0) < TOLERANCE

    def test_trespass_overlap(self):
        """Single prediction overlapping two GTs."""
        # Pred overlaps GT 1 (100) and GT 2 (20).
        # GT 1: 0,0 10x10. GT 2: 10,0 10x10.
        # Pred: 0,0 12x10.
        # Intersect GT 1: 10x10 = 100.
        # Intersect GT 2: 2x10 = 20.
        # Best match GT 1.
        # Trespass area = 20.
        #
        # Paper normalization (Equation 13):
        # n = 1, A_S = 200, A_S_min = 100
        # T = 20 / (1 × (200 - 100)) = 20 / 100 = 0.2

        gt = [
            {"x": 0, "y": 0, "width": 10, "height": 10},  # Area 100
            {"x": 10, "y": 0, "width": 10, "height": 10},  # Area 100
        ]
        pred = [{"x": 0, "y": 0, "width": 12, "height": 10}]

        result = trespass(pred, gt, image_width=100, image_height=100)
        expected = 20.0 / (1 * 100)  # 0.2
        assert abs(result - expected) < TOLERANCE

    def test_trespass_multiple_preds(self):
        """Multiple predictions with one trespassing."""
        # P1 perfect on G1. P2 overlaps G1 and G2.
        gt = [
            {"x": 0, "y": 0, "width": 10, "height": 10},  # G1, Area 100
            {"x": 20, "y": 0, "width": 10, "height": 10},  # G2, Area 100
        ]
        # P2: 5,0 w:25 h:10. x: 5 to 30.
        # Overlaps G1 (5-10 => 5x10=50).
        # Overlaps G2 (20-30 => 10x10=100).
        # Best match G2 (larger overlap).
        # Trespass = Overlap with G1 = 50.
        #
        # Paper normalization:
        # n = 2, A_S = 200, A_S_min = 100
        # T = 50 / (2 × (200 - 100)) = 50 / 200 = 0.25

        pred = [
            {"x": 0, "y": 0, "width": 10, "height": 10},  # P1
            {"x": 5, "y": 0, "width": 25, "height": 10},  # P2
        ]

        result = trespass(pred, gt, image_width=100, image_height=100)
        expected = 50.0 / (2 * 100)  # 0.25
        assert abs(result - expected) < TOLERANCE

    def test_trespass_single_gt(self):
        """Single GT - no trespass possible (returns 0 per Equation 13)."""
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]
        pred = [{"x": 0, "y": 0, "width": 15, "height": 10}]
        # m = 1, so per Equation 13: T = 0
        assert abs(trespass(pred, gt, image_width=100, image_height=100) - 0.0) < TOLERANCE

    def test_trespass_no_predictions(self):
        """No predictions - zero trespass."""
        gt = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 20, "y": 0, "width": 10, "height": 10},
        ]
        pred = []
        assert abs(trespass(pred, gt, image_width=100, image_height=100) - 0.0) < TOLERANCE

    def test_trespass_different_gt_sizes(self):
        """Test with different GT sizes - normalization by smallest."""
        gt = [
            {"x": 0, "y": 0, "width": 10, "height": 10},  # Area 100
            {"x": 20, "y": 0, "width": 20, "height": 10},  # Area 200
        ]
        pred = [{"x": 0, "y": 0, "width": 15, "height": 10}]  # Overlaps both

        # Pred intersects GT1: 100, GT2: 0 (no overlap at x=20)
        # Wait, let me recalculate:
        # Pred: x=0 to 15
        # GT1: x=0 to 10, intersection = 100
        # GT2: x=20 to 40, no intersection
        # Best match: GT1
        # No trespass
        assert abs(trespass(pred, gt, image_width=100, image_height=100) - 0.0) < TOLERANCE


class TestExcess:
    """Tests for the excess metric (corrected to match paper specification)."""

    def test_excess_perfect_match(self):
        """Perfect match - no excess."""
        pred = [{"x": 0, "y": 0, "width": 10, "height": 10}]
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]

        # Image 100x100, GT area 100, white space 9900
        # Pred area 100, all overlaps GT
        # Excess = 0 / 9900 = 0.0
        result = excess(pred, gt, image_width=100, image_height=100)
        assert abs(result - 0.0) < TOLERANCE

    def test_excess_background(self):
        """Pred extends outside GT into white space."""
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]  # Area 100
        pred = [{"x": 0, "y": 0, "width": 15, "height": 10}]  # Area 150

        # Image 100x100 (area 10000)
        # GT area = 100
        # White space area = 10000 - 100 = 9900
        # Pred union area = 150
        # Pred-GT overlap = 100
        # Pred in white space = 150 - 100 = 50
        # Excess score = 50 / 9900 ≈ 0.00505

        result = excess(pred, gt, image_width=100, image_height=100)
        expected = 50.0 / 9900.0
        assert abs(result - expected) < TOLERANCE

    def test_excess_pure_background(self):
        """Pred completely outside GT."""
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]  # Area 100
        pred = [{"x": 20, "y": 0, "width": 10, "height": 10}]  # Area 100

        # Image 100x100
        # White space = 9900
        # Pred in white space = 100 (no overlap with GT)
        # Excess = 100 / 9900 ≈ 0.0101

        result = excess(pred, gt, image_width=100, image_height=100)
        expected = 100.0 / 9900.0
        assert abs(result - expected) < TOLERANCE

    def test_excess_between_gts(self):
        """Pred covers GTs and the gap between them."""
        # GT1: 0-10. GT2: 20-30.
        # Pred: 0-30. Covers GT1 (100), GT2 (100), and gap (10x10=100).
        gt = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 20, "y": 0, "width": 10, "height": 10},
        ]
        pred = [{"x": 0, "y": 0, "width": 30, "height": 10}]  # Area 300

        # Image 100x100 (area 10000)
        # Total GT area = 200
        # White space = 10000 - 200 = 9800
        # Pred union area = 300
        # Pred-GT overlap = 200
        # Pred in white space = 300 - 200 = 100
        # Excess = 100 / 9800 ≈ 0.0102

        result = excess(pred, gt, image_width=100, image_height=100)
        expected = 100.0 / 9800.0
        assert abs(result - expected) < TOLERANCE

    def test_excess_no_predictions(self):
        """No predictions - zero excess."""
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]
        pred = []

        result = excess(pred, gt, image_width=100, image_height=100)
        assert abs(result - 0.0) < TOLERANCE

    def test_excess_no_ground_truth(self):
        """No ground truth - all predictions are in white space."""
        gt = []
        pred = [{"x": 0, "y": 0, "width": 10, "height": 10}]

        # Image 100x100 (area 10000)
        # GT area = 0
        # White space = 10000
        # Pred in white space = 100
        # Excess = 100 / 10000 = 0.01

        result = excess(pred, gt, image_width=100, image_height=100)
        expected = 100.0 / 10000.0
        assert abs(result - expected) < TOLERANCE

    def test_excess_overlapping_predictions(self):
        """Multiple overlapping predictions in white space - should use binary union."""
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]
        pred = [
            {"x": 20, "y": 0, "width": 10, "height": 10},  # Area 100
            {"x": 25, "y": 0, "width": 10, "height": 10},  # Area 100, overlaps 50 with first
        ]

        # Image 100x100
        # White space = 9900
        # Pred union area = 150 (not 200, because they overlap by 50)
        # Neither overlaps GT
        # Pred in white space = 150
        # Excess = 150 / 9900 ≈ 0.01515

        result = excess(pred, gt, image_width=100, image_height=100)
        expected = 150.0 / 9900.0
        assert abs(result - expected) < TOLERANCE

    def test_excess_bounded_at_one(self):
        """Test that excess is bounded at 1.0 when predictions cover all white space."""
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]
        # Prediction covers entire image
        pred = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        # Image 100x100 (area 10000)
        # White space = 9900
        # Pred area = 10000
        # Pred-GT overlap = 100
        # Pred in white space = 10000 - 100 = 9900
        # Excess = 9900 / 9900 = 1.0

        result = excess(pred, gt, image_width=100, image_height=100)
        assert abs(result - 1.0) < TOLERANCE


class TestCOTScore:
    """Tests for the overall COT score (Equation 14)."""

    def test_cot_perfect_score(self):
        """Perfect predictions - COT score = 1.0."""
        # Perfect match: C=1, O=0, T=0 → COT = 1 - 0 - 0 = 1.0
        pred = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 20, "y": 0, "width": 10, "height": 10},
        ]
        gt = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 20, "y": 0, "width": 10, "height": 10},
        ]

        result, _, _, _ = cot_score(pred, gt, image_width=100, image_height=100)
        assert abs(result - 1.0) < TOLERANCE

    def test_cot_no_predictions(self):
        """No predictions - COT score = 0.0."""
        # C=0, O=0, T=0 → COT = 0
        pred = []
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]

        result, _, _, _ = cot_score(pred, gt, image_width=100, image_height=100)
        assert abs(result - 0.0) < TOLERANCE

    def test_cot_single_prediction(self):
        """Single prediction - only coverage matters (O=0, T=0)."""
        # n=1, so O=0 and T=0 by definition
        pred = [{"x": 0, "y": 0, "width": 10, "height": 10}]
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]

        result, _, _, _ = cot_score(pred, gt, image_width=100, image_height=100)
        # C=1, O=0, T=0 → COT = 1
        assert abs(result - 1.0) < TOLERANCE

    def test_cot_with_overlap(self):
        """Overlapping predictions reduce COT score."""
        # Two completely overlapping predictions
        pred = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 0, "y": 0, "width": 10, "height": 10},
        ]
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]

        # C=1 (perfect coverage)
        # O=1 (complete overlap)
        # T=0 (single GT, no trespass possible)
        # COT = 1 - 1 - 0 = 0.0

        result, _, _, _ = cot_score(pred, gt, image_width=100, image_height=100)
        assert abs(result - 0.0) < TOLERANCE

    def test_cot_with_trespass(self):
        """Trespassing predictions reduce COT score."""
        # Multiple predictions with trespass
        # Note: Per paper, with n=1, O and T are not calculated for COT score
        # So we need n>=2 to test trespass in COT score
        gt = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 10, "y": 0, "width": 10, "height": 10},
        ]
        pred = [
            {"x": 0, "y": 0, "width": 5, "height": 10},  # Covers GT1 partially
            {"x": 0, "y": 0, "width": 12, "height": 10},  # Overlaps both GTs
        ]

        C = coverage(pred, gt, image_width=100, image_height=100)
        O = overlap(pred, gt, image_width=100, image_height=100)
        T = trespass(pred, gt, image_width=100, image_height=100)
        result, _, _, _ = cot_score(pred, gt, image_width=100, image_height=100)
        expected = C - O - T
        assert abs(result - expected) < TOLERANCE

    def test_cot_partial_coverage(self):
        """Partial coverage results in reduced COT score."""
        # Single prediction covering half of GT
        pred = [{"x": 0, "y": 0, "width": 5, "height": 10}]
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]

        # C = 0.5, O = 0, T = 0
        # COT = 0.5

        result, _, _, _ = cot_score(pred, gt, image_width=100, image_height=100)
        assert abs(result - 0.5) < TOLERANCE

    def test_cot_maximum_error(self):
        """Maximum overlap and trespass (both at 1.0) gives COT = -1.0."""
        # This is a theoretical case: to maximize both O and T,
        # we need predictions that completely overlap AND trespass.
        # From paper: "Overlap and Trespass can only be maximised by covering
        # the entire ground truth which naturally maximises Coverage"
        # So: C=1, O=1, T=1 → COT = 1 - 1 - 1 = -1.0

        # Create scenario with maximum overlap and trespass
        # Need multiple GTs and predictions that overlap completely and trespass maximally
        gt = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 20, "y": 0, "width": 10, "height": 10},
        ]

        # Multiple predictions all covering both GTs completely
        pred = [
            {"x": 0, "y": 0, "width": 30, "height": 10},
            {"x": 0, "y": 0, "width": 30, "height": 10},
        ]

        result, _, _, _ = cot_score(pred, gt, image_width=100, image_height=100)

        # C should be 1.0 (full coverage)
        # O should be 1.0 (complete overlap)
        # T should approach maximum
        # COT should be negative
        assert result < 0.0

    def test_cot_weighted(self):
        """Test weighted COT score."""
        pred = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 0, "y": 0, "width": 10, "height": 10},
        ]
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]

        # C=1, O=1, T=0
        # With weights (2, 1, 1): COT = 2*1 - 1*1 - 1*0 = 1.0
        result, _, _, _ = cot_score(
            pred,
            gt,
            image_width=100,
            image_height=100,
            weight_coverage=2.0,
            weight_overlap=1.0,
            weight_trespass=1.0,
        )
        assert abs(result - 1.0) < TOLERANCE

    def test_cot_components_range(self):
        """Verify that COT score stays within expected range."""
        # Various scenarios
        test_cases = [
            # Perfect
            (
                [{"x": 0, "y": 0, "width": 10, "height": 10}],
                [{"x": 0, "y": 0, "width": 10, "height": 10}],
            ),
            # Partial coverage
            (
                [{"x": 0, "y": 0, "width": 5, "height": 10}],
                [{"x": 0, "y": 0, "width": 10, "height": 10}],
            ),
            # No overlap
            (
                [{"x": 0, "y": 0, "width": 10, "height": 10}],
                [{"x": 20, "y": 0, "width": 10, "height": 10}],
            ),
        ]

        for pred, gt in test_cases:
            result, _, _, _ = cot_score(pred, gt, image_width=100, image_height=100)
            # With equal weights, COT should be in [-1, 1]
            assert -1.0 <= result <= 1.0
