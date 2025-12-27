"""Comprehensive tests for the metrics module."""

import pytest
import numpy as np
from cot_score.metrics import coverage, overlap, iou, mean_iou
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
        pred = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]
        gt = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

        result = coverage(pred, gt)
        reference = ref_coverage(pred, gt)

        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_coverage_no_overlap(self):
        """Test coverage with no overlap between predicted and ground truth."""
        pred = [{'x': 0, 'y': 0, 'width': 10, 'height': 10}]
        gt = [{'x': 100, 'y': 100, 'width': 10, 'height': 10}]

        result = coverage(pred, gt)
        reference = ref_coverage(pred, gt)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_coverage_partial_overlap(self):
        """Test coverage with partial overlap."""
        # Prediction covers half of ground truth
        pred = [{'x': 0, 'y': 0, 'width': 50, 'height': 100}]
        gt = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

        result = coverage(pred, gt)
        reference = ref_coverage(pred, gt)

        assert abs(result - 0.5) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_coverage_multiple_predictions(self):
        """Test coverage with multiple predictions covering different parts."""
        pred = [
            {'x': 0, 'y': 0, 'width': 60, 'height': 100},
            {'x': 50, 'y': 0, 'width': 60, 'height': 100},
        ]
        gt = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

        result = coverage(pred, gt)
        reference = ref_coverage(pred, gt)

        # Should cover 110% (overlapping predictions), but capped at 100%
        assert result > 0.99  # Close to 1.0
        assert abs(result - reference) < TOLERANCE

    def test_coverage_multiple_ground_truth(self):
        """Test coverage with multiple ground truth boxes."""
        pred = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]
        gt = [
            {'x': 0, 'y': 0, 'width': 100, 'height': 100},
            {'x': 200, 'y': 200, 'width': 100, 'height': 100},
        ]

        result = coverage(pred, gt)
        reference = ref_coverage(pred, gt)

        # Covers first GT completely, second GT not at all = 50%
        assert abs(result - 0.5) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_coverage_empty_predictions(self):
        """Test coverage with empty predictions."""
        pred = []
        gt = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

        result = coverage(pred, gt)
        reference = ref_coverage(pred, gt)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_coverage_empty_ground_truth(self):
        """Test coverage with empty ground truth."""
        pred = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]
        gt = []

        result = coverage(pred, gt)
        reference = ref_coverage(pred, gt)

        # When GT is empty and predictions exist, coverage is 0
        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_coverage_both_empty(self):
        """Test coverage when both are empty."""
        pred = []
        gt = []

        result = coverage(pred, gt)
        reference = ref_coverage(pred, gt)

        # When both are empty, coverage is perfect (1.0)
        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_coverage_complex_scenario(self):
        """Test coverage with complex multi-box scenario."""
        pred = [
            {'x': 10, 'y': 20, 'width': 100, 'height': 50},
            {'x': 120, 'y': 25, 'width': 80, 'height': 60},
        ]
        gt = [
            {'x': 12, 'y': 22, 'width': 95, 'height': 48},
            {'x': 118, 'y': 23, 'width': 85, 'height': 62},
        ]

        result = coverage(pred, gt)
        reference = ref_coverage(pred, gt)

        assert 0.0 < result < 1.0  # Should be partial coverage
        assert abs(result - reference) < TOLERANCE


class TestOverlap:
    """Tests for the overlap metric."""

    def test_overlap_no_predictions(self):
        """Test overlap with no predictions."""
        pred = []
        gt = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

        result = overlap(pred, gt)
        reference = ref_overlap(pred, gt)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_overlap_single_prediction(self):
        """Test overlap with single prediction."""
        pred = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]
        gt = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

        result = overlap(pred, gt)
        reference = ref_overlap(pred, gt)

        # Single prediction cannot overlap with itself
        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_overlap_no_overlap_between_predictions(self):
        """Test overlap when predictions don't overlap each other."""
        pred = [
            {'x': 0, 'y': 0, 'width': 10, 'height': 10},
            {'x': 20, 'y': 20, 'width': 10, 'height': 10},
        ]
        gt = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

        result = overlap(pred, gt)
        reference = ref_overlap(pred, gt)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_overlap_complete_overlap(self):
        """Test overlap with completely overlapping predictions."""
        pred = [
            {'x': 0, 'y': 0, 'width': 100, 'height': 100},
            {'x': 0, 'y': 0, 'width': 100, 'height': 100},
        ]
        gt = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

        result = overlap(pred, gt)
        reference = ref_overlap(pred, gt)

        # Complete overlap
        assert result > 0.99
        assert abs(result - reference) < TOLERANCE

    def test_overlap_partial_overlap(self):
        """Test overlap with partially overlapping predictions."""
        pred = [
            {'x': 0, 'y': 0, 'width': 60, 'height': 100},
            {'x': 40, 'y': 0, 'width': 60, 'height': 100},
        ]
        gt = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

        result = overlap(pred, gt)
        reference = ref_overlap(pred, gt)

        # Partial overlap
        assert 0.0 < result < 1.0
        assert abs(result - reference) < TOLERANCE

    def test_overlap_empty_ground_truth(self):
        """Test overlap with empty ground truth."""
        pred = [
            {'x': 0, 'y': 0, 'width': 10, 'height': 10},
            {'x': 5, 'y': 5, 'width': 10, 'height': 10},
        ]
        gt = []

        result = overlap(pred, gt)
        reference = ref_overlap(pred, gt)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_overlap_three_predictions(self):
        """Test overlap with three overlapping predictions."""
        pred = [
            {'x': 0, 'y': 0, 'width': 50, 'height': 50},
            {'x': 25, 'y': 25, 'width': 50, 'height': 50},
            {'x': 50, 'y': 50, 'width': 50, 'height': 50},
        ]
        gt = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

        result = overlap(pred, gt)
        reference = ref_overlap(pred, gt)

        assert 0.0 < result < 1.0
        assert abs(result - reference) < TOLERANCE


class TestIOU:
    """Tests for the IOU metric."""

    def test_iou_perfect_match(self):
        """Test IOU with identical boxes."""
        box1 = {'x': 0, 'y': 0, 'width': 100, 'height': 100}
        box2 = {'x': 0, 'y': 0, 'width': 100, 'height': 100}

        result = iou(box1, box2)
        reference = ref_iou(box1, box2)

        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_iou_no_overlap(self):
        """Test IOU with non-overlapping boxes."""
        box1 = {'x': 0, 'y': 0, 'width': 10, 'height': 10}
        box2 = {'x': 20, 'y': 20, 'width': 10, 'height': 10}

        result = iou(box1, box2)
        reference = ref_iou(box1, box2)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_iou_partial_overlap(self):
        """Test IOU with partially overlapping boxes."""
        box1 = {'x': 0, 'y': 0, 'width': 20, 'height': 20}
        box2 = {'x': 10, 'y': 10, 'width': 20, 'height': 20}

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
        box1 = {'x': 10, 'y': 10, 'width': 20, 'height': 20}
        box2 = {'x': 0, 'y': 0, 'width': 100, 'height': 100}

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
        box1 = {'x': 0, 'y': 0, 'width': 10, 'height': 10}
        box2 = {'x': 10, 'y': 0, 'width': 10, 'height': 10}

        result = iou(box1, box2)
        reference = ref_iou(box1, box2)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_iou_floating_point_coords(self):
        """Test IOU with floating point coordinates."""
        box1 = {'x': 10.5, 'y': 20.3, 'width': 100.7, 'height': 50.2}
        box2 = {'x': 12.1, 'y': 22.8, 'width': 95.3, 'height': 48.9}

        result = iou(box1, box2)
        reference = ref_iou(box1, box2)

        assert 0.0 < result < 1.0
        assert abs(result - reference) < TOLERANCE


class TestMeanIOU:
    """Tests for the mean_iou metric."""

    def test_mean_iou_perfect_match(self):
        """Test mean_iou with perfect matches."""
        pred = [
            {'x': 0, 'y': 0, 'width': 100, 'height': 100},
            {'x': 200, 'y': 200, 'width': 50, 'height': 50},
        ]
        gt = [
            {'x': 0, 'y': 0, 'width': 100, 'height': 100},
            {'x': 200, 'y': 200, 'width': 50, 'height': 50},
        ]

        result = mean_iou(pred, gt)
        reference = ref_mean_iou(pred, gt)

        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_mean_iou_no_overlap(self):
        """Test mean_iou with no overlap."""
        pred = [{'x': 0, 'y': 0, 'width': 10, 'height': 10}]
        gt = [{'x': 100, 'y': 100, 'width': 10, 'height': 10}]

        result = mean_iou(pred, gt)
        reference = ref_mean_iou(pred, gt)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_mean_iou_multiple_boxes(self):
        """Test mean_iou with multiple boxes."""
        pred = [
            {'x': 0, 'y': 0, 'width': 100, 'height': 100},
            {'x': 150, 'y': 150, 'width': 50, 'height': 50},
        ]
        gt = [
            {'x': 0, 'y': 0, 'width': 100, 'height': 100},
            {'x': 200, 'y': 200, 'width': 50, 'height': 50},
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
            {'x': 0, 'y': 0, 'width': 100, 'height': 100},
            {'x': 5, 'y': 5, 'width': 100, 'height': 100},
            {'x': 10, 'y': 10, 'width': 100, 'height': 100},
        ]
        gt = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

        result = mean_iou(pred, gt)
        reference = ref_mean_iou(pred, gt)

        # Should select the best match (first prediction)
        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_mean_iou_empty_predictions(self):
        """Test mean_iou with empty predictions."""
        pred = []
        gt = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

        result = mean_iou(pred, gt)
        reference = ref_mean_iou(pred, gt)

        assert abs(result - 0.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_mean_iou_empty_ground_truth(self):
        """Test mean_iou with empty ground truth."""
        pred = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]
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
            boxes.append({'x': x, 'y': y, 'width': width, 'height': height})
        return boxes

    def test_coverage_random_small(self):
        """Test coverage on small random dataset."""
        pred = self._generate_random_boxes(5, seed=1)
        gt = self._generate_random_boxes(5, seed=2)

        result = coverage(pred, gt)
        reference = ref_coverage(pred, gt)

        assert abs(result - reference) < TOLERANCE

    def test_coverage_random_medium(self):
        """Test coverage on medium random dataset."""
        pred = self._generate_random_boxes(20, seed=3)
        gt = self._generate_random_boxes(20, seed=4)

        result = coverage(pred, gt)
        reference = ref_coverage(pred, gt)

        assert abs(result - reference) < TOLERANCE

    def test_overlap_random_small(self):
        """Test overlap on small random dataset."""
        pred = self._generate_random_boxes(5, seed=5)
        gt = self._generate_random_boxes(5, seed=6)

        result = overlap(pred, gt)
        reference = ref_overlap(pred, gt)

        assert abs(result - reference) < TOLERANCE

    def test_overlap_random_medium(self):
        """Test overlap on medium random dataset."""
        pred = self._generate_random_boxes(15, seed=7)
        gt = self._generate_random_boxes(15, seed=8)

        result = overlap(pred, gt)
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
        pred = [{'x': 0, 'y': 0, 'width': 0, 'height': 100}]
        gt = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

        # Should handle gracefully
        cov = coverage(pred, gt)
        ref_cov = ref_coverage(pred, gt)
        assert abs(cov - ref_cov) < TOLERANCE

    def test_very_small_boxes(self):
        """Test with very small boxes."""
        pred = [{'x': 0, 'y': 0, 'width': 0.001, 'height': 0.001}]
        gt = [{'x': 0, 'y': 0, 'width': 0.001, 'height': 0.001}]

        result = iou(pred[0], gt[0])
        reference = ref_iou(pred[0], gt[0])

        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_very_large_boxes(self):
        """Test with very large boxes."""
        pred = [{'x': 0, 'y': 0, 'width': 10000, 'height': 10000}]
        gt = [{'x': 0, 'y': 0, 'width': 10000, 'height': 10000}]

        result = coverage(pred, gt)
        reference = ref_coverage(pred, gt)

        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        pred = [{'x': -50, 'y': -50, 'width': 100, 'height': 100}]
        gt = [{'x': -50, 'y': -50, 'width': 100, 'height': 100}]

        result = coverage(pred, gt)
        reference = ref_coverage(pred, gt)

        assert abs(result - 1.0) < TOLERANCE
        assert abs(result - reference) < TOLERANCE
