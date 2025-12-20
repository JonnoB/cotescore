"""Tests for the metrics module."""

import pytest
from cot_score.metrics import coverage, overlap, iou


class TestCoverage:
    """Tests for the coverage metric."""

    def test_coverage_perfect_match(self):
        """Test coverage with perfect match between predicted and ground truth."""
        # TODO: Implement test
        pass

    def test_coverage_no_overlap(self):
        """Test coverage with no overlap between predicted and ground truth."""
        # TODO: Implement test
        pass

    def test_coverage_partial_overlap(self):
        """Test coverage with partial overlap."""
        # TODO: Implement test
        pass


class TestOverlap:
    """Tests for the overlap metric."""

    def test_overlap_perfect_match(self):
        """Test overlap with perfect match between predicted and ground truth."""
        # TODO: Implement test
        pass

    def test_overlap_no_overlap(self):
        """Test overlap with no overlap between predicted and ground truth."""
        # TODO: Implement test
        pass

    def test_overlap_partial_overlap(self):
        """Test overlap with partial overlap."""
        # TODO: Implement test
        pass


class TestIOU:
    """Tests for the IOU metric."""

    def test_iou_perfect_match(self):
        """Test IOU with identical boxes."""
        # TODO: Implement test
        pass

    def test_iou_no_overlap(self):
        """Test IOU with non-overlapping boxes."""
        # TODO: Implement test
        pass

    def test_iou_partial_overlap(self):
        """Test IOU with partially overlapping boxes."""
        # TODO: Implement test
        pass
