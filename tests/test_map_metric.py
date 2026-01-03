
"""
Tests for the MAPMetric class using torchmetrics.
"""

import pytest
import torch
from cot_score.map_metric import MAPMetric

class TestMAPMetric:

    def test_perfect_match(self):
        """Test mAP with identical predictions and ground truth."""
        metric = MAPMetric()

        # 2 boxes, perfectly matching
        preds = [
            {'x': 10, 'y': 10, 'width': 50, 'height': 50, 'class': 'text', 'confidence': 1.0},
            {'x': 100, 'y': 100, 'width': 30, 'height': 30, 'class': 'figure', 'confidence': 1.0}
        ]

        gt = [
            {'x': 10, 'y': 10, 'width': 50, 'height': 50, 'class': 'text'},
            {'x': 100, 'y': 100, 'width': 30, 'height': 30, 'class': 'figure'}
        ]

        metric.update(preds, gt)
        results = metric.compute()

        assert results['map'] == 1.0
        assert results['map_50'] == 1.0
        assert results['map_75'] == 1.0

        # Check per-class
        assert results['classes']['text'] == 1.0
        assert results['classes']['figure'] == 1.0

    def test_no_overlap(self):
        """Test mAP with no overlapping boxes."""
        metric = MAPMetric()

        preds = [
            {'x': 10, 'y': 10, 'width': 20, 'height': 20, 'class': 'text', 'confidence': 1.0}
        ]

        gt = [
            {'x': 100, 'y': 100, 'width': 20, 'height': 20, 'class': 'text'}
        ]

        metric.update(preds, gt)
        results = metric.compute()

        assert results['map'] == 0.0
        assert results['map_50'] == 0.0

    def test_class_mismatch(self):
        """Test mAP when boxes overlap but classes differ."""
        metric = MAPMetric()

        # Box matches location but class is wrong
        preds = [
            {'x': 10, 'y': 10, 'width': 50, 'height': 50, 'class': 'figure', 'confidence': 1.0}
        ]

        gt = [
            {'x': 10, 'y': 10, 'width': 50, 'height': 50, 'class': 'text'}
        ]

        metric.update(preds, gt)
        results = metric.compute()

        # Should be 0 because class mismatch counts as false positive for 'figure' and false negative for 'text'
        assert results['map'] == 0.0
        assert 'figure' in results['classes']
        # 'text' might be in results depending on how torchmetrics handles missing predictions for a GT class
        # (It infers expected classes from both targets and preds)

    def test_partial_match(self):
        """Test mAP with imperfect overlaps."""
        metric = MAPMetric()

        # Box shifted by 10 pixels.
        # Overlap area: 40x40 = 1600. Union: 50x50 + 50x50 - 1600 = 3400. IoU = 1600/3400 = 0.47
        # This is < 0.5, so map_50 should be 0.

        # Let's try a smaller shift for > 0.5 IoU
        # Shift by 5. Overlap 45x45=2025. Union=2500+2500-2025=2975. IoU=0.68.
        # Should pass map_50 but fail map_75

        preds = [
            {'x': 15, 'y': 15, 'width': 50, 'height': 50, 'class': 'text', 'confidence': 1.0}
        ]

        gt = [
            {'x': 10, 'y': 10, 'width': 50, 'height': 50, 'class': 'text'}
        ]

        metric.update(preds, gt)
        results = metric.compute()


        assert results['map_50'] == 1.0
        assert results['map_75'] == 0.0
        assert 0.0 < results['map'] < 1.0  # COCO averages 0.50:0.05:0.95

