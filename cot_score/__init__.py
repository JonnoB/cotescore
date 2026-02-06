"""
COT-Score: Coverage and Overlap Testing for Document Layout Analysis

A library for evaluating document layout analysis models using Coverage and Overlap metrics.
"""

__version__ = "0.1.0"

from .metrics import cote_score, coverage, overlap, iou, mean_iou, cdd

__all__ = ["cote_score", "coverage", "overlap", "iou", "mean_iou", "cdd"]
