"""
COTe-Score: Coverage, Overlap, Trespass, Excess

A library for evaluating document layout analysis models using Coverage, Overlap,
Trespass, and Excess metrics.
"""

__version__ = "0.1.0"

from .layout import cote_score, coverage, overlap, iou, mean_iou
from .ocr import (
    shannon_entropy,
    jensen_shannon_divergence,
    text_to_counter,
    jsd_distance,
    spacer,
    spacer_micro,
    cdd_decomp,
    spacer_decomp,
    cdd_decomp_spatial,
    spacer_decomp_spatial,
)
from .class_metrics import cote_class, coverage_matrix, overlap_matrix, trespass_matrix
from .types import ClassCOTeResult, TokenPositions, RegionChars, RegionPixels, CDDDecomposition, SpACERDecomposition
from .adapters import boxes_to_region_pixels
from .visualisation import compute_cote_masks, visualize_cote_states
from .dataset import load_limerick_example, extract_ssu_boxes
from .alto_ssu_tagger import ALTOSSUTagger, assign_alto_ssu

__all__ = [
    "cote_score",
    "coverage",
    "overlap",
    "iou",
    "mean_iou",
    "shannon_entropy",
    "jensen_shannon_divergence",
    "text_to_counter",
    "jsd_distance",
    "spacer",
    "spacer_micro",
    "cdd_decomp",
    "spacer_decomp",
    "cdd_decomp_spatial",
    "spacer_decomp_spatial",
    "cote_class",
    "coverage_matrix",
    "overlap_matrix",
    "trespass_matrix",
    "ClassCOTeResult",
    "TokenPositions",
    "RegionChars",
    "RegionPixels",
    "CDDDecomposition",
    "SpACERDecomposition",
    "boxes_to_region_pixels",
    "compute_cote_masks",
    "visualize_cote_states",
    "load_limerick_example",
    "extract_ssu_boxes",
    "ALTOSSUTagger",
    "assign_alto_ssu",
]
