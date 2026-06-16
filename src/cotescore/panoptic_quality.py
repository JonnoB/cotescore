"""Panoptic Quality (PQ) metric computation.

Provides the standard PQ = SQ × RQ decomposition for evaluating panoptic
segmentation outputs.  Inputs are a ground-truth panoptic label map and a list
of :class:`~cotescore.types.MaskInstance` predictions (as produced by the
existing HuggingFace adapters).

Typical usage::

    from cotescore.panoptic_quality import panoptic_quality
    from cotescore.adapters import hf_panoptic_seg_to_masks

    # Ground truth: integer map where each unique id is one segment
    gt_panoptic_map = ...  # np.ndarray of shape (H, W), dtype int32
    gt_segments_info = [
        {"id": 1, "label_id": 0},
        {"id": 2, "label_id": 1},
        ...
    ]

    # Predictions from hf_panoptic_seg_to_masks
    pred_instances = hf_panoptic_seg_to_masks(result, things_only=True)

    pq, sq, rq = panoptic_quality(gt_panoptic_map, gt_segments_info, pred_instances)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from cotescore.types import Label, MaskInstance


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean masks."""
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return float(intersection / union) if union > 0 else 0.0


def _match_segments(
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int, List[Tuple[int, int, float]]]:
    """Match predicted segments to GT segments by IoU (greedy, best-first).

    Returns ``(tp, fp, fn, matched_pairs)`` where ``matched_pairs`` is a list
    of ``(gt_idx, pred_idx, iou)`` for each true positive.
    """
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)

    # Build IoU matrix
    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i, g in enumerate(gt_masks):
        for j, p in enumerate(pred_masks):
            iou_matrix[i, j] = _mask_iou(g, p)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matched_pairs: list[tuple[int, int, float]] = []

    # Greedy matching: always pick the highest remaining IoU
    flat = iou_matrix.ravel()
    indices = np.argsort(-flat)
    for flat_idx in indices:
        i = int(flat_idx // n_pred)
        j = int(flat_idx % n_pred)
        if iou_matrix[i, j] < iou_threshold:
            break
        if i in matched_gt or j in matched_pred:
            continue
        matched_gt.add(i)
        matched_pred.add(j)
        matched_pairs.append((i, j, float(iou_matrix[i, j])))

    tp = len(matched_pairs)
    fp = n_pred - len(matched_pred)
    fn = n_gt - len(matched_gt)
    return tp, fp, fn, matched_pairs


def panoptic_quality(
    gt_panoptic_map: np.ndarray,
    gt_segments_info: Sequence[Dict[str, int]],
    preds: Sequence[MaskInstance],
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and
    Recognition Quality (RQ) for a single image.

    Args:
        gt_panoptic_map: 2D integer array of shape ``(H, W)`` where each
            unique non-zero value is a ground-truth segment id.
        gt_segments_info: Sequence of dicts, each with ``"id"`` (segment id
            in the map) and ``"label_id"`` (class label).  Segments with
            id 0 are treated as background and ignored.
        preds: Predicted segments as :class:`~cotescore.types.MaskInstance`
            objects (e.g. from :func:`~cotescore.adapters.hf_panoptic_seg_to_masks`).
        iou_threshold: Minimum IoU for a match to count as a true positive
            (default 0.5, the standard PQ threshold).

    Returns:
        A 3-tuple ``(pq, sq, rq)``.  All values are in ``[0.0, 1.0]``.
        Returns ``(0.0, 0.0, 0.0)`` when there are no ground-truth segments.
    """
    gt_panoptic_map = np.asarray(gt_panoptic_map)

    # Build GT masks from the panoptic map
    gt_masks: list[np.ndarray] = []
    for seg in gt_segments_info:
        seg_id = int(seg["id"])
        if seg_id == 0:
            continue
        gt_masks.append(gt_panoptic_map == seg_id)

    pred_masks = [p.mask for p in preds]

    if not gt_masks:
        return 0.0, 0.0, 0.0

    if not pred_masks:
        # No predictions → RQ = 0, SQ is undefined but conventionally 0
        return 0.0, 0.0, 0.0

    tp, fp, fn, matched_pairs = _match_segments(gt_masks, pred_masks, iou_threshold)

    if tp == 0:
        return 0.0, 0.0, 0.0

    sq = sum(iou for _, _, iou in matched_pairs) / tp
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    pq = sq * rq

    return float(pq), float(sq), float(rq)


def panoptic_quality_per_class(
    gt_panoptic_map: np.ndarray,
    gt_segments_info: Sequence[Dict[str, int]],
    preds: Sequence[MaskInstance],
    iou_threshold: float = 0.5,
) -> Dict[Label, Tuple[float, float, float]]:
    """Compute PQ/SQ/RQ broken down by ground-truth class label.

    Returns a dict mapping class label → ``(pq, sq, rq)`` for that class.
    """
    gt_panoptic_map = np.asarray(gt_panoptic_map)

    # Group GT segments by class
    class_to_gt: Dict[Label, list[tuple[int, np.ndarray]]] = {}
    for seg in gt_segments_info:
        seg_id = int(seg["id"])
        if seg_id == 0:
            continue
        label = seg["label_id"]
        mask = gt_panoptic_map == seg_id
        class_to_gt.setdefault(label, []).append((seg_id, mask))

    # Group predictions by class
    class_to_pred: Dict[Label, list[np.ndarray]] = {}
    for p in preds:
        label = p.label
        if label is None:
            continue
        class_to_pred.setdefault(label, []).append(p.mask)

    results: Dict[Label, Tuple[float, float, float]] = {}
    all_classes = set(class_to_gt.keys()) | set(class_to_pred.keys())

    for cls in sorted(all_classes, key=lambda x: (isinstance(x, str), x)):
        gt_masks = [m for _, m in class_to_gt.get(cls, [])]
        pred_masks = class_to_pred.get(cls, [])

        if not gt_masks:
            # No GT for this class but predictions exist → PQ = 0
            results[cls] = (0.0, 0.0, 0.0)
            continue

        if not pred_masks:
            results[cls] = (0.0, 0.0, 0.0)
            continue

        tp, fp, fn, matched_pairs = _match_segments(
            gt_masks, pred_masks, iou_threshold
        )

        if tp == 0:
            results[cls] = (0.0, 0.0, 0.0)
            continue

        sq = sum(iou for _, _, iou in matched_pairs) / tp
        rq = tp / (tp + 0.5 * fp + 0.5 * fn)
        results[cls] = (float(sq * rq), float(sq), float(rq))

    return results
