"""Class-level COTe metrics.

This module implements the multi-class extension of the COTe decomposition,
producing K×K interaction matrices and K-vector class-share quantities.

The scalar (class-agnostic) COTe metrics live in ``metrics.py``.

Notation (following the formulation):
  M^S_l          — binary GT mask for class l
  M^S            — binary GT mask across all classes
  M^p,b_k        — binary union mask of all class-k predictions
  M^p_k          — count mask of class-k predictions (pixel = # preds covering it)
  A^P_k          = sum(M^p,b_k) — total pixel area of class-k predictions
  A^O_k          = sum(M^S & (M^p_k − M^p,b_k)) — total redundant class-k pred area on GT
  A^S            = sum(M^S)
  i(j)           — SSU that owns prediction j (majority-pixel assignment)
  M^S_{l\\i(j)} — class-l GT mask with owner SSU i(j)'s pixels removed

Interaction matrices (K×K, rows = pred class k, cols = GT class l):

  C[k,l] = sum(M^S_l & M^p,b_k) / A^P_k
    Coverage confusion matrix. "Of all class-k prediction area, what fraction
    lands on class-l GT?" Diagonal = correct coverage; off-diagonal = misclassification.

  O[k,l] = sum(M^S_l & (M^p_k − M^p,b_k)) / A^O_k
    Overlap confusion matrix. "Of all class-k redundant prediction area on GT,
    what fraction lands on class-l GT?" Asymmetric. Row is zero when A^O_k=0.

  T[k,l] = sum_{j in k} sum(M^S_{l\\i(j)} & M^p_j) / A^P_k
    Trespass confusion matrix. "Of all class-k prediction area, what fraction
    trespasses on class-l GT (excluding the owner SSU)?"
    Diagonal is non-zero: within-class trespass against other SSUs of same class.

Class shares (K-vectors summing to 1):
  C_share[k]  fraction of total coverage attributable to class k
  O_share[k]  fraction of total overlap attributable to class k
  T_share[k]  fraction of total trespass attributable to class k
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from cotescore._core import (
    _check_gt_map,
    _ms_mask,
    _area_s,
    _owner_ssu_id,
)
from cotescore.types import ClassCOTeCounts, ClassCOTeResult, Label, MaskInstance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_preds_have_labels(preds: Sequence[MaskInstance]) -> None:
    """Ensure every prediction carries a class label.

    Args:
        preds: Sequence of :class:`~cot_score.types.MaskInstance` predictions.

    Raises:
        ValueError: If any prediction has ``label=None``.
    """
    for i, p in enumerate(preds):
        if p.label is None:
            raise ValueError(
                f"Prediction at index {i} has label=None; all predictions must "
                "carry a class label for class-level metrics."
            )


def _build_class_gt_masks(
    gt_ssu_map: np.ndarray,
    ssu_to_class: Dict[int, Label],
    classes: List[Label],
) -> Dict[Label, np.ndarray]:
    """Return one boolean GT mask per class (pixels belonging to that class's SSUs)."""
    result: Dict[Label, np.ndarray] = {}
    for cls in classes:
        ssu_ids = {sid for sid, c in ssu_to_class.items() if c == cls}
        mask = np.zeros(gt_ssu_map.shape, dtype=bool)
        for sid in ssu_ids:
            mask |= gt_ssu_map == sid
        result[cls] = mask
    return result


def _group_preds_by_class(
    preds: Sequence[MaskInstance],
) -> Dict[Label, List[np.ndarray]]:
    """Group prediction boolean masks by their class label."""
    groups: Dict[Label, List[np.ndarray]] = {}
    for p in preds:
        lbl = p.label  # validated non-None upstream
        m = p.mask.astype(bool, copy=False)
        if m.ndim != 2:
            raise ValueError("Prediction mask must be a 2D array")
        groups.setdefault(lbl, []).append(m)
    return groups


def _class_binary_pred_mask(
    pred_group: List[np.ndarray],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Union binary mask of all predictions in a class group."""
    mp_b = np.zeros(shape, dtype=bool)
    for m in pred_group:
        mp_b |= m
    return mp_b


def _class_count_pred_mask(
    pred_group: List[np.ndarray],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Count mask of all predictions in a class group."""
    mp = np.zeros(shape, dtype=np.int32)
    for m in pred_group:
        mp += m.astype(np.int32, copy=False)
    return mp


# ---------------------------------------------------------------------------
# Public matrix functions
# ---------------------------------------------------------------------------


def coverage_matrix(
    gt_ssu_map: np.ndarray,
    ssu_to_class: Dict[int, Label],
    preds: Sequence[MaskInstance],
) -> Tuple[np.ndarray, List[Label]]:
    """Compute the K×K coverage interaction matrix.

    C[k, l] = sum(M^S_l & M^p,b_k) / A^P_k

    "Of all class-k prediction area, what fraction lands on class-l GT?"
    Diagonal entries give correct within-class coverage fraction.
    Off-diagonal entries indicate classification errors.
    Row k is all-zero when class k has no predictions.

    Args:
        gt_ssu_map: 2D integer array of SSU ids (0 = background).
        ssu_to_class: Mapping from SSU id to class label.
        preds: Sequence of MaskInstance, each with a non-None ``label``.

    Returns:
        (matrix, classes) where matrix is K×K float64 and classes is the
        ordered list of class labels defining row/column meaning.
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    _validate_preds_have_labels(preds)

    classes = sorted(set(ssu_to_class.values()))
    K = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}

    class_gt = _build_class_gt_masks(gt_ssu_map, ssu_to_class, classes)
    pred_groups = _group_preds_by_class(preds)

    C = np.zeros((K, K), dtype=np.float64)
    for k_cls, k in cls_idx.items():
        group = pred_groups.get(k_cls, [])
        if not group:
            continue
        mp_k_b = _class_binary_pred_mask(group, gt_ssu_map.shape)
        a_p_k = int(np.sum(mp_k_b))
        if a_p_k == 0:
            continue
        for l_cls, l in cls_idx.items():
            ms_l = class_gt[l_cls]
            C[k, l] = float(np.sum(ms_l & mp_k_b)) / a_p_k

    return C, classes


def overlap_matrix(
    gt_ssu_map: np.ndarray,
    ssu_to_class: Dict[int, Label],
    preds: Sequence[MaskInstance],
) -> Tuple[np.ndarray, List[Label]]:
    """Compute the K×K overlap interaction matrix.

    O[k, l] = sum(M^S_l & (M^p_k − M^p,b_k)) / A^O_k
    where A^O_k = sum(M^S & (M^p_k − M^p,b_k))

    "Of all class-k redundant prediction area on GT, what fraction lands
    on class-l GT?" The matrix is asymmetric. Row k is all-zero when class k
    has no redundant predictions (A^O_k = 0).

    Args:
        gt_ssu_map: 2D integer array of SSU ids (0 = background).
        ssu_to_class: Mapping from SSU id to class label.
        preds: Sequence of MaskInstance, each with a non-None ``label``.

    Returns:
        (matrix, classes) where matrix is K×K float64 and classes is the
        ordered list of class labels defining row/column meaning.
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    _validate_preds_have_labels(preds)

    ms = _ms_mask(gt_ssu_map)
    classes = sorted(set(ssu_to_class.values()))
    K = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}

    class_gt = _build_class_gt_masks(gt_ssu_map, ssu_to_class, classes)
    pred_groups = _group_preds_by_class(preds)

    O = np.zeros((K, K), dtype=np.float64)
    for k_cls, k in cls_idx.items():
        group = pred_groups.get(k_cls, [])
        if not group:
            continue
        mp_k_b = _class_binary_pred_mask(group, gt_ssu_map.shape)
        mp_k = _class_count_pred_mask(group, gt_ssu_map.shape)
        mp_k_redundant = mp_k - mp_k_b.astype(np.int32)  # pixels covered by 2+ preds
        a_o_k = int(np.sum(ms.astype(np.int32) * mp_k_redundant))
        if a_o_k == 0:
            continue
        for l_cls, l in cls_idx.items():
            ms_l = class_gt[l_cls]
            O[k, l] = float(np.sum(ms_l.astype(np.int32) * mp_k_redundant)) / a_o_k

    return O, classes


def trespass_matrix(
    gt_ssu_map: np.ndarray,
    ssu_to_class: Dict[int, Label],
    preds: Sequence[MaskInstance],
) -> Tuple[np.ndarray, List[Label]]:
    """Compute the K×K trespass interaction matrix.

    T[k, l] = sum_{j in k} sum(M^S_{l\\i(j)} & M^p_j) / A^P_k

    "Of all class-k prediction area, what fraction trespasses on class-l GT
    (excluding the owner SSU)?"

    For off-diagonal entries (k≠l): since SSUs are class-pure, the owner SSU
    of a class-k prediction has no pixels in class-l GT, so M^S_{l\\i(j)} = M^S_l.

    For diagonal entries (k=l): M^S_{k\\i(j)} = class-k GT minus owner SSU pixels.
    This captures trespass against other class-k SSUs. Diagonal is non-zero
    when a prediction covers GT belonging to a different SSU of the same class.

    Row k is all-zero when class k has no predictions.

    Args:
        gt_ssu_map: 2D integer array of SSU ids (0 = background).
        ssu_to_class: Mapping from SSU id to class label.
        preds: Sequence of MaskInstance, each with a non-None ``label``.

    Returns:
        (matrix, classes) where matrix is K×K float64 and classes is the
        ordered list of class labels defining row/column meaning.
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    _validate_preds_have_labels(preds)

    classes = sorted(set(ssu_to_class.values()))
    K = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}

    class_gt = _build_class_gt_masks(gt_ssu_map, ssu_to_class, classes)
    pred_groups = _group_preds_by_class(preds)

    T = np.zeros((K, K), dtype=np.float64)
    for k_cls, k in cls_idx.items():
        group = pred_groups.get(k_cls, [])
        if not group:
            continue
        mp_k_b = _class_binary_pred_mask(group, gt_ssu_map.shape)
        a_p_k = int(np.sum(mp_k_b))
        if a_p_k == 0:
            continue

        for l_cls, l in cls_idx.items():
            ms_l = class_gt[l_cls]
            if k == l:
                # Diagonal: owner SSU excluded from class-k GT for each prediction
                pixels = 0
                for pm in group:
                    owner = _owner_ssu_id(gt_ssu_map, pm)
                    if owner is None:
                        ms_eff = ms_l
                    else:
                        ms_eff = ms_l & (gt_ssu_map != owner)
                    pixels += int(np.sum(pm & ms_eff))
                T[k, l] = float(pixels) / a_p_k
            else:
                # Off-diagonal: owner is class-k, has no pixels in class-l GT
                pixels = sum(int(np.sum(pm & ms_l)) for pm in group)
                T[k, l] = float(pixels) / a_p_k

    return T, classes


# ---------------------------------------------------------------------------
# Combined wrapper
# ---------------------------------------------------------------------------


def cote_class(
    gt_ssu_map: np.ndarray,
    ssu_to_class: Dict[int, Label],
    preds: Sequence[MaskInstance],
) -> ClassCOTeResult:
    """Compute the full class-level COTe decomposition in a single pass.

    Returns a :class:`~cot_score.types.ClassCOTeResult` containing all three
    K×K interaction matrices and all three K-vector share quantities.

    Share vectors each sum to 1.0 (when the corresponding global total is > 0).
    When a global total is zero, the corresponding share vector is all zeros.

    Args:
        gt_ssu_map: 2D integer array of SSU ids (0 = background).
        ssu_to_class: Mapping from SSU id to class label.
        preds: Sequence of MaskInstance, each with a non-None ``label``.

    Returns:
        ClassCOTeResult with all matrices, share vectors, and the ordered
        ``classes`` list.
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    _validate_preds_have_labels(preds)

    classes = sorted(set(ssu_to_class.values()))
    K = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}

    ms = _ms_mask(gt_ssu_map)
    a_s = _area_s(ms)

    class_gt = _build_class_gt_masks(gt_ssu_map, ssu_to_class, classes)
    pred_groups = _group_preds_by_class(preds)

    # Pre-compute per-class binary union and count masks
    class_mp_b: Dict[Label, np.ndarray] = {}
    class_mp: Dict[Label, np.ndarray] = {}
    for cls in classes:
        group = pred_groups.get(cls, [])
        class_mp_b[cls] = _class_binary_pred_mask(group, gt_ssu_map.shape)
        class_mp[cls] = _class_count_pred_mask(group, gt_ssu_map.shape)

    # Global binary union and count masks across all classes
    mp_b_global = np.zeros(gt_ssu_map.shape, dtype=bool)
    mp_global = np.zeros(gt_ssu_map.shape, dtype=np.int32)
    for cls in classes:
        mp_b_global |= class_mp_b[cls]
        mp_global += class_mp[cls]

    # --- Coverage matrix ---
    # C[k,l] = sum(ms_l & mp_k_b) / A^P_k
    C = np.zeros((K, K), dtype=np.float64)
    a_p = np.zeros(K, dtype=np.float64)
    for k_cls, k in cls_idx.items():
        mp_k_b = class_mp_b[k_cls]
        a_p_k = int(np.sum(mp_k_b))
        a_p[k] = a_p_k
        if a_p_k == 0:
            continue
        for l_cls, l in cls_idx.items():
            ms_l = class_gt[l_cls]
            C[k, l] = float(np.sum(ms_l & mp_k_b)) / a_p_k

    # --- Overlap matrix ---
    # O[k,l] = sum(ms_l & (mp_k - mp_k_b)) / A^O_k
    O = np.zeros((K, K), dtype=np.float64)
    for k_cls, k in cls_idx.items():
        mp_k_b = class_mp_b[k_cls]
        mp_k = class_mp[k_cls]
        mp_k_redundant = mp_k - mp_k_b.astype(np.int32)
        a_o_k = int(np.sum(ms.astype(np.int32) * mp_k_redundant))
        if a_o_k == 0:
            continue
        for l_cls, l in cls_idx.items():
            ms_l = class_gt[l_cls]
            O[k, l] = float(np.sum(ms_l.astype(np.int32) * mp_k_redundant)) / a_o_k

    # --- Trespass matrix ---
    # T[k,l] = sum_{j in k} sum(ms_{l\i(j)} & mp_j) / A^P_k
    T = np.zeros((K, K), dtype=np.float64)
    for k_cls, k in cls_idx.items():
        group = pred_groups.get(k_cls, [])
        if not group:
            continue
        mp_k_b = class_mp_b[k_cls]
        a_p_k = int(np.sum(mp_k_b))
        if a_p_k == 0:
            continue
        for l_cls, l in cls_idx.items():
            ms_l = class_gt[l_cls]
            if k == l:
                pixels = 0
                for pm in group:
                    owner = _owner_ssu_id(gt_ssu_map, pm)
                    ms_eff = ms_l & (gt_ssu_map != owner) if owner is not None else ms_l
                    pixels += int(np.sum(pm & ms_eff))
                T[k, l] = float(pixels) / a_p_k
            else:
                pixels = sum(int(np.sum(pm & ms_l)) for pm in group)
                T[k, l] = float(pixels) / a_p_k

    # --- Coverage share ---
    # C_share[k] = sum(ms & mp_k_b) / sum(ms & mp_b_global)
    global_cov_area = int(np.sum(ms & mp_b_global))
    C_share = np.zeros(K, dtype=np.float64)
    if global_cov_area > 0:
        for cls, k in cls_idx.items():
            C_share[k] = float(np.sum(ms & class_mp_b[cls])) / global_cov_area

    # --- Overlap share ---
    # O_share[k] = sum(ms & (mp_k - mp_k_b)) / sum(ms & (mp_global - mp_b_global))
    mp_b_global_int = mp_b_global.astype(np.int32)
    global_overlap_area = int(np.sum(ms.astype(np.int32) * (mp_global - mp_b_global_int)))
    O_share = np.zeros(K, dtype=np.float64)
    if global_overlap_area > 0:
        for cls, k in cls_idx.items():
            mp_k = class_mp[cls]
            mp_k_b_int = class_mp_b[cls].astype(np.int32)
            O_share[k] = (
                float(np.sum(ms.astype(np.int32) * (mp_k - mp_k_b_int))) / global_overlap_area
            )

    # --- Trespass share ---
    # T_share[k] = sum_{j in k} sum(ms_{excluding owner(j)} & mp_j) / (T_scalar * A_S)
    T_share = np.zeros(K, dtype=np.float64)
    if a_s > 0:
        global_trespass_pixels = 0
        per_class_trespass_pixels = {cls: 0 for cls in classes}
        for cls, k in cls_idx.items():
            group = pred_groups.get(cls, [])
            for pm in group:
                owner = _owner_ssu_id(gt_ssu_map, pm)
                if owner is None:
                    continue
                trespass_pixels = int(np.sum(pm & ms & (gt_ssu_map != owner)))
                global_trespass_pixels += trespass_pixels
                per_class_trespass_pixels[cls] += trespass_pixels

        if global_trespass_pixels > 0:
            for cls, k in cls_idx.items():
                T_share[k] = float(per_class_trespass_pixels[cls]) / global_trespass_pixels

    # --- Coverage-derived precision / recall / F1 ---
    # precision_k = TP_k / A^P_k (= diag(C), already normalised by predicted area)
    # recall_k    = TP_k / A^S_k (GT-area-normalised — not available from C alone)
    a_s_k = np.array([float(np.sum(class_gt[cls])) for cls in classes], dtype=np.float64)
    coverage_precision = np.diag(C).copy()
    tp = coverage_precision * a_p
    coverage_recall = np.zeros(K, dtype=np.float64)
    nonzero_s = a_s_k > 0
    coverage_recall[nonzero_s] = tp[nonzero_s] / a_s_k[nonzero_s]
    coverage_f1 = np.zeros(K, dtype=np.float64)
    denom = coverage_precision + coverage_recall
    nonzero_f1 = denom > 0
    coverage_f1[nonzero_f1] = (
        2 * coverage_precision[nonzero_f1] * coverage_recall[nonzero_f1] / denom[nonzero_f1]
    )

    return ClassCOTeResult(
        classes=classes,
        coverage_matrix=C,
        overlap_matrix=O,
        trespass_matrix=T,
        coverage_share=C_share,
        overlap_share=O_share,
        trespass_share=T_share,
        coverage_precision=coverage_precision,
        coverage_recall=coverage_recall,
        coverage_f1=coverage_f1,
    )


# ---------------------------------------------------------------------------
# Multi-image accumulation
# ---------------------------------------------------------------------------


def class_confusion_counts(
    gt_ssu_map: np.ndarray,
    ssu_to_class: Dict[int, Label],
    preds: Sequence[MaskInstance],
    classes: Sequence[Label],
) -> ClassCOTeCounts:
    """Compute raw, unnormalised per-class pixel counts for one image.

    Unlike :func:`cote_class`, ``classes`` is a required, externally-fixed
    taxonomy (not derived from this image's ``ssu_to_class``), so that counts
    from many images can be summed elementwise with :func:`sum_class_counts`
    before normalising once with :func:`finalize_class_counts`. Averaging
    already-normalised per-image matrices instead would let pages with little
    or no GT for a class distort the average — accumulate raw pixel sums
    across the whole dataset and divide once.

    Predictions whose label is not in ``classes`` are ignored entirely. If
    such predictions should still register as imprecision (e.g. a model's
    "Picture" class with no equivalent in the GT taxonomy), map them onto a
    real bucket in ``classes`` (such as an "other" class) upstream.

    Args:
        gt_ssu_map: 2D integer array of SSU ids (0 = background).
        ssu_to_class: Mapping from SSU id to class label.
        preds: Sequence of MaskInstance, each with a non-None ``label``.
        classes: Fixed, ordered list of class labels defining row/column
            meaning — pass the identical list to every image being
            accumulated together.

    Returns:
        ClassCOTeCounts with raw per-class-pair pixel sums for this image.
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    _validate_preds_have_labels(preds)

    classes = list(classes)
    K = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}

    ms = _ms_mask(gt_ssu_map)
    class_gt = _build_class_gt_masks(gt_ssu_map, ssu_to_class, classes)
    pred_groups = _group_preds_by_class(preds)

    class_mp_b: Dict[Label, np.ndarray] = {}
    class_mp: Dict[Label, np.ndarray] = {}
    for cls in classes:
        group = pred_groups.get(cls, [])
        class_mp_b[cls] = _class_binary_pred_mask(group, gt_ssu_map.shape)
        class_mp[cls] = _class_count_pred_mask(group, gt_ssu_map.shape)

    mp_b_global = np.zeros(gt_ssu_map.shape, dtype=bool)
    mp_global = np.zeros(gt_ssu_map.shape, dtype=np.int32)
    for cls in classes:
        mp_b_global |= class_mp_b[cls]
        mp_global += class_mp[cls]

    coverage_numer = np.zeros((K, K), dtype=np.float64)
    pred_area = np.zeros(K, dtype=np.float64)
    gt_area = np.zeros(K, dtype=np.float64)
    overlap_numer = np.zeros((K, K), dtype=np.float64)
    overlap_area = np.zeros(K, dtype=np.float64)
    trespass_numer = np.zeros((K, K), dtype=np.float64)
    trespass_share_numer = np.zeros(K, dtype=np.float64)

    for cls, k in cls_idx.items():
        mp_k_b = class_mp_b[cls]
        mp_k = class_mp[cls]
        pred_area[k] = float(np.sum(mp_k_b))
        gt_area[k] = float(np.sum(class_gt[cls]))

        mp_k_redundant = mp_k - mp_k_b.astype(np.int32)
        overlap_area[k] = float(np.sum(ms.astype(np.int32) * mp_k_redundant))

        for l_cls, l in cls_idx.items():
            ms_l = class_gt[l_cls]
            coverage_numer[k, l] = float(np.sum(ms_l & mp_k_b))
            overlap_numer[k, l] = float(np.sum(ms_l.astype(np.int32) * mp_k_redundant))

        group = pred_groups.get(cls, [])
        for l_cls, l in cls_idx.items():
            ms_l = class_gt[l_cls]
            if k == l:
                pixels = 0
                for pm in group:
                    owner = _owner_ssu_id(gt_ssu_map, pm)
                    ms_eff = ms_l & (gt_ssu_map != owner) if owner is not None else ms_l
                    pixels += int(np.sum(pm & ms_eff))
                trespass_numer[k, l] = pixels
            else:
                trespass_numer[k, l] = sum(int(np.sum(pm & ms_l)) for pm in group)

        # trespass_share_numer uses class-agnostic owner exclusion (matches
        # cote_class's per_class_trespass_pixels), which differs from the
        # row-sum of trespass_numer above whenever a prediction's
        # majority-overlap SSU belongs to a different class than the
        # prediction itself.
        for pm in group:
            owner = _owner_ssu_id(gt_ssu_map, pm)
            if owner is None:
                continue
            trespass_share_numer[k] += float(np.sum(pm & ms & (gt_ssu_map != owner)))

    global_coverage_area = int(np.sum(ms & mp_b_global))
    mp_b_global_int = mp_b_global.astype(np.int32)
    global_overlap_area = int(np.sum(ms.astype(np.int32) * (mp_global - mp_b_global_int)))
    global_trespass_pixels = int(np.sum(trespass_share_numer))

    return ClassCOTeCounts(
        classes=classes,
        coverage_numer=coverage_numer,
        pred_area=pred_area,
        gt_area=gt_area,
        overlap_numer=overlap_numer,
        overlap_area=overlap_area,
        trespass_numer=trespass_numer,
        trespass_share_numer=trespass_share_numer,
        global_coverage_area=global_coverage_area,
        global_overlap_area=global_overlap_area,
        global_trespass_pixels=global_trespass_pixels,
    )


def sum_class_counts(a: ClassCOTeCounts, b: ClassCOTeCounts) -> ClassCOTeCounts:
    """Accumulate two :class:`~cotescore.types.ClassCOTeCounts` elementwise.

    Args:
        a: Running total (or the first image's counts).
        b: The next image's counts. Must share the same ``classes`` list, in
            the same order, as ``a``.

    Returns:
        A new ClassCOTeCounts with every field summed.

    Raises:
        ValueError: If ``a.classes != b.classes``.
    """
    if a.classes != b.classes:
        raise ValueError(
            "Cannot sum ClassCOTeCounts with different class lists; pass the "
            "same fixed `classes` to every class_confusion_counts call."
        )
    return ClassCOTeCounts(
        classes=a.classes,
        coverage_numer=a.coverage_numer + b.coverage_numer,
        pred_area=a.pred_area + b.pred_area,
        gt_area=a.gt_area + b.gt_area,
        overlap_numer=a.overlap_numer + b.overlap_numer,
        overlap_area=a.overlap_area + b.overlap_area,
        trespass_numer=a.trespass_numer + b.trespass_numer,
        trespass_share_numer=a.trespass_share_numer + b.trespass_share_numer,
        global_coverage_area=a.global_coverage_area + b.global_coverage_area,
        global_overlap_area=a.global_overlap_area + b.global_overlap_area,
        global_trespass_pixels=a.global_trespass_pixels + b.global_trespass_pixels,
    )


def finalize_class_counts(counts: ClassCOTeCounts) -> ClassCOTeResult:
    """Normalise accumulated :class:`~cotescore.types.ClassCOTeCounts` into
    the final matrices, shares, and precision/recall/F1.

    This is the division step shared by :func:`cote_class` (single image) and
    the multi-image accumulation path (:func:`class_confusion_counts` +
    :func:`sum_class_counts`): call it once, after summing every image's
    counts, not per image.

    Args:
        counts: Raw pixel sums, typically the result of summing
            :func:`class_confusion_counts` across every image in a dataset.

    Returns:
        ClassCOTeResult with all matrices, share vectors, and
        precision/recall/F1.
    """
    classes = list(counts.classes)
    K = len(classes)
    pred_area = counts.pred_area
    gt_area = counts.gt_area

    C = np.zeros((K, K), dtype=np.float64)
    nonzero_p = pred_area > 0
    C[nonzero_p, :] = counts.coverage_numer[nonzero_p, :] / pred_area[nonzero_p, None]

    O = np.zeros((K, K), dtype=np.float64)
    nonzero_o = counts.overlap_area > 0
    O[nonzero_o, :] = counts.overlap_numer[nonzero_o, :] / counts.overlap_area[nonzero_o, None]

    T = np.zeros((K, K), dtype=np.float64)
    T[nonzero_p, :] = counts.trespass_numer[nonzero_p, :] / pred_area[nonzero_p, None]

    C_share = np.zeros(K, dtype=np.float64)
    if counts.global_coverage_area > 0:
        # Row-sum is valid here: GT classes partition M^S, so summing
        # coverage_numer[k, :] over l recovers sum(ms & mp_k_b) exactly.
        C_share = counts.coverage_numer.sum(axis=1) / counts.global_coverage_area

    O_share = np.zeros(K, dtype=np.float64)
    if counts.global_overlap_area > 0:
        O_share = counts.overlap_numer.sum(axis=1) / counts.global_overlap_area

    T_share = np.zeros(K, dtype=np.float64)
    if counts.global_trespass_pixels > 0:
        T_share = counts.trespass_share_numer / counts.global_trespass_pixels

    coverage_precision = np.diag(C).copy()
    tp = np.diag(counts.coverage_numer)
    coverage_recall = np.zeros(K, dtype=np.float64)
    nonzero_s = gt_area > 0
    coverage_recall[nonzero_s] = tp[nonzero_s] / gt_area[nonzero_s]

    coverage_f1 = np.zeros(K, dtype=np.float64)
    denom = coverage_precision + coverage_recall
    nonzero_f1 = denom > 0
    coverage_f1[nonzero_f1] = (
        2 * coverage_precision[nonzero_f1] * coverage_recall[nonzero_f1] / denom[nonzero_f1]
    )

    return ClassCOTeResult(
        classes=classes,
        coverage_matrix=C,
        overlap_matrix=O,
        trespass_matrix=T,
        coverage_share=C_share,
        overlap_share=O_share,
        trespass_share=T_share,
        coverage_precision=coverage_precision,
        coverage_recall=coverage_recall,
        coverage_f1=coverage_f1,
    )


__all__ = [
    "coverage_matrix",
    "overlap_matrix",
    "trespass_matrix",
    "cote_class",
    "class_confusion_counts",
    "sum_class_counts",
    "finalize_class_counts",
]
