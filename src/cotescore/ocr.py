
from collections import Counter
from typing import List, Dict, Any, Union, Tuple, Optional, Iterable, Sequence, overload
import numpy as np
import collections

from cotescore.types import MaskInstance, TokenPositions, CDDDecomposition
from cotescore.adapters import calculate_intersection_area as _calculate_intersection_area
from cotescore._core import (
    _as_pred_masks,
    _check_gt_map,
    _ms_mask,
    _area_s,
    _compose_pred_count,
    _owner_ssu_id,
)
from cotescore._distributions import build_Q, build_S, build_S_star, build_R



# =============================================================================
# The Character Distribution Divergence
#
# This is basically a wrapper around the jensen-shannon divergence
# which has been implemented using basic shannon entropy.
#
# =============================================================================


def shannon_entropy(p):
    """
    Calculates the Shannon Entropy for a probability distribution.
    Args:
        p (np.ndarray): A 1D numpy array of probabilities (must sum to 1).
    Returns:
        float: The Shannon Entropy in bits.
    """
    p_nonzero = p[p > 0]
    return -np.sum(p_nonzero * np.log2(p_nonzero))


def jensen_shannon_divergence(p, q):
    """
    Calculates the Jensen-Shannon Divergence between two probability distributions.
    Args:
        p (np.ndarray): A 1D numpy array for the first distribution.
        q (np.ndarray): A 1D numpy array for the second distribution.
                         p and q must be of the same length.
    Returns:
        float: The JSD value.
    """
    p = np.asarray(p)
    q = np.asarray(q)

    if not np.isclose(p.sum(), 1.0) and p.sum() > 0:
        p = p / p.sum()
    if not np.isclose(q.sum(), 1.0) and q.sum() > 0:
        q = q / q.sum()

    # Handle cases where a distribution might be all zeros after initial conversion (e.g., empty text)
    # If a distribution sums to 0, it means it has no characters, so its probabilities are all 0.
    # The JSD definition assumes valid probability distributions.
    # We ensure p and q are of the same length.
    if p.sum() == 0 and q.sum() == 0:
        return 0.0  # Both empty, no divergence
    elif p.sum() == 0:
        p = np.zeros_like(q)  # Make p a zero-distribution of same size as q
    elif q.sum() == 0:
        q = np.zeros_like(p)  # Make q a zero-distribution of same size as p

    m = 0.5 * (p + q)
    jsd = shannon_entropy(m) - 0.5 * (shannon_entropy(p) + shannon_entropy(q))

    return jsd


def cdd(gt_text_list, ocr_text_list):
    """
    Calculates the Character Distribution Divergence (CDD) between ground truth
    and OCR output, which is essentially the Jensen-Shannon Divergence of
    their character frequency distributions.

    Args:
        gt_text_list (list of str): A list of strings representing the ground truth text.
        ocr_text_list (list of str): A list of strings representing the OCR output text.

    Returns:
        tuple: A tuple containing:
            - float: The CDD (Jensen-Shannon Divergence) value.
            - dict: A dictionary where keys are unique characters and values
                    are lists/tuples of [gt_count, ocr_count].
    """
    # 1. Flatten the lists of strings into single strings
    gt_full_text = "".join(gt_text_list)
    ocr_full_text = "".join(ocr_text_list)

    # 2. Count character frequencies
    gt_counts = collections.Counter(gt_full_text)
    ocr_counts = collections.Counter(ocr_full_text)

    # 3. Identify all unique characters present in either text
    # This step is still necessary to define the shared vocabulary and order.
    all_unique_chars = sorted(list(set(gt_counts.keys()).union(set(ocr_counts.keys()))))

    # Handle cases where both texts might be completely empty
    if not all_unique_chars:
        return 0.0, {}, []

    # 4. Create aligned count arrays using list comprehensions
    # This is more efficient than a manual for loop for filling arrays.
    gt_aligned_counts_list = [gt_counts.get(char, 0) for char in all_unique_chars]
    ocr_aligned_counts_list = [ocr_counts.get(char, 0) for char in all_unique_chars]

    gt_aligned_counts = np.array(gt_aligned_counts_list, dtype=int)
    ocr_aligned_counts = np.array(ocr_aligned_counts_list, dtype=int)

    # 5. Create the character counts dictionary using a dictionary comprehension
    # This is also more efficient for constructing the dictionary.
    char_counts_dict = {
        char: [gt_counts.get(char, 0), ocr_counts.get(char, 0)] for char in all_unique_chars
    }

    # 6. Convert counts to probability distributions
    gt_total = np.sum(gt_aligned_counts)
    ocr_total = np.sum(ocr_aligned_counts)

    # If a total is zero, it means no characters of that type.
    # This means the distribution is effectively all zeros, which will result in 0 entropy.
    p_gt = (
        gt_aligned_counts / gt_total
        if gt_total > 0
        else np.zeros_like(gt_aligned_counts, dtype=float)
    )
    p_ocr = (
        ocr_aligned_counts / ocr_total
        if ocr_total > 0
        else np.zeros_like(ocr_aligned_counts, dtype=float)
    )

    # 7. Calculate JSD
    cdd_value = np.sqrt(jensen_shannon_divergence(p_gt, p_ocr))

    return cdd_value, char_counts_dict


# =============================================================================
# CDD Decomposition
# =============================================================================


def _counters_to_cdd(counter_p: Counter, counter_q: Counter) -> float:
    """Compute sqrt(JSD) between two token-frequency Counter objects.

    Args:
        counter_p: First token frequency distribution.
        counter_q: Second token frequency distribution.

    Returns:
        CDD value (sqrt-JSD) in [0, 1]. Returns 0.0 when both are empty.
    """
    all_tokens = sorted(set(counter_p) | set(counter_q))
    if not all_tokens:
        return 0.0

    p_arr = np.array([counter_p.get(t, 0) for t in all_tokens], dtype=np.float64)
    q_arr = np.array([counter_q.get(t, 0) for t in all_tokens], dtype=np.float64)

    p_total = p_arr.sum()
    q_total = q_arr.sum()

    p_dist = p_arr / p_total if p_total > 0 else np.zeros_like(p_arr)
    q_dist = q_arr / q_total if q_total > 0 else np.zeros_like(q_arr)

    return float(np.sqrt(jensen_shannon_divergence(p_dist, q_dist)))


def cdd_decompose(
    Q: Counter,
    R: Counter,
    S_star: Counter,
    S: Counter,
) -> CDDDecomposition:
    """Compute the four-way CDD decomposition from pre-built distributions.

    Token-agnostic: works at character or word level depending on how the
    input Counters were built. Use the builders in cotescore._distributions
    to construct Q, R, S_star, and S.

    Args:
        Q:      GT token distribution (from build_Q).
        R:      Parsed GT token distribution (from build_R).
        S_star: OCR-on-GT-regions distribution (from build_S_star).
        S:      OCR-on-pred-regions distribution (from build_S).

    Returns:
        CDDDecomposition with d_pars, d_ocr, d_int, d_total.
    """
    return CDDDecomposition(
        d_pars=_counters_to_cdd(R, Q),
        d_ocr=_counters_to_cdd(S_star, Q),
        d_int=_counters_to_cdd(S, R),
        d_total=_counters_to_cdd(S, Q),
    )


def cdd_decompose_from_raw(
    token_positions: TokenPositions,
    pred_masks: Sequence[Union[np.ndarray, MaskInstance]],
    pred_token_lists: Sequence[Sequence[str]],
    gt_token_lists: Sequence[Sequence[str]],
) -> CDDDecomposition:
    """Compute full CDD decomposition from raw inputs.

    Builds all four distributions then delegates to cdd_decompose.
    The caller is responsible for pre-splitting text into token lists:
      - Character level: ``[list(text) for text in texts]``
      - Word level:      ``[text.split() for text in texts]``

    Args:
        token_positions:  GT token pixel positions.
        pred_masks:       Predicted region masks (same coordinate space).
        pred_token_lists: Pre-split token lists, one per predicted region.
        gt_token_lists:   Pre-split token lists, one per GT region.

    Returns:
        CDDDecomposition with all four error components.
    """
    Q = build_Q(token_positions)
    R = build_R(token_positions, pred_masks)
    S_star = build_S_star(gt_token_lists)
    S = build_S(pred_token_lists)
    return cdd_decompose(Q, R, S_star, S)


def cdd_decompose_low_info(
    gt_token_lists: Sequence[Sequence[str]],
    pred_token_lists: Sequence[Sequence[str]],
    gt_ocr_token_lists: Sequence[Sequence[str]],
) -> tuple[float, float, float]:
    """Compute CDD components without character position information.

    In low-information mode, R is unavailable so d_pars and d_int cannot
    be computed. Only d_total, d_ocr, and a triage ratio are returned.

    The triage heuristic (from the paper):
      - d_total << 2 * d_ocr  → OCR is the main error source
      - d_total >> 2 * d_ocr  → Parsing is the main error source

    Args:
        gt_token_lists:      Pre-split GT token lists (used to build Q).
        pred_token_lists:    Pre-split token lists from OCR on predicted regions.
        gt_ocr_token_lists:  Pre-split token lists from OCR on GT regions.

    Returns:
        Tuple of (d_total, d_ocr, triage_ratio) where
        triage_ratio = d_total / (2 * d_ocr), or inf if d_ocr == 0.
    """
    Q: Counter = Counter(token for tokens in gt_token_lists for token in tokens)
    S = build_S(pred_token_lists)
    S_star = build_S_star(gt_ocr_token_lists)

    d_total = _counters_to_cdd(S, Q)
    d_ocr = _counters_to_cdd(S_star, Q)
    triage = d_total / (2.0 * d_ocr) if d_ocr > 0.0 else float("inf")
    return d_total, d_ocr, triage