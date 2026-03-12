"""Tests for the class-level COTe metrics (class_metrics module)."""

import pytest
import numpy as np

from cotescore.class_metrics import coverage_matrix, overlap_matrix, trespass_matrix, cote_class
from cotescore.types import MaskInstance, ClassCOTeResult


TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_gt_ssu_map(h: int, w: int, regions: dict) -> np.ndarray:
    """Build a gt_ssu_map from {ssu_id: (row_start, row_end, col_start, col_end)}."""
    gt = np.zeros((h, w), dtype=np.int32)
    for ssu_id, (r0, r1, c0, c1) in regions.items():
        gt[r0:r1, c0:c1] = ssu_id
    return gt


def _mask(h: int, w: int, r0: int, r1: int, c0: int, c1: int, label: str) -> MaskInstance:
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1, c0:c1] = True
    return MaskInstance(mask=m, label=label)


# Canonical 2-class 10×10 scenario:
#   SSU 1 → class "A": rows 0-4 (top half, 50 pixels)
#   SSU 2 → class "B": rows 5-9 (bottom half, 50 pixels)
#   Total GT area A^S = 100 px
SSU_TO_CLASS_2 = {1: "A", 2: "B"}
H, W = 10, 10
GT_2CLASS = _make_gt_ssu_map(H, W, {1: (0, 5, 0, 10), 2: (5, 10, 0, 10)})


# ---------------------------------------------------------------------------
# TestCoverageMatrix
# ---------------------------------------------------------------------------

class TestCoverageMatrix:
    """C[k,l] = sum(ms_l & mp_k_b) / A^P_k
    "Of all class-k prediction area, what fraction lands on class-l GT?"
    """

    def test_perfect_within_class_coverage(self):
        """When each pred exactly matches its class GT, diagonal = 1.0."""
        # A pred = 50 px, all on A GT → C[A,A] = 50/50 = 1.0
        # B pred = 50 px, all on B GT → C[B,B] = 50/50 = 1.0
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 5, 10, 0, 10, "B"),
        ]
        C, classes = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        assert abs(C[a, a] - 1.0) < TOLERANCE
        assert abs(C[b, b] - 1.0) < TOLERANCE
        assert abs(C[a, b] - 0.0) < TOLERANCE
        assert abs(C[b, a] - 0.0) < TOLERANCE

    def test_no_predictions(self):
        """All rows are zero when there are no predictions."""
        C, classes = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, [])
        assert C.shape == (2, 2)
        assert np.all(C == 0.0)

    def test_cross_class_coverage(self):
        """A pred covering B GT gives off-diagonal entry."""
        # A pred of 50 px all on B GT → C[A,B] = 50/50 = 1.0, C[A,A] = 0
        preds = [_mask(H, W, 5, 10, 0, 10, "A")]
        C, classes = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        assert abs(C[a, b] - 1.0) < TOLERANCE
        assert abs(C[a, a] - 0.0) < TOLERANCE

    def test_partial_coverage_normalised_by_pred_area(self):
        """Normalization is A^P_k (pred area), not GT area.
        An A pred of 25 px all on A GT → C[A,A] = 25/25 = 1.0 (not 0.5).
        """
        preds = [_mask(H, W, 0, 5, 0, 5, "A")]  # 25 px, all on A's GT
        C, classes = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a = classes.index("A")
        assert abs(C[a, a] - 1.0) < TOLERANCE

    def test_pred_split_across_classes(self):
        """A pred spanning both classes: fractions sum to ≤1 (remainder on background)."""
        # A pred covering rows 0-9, cols 0-10 = 100 px
        # 50 px on A GT, 50 px on B GT → C[A,A]=0.5, C[A,B]=0.5
        preds = [_mask(H, W, 0, 10, 0, 10, "A")]
        C, classes = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        assert abs(C[a, a] - 0.5) < TOLERANCE
        assert abs(C[a, b] - 0.5) < TOLERANCE

    def test_matrix_shape(self):
        preds = [_mask(H, W, 0, 5, 0, 10, "A")]
        C, classes = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert C.shape == (len(classes), len(classes))

    def test_unlabelled_prediction_raises(self):
        m = np.zeros((H, W), dtype=bool)
        preds = [MaskInstance(mask=m, label=None)]
        with pytest.raises(ValueError, match="label=None"):
            coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)


# ---------------------------------------------------------------------------
# TestOverlapMatrix
# ---------------------------------------------------------------------------

class TestOverlapMatrix:
    """O[k,l] = sum(ms_l & (mp_k - mp_k_b)) / A^O_k
    "Of all class-k redundant prediction area on GT, what fraction lands on class-l GT?"
    Row k is zero when class k has no redundant predictions.
    """

    def test_single_pred_per_class_no_redundancy(self):
        """A single prediction per class produces no redundant area → all rows zero."""
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 5, 10, 0, 10, "B"),
        ]
        O, classes = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert np.allclose(O, 0.0)

    def test_no_predictions_all_zero(self):
        O, _ = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, [])
        assert np.all(O == 0.0)

    def test_within_class_overlap_diagonal(self):
        """Two identical A preds covering A GT: all redundancy on A GT.
        A^O_A = 50 px (rows 0-4 all covered twice). O[A,A] = 50/50 = 1.0.
        """
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 0, 5, 0, 10, "A"),
        ]
        O, classes = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a = classes.index("A")
        assert abs(O[a, a] - 1.0) < TOLERANCE
        # B row stays zero (no B preds)
        b = classes.index("B")
        assert np.allclose(O[b, :], 0.0)

    def test_within_class_overlap_spanning_two_classes(self):
        """Two A preds both spanning rows 0-9: redundant area = 100 px on GT.
        50 px on A GT, 50 px on B GT.
        O[A,A] = 50/100 = 0.5, O[A,B] = 50/100 = 0.5.
        """
        preds = [
            _mask(H, W, 0, 10, 0, 10, "A"),
            _mask(H, W, 0, 10, 0, 10, "A"),
        ]
        O, classes = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        assert abs(O[a, a] - 0.5) < TOLERANCE
        assert abs(O[a, b] - 0.5) < TOLERANCE

    def test_matrix_not_symmetric_in_general(self):
        """Overlap matrix is not required to be symmetric."""
        # Two A preds overlapping only on A GT
        # One B pred (no redundancy)
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 5, 10, 0, 10, "B"),
        ]
        O, classes = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        # O[A,B] > 0 only if A redundancy lands on B GT (it doesn't here)
        assert abs(O[a, b] - 0.0) < TOLERANCE
        # O[B,A] = 0 (no B redundancy)
        assert abs(O[b, a] - 0.0) < TOLERANCE

    def test_matrix_shape(self):
        O, classes = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, [])
        assert O.shape == (len(classes), len(classes))


# ---------------------------------------------------------------------------
# TestTrespassMatrix
# ---------------------------------------------------------------------------

class TestTrespassMatrix:
    """T[k,l] = sum_{j in k} sum(ms_{l\\i(j)} & mp_j) / A^P_k
    Diagonal is non-zero: within-class trespass against other SSUs.
    """

    def test_no_predictions(self):
        T, _ = trespass_matrix(GT_2CLASS, SSU_TO_CLASS_2, [])
        assert np.all(T == 0.0)

    def test_no_trespass_when_pred_within_owner_ssu(self):
        """Pred exactly matching its owner SSU has zero trespass on all columns."""
        # A pred covers exactly SSU 1 (A's GT). Owner = SSU 1, no pixels outside it.
        preds = [_mask(H, W, 0, 5, 0, 10, "A")]
        T, classes = trespass_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert np.allclose(T, 0.0)

    def test_off_diagonal_trespass(self):
        """A pred (50 px) entirely on B GT → T[A,B] = 50/50 = 1.0."""
        preds = [_mask(H, W, 5, 10, 0, 10, "A")]
        T, classes = trespass_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        assert abs(T[a, b] - 1.0) < TOLERANCE
        assert abs(T[b, a] - 0.0) < TOLERANCE  # no B preds

    def test_partial_off_diagonal_trespass(self):
        """A pred spanning rows 2-6 (50 px total):
          - rows 2-4 → 30 px on A GT (SSU 1)
          - rows 5-6 → 20 px on B GT (SSU 2)
        Owner = SSU 1 (30 px overlap > 20 px). A^P_A = 50.
        T[A,B] = 20/50 = 0.4 (off-diagonal).
        T[A,A] = 0 (only one A SSU, so no within-class trespass possible).
        """
        preds = [_mask(H, W, 2, 7, 0, 10, "A")]  # rows 2-6: 30 on A GT, 20 on B GT
        T, classes = trespass_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        assert abs(T[a, b] - 0.4) < TOLERANCE
        assert abs(T[a, a] - 0.0) < TOLERANCE  # only one A SSU, no within-class trespass

    def test_diagonal_nonzero_within_class_trespass(self):
        """Two A SSUs exist. An A pred covers both → diagonal T[A,A] > 0.
        Setup: 4-class-region map with two separate A SSUs.
          SSU 1 (A): rows 0-2
          SSU 2 (A): rows 3-4
          SSU 3 (B): rows 5-9
        A pred covering rows 0-4 (50 px): owner = SSU 1 or SSU 2 depending on overlap.
        """
        gt = _make_gt_ssu_map(H, W, {1: (0, 3, 0, 10), 2: (3, 5, 0, 10), 3: (5, 10, 0, 10)})
        ssu_to_class = {1: "A", 2: "A", 3: "B"}
        # Pred covers rows 0-4 (50 px). SSU 1 has 30 px, SSU 2 has 20 px → owner = SSU 1.
        # A GT minus SSU 1 = rows 3-4 (SSU 2, 20 px). Pred covers those 20 px.
        # T[A,A] = 20 / 50 = 0.4
        preds = [_mask(H, W, 0, 5, 0, 10, "A")]
        T, classes = trespass_matrix(gt, ssu_to_class, preds)
        a = classes.index("A")
        assert abs(T[a, a] - 0.4) < TOLERANCE

    def test_matrix_shape(self):
        T, classes = trespass_matrix(GT_2CLASS, SSU_TO_CLASS_2, [])
        assert T.shape == (len(classes), len(classes))


# ---------------------------------------------------------------------------
# TestCOTeClass (combined wrapper)
# ---------------------------------------------------------------------------

class TestCOTeClass:

    def test_returns_correct_type(self):
        preds = [_mask(H, W, 0, 5, 0, 10, "A")]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert isinstance(result, ClassCOTeResult)

    def test_classes_ordered(self):
        preds = [_mask(H, W, 0, 5, 0, 10, "A")]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert result.classes == sorted(result.classes)

    def test_coverage_share_sums_to_one(self):
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 5, 10, 0, 10, "B"),
        ]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert abs(result.coverage_share.sum() - 1.0) < TOLERANCE

    def test_overlap_share_sums_to_one(self):
        # Two A preds create redundancy
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 0, 5, 0, 10, "A"),
        ]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        if result.overlap_share.sum() > 0:
            assert abs(result.overlap_share.sum() - 1.0) < TOLERANCE

    def test_trespass_share_sums_to_one(self):
        preds = [
            _mask(H, W, 5, 10, 0, 10, "A"),  # A pred on B GT
            _mask(H, W, 0, 5, 0, 10, "B"),   # B pred on A GT
        ]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        if result.trespass_share.sum() > 0:
            assert abs(result.trespass_share.sum() - 1.0) < TOLERANCE

    def test_share_vectors_all_zero_when_no_preds(self):
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, [])
        assert np.all(result.coverage_share == 0.0)
        assert np.all(result.overlap_share == 0.0)
        assert np.all(result.trespass_share == 0.0)

    def test_matrices_have_correct_shape(self):
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, [])
        K = len(result.classes)
        assert result.coverage_matrix.shape == (K, K)
        assert result.overlap_matrix.shape == (K, K)
        assert result.trespass_matrix.shape == (K, K)
        assert result.coverage_share.shape == (K,)
        assert result.overlap_share.shape == (K,)
        assert result.trespass_share.shape == (K,)

    def test_integration_perfect_coverage_no_overlap_no_trespass(self):
        """Perfect non-overlapping coverage: diagonal C=1, O=0, T=0."""
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 5, 10, 0, 10, "B"),
        ]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = result.classes.index("A"), result.classes.index("B")

        # Coverage: C[A,A] = 50/50 = 1.0, C[B,B] = 1.0, off-diagonal = 0
        assert abs(result.coverage_matrix[a, a] - 1.0) < TOLERANCE
        assert abs(result.coverage_matrix[b, b] - 1.0) < TOLERANCE
        assert abs(result.coverage_matrix[a, b] - 0.0) < TOLERANCE
        assert abs(result.coverage_matrix[b, a] - 0.0) < TOLERANCE

        # No redundant preds → overlap matrix all zero
        assert np.allclose(result.overlap_matrix, 0.0)

        # No trespass (each pred within its own class GT, owner SSU)
        assert np.allclose(result.trespass_matrix, 0.0)

        # Coverage shares: each class contributes 50/100 of covered GT
        assert abs(result.coverage_share[a] - 0.5) < TOLERANCE
        assert abs(result.coverage_share[b] - 0.5) < TOLERANCE

    def test_coverage_matrix_matches_standalone(self):
        preds = [_mask(H, W, 0, 7, 0, 10, "A"), _mask(H, W, 3, 10, 0, 10, "B")]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        C_standalone, _ = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert np.allclose(result.coverage_matrix, C_standalone)

    def test_overlap_matrix_matches_standalone(self):
        preds = [_mask(H, W, 0, 5, 0, 10, "A"), _mask(H, W, 0, 5, 0, 10, "A")]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        O_standalone, _ = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert np.allclose(result.overlap_matrix, O_standalone)

    def test_trespass_matrix_matches_standalone(self):
        preds = [_mask(H, W, 5, 10, 0, 10, "A")]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        T_standalone, _ = trespass_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert np.allclose(result.trespass_matrix, T_standalone)
