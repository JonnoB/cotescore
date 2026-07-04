"""Tests for the analytic bounding-box fast path (GTBoxes / bbox-mode cote_score)."""

import numpy as np
import pytest

from cotescore import GTBoxes, cote_score
from cotescore.types import MaskInstance
from cotescore.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks
from cotescore._core_bbox import (
    _build_compressed_grid,
    _box_to_cells,
    bbox_cote_components,
)


TOLERANCE = 1e-5
# Cross-representation tolerance for float-coordinate agreement tests
# (matches the existing repo convention in TestVectorizedCorrectness).
CROSS_TOLERANCE = 0.02


def _gt_boxes(boxes, image_width=100, image_height=100):
    """Build a GTBoxes from a list of [x, y, w, h] boxes, ssu_ids 1..n."""
    arr = np.array(boxes, dtype=float) if boxes else np.zeros((0, 4))
    ids = np.arange(1, len(boxes) + 1, dtype=int)
    return GTBoxes(boxes=arr, ssu_ids=ids, image_width=image_width, image_height=image_height)


def _pred_boxes(boxes):
    """Build an (M, 4) predicted-box array from a list of [x, y, w, h] boxes."""
    return np.array(boxes, dtype=float) if boxes else np.zeros((0, 4))


def _dict_boxes_to_gt_ssu_map(gt_boxes, image_width, image_height):
    gt_boxes_with_id = []
    for idx, g in enumerate(gt_boxes, start=1):
        gg = dict(g)
        gg["ssu_id"] = idx
        gt_boxes_with_id.append(gg)
    return boxes_to_gt_ssu_map(gt_boxes_with_id, image_width, image_height, image_width, image_height)


def _dict_boxes_to_pred_masks(pred_boxes, image_width, image_height):
    instances = boxes_to_pred_masks(pred_boxes, image_width, image_height, image_width, image_height)
    return [inst.mask for inst in instances]


class TestGTBoxes:
    """Construction and __post_init__ validation for GTBoxes."""

    def test_valid_construction(self):
        gt = GTBoxes(
            boxes=np.array([[0, 0, 10, 10]], dtype=float),
            ssu_ids=np.array([1]),
            image_width=100,
            image_height=100,
        )
        assert gt.boxes.shape == (1, 4)
        assert gt.ssu_ids.tolist() == [1]

    def test_empty_construction(self):
        gt = GTBoxes(
            boxes=np.zeros((0, 4)),
            ssu_ids=np.zeros((0,), dtype=int),
            image_width=100,
            image_height=100,
        )
        assert len(gt.boxes) == 0

    def test_bad_box_shape_raises(self):
        with pytest.raises(ValueError):
            GTBoxes(
                boxes=np.array([[0, 0, 10]], dtype=float),
                ssu_ids=np.array([1]),
                image_width=100,
                image_height=100,
            )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            GTBoxes(
                boxes=np.array([[0, 0, 10, 10], [0, 0, 5, 5]], dtype=float),
                ssu_ids=np.array([1]),
                image_width=100,
                image_height=100,
            )

    def test_zero_ssu_id_raises(self):
        with pytest.raises(ValueError):
            GTBoxes(
                boxes=np.array([[0, 0, 10, 10]], dtype=float),
                ssu_ids=np.array([0]),
                image_width=100,
                image_height=100,
            )

    def test_negative_ssu_id_raises(self):
        with pytest.raises(ValueError):
            GTBoxes(
                boxes=np.array([[0, 0, 10, 10]], dtype=float),
                ssu_ids=np.array([-1]),
                image_width=100,
                image_height=100,
            )

    def test_float_ssu_id_raises(self):
        with pytest.raises(TypeError):
            GTBoxes(
                boxes=np.array([[0, 0, 10, 10]], dtype=float),
                ssu_ids=np.array([1.0]),
                image_width=100,
                image_height=100,
            )

    def test_non_positive_image_dims_raise(self):
        with pytest.raises(ValueError):
            GTBoxes(
                boxes=np.array([[0, 0, 10, 10]], dtype=float),
                ssu_ids=np.array([1]),
                image_width=0,
                image_height=100,
            )
        with pytest.raises(ValueError):
            GTBoxes(
                boxes=np.array([[0, 0, 10, 10]], dtype=float),
                ssu_ids=np.array([1]),
                image_width=100,
                image_height=-5,
            )


class TestCompressedGrid:
    """Unit tests for the coordinate-compression grid builder."""

    def test_edges_are_union_of_box_bounds(self):
        gt_boxes = np.array([[0, 0, 10, 10]], dtype=float)
        pred_boxes = np.array([[5, 5, 10, 10]], dtype=float)
        grid = _build_compressed_grid(gt_boxes, pred_boxes)
        assert list(grid.edges_x) == [0.0, 5.0, 10.0, 15.0]
        assert list(grid.edges_y) == [0.0, 5.0, 10.0, 15.0]

    def test_cell_area_sums_to_bounding_extent(self):
        gt_boxes = np.array([[0, 0, 10, 10]], dtype=float)
        pred_boxes = np.array([[5, 5, 10, 10]], dtype=float)
        grid = _build_compressed_grid(gt_boxes, pred_boxes)
        # bounding extent is 15x15 = 225
        assert abs(grid.cell_area.sum() - 225.0) < TOLERANCE

    def test_box_maps_to_exact_cells(self):
        gt_boxes = np.array([[0, 0, 10, 10], [10, 0, 10, 10]], dtype=float)
        pred_boxes = np.zeros((0, 4))
        grid = _build_compressed_grid(gt_boxes, pred_boxes)
        iy1, iy2, ix1, ix2 = _box_to_cells(grid, gt_boxes[0])
        area = grid.cell_area[iy1:iy2, ix1:ix2].sum()
        assert abs(area - 100.0) < TOLERANCE

    def test_zero_area_box_yields_empty_range(self):
        gt_boxes = np.array([[0, 0, 10, 10]], dtype=float)
        grid = _build_compressed_grid(gt_boxes, np.zeros((0, 4)))
        iy1, iy2, ix1, ix2 = _box_to_cells(grid, np.array([5.0, 5.0, 0.0, 5.0]))
        assert ix2 <= ix1 or iy2 <= iy1

    def test_adjoining_boxes_dedupe_shared_edge(self):
        gt_boxes = np.array([[0, 0, 10, 10], [10, 0, 10, 10]], dtype=float)
        grid = _build_compressed_grid(gt_boxes, np.zeros((0, 4)))
        # shared edge at x=10 should not be duplicated
        assert list(grid.edges_x) == [0.0, 10.0, 20.0]


class TestBboxCoteScoreBasic:
    """Hand-verified exact scenarios, reusing test_metrics.py's worked examples."""

    def test_perfect_match(self):
        # mirrors test_cot_perfect_score
        gt = _gt_boxes([[0, 0, 10, 10], [20, 0, 10, 10]])
        preds = _pred_boxes([[0, 0, 10, 10], [20, 0, 10, 10]])
        result = cote_score(gt, preds)
        assert abs(result[0] - 1.0) < TOLERANCE

    def test_partial_coverage(self):
        # mirrors test_cot_partial_coverage
        gt = _gt_boxes([[0, 0, 10, 10]])
        preds = _pred_boxes([[0, 0, 5, 10]])
        result = cote_score(gt, preds)
        assert abs(result[0] - 0.5) < TOLERANCE
        assert abs(result[1] - 0.5) < TOLERANCE

    def test_trespass_overlap(self):
        # mirrors test_trespass_overlap: expected T = 20/200 = 0.1
        gt = _gt_boxes([[0, 0, 10, 10], [10, 0, 10, 10]])
        preds = _pred_boxes([[0, 0, 12, 10]])
        _, _, _, T, _ = cote_score(gt, preds)
        assert abs(T - 0.1) < TOLERANCE

    def test_trespass_multiple_preds(self):
        # mirrors test_trespass_multiple_preds: expected T = 50/200 = 0.25
        gt = _gt_boxes([[0, 0, 10, 10], [20, 0, 10, 10]])
        preds = _pred_boxes([[0, 0, 10, 10], [5, 0, 25, 10]])
        _, _, _, T, _ = cote_score(gt, preds)
        assert abs(T - 0.25) < TOLERANCE

    def test_trespass_different_gt_sizes(self):
        # mirrors test_trespass_different_gt_sizes: expected T = 0.0
        gt = _gt_boxes([[0, 0, 10, 10], [20, 0, 20, 10]])
        preds = _pred_boxes([[0, 0, 15, 10]])
        _, _, _, T, _ = cote_score(gt, preds)
        assert abs(T - 0.0) < TOLERANCE

    def test_excess_background(self):
        # mirrors test_excess_background: expected E = 50/9900
        gt = _gt_boxes([[0, 0, 10, 10]])
        preds = _pred_boxes([[0, 0, 15, 10]])
        _, _, _, _, E = cote_score(gt, preds)
        assert abs(E - 50.0 / 9900.0) < TOLERANCE

    def test_excess_between_gts(self):
        # mirrors test_excess_between_gts: expected E = 100/9800
        gt = _gt_boxes([[0, 0, 10, 10], [20, 0, 10, 10]])
        preds = _pred_boxes([[0, 0, 30, 10]])
        _, _, _, _, E = cote_score(gt, preds)
        assert abs(E - 100.0 / 9800.0) < TOLERANCE

    def test_excess_overlapping_predictions(self):
        # mirrors test_excess_overlapping_predictions: expected E = 150/9900
        gt = _gt_boxes([[0, 0, 10, 10]])
        preds = _pred_boxes([[20, 0, 10, 10], [25, 0, 10, 10]])
        _, _, _, _, E = cote_score(gt, preds)
        assert abs(E - 150.0 / 9900.0) < TOLERANCE

    def test_excess_bounded_at_one(self):
        # mirrors test_excess_bounded_at_one: expected E = 1.0
        gt = _gt_boxes([[0, 0, 10, 10]])
        preds = _pred_boxes([[0, 0, 100, 100]])
        _, _, _, _, E = cote_score(gt, preds)
        assert abs(E - 1.0) < TOLERANCE

    def test_overlap_three_stacked_predictions(self):
        """Overlap must match production's per-cell max(k-1,0), NOT the pairwise
        C(k,2)/(n-1) formula from tests/reference_metrics.py. With 3 identical
        preds fully covering one GT box: production -> max(3-1,0) = 2.0;
        pairwise formula would give C(3,2)/(3-1) = 3/2 = 1.5, clamped to 1.0.
        Picking numbers where these clearly diverge."""
        gt = _gt_boxes([[0, 0, 10, 10]])
        preds = _pred_boxes([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]])
        _, _, O, _, _ = cote_score(gt, preds)
        assert abs(O - 2.0) < TOLERANCE

    def test_gt_ownership_tie_break_is_array_order(self):
        """Two fully-overlapping GT boxes, ssu_ids=[5, 2] in that array order.
        Owner must be 5 (array-order-first / first-write-wins), not 2."""
        gt = GTBoxes(
            boxes=np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float),
            ssu_ids=np.array([5, 2]),
            image_width=100,
            image_height=100,
        )
        # A prediction only partially covering GT box 5 so coverage/trespass
        # are observable; use trespass to surface which owner was assigned by
        # constructing a second GT region owned by a different id.
        # Simplest direct probe: paint owner and check its value directly.
        from cotescore._core_bbox import _build_compressed_grid, _paint_gt_owner

        grid = _build_compressed_grid(gt.boxes, np.zeros((0, 4)))
        owner = _paint_gt_owner(grid, gt.boxes, gt.ssu_ids)
        assert owner[owner != 0].tolist() == [5] * owner[owner != 0].size

    def test_dominant_owner_tie_break_is_lowest_ssu_id(self):
        """Two GT boxes with an exact-area tie under one prediction, with
        ssu_ids=[7, 3] (array-order-first has the LARGER id). Dominant owner
        for trespass must be 3 (lowest value), not 7 -- this catches
        conflating the array-order tie-break (ownership painting) with the
        lowest-value tie-break (trespass dominant owner)."""
        # Two disjoint, equal-area GT boxes, both fully covered by one pred.
        gt = GTBoxes(
            boxes=np.array([[0, 0, 10, 10], [10, 0, 10, 10]], dtype=float),
            ssu_ids=np.array([7, 3]),
            image_width=100,
            image_height=100,
        )
        preds = _pred_boxes([[0, 0, 20, 10]])
        # Trespass should be 0 here (pred equally covers both, dominant owner
        # picked among tied areas is 3; the "other" GT area (owned by 7) is
        # then trespassed). Cross-check against a mask-mode equivalent built
        # with ssu ids assigned in the same order via boxes_to_gt_ssu_map,
        # which assigns ssu_id from the ssu_id field directly (not tie-broken
        # by value), so mask mode's _owner_ssu_id is the authority here.
        gt_dicts = [
            {"x": 0, "y": 0, "width": 10, "height": 10, "ssu_id": 7},
            {"x": 10, "y": 0, "width": 10, "height": 10, "ssu_id": 3},
        ]
        gt_map = boxes_to_gt_ssu_map(gt_dicts, 100, 100, 100, 100)
        pred_masks = _dict_boxes_to_pred_masks(
            [{"x": 0, "y": 0, "width": 20, "height": 10}], 100, 100
        )
        _, _, _, T_mask, _ = cote_score(gt_map, pred_masks)
        _, _, _, T_bbox, _ = cote_score(gt, preds)
        assert abs(T_bbox - T_mask) < TOLERANCE

    def test_zero_gt_boxes(self):
        gt = _gt_boxes([])
        preds = _pred_boxes([[0, 0, 10, 10]])
        C, O, T, E = bbox_cote_components(gt, preds)
        assert C == 0.0
        assert O == 0.0
        assert T == 0.0

    def test_zero_predicted_boxes(self):
        gt = _gt_boxes([[0, 0, 10, 10]])
        preds = _pred_boxes([])
        C, O, T, E = bbox_cote_components(gt, preds)
        assert C == 0.0
        assert O == 0.0
        assert T == 0.0
        assert E == 0.0

    def test_zero_area_predicted_box_contributes_nothing(self):
        gt = _gt_boxes([[0, 0, 10, 10]])
        preds = _pred_boxes([[5, 5, 0, 5]])  # zero width
        C, O, T, E = bbox_cote_components(gt, preds)
        assert C == 0.0
        assert O == 0.0
        assert T == 0.0
        assert E == 0.0

    def test_gt_box_extending_past_image_bounds_is_clamped(self):
        """Regression test for a real HierText page: a GT box with negative
        x/y and width/height that pushes its far edge past the declared
        image dimensions (e.g. x=-9, width=1138 on a 1124-wide image) has an
        *unclamped* area exceeding the whole image's area. Mask mode always
        clamps box coordinates to the canvas before painting
        (boxes_to_gt_ssu_map), so bbox mode must too, or `excess`'s
        background_area = image_area - gt_area goes negative and produces a
        wildly wrong result (silently clamped to 0.0) instead of matching
        mask mode's correctly-clamped value."""
        image_width, image_height = 1124, 1600
        gt = GTBoxes(
            boxes=np.array([[-9, -6, 1138, 1603]], dtype=float),
            ssu_ids=np.array([1]),
            image_width=image_width,
            image_height=image_height,
        )
        preds = _pred_boxes([[0, 176.244354, 1124.0, 1423.755615]])

        gt_dicts = [{"x": -9, "y": -6, "width": 1138, "height": 1603, "ssu_id": 1}]
        gt_map = boxes_to_gt_ssu_map(gt_dicts, image_width, image_height, image_width, image_height)
        pred_masks = _dict_boxes_to_pred_masks(
            [{"x": 0, "y": 176.244354, "width": 1124.0, "height": 1423.755615}],
            image_width,
            image_height,
        )

        bbox_result = cote_score(gt, preds)
        mask_result = cote_score(gt_map, pred_masks)
        # Non-integer box coordinates -> mask mode still rounds to pixel
        # boundaries even at native resolution, so use the cross-representation
        # tolerance, not exact.
        for a, b in zip(bbox_result, mask_result):
            assert abs(a - b) < CROSS_TOLERANCE, f"{bbox_result} vs {mask_result}"

    def test_gt_box_extending_past_image_bounds_integer_exact(self):
        """Integer-coordinate variant of the out-of-bounds regression above,
        isolating the clamping fix from float-rounding noise: exact
        agreement expected."""
        image_width, image_height = 100, 100
        gt = GTBoxes(
            boxes=np.array([[-5, -5, 110, 110]], dtype=float),  # extends 5px past every edge
            ssu_ids=np.array([1]),
            image_width=image_width,
            image_height=image_height,
        )
        preds = _pred_boxes([[0, 0, 100, 100]])

        gt_dicts = [{"x": -5, "y": -5, "width": 110, "height": 110, "ssu_id": 1}]
        gt_map = boxes_to_gt_ssu_map(gt_dicts, image_width, image_height, image_width, image_height)
        pred_masks = _dict_boxes_to_pred_masks(
            [{"x": 0, "y": 0, "width": 100, "height": 100}], image_width, image_height
        )

        bbox_result = cote_score(gt, preds)
        mask_result = cote_score(gt_map, pred_masks)
        for a, b in zip(bbox_result, mask_result):
            assert abs(a - b) < TOLERANCE, f"{bbox_result} vs {mask_result}"
        # GT clamps to the full 100x100 image -> coverage 1.0, excess 0.0
        assert abs(bbox_result[1] - 1.0) < TOLERANCE
        assert abs(bbox_result[4] - 0.0) < TOLERANCE

    def test_adjoining_grid_no_double_counting(self):
        """3x3 grid of edge-adjoining GT boxes plus predictions whose edges
        coincide with several GT edges -- validates edge-dedup/half-open
        correctness at shared boundaries (no double-counted or dropped
        slivers)."""
        gt_list = []
        for row in range(3):
            for col in range(3):
                gt_list.append([col * 10, row * 10, 10, 10])
        gt = _gt_boxes(gt_list, image_width=100, image_height=100)
        # One big prediction covering the whole 3x3 block exactly.
        preds = _pred_boxes([[0, 0, 30, 30]])
        C, O, T, E = bbox_cote_components(gt, preds)
        assert abs(C - 1.0) < TOLERANCE
        assert abs(O - 0.0) < TOLERANCE


class TestBboxModeAgreesWithMaskMode:
    """Randomized agreement between bbox mode and mask mode."""

    def _random_boxes(self, rng, n, max_coord, max_size, integer=True):
        boxes = []
        for _ in range(n):
            x = rng.uniform(0, max_coord)
            y = rng.uniform(0, max_coord)
            w = rng.uniform(1, max_size)
            h = rng.uniform(1, max_size)
            if integer:
                x, y, w, h = round(x), round(y), round(max(1, round(w))), round(max(1, round(h)))
            boxes.append([x, y, w, h])
        return boxes

    def _compare(self, gt_list, pred_list, image_dim, tolerance):
        gt = _gt_boxes(gt_list, image_width=image_dim, image_height=image_dim)
        preds = _pred_boxes(pred_list)
        bbox_result = cote_score(gt, preds)

        gt_map = _dict_boxes_to_gt_ssu_map(
            [{"x": b[0], "y": b[1], "width": b[2], "height": b[3]} for b in gt_list],
            image_dim,
            image_dim,
        )
        pred_masks = _dict_boxes_to_pred_masks(
            [{"x": b[0], "y": b[1], "width": b[2], "height": b[3]} for b in pred_list],
            image_dim,
            image_dim,
        )
        mask_result = cote_score(gt_map, pred_masks)

        for a, b in zip(bbox_result, mask_result):
            assert abs(a - b) < tolerance, f"{bbox_result} vs {mask_result}"

    def test_exact_agreement_small_integer_boxes(self):
        rng = np.random.default_rng(0)
        gt_list = self._random_boxes(rng, 5, 80, 20)
        pred_list = self._random_boxes(rng, 5, 80, 20)
        self._compare(gt_list, pred_list, 100, TOLERANCE)

    def test_exact_agreement_medium_integer_boxes(self):
        rng = np.random.default_rng(1)
        gt_list = self._random_boxes(rng, 20, 400, 60)
        pred_list = self._random_boxes(rng, 20, 400, 60)
        self._compare(gt_list, pred_list, 500, TOLERANCE)

    def test_exact_agreement_large_smoke(self):
        rng = np.random.default_rng(2)
        gt_list = self._random_boxes(rng, 75, 1800, 150)
        pred_list = self._random_boxes(rng, 75, 1800, 150)
        self._compare(gt_list, pred_list, 2000, TOLERANCE)

    def test_tolerance_agreement_float_boxes(self):
        rng = np.random.default_rng(3)
        gt_list = self._random_boxes(rng, 10, 400, 60, integer=False)
        pred_list = self._random_boxes(rng, 10, 400, 60, integer=False)
        self._compare(gt_list, pred_list, 2000, CROSS_TOLERANCE)


class TestMixedModeRaises:
    def test_boxes_gt_with_mask_sequence_raises(self):
        gt = _gt_boxes([[0, 0, 10, 10]])
        pred_masks = [np.zeros((100, 100), dtype=bool)]
        with pytest.raises(TypeError):
            cote_score(gt, pred_masks)

    def test_boxes_gt_with_mask_instance_raises(self):
        gt = _gt_boxes([[0, 0, 10, 10]])
        pred_masks = [MaskInstance(mask=np.zeros((100, 100), dtype=bool))]
        with pytest.raises(TypeError):
            cote_score(gt, pred_masks)

    def test_mask_gt_with_box_array_raises(self):
        gt_map = _dict_boxes_to_gt_ssu_map(
            [{"x": 0, "y": 0, "width": 10, "height": 10}], 100, 100
        )
        preds = _pred_boxes([[0, 0, 10, 10]])
        with pytest.raises(TypeError):
            cote_score(gt_map, preds)
