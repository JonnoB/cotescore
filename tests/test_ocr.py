"""Tests for the OCR metric functions: text_to_counter, jsd_distance,
spacer, spacer_micro, cdd_decomp, spacer_decomp, cdd_decomp_spatial,
and spacer_decomp_spatial."""

from collections import Counter

import numpy as np
import pytest

from cotescore import (
    CDDDecomposition,
    SpACERDecomposition,
    RegionChars,
    RegionPixels,
    boxes_to_region_pixels,
    cdd_decomp,
    cdd_decomp_spatial,
    jsd_distance,
    spacer,
    spacer_micro,
    spacer_decomp,
    spacer_decomp_spatial,
    text_to_counter,
)
from cotescore._distributions import build_R_from_region_pixels
from cotescore.ocr import _spacer_counts


# =============================================================================
# text_to_counter
# =============================================================================


class TestTextToCounter:
    def test_char_mode_string(self):
        c = text_to_counter("aab", mode="char")
        assert c == Counter({"a": 2, "b": 1})

    def test_char_mode_list(self):
        c = text_to_counter(["aa", "b"], mode="char")
        assert c == Counter({"a": 2, "b": 1})

    def test_word_mode_string(self):
        c = text_to_counter("hello world hello", mode="word")
        assert c == Counter({"hello": 2, "world": 1})

    def test_word_mode_list(self):
        c = text_to_counter(["hello world", "hello"], mode="word")
        assert c == Counter({"hello": 2, "world": 1})

    def test_empty_string(self):
        assert text_to_counter("") == Counter()

    def test_empty_list(self):
        assert text_to_counter([]) == Counter()

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            text_to_counter("abc", mode="token")


# =============================================================================
# jsd_distance
# =============================================================================


class TestJSDDistance:
    def test_identical_distributions(self):
        c = Counter({"a": 3, "b": 2})
        assert jsd_distance(c, c) == pytest.approx(0.0, abs=1e-9)

    def test_completely_disjoint(self):
        p = Counter({"a": 1})
        q = Counter({"b": 1})
        assert jsd_distance(p, q) == pytest.approx(1.0, abs=1e-9)

    def test_symmetric(self):
        p = Counter({"a": 3, "b": 1})
        q = Counter({"a": 1, "b": 3})
        assert jsd_distance(p, q) == pytest.approx(jsd_distance(q, p), abs=1e-9)

    def test_both_empty(self):
        assert jsd_distance(Counter(), Counter()) == pytest.approx(0.0)

    def test_range(self):
        p = Counter({"a": 5, "b": 2, "c": 1})
        q = Counter({"a": 1, "b": 4, "d": 2})
        assert 0.0 <= jsd_distance(p, q) <= 1.0

    def test_one_empty(self):
        # One side empty: JSD(single-atom, empty) = 0.5 bits → sqrt = 1/sqrt(2)
        result = jsd_distance(Counter({"a": 3}), Counter())
        assert result == pytest.approx(1.0 / 2 ** 0.5, abs=1e-9)


# =============================================================================
# _spacer_counts (internal helper — tested directly for precision)
# =============================================================================


class TestSpACERCounts:
    def test_perfect_match_single_box(self):
        c = Counter({"a": 3, "b": 2})
        E_hat, C, D_macro, D_micro = _spacer_counts([c], [c])
        assert E_hat == pytest.approx(0.0)
        assert C == pytest.approx(5.0)
        assert D_macro == pytest.approx(0.0)
        assert D_micro == pytest.approx(0.0)

    def test_deletion_only(self):
        ref = Counter({"a": 4})
        pred = Counter({"a": 2})
        E_hat, C, D_macro, D_micro = _spacer_counts([ref], [pred])
        # |4 - 2| = 2
        assert E_hat == pytest.approx(2.0)
        assert C == pytest.approx(4.0)
        assert D_macro == pytest.approx(2.0)
        assert D_micro == pytest.approx(2.0)

    def test_insertion_only(self):
        ref = Counter({"a": 2})
        pred = Counter({"a": 4})
        E_hat, C, D_macro, D_micro = _spacer_counts([ref], [pred])
        assert E_hat == pytest.approx(2.0)
        assert C == pytest.approx(2.0)
        assert D_macro == pytest.approx(0.0)   # no net deletion at page level
        assert D_micro == pytest.approx(0.0)   # no per-box deletion either

    def test_micro_vs_macro_differ(self):
        # Box 0: ref=3, pred=1 → deletion of 2
        # Box 1: ref=1, pred=3 → insertion of 2 (masks macro deletion)
        ref0 = Counter({"a": 3})
        pred0 = Counter({"a": 1})
        ref1 = Counter({"a": 1})
        pred1 = Counter({"a": 3})
        E_hat, C, D_macro, D_micro = _spacer_counts([ref0, ref1], [pred0, pred1])
        # Aggregate: ref=4, pred=4 → D_macro = 0
        assert D_macro == pytest.approx(0.0)
        # Per box: box0 del=2, box1 del=0 → D_micro = 2
        assert D_micro == pytest.approx(2.0)
        assert C == pytest.approx(4.0)


# =============================================================================
# spacer
# =============================================================================


class TestSpacer:
    def test_perfect_match(self):
        c = Counter({"a": 3, "b": 2})
        assert spacer(c, c) == pytest.approx(0.0)

    def test_known_value(self):
        # ref="aaaa" (C=4), pred="aa" (2 deletions)
        # E_hat=2, D_macro=2  → (2+2)/(2*4) = 0.5
        ref = Counter({"a": 4})
        pred = Counter({"a": 2})
        assert spacer(ref, pred) == pytest.approx(0.5)

    def test_empty_ref_returns_zero(self):
        assert spacer(Counter(), Counter({"a": 3})) == pytest.approx(0.0)

    def test_can_exceed_one(self):
        # Completely wrong characters: E_hat = C (all chars differ)
        # D_macro = C (all deleted from ref perspective if pred uses wrong chars)
        # Actually: ref=Counter(a:2), pred=Counter(b:4)
        # E_hat = |2-0| + |0-4| = 6, C=2, D_macro=max(0,2-4)=0
        # SpACER = (0+6)/(2*2) = 1.5
        ref = Counter({"a": 2})
        pred = Counter({"b": 4})
        assert spacer(ref, pred) > 1.0


# =============================================================================
# spacer_micro
# =============================================================================


class TestSpacerMicro:
    def test_single_box_matches_spacer(self):
        ref = Counter({"a": 3, "b": 1})
        pred = Counter({"a": 2, "b": 2})
        assert spacer_micro([ref], [pred]) == pytest.approx(spacer(ref, pred))

    def test_micro_higher_than_macro_when_cancellation(self):
        ref0 = Counter({"a": 3})
        pred0 = Counter({"a": 1})
        ref1 = Counter({"a": 1})
        pred1 = Counter({"a": 3})
        macro_val = spacer(ref0 + ref1, pred0 + pred1)
        micro_val = spacer_micro([ref0, ref1], [pred0, pred1])
        assert micro_val > macro_val


# =============================================================================
# cdd_decomp
# =============================================================================


class TestCDDDecomp:
    def test_all_keys_present(self):
        d = cdd_decomp({"gt": "abc", "parsing": "abc", "ocr": "abc", "total": "abc"})
        assert isinstance(d, CDDDecomposition)
        assert d.d_pars == pytest.approx(0.0)
        assert d.d_ocr == pytest.approx(0.0)
        assert d.d_int == pytest.approx(0.0)
        assert d.d_total == pytest.approx(0.0)

    def test_missing_parsing_gives_none(self):
        d = cdd_decomp({"gt": "abc", "ocr": "abc", "total": "abc"})
        assert d.d_pars is None
        assert d.d_int is None
        assert d.d_ocr is not None
        assert d.d_total is not None

    def test_missing_ocr_gives_none(self):
        d = cdd_decomp({"gt": "abc", "parsing": "abc", "total": "abc"})
        assert d.d_ocr is None
        assert d.d_pars is not None

    def test_missing_total_gives_none(self):
        d = cdd_decomp({"gt": "abc", "parsing": "abc", "ocr": "abc"})
        assert d.d_int is None
        assert d.d_total is None

    def test_custom_metric(self):
        def zero_metric(p, q):
            return 0.0

        d = cdd_decomp(
            {"gt": "abc", "parsing": "xyz", "ocr": "xyz", "total": "xyz"},
            metric=zero_metric,
        )
        assert d.d_pars == pytest.approx(0.0)
        assert d.d_ocr == pytest.approx(0.0)
        assert d.d_total == pytest.approx(0.0)

    def test_word_mode(self):
        d = cdd_decomp(
            {"gt": "hello world", "total": "hello world"},
            mode="word",
        )
        assert d.d_total == pytest.approx(0.0)

    def test_list_values(self):
        d = cdd_decomp({"gt": ["ab", "c"], "total": ["ab", "c"]})
        assert d.d_total == pytest.approx(0.0)

    def test_counter_passthrough_identity(self):
        # Pre-built Counter produces identical result to the equivalent string
        gt = Counter({"a": 3, "b": 2})    # "aaabb"
        total = Counter({"a": 1, "b": 4}) # "abbbb"
        from_counters = cdd_decomp({"gt": gt, "total": total})
        from_strings = cdd_decomp({"gt": "aaabb", "total": "abbbb"})
        assert from_counters.d_total == pytest.approx(from_strings.d_total, abs=1e-9)

    def test_counter_passthrough_perfect_match(self):
        gt = Counter({"a": 3, "b": 2})
        d = cdd_decomp({"gt": gt, "total": gt})
        assert d.d_total == pytest.approx(0.0)

    def test_counter_and_string_mixed(self):
        # Counter and equivalent string should give the same result
        gt = Counter({"a": 3, "b": 2})
        d = cdd_decomp({"gt": gt, "total": "aaabb"})
        assert d.d_total == pytest.approx(0.0)

    def test_all_counter_values_full_decomp(self):
        # Simulate spatial use case: all four slots as pre-built Counters
        Q = Counter({"a": 4, "b": 3, "c": 2})   # gt
        R = Counter({"a": 4, "b": 3, "c": 2})   # parsing == gt → d_pars = 0
        S_star = Counter({"a": 4, "b": 2, "c": 1})  # ocr has some error
        S = Counter({"a": 4, "b": 2, "c": 1})   # total == ocr
        d = cdd_decomp({"gt": Q, "parsing": R, "ocr": S_star, "total": S})
        assert d.d_pars == pytest.approx(0.0)
        assert d.d_ocr is not None and d.d_ocr > 0.0
        assert d.d_int is not None
        assert d.d_total == pytest.approx(d.d_ocr, abs=1e-9)

    def test_counter_does_not_call_text_to_counter(self):
        # A Counter with non-string keys (e.g. from a custom tokeniser) should
        # pass straight through without triggering text_to_counter
        gt = Counter({1: 3, 2: 2})
        total = Counter({1: 3, 2: 2})
        d = cdd_decomp({"gt": gt, "total": total})
        assert d.d_total == pytest.approx(0.0)

    def test_jsd_distance_is_default(self):
        # Explicit metric=jsd_distance should match omitting the argument
        d_default = cdd_decomp({"gt": "abcde", "total": "aabcd"})
        d_explicit = cdd_decomp({"gt": "abcde", "total": "aabcd"}, metric=jsd_distance)
        assert d_default.d_total == pytest.approx(d_explicit.d_total, abs=1e-12)


# =============================================================================
# spacer_decomp
# =============================================================================


class TestSpACERDecomp:
    def test_all_keys_perfect_match(self):
        d = spacer_decomp({"gt": "abc", "parsing": "abc", "ocr": "abc", "total": "abc"})
        assert isinstance(d, SpACERDecomposition)
        assert d.d_total_macro == pytest.approx(0.0)
        assert d.d_total_micro == pytest.approx(0.0)

    def test_missing_parsing_gives_none(self):
        d = spacer_decomp({"gt": "abc", "ocr": "abc", "total": "abc"})
        assert d.d_pars_macro is None
        assert d.d_pars_micro is None
        assert d.d_int_macro is None
        assert d.d_int_micro is None
        assert d.d_ocr_macro is not None
        assert d.d_total_macro is not None

    def test_missing_total_gives_none(self):
        d = spacer_decomp({"gt": "abc", "parsing": "abc", "ocr": "abc"})
        assert d.d_int_macro is None
        assert d.d_total_macro is None

    def test_micro_macro_same_for_single_string(self):
        d = spacer_decomp({"gt": "aaaa", "total": "aa"})
        assert d.d_total_macro == pytest.approx(d.d_total_micro)

    def test_micro_differs_from_macro_with_boxes(self):
        # Box 0: gt=aaa, total=a  (deletion 2)
        # Box 1: gt=a,   total=aaa (insertion 2, masks macro)
        d = spacer_decomp({"gt": ["aaa", "a"], "total": ["a", "aaa"]})
        assert d.d_total_macro == pytest.approx(0.0, abs=1e-9)
        assert d.d_total_micro is not None
        assert d.d_total_micro > 0.0

    def test_word_mode(self):
        d = spacer_decomp({"gt": "hello world", "total": "hello world"}, mode="word")
        assert d.d_total_macro == pytest.approx(0.0)

    def test_known_value_total_macro(self):
        # gt="aaaa" (C=4), total="aa" → E_hat=2, D_macro=2
        # SpACER = (2+2)/(2*4) = 0.5
        d = spacer_decomp({"gt": "aaaa", "total": "aa"})
        assert d.d_total_macro == pytest.approx(0.5)


# =============================================================================
# RegionChars / RegionPixels
# =============================================================================


class TestRegionChars:
    def test_valid_construction(self):
        rc = RegionChars(
            tokens=np.array(["a", "b", "c"]),
            xs=np.array([10, 20, 30]),
            ys=np.array([5, 15, 25]),
            region_ids=np.array([1, 1, 2]),
        )
        assert len(rc.tokens) == 3

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            RegionChars(
                tokens=np.array(["a", "b"]),
                xs=np.array([10]),
                ys=np.array([5, 15]),
                region_ids=np.array([1, 1]),
            )

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            RegionChars(
                tokens=np.array([["a", "b"]]),
                xs=np.array([10, 20]),
                ys=np.array([5, 15]),
                region_ids=np.array([1, 1]),
            )


class TestRegionPixels:
    def test_valid_construction(self):
        rp = RegionPixels(
            region_ids=np.array([0, 0, 1]),
            xs=np.array([10, 11, 20]),
            ys=np.array([5, 5, 15]),
        )
        assert len(rp.region_ids) == 3

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            RegionPixels(
                region_ids=np.array([0, 1]),
                xs=np.array([10]),
                ys=np.array([5, 15]),
            )


# =============================================================================
# build_R_from_region_pixels
# =============================================================================


def _make_gt_chars(chars_xy_rid):
    """Helper: list of (char, x, y, region_id) -> RegionChars."""
    tokens = np.array([r[0] for r in chars_xy_rid])
    xs = np.array([r[1] for r in chars_xy_rid])
    ys = np.array([r[2] for r in chars_xy_rid])
    rids = np.array([r[3] for r in chars_xy_rid])
    return RegionChars(tokens=tokens, xs=xs, ys=ys, region_ids=rids)


def _make_pred_pixels(rid_xy):
    """Helper: list of (region_id, x, y) -> RegionPixels."""
    rids = np.array([r[0] for r in rid_xy])
    xs = np.array([r[1] for r in rid_xy])
    ys = np.array([r[2] for r in rid_xy])
    return RegionPixels(region_ids=rids, xs=xs, ys=ys)


class TestBuildRFromRegionPixels:
    def test_perfect_coverage(self):
        # All GT chars fall in one predicted region.
        gt = _make_gt_chars([("a", 10, 5, 1), ("b", 20, 5, 1), ("a", 30, 5, 1)])
        pred = _make_pred_pixels([(0, 10, 5), (0, 20, 5), (0, 30, 5)])
        agg, per_region = build_R_from_region_pixels(gt, pred)
        assert agg == Counter({"a": 2, "b": 1})
        assert per_region[0] == Counter({"a": 2, "b": 1})

    def test_missed_chars_absent(self):
        # One GT char not in any predicted pixel → not counted in R.
        gt = _make_gt_chars([("a", 10, 5, 1), ("b", 99, 99, 1)])
        pred = _make_pred_pixels([(0, 10, 5)])
        agg, per_region = build_R_from_region_pixels(gt, pred)
        assert agg == Counter({"a": 1})
        assert "b" not in agg

    def test_overlap_double_counts(self):
        # GT char at (10,5) covered by two pred regions → counted twice.
        gt = _make_gt_chars([("a", 10, 5, 1)])
        pred = _make_pred_pixels([(0, 10, 5), (1, 10, 5)])
        agg, per_region = build_R_from_region_pixels(gt, pred)
        assert agg["a"] == 2
        assert per_region[0]["a"] == 1
        assert per_region[1]["a"] == 1

    def test_empty_pred_pixels(self):
        gt = _make_gt_chars([("a", 10, 5, 1)])
        pred = RegionPixels(
            region_ids=np.array([], dtype=np.int64),
            xs=np.array([], dtype=np.int64),
            ys=np.array([], dtype=np.int64),
        )
        agg, per_region = build_R_from_region_pixels(gt, pred)
        assert agg == Counter()
        assert per_region == {}

    def test_empty_gt_chars(self):
        gt = RegionChars(
            tokens=np.array([]),
            xs=np.array([], dtype=np.int64),
            ys=np.array([], dtype=np.int64),
            region_ids=np.array([], dtype=np.int64),
        )
        pred = _make_pred_pixels([(0, 10, 5)])
        agg, per_region = build_R_from_region_pixels(gt, pred)
        assert agg == Counter()


# =============================================================================
# cdd_decomp_spatial
# =============================================================================


class TestCDDDecompSpatial:
    def _perfect_setup(self):
        gt = _make_gt_chars([("a", 0, 0, 1), ("b", 1, 0, 1), ("c", 2, 0, 2)])
        pred = _make_pred_pixels([(0, 0, 0), (0, 1, 0), (0, 2, 0)])
        ocr_gt = {1: "ab", 2: "c"}
        ocr_parse = {0: "abc"}
        return gt, pred, ocr_gt, ocr_parse

    def test_perfect_d_pars_zero(self):
        gt, pred, ocr_gt, ocr_parse = self._perfect_setup()
        d = cdd_decomp_spatial(gt, pred, ocr_gt, ocr_parse)
        assert isinstance(d, CDDDecomposition)
        assert d.d_pars == pytest.approx(0.0, abs=1e-9)

    def test_perfect_d_ocr_zero(self):
        gt, pred, ocr_gt, ocr_parse = self._perfect_setup()
        d = cdd_decomp_spatial(gt, pred, ocr_gt, ocr_parse)
        assert d.d_ocr == pytest.approx(0.0, abs=1e-9)

    def test_perfect_d_total_zero(self):
        gt, pred, ocr_gt, ocr_parse = self._perfect_setup()
        d = cdd_decomp_spatial(gt, pred, ocr_gt, ocr_parse)
        assert d.d_total == pytest.approx(0.0, abs=1e-9)

    def test_missing_char_raises_d_pars(self):
        # One GT char missed by parsing → R ≠ Q → d_pars > 0
        gt = _make_gt_chars([("a", 0, 0, 1), ("b", 99, 99, 1)])
        pred = _make_pred_pixels([(0, 0, 0)])
        d = cdd_decomp_spatial(gt, pred, {}, {})
        assert d.d_pars is not None and d.d_pars > 0.0

    def test_empty_ocr_gives_none_components(self):
        gt = _make_gt_chars([("a", 0, 0, 1)])
        pred = _make_pred_pixels([(0, 0, 0)])
        d = cdd_decomp_spatial(gt, pred, {}, {})
        # S* and S are empty Counters, so d_ocr and d_total are computed
        # (comparing Q to empty distributions) but not None.
        assert d.d_pars is not None

    def test_custom_metric(self):
        gt, pred, ocr_gt, ocr_parse = self._perfect_setup()
        d = cdd_decomp_spatial(gt, pred, ocr_gt, ocr_parse, metric=lambda p, q: 42.0)
        assert d.d_pars == pytest.approx(42.0)


# =============================================================================
# spacer_decomp_spatial
# =============================================================================


class TestSpACERDecompSpatial:
    def _perfect_setup(self):
        gt = _make_gt_chars([("a", 0, 0, 1), ("b", 1, 0, 1), ("c", 2, 0, 2)])
        pred = _make_pred_pixels([(0, 0, 0), (0, 1, 0), (0, 2, 0)])
        ocr_gt = {1: "ab", 2: "c"}
        ocr_parse = {0: "abc"}
        return gt, pred, ocr_gt, ocr_parse

    def test_returns_spacer_decomposition(self):
        gt, pred, ocr_gt, ocr_parse = self._perfect_setup()
        d = spacer_decomp_spatial(gt, pred, ocr_gt, ocr_parse)
        assert isinstance(d, SpACERDecomposition)

    def test_d_pars_micro_always_none(self):
        gt, pred, ocr_gt, ocr_parse = self._perfect_setup()
        d = spacer_decomp_spatial(gt, pred, ocr_gt, ocr_parse)
        assert d.d_pars_micro is None

    def test_perfect_d_ocr_zero(self):
        gt, pred, ocr_gt, ocr_parse = self._perfect_setup()
        d = spacer_decomp_spatial(gt, pred, ocr_gt, ocr_parse)
        assert d.d_ocr_macro == pytest.approx(0.0, abs=1e-9)
        assert d.d_ocr_micro == pytest.approx(0.0, abs=1e-9)

    def test_perfect_d_pars_macro_zero(self):
        gt, pred, ocr_gt, ocr_parse = self._perfect_setup()
        d = spacer_decomp_spatial(gt, pred, ocr_gt, ocr_parse)
        assert d.d_pars_macro == pytest.approx(0.0, abs=1e-9)

    def test_overlap_increases_r_counts(self):
        # GT char at (0,0) covered by two pred regions — R has double count.
        gt = _make_gt_chars([("a", 0, 0, 1)])
        pred = _make_pred_pixels([(0, 0, 0), (1, 0, 0)])
        d = spacer_decomp_spatial(gt, pred, {}, {0: "aa", 1: "a"})
        # R_agg has "a":2 (double counted), S_agg has "a":3.
        # d_pars compares R(a:2) vs Q(a:1): E_hat=1, D_macro=0
        # → SpACER = (0+1)/(2*1) = 0.5
        assert d.d_pars_macro == pytest.approx(0.5)

    def test_missing_region_in_ocr_gt_handled(self):
        # pred_gt_ocr has no entry for region 2 — should not raise.
        gt = _make_gt_chars([("a", 0, 0, 1), ("c", 2, 0, 2)])
        pred = _make_pred_pixels([(0, 0, 0), (0, 2, 0)])
        d = spacer_decomp_spatial(gt, pred, {1: "a"}, {0: "ac"})
        assert d.d_ocr_macro is not None

    def test_empty_ocr_inputs(self):
        gt = _make_gt_chars([("a", 0, 0, 1)])
        pred = _make_pred_pixels([(0, 0, 0)])
        d = spacer_decomp_spatial(gt, pred, {}, {})
        assert d.d_ocr_macro is None
        assert d.d_int_macro is None
        assert d.d_total_macro is None
        assert d.d_pars_macro is not None
