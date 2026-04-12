"""Tests for ALTOSSUTagger."""

import os
import tempfile
import xml.etree.ElementTree as ET

import pytest

from cotescore.alto_ssu_tagger import ALTOSSUTagger, assign_alto_ssu

ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"


# ---------------------------------------------------------------------------
# XML construction helpers
# ---------------------------------------------------------------------------


def _make_alto(print_space_content: str) -> str:
    """Wrap PrintSpace content in a minimal valid ALTO XML document."""
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<alto xmlns="{ALTO_NS}">\n'
        f"  <Layout>\n"
        f'    <Page HEIGHT="3508" WIDTH="2479">\n'
        f'      <PrintSpace HEIGHT="3508" WIDTH="2479" VPOS="0" HPOS="0">\n'
        f"{print_space_content}"
        f"      </PrintSpace>\n"
        f"    </Page>\n"
        f"  </Layout>\n"
        f"</alto>\n"
    )


def _block(bid, hpos, vpos, width, height, lines=""):
    return (
        f'        <TextBlock ID="{bid}" HPOS="{hpos}" VPOS="{vpos}"'
        f' WIDTH="{width}" HEIGHT="{height}">\n'
        f"{lines}"
        f"        </TextBlock>\n"
    )


def _line(lid, hpos, vpos, width, height, strings=""):
    return (
        f'          <TextLine ID="{lid}" HPOS="{hpos}" VPOS="{vpos}"'
        f' WIDTH="{width}" HEIGHT="{height}">\n'
        f"{strings}"
        f"          </TextLine>\n"
    )


def _string(content, height=45):
    return f'            <String CONTENT="{content}" HEIGHT="{height}" WIDTH="80" HPOS="0" VPOS="0"/>\n'


def _write_tmp(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return f.name


def _make_line_dict(lid, hpos, vpos, width, height, strings):
    """Build a line dict in the format produced by _extract_blocks."""
    return {
        "id": lid,
        "hpos": hpos,
        "vpos": vpos,
        "width": width,
        "height": height,
        "bbox": (hpos, vpos, hpos + width, vpos + height),
        "strings": strings,
    }


def _make_block_dict(bid, hpos, vpos, width, height, lines=None):
    return {
        "id": bid,
        "hpos": hpos,
        "vpos": vpos,
        "width": width,
        "height": height,
        "bbox": (hpos, vpos, hpos + width, vpos + height),
        "lines": lines or [],
    }


# ---------------------------------------------------------------------------
# TestHeadingDetectionRules
# ---------------------------------------------------------------------------


class TestHeadingDetectionRules:
    """Unit tests for _detect_heading — each rule tested in isolation."""

    def setup_method(self):
        self.tagger = ALTOSSUTagger.__new__(ALTOSSUTagger)
        self.tagger.min_heading_score = 2
        # A block wide enough to support centering tests
        self.block = _make_block_dict("b1", hpos=100, vpos=100, width=800, height=600)
        self.page_median = 45.0

    def _score(self, line, block=None, n_block_lines=3, page_median=None):
        b = block if block is not None else self.block
        pm = page_median if page_median is not None else self.page_median
        _, score = self.tagger._detect_heading(line, b, n_block_lines, pm)
        return score

    def test_r1_all_caps_scores(self):
        line = _make_line_dict(
            "l1", 300, 200, 400, 50,
            [{"content": "HEADING", "height": 45}, {"content": "TEXT", "height": 45}],
        )
        assert self._score(line) >= 1

    def test_r1_mixed_case_no_score(self):
        line = _make_line_dict(
            "l1", 100, 200, 800, 50,
            [{"content": "This", "height": 45}, {"content": "is", "height": 45},
             {"content": "body", "height": 45}],
        )
        assert self._score(line) == 0

    def test_r2_centered_line_scores(self):
        # Block: HPOS=100, WIDTH=800 → right edge at 900
        # Line: HPOS=300, WIDTH=400 → left_margin=200, right_margin=200
        line = _make_line_dict(
            "l1", 300, 200, 400, 50,
            [{"content": "text", "height": 45}],
        )
        assert self._score(line) >= 1

    def test_r2_flush_left_no_score(self):
        # left_margin = 0, so R2 fails
        line = _make_line_dict(
            "l1", 100, 200, 600, 50,
            [{"content": "text", "height": 45}],
        )
        assert self._score(line) == 0

    def test_r2_asymmetric_no_score(self):
        # left_margin=50, right_margin=150 → smaller/larger = 50/150 ≈ 0.33 < 0.6
        # Use WIDTH=600 so R4 (narrow) doesn't also fire (600/800 = 0.75, not < 0.75)
        line = _make_line_dict(
            "l1", 150, 200, 600, 50,
            [{"content": "text", "height": 45}],
        )
        assert self._score(line) == 0

    def test_r3_standalone_block_scores(self):
        # Block with only 1 line → R3 fires
        line = _make_line_dict(
            "l1", 100, 200, 800, 50,
            [{"content": "text", "height": 45}],
        )
        assert self._score(line, n_block_lines=1) >= 1

    def test_r3_multi_line_block_no_score(self):
        line = _make_line_dict(
            "l1", 100, 200, 800, 50,
            [{"content": "text", "height": 45}],
        )
        assert self._score(line, n_block_lines=5) == 0

    def test_r4_narrow_line_scores(self):
        # line_width=300, block_width=800 → 0.375 < 0.75
        line = _make_line_dict(
            "l1", 100, 200, 300, 50,
            [{"content": "text", "height": 45}],
        )
        assert self._score(line) >= 1

    def test_r4_full_width_line_no_score(self):
        # line_width=800, block_width=800 → 1.0 ≥ 0.75
        line = _make_line_dict(
            "l1", 100, 200, 800, 50,
            [{"content": "text", "height": 45}],
        )
        assert self._score(line) == 0

    def test_r5_tall_text_scores(self):
        # string height=60, page_median=45 → 60 > 45*1.2=54
        line = _make_line_dict(
            "l1", 100, 200, 400, 65,
            [{"content": "TEXT", "height": 60}],
        )
        assert self._score(line, page_median=45.0) >= 1

    def test_r5_normal_height_no_score(self):
        # string height=45, page_median=45 → 45 ≤ 54
        # Use full block width so R4 (narrow) doesn't also fire
        line = _make_line_dict(
            "l1", 100, 200, 800, 50,
            [{"content": "text", "height": 45}],
        )
        assert self._score(line, page_median=45.0) == 0

    def test_is_heading_when_score_meets_threshold(self):
        # R1 (all caps) + R4 (narrow) → score=2, default min=2 → heading
        line = _make_line_dict(
            "l1", 100, 200, 300, 50,
            [{"content": "HEADING", "height": 45}],
        )
        is_heading, score = self.tagger._detect_heading(line, self.block, 3, self.page_median)
        assert score >= 2
        assert is_heading is True

    def test_is_not_heading_below_threshold(self):
        # body text: score = 0
        line = _make_line_dict(
            "l1", 100, 200, 800, 50,
            [{"content": "body text", "height": 45}],
        )
        is_heading, _ = self.tagger._detect_heading(line, self.block, 3, self.page_median)
        assert is_heading is False


# ---------------------------------------------------------------------------
# TestStructuralBinning
# ---------------------------------------------------------------------------


class TestStructuralBinning:
    """Lines in a two-column layout receive distinct col_k structural units."""

    def test_two_column_assignment(self):
        # Two blocks in left column (~HPOS=100), two in right column (~HPOS=550)
        xml = _make_alto(
            _block("b1", 100, 200, 350, 200,
                   _line("l1", 100, 200, 350, 45, _string("body")))
            + _block("b2", 100, 450, 350, 200,
                     _line("l2", 100, 450, 350, 45, _string("body")))
            + _block("b3", 550, 200, 350, 200,
                     _line("l3", 550, 200, 350, 45, _string("body")))
            + _block("b4", 550, 450, 350, 200,
                     _line("l4", 550, 450, 350, 45, _string("body")))
        )
        path = _write_tmp(xml)
        try:
            result = assign_alto_ssu(path)
            col_l1 = result["ssu_metadata"][result["line_to_ssu"]["l1"]]["structural_unit_id"]
            col_l3 = result["ssu_metadata"][result["line_to_ssu"]["l3"]]["structural_unit_id"]
            assert col_l1 != col_l3
            assert col_l1.startswith("col_")
            assert col_l3.startswith("col_")
        finally:
            os.unlink(path)

    def test_same_column_same_structural_unit(self):
        xml = _make_alto(
            _block("b1", 100, 200, 350, 200,
                   _line("l1", 100, 200, 350, 45, _string("body")))
            + _block("b2", 100, 450, 350, 200,
                     _line("l2", 100, 450, 350, 45, _string("body")))
            + _block("b3", 550, 200, 350, 200,
                     _line("l3", 550, 200, 350, 45, _string("body")))
            + _block("b4", 550, 450, 350, 200,
                     _line("l4", 550, 450, 350, 45, _string("body")))
        )
        path = _write_tmp(xml)
        try:
            result = assign_alto_ssu(path)
            col_l1 = result["ssu_metadata"][result["line_to_ssu"]["l1"]]["structural_unit_id"]
            col_l2 = result["ssu_metadata"][result["line_to_ssu"]["l2"]]["structural_unit_id"]
            assert col_l1 == col_l2
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestSemanticSplitMidBlock
# ---------------------------------------------------------------------------


class TestSemanticSplitMidBlock:
    """A heading TextLine mid-block splits the block into two SSUs."""

    def _make_xml(self):
        # Block with 3 lines: body, heading (all-caps + narrow), body
        # Block: HPOS=100, WIDTH=800 → body lines fill width, heading is narrow
        lines = (
            _line("l1", 100, 200, 800, 45, _string("body text here"))
            + _line("l2", 300, 260, 300, 45, _string("ARTICLE HEADING"))
            + _line("l3", 100, 320, 800, 45, _string("more body text"))
        )
        return _make_alto(_block("b1", 100, 100, 800, 300, lines))

    def test_heading_creates_new_ssu(self):
        path = _write_tmp(self._make_xml())
        try:
            result = assign_alto_ssu(path)
            ssu_l1 = result["line_to_ssu"]["l1"]
            ssu_l2 = result["line_to_ssu"]["l2"]
            ssu_l3 = result["line_to_ssu"]["l3"]
            # heading and body-before-heading are in different SSUs
            assert ssu_l1 != ssu_l2
            # heading and body-after-heading share an SSU
            assert ssu_l2 == ssu_l3
        finally:
            os.unlink(path)

    def test_heading_line_is_marked(self):
        path = _write_tmp(self._make_xml())
        try:
            result = assign_alto_ssu(path)
            assert result["line_metadata"]["l1"]["is_heading"] is False
            assert result["line_metadata"]["l2"]["is_heading"] is True
            assert result["line_metadata"]["l3"]["is_heading"] is False
        finally:
            os.unlink(path)

    def test_semantic_id_increments(self):
        path = _write_tmp(self._make_xml())
        try:
            result = assign_alto_ssu(path)
            sem_l1 = result["ssu_metadata"][result["line_to_ssu"]["l1"]]["semantic_id"]
            sem_l2 = result["ssu_metadata"][result["line_to_ssu"]["l2"]]["semantic_id"]
            assert sem_l2 == sem_l1 + 1
        finally:
            os.unlink(path)

    def test_same_block_different_ssu(self):
        path = _write_tmp(self._make_xml())
        try:
            result = assign_alto_ssu(path)
            # Both lines share the block but are in different SSUs
            assert result["line_metadata"]["l1"]["block_id"] == "b1"
            assert result["line_metadata"]["l2"]["block_id"] == "b1"
            assert result["line_to_ssu"]["l1"] != result["line_to_ssu"]["l2"]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestNoHeadings
# ---------------------------------------------------------------------------


class TestNoHeadings:
    """When no headings are detected all lines share semantic_id=1."""

    def test_all_lines_semantic_id_one(self):
        xml = _make_alto(
            _block("b1", 100, 100, 800, 200,
                   _line("l1", 100, 100, 800, 45, _string("body text"))
                   + _line("l2", 100, 160, 800, 45, _string("more body")))
        )
        path = _write_tmp(xml)
        try:
            result = assign_alto_ssu(path)
            sem_ids = {
                result["ssu_metadata"][ssu_id]["semantic_id"]
                for ssu_id in result["ssu_metadata"]
            }
            assert sem_ids == {1}
        finally:
            os.unlink(path)

    def test_single_ssu_for_single_column(self):
        xml = _make_alto(
            _block("b1", 100, 100, 800, 200,
                   _line("l1", 100, 100, 800, 45, _string("body text"))
                   + _line("l2", 100, 160, 800, 45, _string("more body")))
        )
        path = _write_tmp(xml)
        try:
            result = assign_alto_ssu(path)
            assert len(result["ssu_to_lines"]) == 1
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestMultipleHeadings
# ---------------------------------------------------------------------------


class TestMultipleHeadings:
    """Each heading increments the semantic_id by 1."""

    def test_three_articles(self):
        # heading → body → heading → body → heading → body
        lines = (
            _line("l1", 250, 100, 300, 45, _string("FIRST ARTICLE"))
            + _line("l2", 100, 160, 800, 45, _string("first body"))
            + _line("l3", 250, 220, 300, 45, _string("SECOND ARTICLE"))
            + _line("l4", 100, 280, 800, 45, _string("second body"))
            + _line("l5", 250, 340, 300, 45, _string("THIRD ARTICLE"))
            + _line("l6", 100, 400, 800, 45, _string("third body"))
        )
        xml = _make_alto(_block("b1", 100, 100, 800, 400, lines))
        path = _write_tmp(xml)
        try:
            result = assign_alto_ssu(path)
            sem = {
                lid: result["ssu_metadata"][result["line_to_ssu"][lid]]["semantic_id"]
                for lid in ["l1", "l2", "l3", "l4", "l5", "l6"]
            }
            # l1 heading → sem=1, l2 body → sem=1
            assert sem["l1"] == sem["l2"]
            # l3 heading increments to 2
            assert sem["l3"] == sem["l1"] + 1
            assert sem["l3"] == sem["l4"]
            # l5 heading increments to 3
            assert sem["l5"] == sem["l3"] + 1
            assert sem["l5"] == sem["l6"]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestStandaloneHeadingBlock
# ---------------------------------------------------------------------------


class TestStandaloneHeadingBlock:
    """A single-line TextBlock with all-caps content scores >= 2."""

    def test_standalone_heading_block_detected(self):
        # 1-line block: R3 (standalone) + R1 (caps) → score=2 → heading
        xml = _make_alto(
            _block("header", 200, 100, 600, 50,
                   _line("l_hdr", 200, 100, 600, 45, _string("ALL CAPS TITLE")))
            + _block("body", 100, 200, 800, 200,
                     _line("l1", 100, 200, 800, 45, _string("body text"))
                     + _line("l2", 100, 260, 800, 45, _string("more text")))
        )
        path = _write_tmp(xml)
        try:
            result = assign_alto_ssu(path)
            assert result["line_metadata"]["l_hdr"]["is_heading"] is True
        finally:
            os.unlink(path)

    def test_standalone_heading_starts_new_semantic_unit(self):
        xml = _make_alto(
            _block("header", 200, 100, 600, 50,
                   _line("l_hdr", 200, 100, 600, 45, _string("ALL CAPS TITLE")))
            + _block("body", 100, 200, 800, 200,
                     _line("l1", 100, 200, 800, 45, _string("body text"))
                     + _line("l2", 100, 260, 800, 45, _string("more text")))
        )
        path = _write_tmp(xml)
        try:
            result = assign_alto_ssu(path)
            sem_hdr = result["ssu_metadata"][result["line_to_ssu"]["l_hdr"]]["semantic_id"]
            sem_l1 = result["ssu_metadata"][result["line_to_ssu"]["l1"]]["semantic_id"]
            # heading and body share semantic unit (heading opens it)
            assert sem_hdr == sem_l1
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestBlockMembership
# ---------------------------------------------------------------------------


class TestBlockMembership:
    """line_metadata carries the correct block_id for each line."""

    def test_line_metadata_block_ids(self):
        xml = _make_alto(
            _block("b1", 100, 100, 350, 200,
                   _line("l1", 100, 100, 350, 45, _string("body"))
                   + _line("l2", 100, 160, 350, 45, _string("body")))
            + _block("b2", 550, 100, 350, 200,
                     _line("l3", 550, 100, 350, 45, _string("body")))
        )
        path = _write_tmp(xml)
        try:
            result = assign_alto_ssu(path)
            assert result["line_metadata"]["l1"]["block_id"] == "b1"
            assert result["line_metadata"]["l2"]["block_id"] == "b1"
            assert result["line_metadata"]["l3"]["block_id"] == "b2"
        finally:
            os.unlink(path)

    def test_line_metadata_bbox(self):
        xml = _make_alto(
            _block("b1", 100, 100, 800, 200,
                   _line("l1", 120, 110, 760, 45, _string("body")))
        )
        path = _write_tmp(xml)
        try:
            result = assign_alto_ssu(path)
            bbox = result["line_metadata"]["l1"]["bbox"]
            assert bbox == (120, 110, 760, 45)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestXmlOutput
# ---------------------------------------------------------------------------


class TestXmlOutput:
    """SSU attribute is written to TextLine elements when output_path is given."""

    def test_ssu_attribute_written(self):
        xml = _make_alto(
            _block("b1", 100, 100, 800, 100,
                   _line("l1", 100, 100, 800, 45, _string("body text")))
        )
        src = _write_tmp(xml)
        out = _write_tmp("")
        try:
            result = assign_alto_ssu(src, output_path=out)
            tree = ET.parse(out)
            ns = {"alto": ALTO_NS}
            lines = tree.findall(".//alto:TextLine", ns)
            assert len(lines) == 1
            assert lines[0].get("SSU") == result["line_to_ssu"]["l1"]
        finally:
            os.unlink(src)
            os.unlink(out)

    def test_modify_in_place(self):
        xml = _make_alto(
            _block("b1", 100, 100, 800, 100,
                   _line("l1", 100, 100, 800, 45, _string("body")))
        )
        path = _write_tmp(xml)
        try:
            result = assign_alto_ssu(path, modify_in_place=True)
            tree = ET.parse(path)
            ns = {"alto": ALTO_NS}
            line = tree.find(".//alto:TextLine", ns)
            assert line.get("SSU") == result["line_to_ssu"]["l1"]
        finally:
            os.unlink(path)

    def test_no_output_path_leaves_file_unchanged(self):
        xml = _make_alto(
            _block("b1", 100, 100, 800, 100,
                   _line("l1", 100, 100, 800, 45, _string("body")))
        )
        path = _write_tmp(xml)
        original = open(path).read()
        try:
            assign_alto_ssu(path)
            assert open(path).read() == original
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestMinScoreArgument
# ---------------------------------------------------------------------------


class TestMinScoreArgument:
    """min_heading_score controls how many rules must fire for heading detection."""

    def _make_xml(self):
        # l0: body before heading (needed to show a split when heading is detected)
        # l1: heading — scores R1 (all caps) + R4 (narrow) = 2
        # l2: body after heading
        lines = (
            _line("l0", 100, 100, 800, 45, _string("body text before"))
            + _line("l1", 100, 160, 300, 45, _string("HEADING LINE"))
            + _line("l2", 100, 220, 800, 45, _string("body text here"))
        )
        return _make_alto(_block("b1", 100, 100, 800, 180, lines))

    def test_min_score_2_detects_heading(self):
        path = _write_tmp(self._make_xml())
        try:
            result = assign_alto_ssu(path, min_heading_score=2)
            assert result["line_metadata"]["l1"]["is_heading"] is True
        finally:
            os.unlink(path)

    def test_min_score_5_no_heading_detected(self):
        path = _write_tmp(self._make_xml())
        try:
            result = assign_alto_ssu(path, min_heading_score=5)
            assert result["line_metadata"]["l1"]["is_heading"] is False
            assert result["line_metadata"]["l2"]["is_heading"] is False
        finally:
            os.unlink(path)

    def test_min_score_affects_semantic_splitting(self):
        path = _write_tmp(self._make_xml())
        try:
            result_2 = assign_alto_ssu(path, min_heading_score=2)
            result_5 = assign_alto_ssu(path, min_heading_score=5)
            # At score 2, l1 is a heading → l0 and l1 are in different SSUs
            assert result_2["line_to_ssu"]["l0"] != result_2["line_to_ssu"]["l1"]
            # l1 (heading) and l2 (body after) share the same SSU
            assert result_2["line_to_ssu"]["l1"] == result_2["line_to_ssu"]["l2"]
            # At score 5, nothing is a heading → all lines share one SSU
            assert result_5["line_to_ssu"]["l0"] == result_5["line_to_ssu"]["l1"]
            assert result_5["line_to_ssu"]["l1"] == result_5["line_to_ssu"]["l2"]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestReturnShape
# ---------------------------------------------------------------------------


class TestReturnShape:
    """The return dict has all expected keys and consistent cross-references."""

    def test_return_keys(self):
        xml = _make_alto(
            _block("b1", 100, 100, 800, 100,
                   _line("l1", 100, 100, 800, 45, _string("body")))
        )
        path = _write_tmp(xml)
        try:
            result = assign_alto_ssu(path)
            assert set(result) == {"line_to_ssu", "ssu_to_lines", "ssu_metadata", "line_metadata"}
        finally:
            os.unlink(path)

    def test_line_to_ssu_and_ssu_to_lines_consistent(self):
        xml = _make_alto(
            _block("b1", 100, 100, 800, 200,
                   _line("l1", 100, 100, 800, 45, _string("body"))
                   + _line("l2", 100, 160, 800, 45, _string("more")))
        )
        path = _write_tmp(xml)
        try:
            result = assign_alto_ssu(path)
            for line_id, ssu_id in result["line_to_ssu"].items():
                assert line_id in result["ssu_to_lines"][ssu_id]
        finally:
            os.unlink(path)

    def test_ssu_metadata_structural_unit_id_present(self):
        xml = _make_alto(
            _block("b1", 100, 100, 800, 100,
                   _line("l1", 100, 100, 800, 45, _string("body")))
        )
        path = _write_tmp(xml)
        try:
            result = assign_alto_ssu(path)
            for meta in result["ssu_metadata"].values():
                assert "semantic_id" in meta
                assert "structural_unit_id" in meta
                assert "block_id" in meta
        finally:
            os.unlink(path)

    def test_missing_print_space_returns_empty(self):
        xml = (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<alto xmlns="{ALTO_NS}">\n'
            f"  <Layout>\n"
            f'    <Page HEIGHT="1000" WIDTH="1000"/>\n'
            f"  </Layout>\n"
            f"</alto>\n"
        )
        path = _write_tmp(xml)
        try:
            result = assign_alto_ssu(path)
            assert result["line_to_ssu"] == {}
            assert result["ssu_to_lines"] == {}
        finally:
            os.unlink(path)
