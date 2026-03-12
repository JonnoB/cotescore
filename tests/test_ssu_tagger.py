"""Tests for the SSU tagger module."""

import os
import tempfile
import xml.etree.ElementTree as ET

import pytest

from cotescore.ssu_tagger import SSUTagger, assign_ssu

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19"


def _make_xml(page_content: str) -> str:
    """Wrap a <Page> body in a minimal PAGE XML document."""
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<PcGts xmlns="{NS}">\n'
        f'  <Page imageFilename="test.tif" imageWidth="1000" imageHeight="1000">\n'
        f"{page_content}"
        f"  </Page>\n"
        f"</PcGts>\n"
    )


def _write_tmp(content: str) -> str:
    """Write XML content to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Fixtures — minimal PAGE XML strings
# ---------------------------------------------------------------------------

SINGLE_GROUP_WITH_HEADING = _make_xml(
    """\
<ReadingOrder>
  <UnorderedGroup id="ug1">
    <OrderedGroup id="g1">
      <RegionRefIndexed regionRef="r1" index="0"/>
      <RegionRefIndexed regionRef="r2" index="1"/>
      <RegionRefIndexed regionRef="r3" index="2"/>
    </OrderedGroup>
  </UnorderedGroup>
</ReadingOrder>
<TextRegion id="r1" type="paragraph"/>
<TextRegion id="r2" type="heading"/>
<TextRegion id="r3" type="paragraph"/>
"""
)

MULTI_GROUP = _make_xml(
    """\
<ReadingOrder>
  <UnorderedGroup id="ug1">
    <OrderedGroup id="g1">
      <RegionRefIndexed regionRef="r1" index="0"/>
      <RegionRefIndexed regionRef="r2" index="1"/>
    </OrderedGroup>
    <OrderedGroup id="g2">
      <RegionRefIndexed regionRef="r3" index="0"/>
      <RegionRefIndexed regionRef="r4" index="1"/>
    </OrderedGroup>
  </UnorderedGroup>
</ReadingOrder>
<TextRegion id="r1" type="paragraph"/>
<TextRegion id="r2" type="paragraph"/>
<TextRegion id="r3" type="heading"/>
<TextRegion id="r4" type="paragraph"/>
"""
)

UNGROUPED_REGIONS = _make_xml(
    """\
<ReadingOrder>
  <UnorderedGroup id="ug1">
    <OrderedGroup id="g1">
      <RegionRefIndexed regionRef="r1" index="0"/>
    </OrderedGroup>
  </UnorderedGroup>
</ReadingOrder>
<TextRegion id="r1" type="paragraph"/>
<TextRegion id="r2" type="page-number"/>
<TextRegion id="r3" type="header"/>
"""
)

CONSECUTIVE_HEADINGS = _make_xml(
    """\
<ReadingOrder>
  <UnorderedGroup id="ug1">
    <OrderedGroup id="g1">
      <RegionRefIndexed regionRef="r1" index="0"/>
      <RegionRefIndexed regionRef="r2" index="1"/>
      <RegionRefIndexed regionRef="r3" index="2"/>
    </OrderedGroup>
  </UnorderedGroup>
</ReadingOrder>
<TextRegion id="r1" type="heading"/>
<TextRegion id="r2" type="heading"/>
<TextRegion id="r3" type="paragraph"/>
"""
)

NO_TYPE_ATTRIBUTE = _make_xml(
    """\
<ReadingOrder>
  <UnorderedGroup id="ug1">
    <OrderedGroup id="g1">
      <RegionRefIndexed regionRef="r1" index="0"/>
      <RegionRefIndexed regionRef="r2" index="1"/>
    </OrderedGroup>
  </UnorderedGroup>
</ReadingOrder>
<TextRegion id="r1" type="paragraph"/>
<TextRegion id="r2"/>
"""
)

EMPTY_ORDERED_GROUP = _make_xml(
    """\
<ReadingOrder>
  <UnorderedGroup id="ug1">
    <OrderedGroup id="g1"/>
    <OrderedGroup id="g2">
      <RegionRefIndexed regionRef="r1" index="0"/>
    </OrderedGroup>
  </UnorderedGroup>
</ReadingOrder>
<TextRegion id="r1" type="paragraph"/>
"""
)

FLAT_ORDERED_GROUP = _make_xml(
    """\
<ReadingOrder>
  <OrderedGroup id="g1">
    <RegionRefIndexed regionRef="r1" index="0"/>
    <RegionRefIndexed regionRef="r2" index="1"/>
  </OrderedGroup>
</ReadingOrder>
<TextRegion id="r1" type="heading"/>
<TextRegion id="r2" type="paragraph"/>
"""
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleGroupWithHeading:
    """One OrderedGroup: paragraph, heading, paragraph."""

    def setup_method(self):
        self.path = _write_tmp(SINGLE_GROUP_WITH_HEADING)
        self.result = assign_ssu(self.path)

    def teardown_method(self):
        os.unlink(self.path)

    def test_two_ssus_produced(self):
        assert len(self.result["ssu_to_regions"]) == 2

    def test_first_paragraph_in_first_ssu(self):
        # r1 (paragraph at index 0) starts ssu_1_g1
        assert self.result["region_to_ssu"]["r1"] == "ssu_1_g1"

    def test_heading_starts_new_ssu(self):
        # r2 (heading) increments semantic_id → ssu_2_g1
        assert self.result["region_to_ssu"]["r2"] == "ssu_2_g1"

    def test_paragraph_after_heading_inherits_ssu(self):
        assert self.result["region_to_ssu"]["r3"] == "ssu_2_g1"

    def test_ssu_to_regions_mapping(self):
        assert self.result["ssu_to_regions"]["ssu_1_g1"] == ["r1"]
        assert self.result["ssu_to_regions"]["ssu_2_g1"] == ["r2", "r3"]


class TestMultiGroupSemanticIdsIncrement:
    """Two OrderedGroups — semantic IDs must be global (not reset per group)."""

    def setup_method(self):
        self.path = _write_tmp(MULTI_GROUP)
        self.result = assign_ssu(self.path)

    def teardown_method(self):
        os.unlink(self.path)

    def test_two_ssus_total(self):
        # g1 first region → semantic_id=1 (ssu_1_g1 for r1, r2)
        # g2 first region (heading r3) → semantic_id=2 (ssu_2_g2 for r3, r4)
        assert len(self.result["ssu_to_regions"]) == 2

    def test_g1_regions_share_ssu(self):
        assert self.result["region_to_ssu"]["r1"] == "ssu_1_g1"
        assert self.result["region_to_ssu"]["r2"] == "ssu_1_g1"

    def test_g2_semantic_id_continues(self):
        # g2 first region always bumps semantic_id regardless of type
        ssu_r3 = self.result["region_to_ssu"]["r3"]
        assert ssu_r3 == "ssu_2_g2"

    def test_g2_paragraph_inherits_heading_ssu(self):
        assert self.result["region_to_ssu"]["r4"] == "ssu_2_g2"


class TestUngroupedRegions:
    """TextRegions not in any OrderedGroup each get their own ssu_N_ungrouped."""

    def setup_method(self):
        self.path = _write_tmp(UNGROUPED_REGIONS)
        self.result = assign_ssu(self.path)

    def teardown_method(self):
        os.unlink(self.path)

    def test_three_ssus_total(self):
        # 1 grouped → ssu_1_g1; 2 ungrouped → ssu_2_ungrouped, ssu_3_ungrouped
        assert len(self.result["ssu_to_regions"]) == 3

    def test_grouped_region_has_group_id(self):
        assert self.result["region_to_ssu"]["r1"] == "ssu_1_g1"

    def test_ungrouped_regions_get_ungrouped_suffix(self):
        for rid in ("r2", "r3"):
            ssu_id = self.result["region_to_ssu"][rid]
            assert ssu_id.endswith("_ungrouped"), f"{rid} → {ssu_id}"

    def test_ungrouped_regions_are_distinct_ssus(self):
        assert self.result["region_to_ssu"]["r2"] != self.result["region_to_ssu"]["r3"]

    def test_each_ungrouped_ssu_has_one_region(self):
        for rid in ("r2", "r3"):
            ssu_id = self.result["region_to_ssu"][rid]
            assert self.result["ssu_to_regions"][ssu_id] == [rid]


class TestConsecutiveHeadings:
    """Two consecutive headings each produce their own single-region SSU."""

    def setup_method(self):
        self.path = _write_tmp(CONSECUTIVE_HEADINGS)
        self.result = assign_ssu(self.path)

    def teardown_method(self):
        os.unlink(self.path)

    def test_two_ssus(self):
        # r1 (heading, index 0) → ssu_1_g1
        # r2 (heading, index 1) → ssu_2_g1
        # r3 (paragraph, index 2) → ssu_2_g1
        assert len(self.result["ssu_to_regions"]) == 2

    def test_first_heading_own_ssu(self):
        assert self.result["region_to_ssu"]["r1"] == "ssu_1_g1"
        assert self.result["ssu_to_regions"]["ssu_1_g1"] == ["r1"]

    def test_second_heading_new_ssu(self):
        assert self.result["region_to_ssu"]["r2"] == "ssu_2_g1"

    def test_paragraph_after_second_heading_inherits(self):
        assert self.result["region_to_ssu"]["r3"] == "ssu_2_g1"


class TestNoTypeAttribute:
    """TextRegion with no type attribute is treated as paragraph."""

    def setup_method(self):
        self.path = _write_tmp(NO_TYPE_ATTRIBUTE)
        self.result = assign_ssu(self.path)

    def teardown_method(self):
        os.unlink(self.path)

    def test_one_ssu_only(self):
        # r1 (paragraph) → ssu_1_g1; r2 (no type → paragraph) → same ssu_1_g1
        assert len(self.result["ssu_to_regions"]) == 1

    def test_both_regions_same_ssu(self):
        assert self.result["region_to_ssu"]["r1"] == self.result["region_to_ssu"]["r2"]

    def test_region_type_recorded_as_paragraph(self):
        ssu_id = self.result["region_to_ssu"]["r2"]
        types = self.result["ssu_metadata"][ssu_id]["region_types"]
        assert types[1] == "paragraph"


class TestEmptyOrderedGroup:
    """Empty OrderedGroup is skipped without error."""

    def setup_method(self):
        self.path = _write_tmp(EMPTY_ORDERED_GROUP)
        self.result = assign_ssu(self.path)

    def teardown_method(self):
        os.unlink(self.path)

    def test_one_ssu_from_non_empty_group(self):
        assert len(self.result["ssu_to_regions"]) == 1
        assert "r1" in self.result["region_to_ssu"]


class TestOutputXmlCustomAttribute:
    """When output_path is provided, custom attribute is written to each TextRegion."""

    def setup_method(self):
        self.src = _write_tmp(SINGLE_GROUP_WITH_HEADING)
        self.out = self.src + ".out.xml"
        self.result = assign_ssu(self.src, output_path=self.out)

    def teardown_method(self):
        os.unlink(self.src)
        if os.path.exists(self.out):
            os.unlink(self.out)

    def test_output_file_created(self):
        assert os.path.exists(self.out)

    def test_custom_attribute_present_on_all_regions(self):
        tree = ET.parse(self.out)
        root = tree.getroot()
        ns_uri = root.tag[1 : root.tag.index("}")]
        ns = {"page": ns_uri}
        page = root.find("page:Page", ns)
        for region in page.findall("page:TextRegion", ns):
            custom = region.get("custom", "")
            assert "type:ssu" in custom, f"Region {region.get('id')} missing SSU custom attr"

    def test_custom_attribute_contains_ssu_id(self):
        tree = ET.parse(self.out)
        root = tree.getroot()
        ns_uri = root.tag[1 : root.tag.index("}")]
        ns = {"page": ns_uri}
        page = root.find("page:Page", ns)
        r1 = page.find("page:TextRegion[@id='r1']", ns)
        assert "ssu_1_g1" in r1.get("custom", "")

    def test_source_file_not_modified(self):
        tree = ET.parse(self.src)
        root = tree.getroot()
        ns_uri = root.tag[1 : root.tag.index("}")]
        ns = {"page": ns_uri}
        page = root.find("page:Page", ns)
        for region in page.findall("page:TextRegion", ns):
            assert region.get("custom") is None, "Source file should not be modified"


class TestFlatOrderedGroupStructure:
    """ReadingOrder > OrderedGroup directly (no UnorderedGroup wrapper)."""

    def setup_method(self):
        self.path = _write_tmp(FLAT_ORDERED_GROUP)
        self.result = assign_ssu(self.path)

    def teardown_method(self):
        os.unlink(self.path)

    def test_two_ssus(self):
        # r1 (heading, index 0) → ssu_1_g1
        # r2 (paragraph, index 1) → ssu_2_g1 ... wait:
        # index 0 always starts new semantic unit regardless of type → ssu_1_g1
        # r2 (paragraph) → no heading → stays in ssu_1_g1
        # Actually: r1 is heading at index 0 → semantic_id bumps to 1 → ssu_1_g1
        # r2 is paragraph (i>0, not heading) → stays ssu_1_g1
        assert len(self.result["ssu_to_regions"]) == 1

    def test_both_regions_in_same_ssu(self):
        assert self.result["region_to_ssu"]["r1"] == "ssu_1_g1"
        assert self.result["region_to_ssu"]["r2"] == "ssu_1_g1"

    def test_metadata_structural_group_id(self):
        meta = self.result["ssu_metadata"]["ssu_1_g1"]
        assert meta["structural_group_id"] == "g1"
