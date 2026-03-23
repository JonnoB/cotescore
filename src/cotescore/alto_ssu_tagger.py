"""
SSU Tagger for Transkribus ALTO XML files.

Assigns Semantic Structural Unit (SSU) identifiers to TextLines in ALTO XML.
An SSU is the intersection of a semantic unit (article/section, bounded by
detected headings) and a structural unit (column, inferred from TextBlock geometry).

Assignment is at TextLine level to provide tight bounding boxes for ground-truth use.
No reading order is assumed; TextBlocks are processed in document order.
"""

import logging
import statistics
import xml.etree.ElementTree as ET
from typing import Optional

logger = logging.getLogger(__name__)

_CAPS_RATIO_THRESHOLD = 0.85
_CENTERING_MIN_MARGIN_FRACTION = 0.05
_CENTERING_SYMMETRY_RATIO = 0.6
_STANDALONE_MAX_LINES = 2
_NARROW_LINE_RATIO = 0.75
_TALL_TEXT_FACTOR = 1.2


class ALTOSSUTagger:
    """Assigns SSU identifiers to TextLines in a Transkribus ALTO XML file."""

    def __init__(
        self,
        alto_xml_path: str,
        min_heading_score: int = 2,
        min_col_lines: int = 5,
    ) -> None:
        """Initialise the tagger for a single ALTO XML file.

        Args:
            alto_xml_path: Path to the ALTO XML file.
            min_heading_score: Number of heading rules (out of 5) a TextLine must
                satisfy to be classified as a heading. Defaults to 2.
            min_col_lines: Minimum number of TextLines a block must contain to be
                used when inferring column bins.  Short header/preamble blocks
                (typically 1-2 lines) are excluded so they do not distort column
                boundaries.  Falls back to all blocks when fewer than 2 qualify.
                Defaults to 5.
        """
        self.alto_xml_path = alto_xml_path
        self.min_heading_score = min_heading_score
        self.min_col_lines = min_col_lines

    # ------------------------------------------------------------------
    # Namespace helpers
    # ------------------------------------------------------------------

    def _detect_namespace(self, root: ET.Element) -> str:
        """Extract the ALTO namespace URI from the root element tag."""
        tag = root.tag
        if tag.startswith("{"):
            return tag[1: tag.index("}")]
        return ""

    def _q(self, tag: str, ns: dict) -> str:
        """Return a namespace-prefixed tag name when a namespace is present."""
        return f"alto:{tag}" if ns else tag

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_blocks(self, print_space: ET.Element, ns: dict) -> list[dict]:
        """Return a list of block dicts parsed from a PrintSpace element.

        Each block dict:
            {id, hpos, vpos, width, height, bbox: (x1,y1,x2,y2), lines: [...]}

        Each line dict within a block:
            {id, hpos, vpos, width, height, bbox: (x1,y1,x2,y2),
             strings: [{content: str, height: int}]}
        """
        blocks = []
        for block in print_space.findall(self._q("TextBlock", ns), ns):
            bid = block.get("ID", "")
            try:
                bh = int(block.get("HPOS", 0))
                bv = int(block.get("VPOS", 0))
                bw = int(block.get("WIDTH", 0))
                bht = int(block.get("HEIGHT", 0))
            except (ValueError, TypeError):
                continue

            lines = []
            for line in block.findall(self._q("TextLine", ns), ns):
                lid = line.get("ID", "")
                try:
                    lh = int(line.get("HPOS", 0))
                    lv = int(line.get("VPOS", 0))
                    lw = int(line.get("WIDTH", 0))
                    lht = int(line.get("HEIGHT", 0))
                except (ValueError, TypeError):
                    continue

                strings = []
                for s in line.findall(self._q("String", ns), ns):
                    content = s.get("CONTENT", "")
                    try:
                        sh = int(s.get("HEIGHT", 0))
                    except (ValueError, TypeError):
                        sh = 0
                    strings.append({"content": content, "height": sh})

                lines.append({
                    "id": lid,
                    "hpos": lh,
                    "vpos": lv,
                    "width": lw,
                    "height": lht,
                    "bbox": (lh, lv, lh + lw, lv + lht),
                    "strings": strings,
                })

            blocks.append({
                "id": bid,
                "hpos": bh,
                "vpos": bv,
                "width": bw,
                "height": bht,
                "bbox": (bh, bv, bh + bw, bv + bht),
                "lines": lines,
            })
        return blocks

    def _compute_page_median_string_height(self, blocks: list[dict]) -> float:
        """Return the median String HEIGHT across all String elements on the page."""
        heights = []
        for block in blocks:
            for line in block["lines"]:
                for s in line["strings"]:
                    if s["height"] > 0:
                        heights.append(s["height"])
        return statistics.median(heights) if heights else 0.0

    # ------------------------------------------------------------------
    # Heading detection
    # ------------------------------------------------------------------

    def _detect_heading(
        self,
        line: dict,
        block: dict,
        n_block_lines: int,
        page_median_height: float,
    ) -> tuple[bool, int]:
        """Score a TextLine on five heading heuristics.

        Rules:
            R1 caps_ratio:      >= 85% of alphabetic characters are uppercase.
            R2 centering:       line is horizontally centred within its block
                                (both margins > 5% of block width, and the
                                smaller margin is >= 60% of the larger).
            R3 standalone block: the parent TextBlock has <= 2 TextLines.
            R4 narrow line:     line width < 75% of block width.
            R5 tall text:       median String HEIGHT on the line exceeds the
                                page median by a factor of 1.2.

        Returns:
            (is_heading, score) where is_heading = score >= self.min_heading_score.
        """
        score = 0

        # R1: caps ratio
        all_text = "".join(s["content"] for s in line["strings"])
        alpha_chars = [c for c in all_text if c.isalpha()]
        if alpha_chars:
            caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if caps_ratio >= _CAPS_RATIO_THRESHOLD:
                score += 1

        # R2: centering
        block_width = block["width"]
        if block_width > 0:
            left_margin = line["hpos"] - block["hpos"]
            right_margin = (block["hpos"] + block_width) - (line["hpos"] + line["width"])
            min_margin = _CENTERING_MIN_MARGIN_FRACTION * block_width
            if left_margin > min_margin and right_margin > min_margin:
                larger = max(left_margin, right_margin)
                smaller = min(left_margin, right_margin)
                if larger > 0 and (smaller / larger) >= _CENTERING_SYMMETRY_RATIO:
                    score += 1

        # R3: standalone block
        if n_block_lines <= _STANDALONE_MAX_LINES:
            score += 1

        # R4: narrow line
        if block_width > 0 and (line["width"] / block_width) < _NARROW_LINE_RATIO:
            score += 1

        # R5: tall text
        string_heights = [s["height"] for s in line["strings"] if s["height"] > 0]
        if string_heights and page_median_height > 0:
            if statistics.median(string_heights) > page_median_height * _TALL_TEXT_FACTOR:
                score += 1

        return score >= self.min_heading_score, score

    # ------------------------------------------------------------------
    # Structural bin inference (geometry only; duplicated from SSUTagger
    # to avoid coupling between the two format-specific taggers)
    # ------------------------------------------------------------------

    def _infer_structural_bins(
        self,
        group_regions: list[dict],
        min_bin_members: int = 2,
        overlap_eps: float = 0.02,
        max_width_median_factor: float = 1.6,
    ) -> list[dict]:
        """Infer column-like structural bins from region bounding boxes.

        Returns list of {'bin_id': int, 'x1': float, 'x2': float, 'member_ids': [str]}.
        """
        candidates = []
        for r in group_regions:
            bbox = r.get("bbox")
            if bbox is None:
                continue
            if r.get("type", "paragraph") not in {"paragraph", "text"}:
                continue
            x1, _, x2, _ = bbox
            if x2 <= x1:
                continue
            candidates.append({"id": r["id"], "x1": x1, "x2": x2, "width": x2 - x1})

        if not candidates:
            return []

        widths = [r["width"] for r in candidates if r["width"] > 0]
        if widths:
            median_width = statistics.median(widths)
            width_cap = median_width * max_width_median_factor
            filtered = [r for r in candidates if r["width"] <= width_cap]
            if filtered:
                candidates = filtered

        median_width = statistics.median(widths) if widths else 100.0
        center_tolerance = max(30.0, median_width * 0.22)
        width_tolerance = max(40.0, median_width * 0.35)

        candidates.sort(key=lambda r: (r["x1"] + r["x2"]) / 2.0)
        clusters: list[dict] = []

        for region in candidates:
            rx1, rx2 = region["x1"], region["x2"]
            rcenter = (rx1 + rx2) / 2.0
            rwidth = region["width"]
            best_idx = None
            best_score = float("inf")

            for idx, c in enumerate(clusters):
                ccenter = statistics.median(c["center_values"])
                cwidth = statistics.median(c["width_values"])
                if (
                    abs(rcenter - ccenter) <= center_tolerance
                    and abs(rwidth - cwidth) <= width_tolerance
                ):
                    s = abs(rcenter - ccenter) + abs(rwidth - cwidth)
                    if s < best_score:
                        best_score = s
                        best_idx = idx

            if best_idx is None:
                clusters.append({
                    "member_ids": [region["id"]],
                    "x1_values": [rx1],
                    "x2_values": [rx2],
                    "center_values": [rcenter],
                    "width_values": [rwidth],
                })
            else:
                c = clusters[best_idx]
                c["member_ids"].append(region["id"])
                c["x1_values"].append(rx1)
                c["x2_values"].append(rx2)
                c["center_values"].append(rcenter)
                c["width_values"].append(rwidth)

        strong = [c for c in clusters if len(c["member_ids"]) >= min_bin_members]
        weak = [c for c in clusters if len(c["member_ids"]) < min_bin_members]

        if strong:
            for w in weak:
                wcenter = statistics.median(w["center_values"])
                nearest = min(
                    strong,
                    key=lambda c: abs(wcenter - statistics.median(c["center_values"])),
                )
                nearest["member_ids"].extend(w["member_ids"])
                nearest["x1_values"].extend(w["x1_values"])
                nearest["x2_values"].extend(w["x2_values"])
                nearest["center_values"].extend(w["center_values"])
                nearest["width_values"].extend(w["width_values"])
            clusters = strong

        bins = []
        for c in clusters:
            bins.append({
                "x1": statistics.median(c["x1_values"]),
                "x2": statistics.median(c["x2_values"]),
                "member_ids": c["member_ids"],
            })
        bins.sort(key=lambda b: b["x1"])
        for i, b in enumerate(bins, start=1):
            b["bin_id"] = i
        return bins

    def _assign_structural_unit(
        self,
        region: dict,
        bins: list[dict],
        width_overlap_threshold: float = 0.1,
        fallback_body_types: tuple[str, ...] = ("paragraph", "text"),
    ) -> str:
        """Return the structural unit token for a region.

        Returns col_<k> when the region overlaps exactly one bin,
        or span_<id> when it overlaps multiple bins or none.
        """
        bbox = region.get("bbox")
        rid = region.get("id", "")
        if bbox is None or not bins:
            return ""

        x1, _, x2, _ = bbox
        width = max(1e-6, x2 - x1)

        hit_bins = []
        for b in bins:
            inter = max(0.0, min(x2, b["x2"]) - max(x1, b["x1"]))
            if (inter / width) >= width_overlap_threshold:
                hit_bins.append(b["bin_id"])

        if len(hit_bins) == 1:
            return f"col_{hit_bins[0]}"
        if len(hit_bins) == 0 and region.get("type", "paragraph") in fallback_body_types:
            cx = (x1 + x2) / 2.0
            best = min(bins, key=lambda b: abs(cx - (b["x1"] + b["x2"]) / 2.0))
            return f"col_{best['bin_id']}"
        return f"span_{rid}" if rid else "span"

    # ------------------------------------------------------------------
    # SSU assignment
    # ------------------------------------------------------------------

    def _assign_ssus(
        self,
        blocks: list[dict],
        bins: list[dict],
        page_median_height: float,
    ) -> tuple[dict, dict, dict, dict]:
        """Core SSU assignment algorithm.

        Walks TextLines within each TextBlock in VPOS order. When a TextLine
        is detected as a heading the semantic_id increments before that line
        is assigned, so headings belong to the unit they open.

        Returns:
            (line_to_ssu, ssu_to_lines, ssu_metadata, line_metadata)
        """
        line_to_ssu: dict = {}
        ssu_to_lines: dict = {}
        ssu_metadata: dict = {}
        line_metadata: dict = {}
        semantic_id = 0
        prev_was_heading = False

        for block in blocks:
            n_lines = len(block["lines"])
            if n_lines == 0:
                continue

            block_region = {"id": block["id"], "type": "paragraph", "bbox": block["bbox"]}
            structural_unit = self._assign_structural_unit(block_region, bins)
            if not structural_unit:
                structural_unit = f"block_{block['id']}"

            for line in sorted(block["lines"], key=lambda l: l["vpos"]):
                is_heading, _ = self._detect_heading(
                    line, block, n_lines, page_median_height
                )

                if is_heading and not prev_was_heading:
                    semantic_id += 1

                if semantic_id == 0:
                    # Lines appearing before any heading on the page → unit 1
                    semantic_id = 1

                ssu_id = f"ssu_{semantic_id}_{structural_unit}"

                line_to_ssu[line["id"]] = ssu_id
                ssu_to_lines.setdefault(ssu_id, []).append(line["id"])

                if ssu_id not in ssu_metadata:
                    ssu_metadata[ssu_id] = {
                        "semantic_id": semantic_id,
                        "structural_unit_id": structural_unit,
                        # block_id of the first line registered to this SSU.
                        # An SSU may span multiple blocks when no heading
                        # separates successive blocks in the same column;
                        # use line_metadata for authoritative per-line block info.
                        "block_id": block["id"],
                    }

                line_metadata[line["id"]] = {
                    "block_id": block["id"],
                    "bbox": (line["hpos"], line["vpos"], line["width"], line["height"]),
                    "is_heading": is_heading,
                }

                prev_was_heading = is_heading

        return line_to_ssu, ssu_to_lines, ssu_metadata, line_metadata

    # ------------------------------------------------------------------
    # XML output
    # ------------------------------------------------------------------

    def _update_xml(
        self, print_space: ET.Element, ns: dict, line_to_ssu: dict
    ) -> None:
        """Write SSU attribute onto each TextLine element."""
        for block in print_space.findall(self._q("TextBlock", ns), ns):
            for line in block.findall(self._q("TextLine", ns), ns):
                lid = line.get("ID", "")
                ssu_id = line_to_ssu.get(lid)
                if ssu_id is not None:
                    line.set("SSU", ssu_id)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def assign(
        self,
        output_path: Optional[str] = None,
        modify_in_place: bool = False,
    ) -> dict:
        """Parse the ALTO XML, assign SSUs, and optionally write modified XML.

        Args:
            output_path: Write modified XML to this path.
            modify_in_place: If True and output_path is None, overwrite input.

        Returns:
            dict with keys:
                'line_to_ssu'   — line_id → ssu_id
                'ssu_to_lines'  — ssu_id  → [line_id, ...]
                'ssu_metadata'  — ssu_id  → {semantic_id, structural_unit_id, block_id}
                'line_metadata' — line_id → {block_id, bbox, is_heading}
        """
        tree = ET.parse(self.alto_xml_path)
        root = tree.getroot()
        ns_uri = self._detect_namespace(root)
        ns = {"alto": ns_uri} if ns_uri else {}

        layout = root.find(self._q("Layout", ns), ns)
        page = layout.find(self._q("Page", ns), ns) if layout is not None else None
        print_space = page.find(self._q("PrintSpace", ns), ns) if page is not None else None

        empty = {"line_to_ssu": {}, "ssu_to_lines": {}, "ssu_metadata": {}, "line_metadata": {}}
        if print_space is None:
            logger.warning("No <PrintSpace> element found in %s", self.alto_xml_path)
            return empty

        blocks = self._extract_blocks(print_space, ns)
        page_median_height = self._compute_page_median_string_height(blocks)

        col_blocks = [b for b in blocks if len(b["lines"]) >= self.min_col_lines]
        using_col_filter = len(col_blocks) >= 2
        bin_source = col_blocks if using_col_filter else blocks

        block_regions = [
            {"id": b["id"], "type": "paragraph", "bbox": b["bbox"]} for b in bin_source
        ]
        # When blocks are pre-filtered to substantive columns each qualifying
        # block is trusted on its own; lower min_bin_members to 1 so singleton
        # column blocks are not collapsed into their nearest neighbour.
        min_bin_members = 1 if using_col_filter else 2
        bins = self._infer_structural_bins(block_regions, min_bin_members=min_bin_members)

        line_to_ssu, ssu_to_lines, ssu_metadata, line_metadata = self._assign_ssus(
            blocks, bins, page_median_height
        )

        write_path = output_path or (self.alto_xml_path if modify_in_place else None)
        if write_path:
            self._update_xml(print_space, ns, line_to_ssu)
            if ns_uri:
                ET.register_namespace("", ns_uri)
            tree.write(write_path, xml_declaration=True, encoding="unicode")

        return {
            "line_to_ssu": line_to_ssu,
            "ssu_to_lines": ssu_to_lines,
            "ssu_metadata": ssu_metadata,
            "line_metadata": line_metadata,
        }


def assign_alto_ssu(
    alto_xml_path: str,
    output_path: Optional[str] = None,
    modify_in_place: bool = False,
    min_heading_score: int = 2,
    min_col_lines: int = 5,
) -> dict:
    """Assign Semantic Structural Units to TextLines in a Transkribus ALTO XML file.

    Args:
        alto_xml_path: Path to the input ALTO XML file.
        output_path: If provided, write the modified XML here.
        modify_in_place: If True and output_path is None, overwrite the input.
        min_heading_score: Number of heading rules (out of 5) a TextLine must
            satisfy to be classified as a heading. Defaults to 2.

    Returns:
        Dictionary with:
            'line_to_ssu':   dict mapping line_id (str) → ssu_id (str)
            'ssu_to_lines':  dict mapping ssu_id (str) → list of line_id (str)
            'ssu_metadata':  dict mapping ssu_id (str) → {
                                 'semantic_id': int,
                                 'structural_unit_id': str,  # col_k or span_<id>
                                 'block_id': str,            # first contributing block
                             }
            'line_metadata': dict mapping line_id (str) → {
                                 'block_id': str,
                                 'bbox': (hpos, vpos, width, height),
                                 'is_heading': bool,
                             }
    """
    return ALTOSSUTagger(alto_xml_path, min_heading_score, min_col_lines).assign(output_path, modify_in_place)
