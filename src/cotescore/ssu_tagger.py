"""
SSU Tagger for PAGE XML files.

Assigns Semantic Structural Unit (SSU) identifiers to TextRegions in PAGE XML.
An SSU is the intersection of a semantic unit (article/section, bounded by
headings) and a structural unit (reading order group / column).
"""

import logging
import statistics
import xml.etree.ElementTree as ET
from typing import Optional

logger = logging.getLogger(__name__)


class SSUTagger:
    """Assigns SSU identifiers to TextRegions in a PAGE XML file."""

    def __init__(self, page_xml_path: str, unique_per_region: bool = False) -> None:
        """Initialise the tagger for a single PAGE XML file.

        Args:
            page_xml_path: Path to the PAGE XML file whose TextRegions will
                be assigned SSU identifiers.
            unique_per_region: If ``True``, each TextRegion receives a unique
                SSU id regardless of reading-order grouping. Defaults to
                ``False``.
        """
        self.page_xml_path = page_xml_path
        self.unique_per_region = unique_per_region

    def _detect_namespace(self, root: ET.Element) -> str:
        """Extract the PAGE XML namespace URI from the root element tag."""
        tag = root.tag  # e.g. '{http://schema.primaresearch.org/...}PcGts'
        if tag.startswith("{"):
            return tag[1 : tag.index("}")]
        return ""

    def _extract_text_regions(self, page: ET.Element, ns: dict) -> dict:
        """Return {region_id: {"type": str, "element": ET.Element}} for all TextRegions."""
        regions = {}
        for region in page.findall("page:TextRegion", ns):
            rid = region.get("id", "")
            rtype = region.get("type", "paragraph")
            bbox = self._extract_region_bbox(region, ns)
            regions[rid] = {
                "type": rtype,
                "element": region,
                "bbox": bbox,
            }
        return regions

    def _extract_region_bbox(
        self, region: ET.Element, ns: dict
    ) -> Optional[tuple[float, float, float, float]]:
        """Extract (x1, y1, x2, y2) bbox from a TextRegion Coords polygon."""
        coords = region.find("page:Coords", ns)
        if coords is None:
            return None

        xs: list[float] = []
        ys: list[float] = []

        # Newer PAGE format: <Coords points="x,y x,y ..."/>
        points_attr = coords.get("points", "").strip()
        if points_attr:
            for token in points_attr.split():
                if "," not in token:
                    continue
                x_raw, y_raw = token.split(",", 1)
                try:
                    xs.append(float(x_raw))
                    ys.append(float(y_raw))
                except ValueError:
                    continue

        # Older PAGE format: <Coords><Point x="..." y="..."/>...</Coords>
        if not xs or not ys:
            point_elements = coords.findall("page:Point", ns)
            if not point_elements:
                point_elements = coords.findall("Point")

            for pt in point_elements:
                x_raw = pt.get("x")
                y_raw = pt.get("y")
                if x_raw is None or y_raw is None:
                    continue
                try:
                    xs.append(float(x_raw))
                    ys.append(float(y_raw))
                except ValueError:
                    continue

        if not xs or not ys:
            return None
        return (min(xs), min(ys), max(xs), max(ys))

    def _infer_structural_bins(
        self,
        group_regions: list[dict],
        min_bin_members: int = 2,
        overlap_eps: float = 0.02,
        max_width_median_factor: float = 1.6,
    ) -> list[dict]:
        """
        Infer column-like structural bins from body-like grouped regions.

        Returns list of {'bin_id': int, 'x1': float, 'x2': float, 'member_ids': [str, ...]}.
        """
        candidates = []
        for r in group_regions:
            bbox = r.get("bbox")
            if bbox is None:
                continue
            # Keep this conservative for now: body text drives bin inference.
            if r.get("type", "paragraph") not in {"paragraph", "text"}:
                continue
            x1, _, x2, _ = bbox
            if x2 <= x1:
                continue
            candidates.append(
                {
                    "id": r["id"],
                    "x1": x1,
                    "x2": x2,
                    "width": x2 - x1,
                }
            )

        if not candidates:
            return []

        # Prevent unusually wide body boxes from fusing adjacent column bins.
        # Wide boxes are still handled later during structural-unit assignment.
        widths = [r["width"] for r in candidates if r["width"] > 0]
        if widths:
            median_width = statistics.median(widths)
            width_cap = median_width * max_width_median_factor
            filtered_candidates = [r for r in candidates if r["width"] <= width_cap]
            if filtered_candidates:
                candidates = filtered_candidates

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
                    score = abs(rcenter - ccenter) + abs(rwidth - cwidth)
                    if score < best_score:
                        best_score = score
                        best_idx = idx

            if best_idx is None:
                clusters.append(
                    {
                        "member_ids": [region["id"]],
                        "x1_values": [rx1],
                        "x2_values": [rx2],
                        "center_values": [rcenter],
                        "width_values": [rwidth],
                    }
                )
            else:
                c = clusters[best_idx]
                c["member_ids"].append(region["id"])
                c["x1_values"].append(rx1)
                c["x2_values"].append(rx2)
                c["center_values"].append(rcenter)
                c["width_values"].append(rwidth)

        # Absorb tiny clusters into nearest strong cluster to reduce over-segmentation.
        strong_clusters = [c for c in clusters if len(c["member_ids"]) >= min_bin_members]
        weak_clusters = [c for c in clusters if len(c["member_ids"]) < min_bin_members]

        if strong_clusters:
            for weak in weak_clusters:
                wcenter = statistics.median(weak["center_values"])
                nearest = min(
                    strong_clusters,
                    key=lambda c: abs(wcenter - statistics.median(c["center_values"])),
                )
                nearest["member_ids"].extend(weak["member_ids"])
                nearest["x1_values"].extend(weak["x1_values"])
                nearest["x2_values"].extend(weak["x2_values"])
                nearest["center_values"].extend(weak["center_values"])
                nearest["width_values"].extend(weak["width_values"])
            clusters = strong_clusters

        finalized_bins = []
        for c in clusters:
            finalized_bins.append(
                {
                    "x1": statistics.median(c["x1_values"]),
                    "x2": statistics.median(c["x2_values"]),
                    "member_ids": c["member_ids"],
                }
            )

        finalized_bins.sort(key=lambda b: b["x1"])
        for i, b in enumerate(finalized_bins, start=1):
            b["bin_id"] = i

        return finalized_bins

    def _assign_structural_unit(
        self,
        region: dict,
        bins: list[dict],
        width_overlap_threshold: float = 0.1,
        fallback_body_types: tuple[str, ...] = ("paragraph", "text"),
    ) -> str:
        """
        Assign structural unit token for a region using inferred bins.

        Returns:
            - col_<k> when the region overlaps exactly one bin
            - span_<region_id> when it overlaps multiple bins or none
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
            best_bin_id = min(
                bins,
                key=lambda b: abs(cx - ((b["x1"] + b["x2"]) / 2.0)),
            )["bin_id"]
            return f"col_{best_bin_id}"
        return f"span_{rid}" if rid else "span"

    def _find_leaf_ordered_groups(self, element: ET.Element, ns: dict, depth: int = 0) -> list:
        """
        Recursively find all OrderedGroup elements that contain RegionRefIndexed
        children (leaf groups). Warns if nesting deeper than 1 level is found.

        Returns list of (group_id, [(index, region_ref), ...]) tuples.
        """
        results = []
        for group in element.findall("page:OrderedGroup", ns):
            refs = group.findall("page:RegionRefIndexed", ns)
            child_groups = group.findall("page:OrderedGroup", ns)
            if refs:
                if depth > 0:
                    logger.warning(
                        "Nested OrderedGroup detected (depth=%d) in group '%s'. "
                        "Treating as leaf group.",
                        depth,
                        group.get("id", ""),
                    )
                region_refs = [(int(r.get("index", 0)), r.get("regionRef", "")) for r in refs]
                results.append((group.get("id", ""), region_refs))
            elif child_groups:
                results.extend(self._find_leaf_ordered_groups(group, ns, depth + 1))
        return results

    def _extract_reading_order(self, page: ET.Element, ns: dict) -> list:
        """
        Return ordered list of (group_id, [region_id, ...]) for all leaf
        OrderedGroups in the ReadingOrder. Handles both flat structures
        (ReadingOrder > OrderedGroup) and wrapped ones
        (ReadingOrder > UnorderedGroup > OrderedGroup).
        """
        ro = page.find("page:ReadingOrder", ns)
        if ro is None:
            return []

        # Collect from UnorderedGroup children and direct OrderedGroup children
        leaf_groups = []

        # Direct OrderedGroups under ReadingOrder (flat / single-column structure)
        for group in ro.findall("page:OrderedGroup", ns):
            refs = group.findall("page:RegionRefIndexed", ns)
            child_groups = group.findall("page:OrderedGroup", ns)
            if refs:
                region_refs = [(int(r.get("index", 0)), r.get("regionRef", "")) for r in refs]
                leaf_groups.append((group.get("id", ""), region_refs))
            elif child_groups:
                leaf_groups.extend(self._find_leaf_ordered_groups(group, ns, depth=1))

        # OrderedGroups nested under UnorderedGroup(s)
        for ug in ro.findall("page:UnorderedGroup", ns):
            leaf_groups.extend(self._find_leaf_ordered_groups(ug, ns, depth=0))

        # Sort each group's regions by index, return only ids
        ordered_groups = []
        for group_id, region_refs in leaf_groups:
            sorted_ids = [rid for _, rid in sorted(region_refs, key=lambda x: x[0])]
            ordered_groups.append((group_id, sorted_ids))

        return ordered_groups

    def _assign_ssus(
        self,
        text_regions: dict,
        ordered_groups: list,
    ) -> tuple:
        """
        Core SSU assignment algorithm.

        Returns (region_to_ssu, ssu_to_regions, ssu_metadata).
        """
        region_to_ssu: dict = {}
        ssu_to_regions: dict = {}
        ssu_metadata: dict = {}
        semantic_id = 0

        def _register(
            region_id: str,
            ssu_id: str,
            structural_group_id: str,
            structural_unit_id: str,
        ) -> None:
            region_to_ssu[region_id] = ssu_id
            ssu_to_regions.setdefault(ssu_id, []).append(region_id)
            if ssu_id not in ssu_metadata:
                ssu_metadata[ssu_id] = {
                    "semantic_id": semantic_id,
                    "structural_group_id": structural_group_id,
                    "structural_unit_id": structural_unit_id,
                    "region_types": [],
                }
            rtype = text_regions.get(region_id, {}).get("type", "paragraph")
            ssu_metadata[ssu_id]["region_types"].append(rtype)

        grouped_region_ids: set = set()

        for group_id, region_ids in ordered_groups:
            if not region_ids:
                continue

            group_regions = []
            for region_id in region_ids:
                if region_id not in text_regions:
                    continue
                info = text_regions[region_id]
                group_regions.append(
                    {
                        "id": region_id,
                        "type": info.get("type", "paragraph"),
                        "bbox": info.get("bbox"),
                    }
                )

            bins = self._infer_structural_bins(group_regions)
            use_structural_units = bool(bins)

            for i, region_id in enumerate(region_ids):
                rtype = text_regions.get(region_id, {}).get("type", "paragraph")
                if i == 0:
                    semantic_id += 1
                elif rtype == "heading":
                    semantic_id += 1

                structural_unit_id = ""
                if use_structural_units:
                    region_info = {
                        "id": region_id,
                        "type": rtype,
                        "bbox": text_regions.get(region_id, {}).get("bbox"),
                    }
                    structural_unit_id = self._assign_structural_unit(region_info, bins)

                if self.unique_per_region:
                    ssu_id = f"ssu_{region_id}"
                    _register(region_id, ssu_id, group_id, structural_unit_id or group_id)
                elif structural_unit_id:
                    ssu_id = f"ssu_{semantic_id}_{group_id}_{structural_unit_id}"
                    _register(region_id, ssu_id, group_id, structural_unit_id)
                else:
                    # Legacy fallback when geometry is unavailable.
                    ssu_id = f"ssu_{semantic_id}_{group_id}"
                    _register(region_id, ssu_id, group_id, group_id)

                grouped_region_ids.add(region_id)

        # Ungrouped TextRegions — each becomes its own SSU
        for region_id in text_regions:
            if region_id not in grouped_region_ids:
                semantic_id += 1
                ssu_id = (
                    f"ssu_{region_id}" if self.unique_per_region else f"ssu_{semantic_id}_ungrouped"
                )
                _register(region_id, ssu_id, "ungrouped", "ungrouped")

        return region_to_ssu, ssu_to_regions, ssu_metadata

    def _update_xml(self, page: ET.Element, ns: dict, region_to_ssu: dict) -> None:
        """Append SSU custom attribute to each TextRegion element."""
        for region in page.findall("page:TextRegion", ns):
            rid = region.get("id", "")
            ssu_id = region_to_ssu.get(rid)
            if ssu_id is None:
                continue
            ssu_token = f"structure {{type:ssu; id:{ssu_id};}}"
            existing = region.get("custom", "")
            region.set("custom", f"{existing} {ssu_token}".strip() if existing else ssu_token)

    def assign(
        self,
        output_path: Optional[str] = None,
        modify_in_place: bool = False,
    ) -> dict:
        """
        Parse the PAGE XML, assign SSUs, and optionally write modified XML.

        Args:
            output_path: Write modified XML to this path.
            modify_in_place: If True and output_path is None, overwrite input.

        Returns:
            dict with keys 'region_to_ssu', 'ssu_to_regions', 'ssu_metadata'.
        """
        tree = ET.parse(self.page_xml_path)
        root = tree.getroot()
        ns_uri = self._detect_namespace(root)
        ns = {"page": ns_uri} if ns_uri else {}

        # PAGE XML structure: PcGts > Page
        page = root.find("page:Page", ns) if ns else root.find("Page")

        if page is None:
            logger.warning("No <Page> element found in %s", self.page_xml_path)
            return {"region_to_ssu": {}, "ssu_to_regions": {}, "ssu_metadata": {}}

        text_regions = self._extract_text_regions(page, ns)
        ordered_groups = self._extract_reading_order(page, ns)

        region_to_ssu, ssu_to_regions, ssu_metadata = self._assign_ssus(
            text_regions, ordered_groups
        )

        write_path = None
        if output_path:
            write_path = output_path
        elif modify_in_place:
            write_path = self.page_xml_path

        if write_path:
            self._update_xml(page, ns, region_to_ssu)
            if ns_uri:
                ET.register_namespace("", ns_uri)
            tree.write(write_path, xml_declaration=True, encoding="unicode")

        return {
            "region_to_ssu": region_to_ssu,
            "ssu_to_regions": ssu_to_regions,
            "ssu_metadata": ssu_metadata,
        }


def assign_ssu(
    page_xml_path: str,
    output_path: Optional[str] = None,
    modify_in_place: bool = False,
    unique_per_region: bool = False,
) -> dict:
    """
    Assign Semantic Structural Units to TextRegions in a PAGE XML file.

    Args:
        page_xml_path: Path to the input PAGE XML file.
        output_path: If provided, write the modified XML here.
        modify_in_place: If True and output_path is None, overwrite the input.

    Returns:
        Dictionary with:
            - 'region_to_ssu': dict mapping region_id (str) → ssu_id (str)
            - 'ssu_to_regions': dict mapping ssu_id (str) → list of region_id (str)
            - 'ssu_metadata': dict mapping ssu_id (str) → {
                'semantic_id': int,
                'structural_group_id': str,
                'region_types': list of str
              }
    """
    return SSUTagger(page_xml_path, unique_per_region).assign(output_path, modify_in_place)
