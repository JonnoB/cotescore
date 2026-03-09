#!/usr/bin/env python3
"""Count ground-truth annotation boxes in the HNLA2013 dataset.

Counts TextRegion elements in PAGE XML ground truth files (matching the
logic in HNLA2013Dataset._parse_xml) and outputs a CSV summary.
"""

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PAGE_NS = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19"}


def count_xml(xml_path: Path) -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return len(root.findall(".//page:TextRegion", PAGE_NS))


def main():
    parser = argparse.ArgumentParser(description="Count HNLA2013 ground-truth boxes")
    parser.add_argument(
        "--groundtruth",
        type=str,
        default="data/HNLA2013/groundtruth_unique_ssu",
        help="Path to directory containing PAGE XML files "
             "(default: data/HNLA2013/groundtruth_unique_ssu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/hnla2013/box_counts.csv",
        help="Output CSV path (default: results/hnla2013/box_counts.csv)",
    )
    args = parser.parse_args()

    gt_path = Path(args.groundtruth)
    output_path = Path(args.output)

    if not gt_path.exists():
        print(f"Ground truth directory not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(gt_path.glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {gt_path}", file=sys.stderr)
        sys.exit(1)

    total_images = 0
    total_boxes = 0
    per_page = []

    for xml_path in xml_files:
        n = count_xml(xml_path)
        per_page.append((xml_path.stem, n))
        total_images += 1
        total_boxes += n

    print(f"Images: {total_images}")
    print(f"Total boxes (TextRegions): {total_boxes}")
    print(f"Avg per image: {total_boxes / total_images:.1f}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "box_count"])
        for stem, n in per_page:
            writer.writerow([stem, n])
        writer.writerow(["TOTAL", total_boxes])

    print(f"CSV written to {output_path}")


if __name__ == "__main__":
    main()
