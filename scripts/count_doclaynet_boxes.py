#!/usr/bin/env python3
"""Count ground-truth annotation boxes in the DocLayNet test set.

Outputs a CSV with per-class box counts and totals to results/doclaynet/box_counts.csv.
No model inference or image extraction is performed.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


CATEGORY_NAMES = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}


def main():
    parser = argparse.ArgumentParser(description="Count DocLayNet ground-truth boxes")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to directory containing DocLayNet HuggingFace parquet files. "
        "If omitted, the test split is downloaded from HuggingFace.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/doclaynet/box_counts.csv",
        help="Output CSV path (default: results/doclaynet/box_counts.csv)",
    )
    args = parser.parse_args()

    import datasets

    dataset_path = Path(args.dataset) if args.dataset else None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load parquet — same logic as DocLayNetDataset.load()
    if dataset_path is not None:
        local_parquet = sorted(dataset_path.glob("*.parquet"))
        if not local_parquet:
            print(f"No parquet files found in {dataset_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading {len(local_parquet)} parquet file(s) from {dataset_path}...")
        ds = datasets.load_dataset(
            "parquet",
            data_files=[str(p) for p in local_parquet],
            split="train",
        )
    else:
        print("No local path given — downloading DocLayNet-v1.2 test split from HuggingFace...")
        ds = datasets.load_dataset(
            "docling-project/DocLayNet-v1.2",
            data_files="data/test-*.parquet",
            split="train",
            verification_mode="no_checks",
        )

    total_images = 0
    total_boxes = 0
    class_counts: Counter = Counter()

    for row in ds:
        total_images += 1
        category_ids = row.get("category_id", [])
        total_boxes += len(category_ids)
        class_counts.update(category_ids)

    print(f"Images: {total_images}")
    print(f"Total boxes: {total_boxes}")

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "count", "pct_of_total"])
        for cat_id in sorted(CATEGORY_NAMES):
            name = CATEGORY_NAMES[cat_id]
            count = class_counts.get(cat_id, 0)
            pct = 100.0 * count / total_boxes if total_boxes > 0 else 0.0
            writer.writerow([name, count, f"{pct:.2f}"])
        writer.writerow(["TOTAL", total_boxes, "100.00"])

    print(f"CSV written to {output_path}")


if __name__ == "__main__":
    main()
