#!/usr/bin/env python
"""
Assign SSU tags to all PAGE XML files in a folder.

Usage:
    python scripts/tag_ssu.py <input_folder> [output_folder]

If output_folder is omitted, a sibling folder named <input_folder>_with_ssu
is created automatically.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cot_score.ssu_tagger import assign_ssu

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag PAGE XML files with Semantic Structural Unit (SSU) identifiers."
    )
    parser.add_argument("input", type=Path, help="Folder containing PAGE XML files.")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Destination folder (default: <input>_with_ssu alongside the input folder).",
    )
    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    if not input_dir.is_dir():
        logger.error("Input path is not a directory: %s", input_dir)
        sys.exit(1)

    output_dir: Path = (
        args.output.resolve()
        if args.output
        else input_dir.parent / f"{input_dir.name}_with_ssu"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(input_dir.glob("*.xml"))
    if not xml_files:
        logger.warning("No XML files found in %s", input_dir)
        sys.exit(0)

    logger.info("Input:  %s", input_dir)
    logger.info("Output: %s", output_dir)
    logger.info("Files:  %d", len(xml_files))

    ok = 0
    errors = 0
    total_ssus = 0

    for xml_file in xml_files:
        out_file = output_dir / xml_file.name
        try:
            result = assign_ssu(str(xml_file), output_path=str(out_file))
            n_ssus = len(result["ssu_to_regions"])
            total_ssus += n_ssus
            logger.info("  %-40s  %3d SSUs", xml_file.name, n_ssus)
            ok += 1
        except Exception as exc:
            logger.error("  %-40s  FAILED: %s", xml_file.name, exc)
            errors += 1

    logger.info("")
    logger.info("Done: %d/%d files processed, %d SSUs assigned, %d errors.",
                ok, len(xml_files), total_ssus, errors)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
