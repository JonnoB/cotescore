#!/usr/bin/env python3
"""
Convert TIFF scans to PNG at reduced resolution for faster inference.

300 DPI scans (~3600x5000px) are far larger than layout models need.
Downsampling to a max dimension of ~1600px cuts pixels by ~9x with
negligible impact on layout detection quality.

The HNLA2013 dataset loader reads original image dimensions from the
PAGE XML <Page> element, so the TIFFs are not needed after conversion.

Usage:
    python scripts/convert_tiff_to_png.py \\
        --input  data/HNLA2013/HNLA2013/HNLA2013_evaluationSet \\
        --output data/HNLA2013/eval_png_1600 \\
        --max-dim 1600
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert(input_dir: Path, output_dir: Path, max_dim: int, ext: str) -> None:
    images = sorted(input_dir.glob(f"*.{ext}")) + sorted(input_dir.glob(f"*.{ext.upper()}"))
    if not images:
        print(f"No .{ext} files found in {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(images)} images → {output_dir}  (max_dim={max_dim})")

    skipped = 0
    for tif_path in tqdm(images):
        out_path = output_dir / (tif_path.stem + ".png")
        if out_path.exists():
            skipped += 1
            continue

        with Image.open(tif_path) as img:
            w, h = img.size
            scale = min(max_dim / w, max_dim / h, 1.0)  # never upscale
            if scale < 1.0:
                new_w = round(w * scale)
                new_h = round(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(out_path, format="PNG", optimize=False)

    converted = len(images) - skipped
    print(f"Done. Converted: {converted}  Skipped (already exist): {skipped}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Downsample TIFF scans to PNG for faster inference"
    )
    parser.add_argument("--input", required=True, help="Directory containing TIFF files")
    parser.add_argument("--output", required=True, help="Output directory for PNG files")
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1600,
        help="Maximum dimension (width or height) in pixels (default: 1600)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="tif",
        help="Input file extension (default: tif)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    convert(input_dir, Path(args.output), args.max_dim, args.ext)


if __name__ == "__main__":
    main()
