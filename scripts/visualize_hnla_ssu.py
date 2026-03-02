#!/usr/bin/env python3
"""Visualize SSU-tagged PAGE XML ground truth on HNLA2013 images."""

import argparse
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import logging

from PIL import Image, ImageDraw, ImageFont


logger = logging.getLogger(__name__)


def _detect_namespace(root: ET.Element) -> str:
    tag = root.tag
    if tag.startswith("{"):
        return tag[1 : tag.index("}")]
    return ""


def _parse_points(points_str: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for token in points_str.strip().split():
        if "," not in token:
            continue
        x_str, y_str = token.split(",", 1)
        try:
            points.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    return points


def _extract_points_from_coords(coords: ET.Element, ns: dict) -> list[tuple[float, float]]:
    points_attr = coords.get("points", "")
    if points_attr:
        return _parse_points(points_attr)

    points: list[tuple[float, float]] = []
    point_elems = coords.findall("page:Point", ns) if ns else coords.findall("Point")
    for p in point_elems:
        x_str = p.get("x")
        y_str = p.get("y")
        if x_str is None or y_str is None:
            continue
        try:
            points.append((float(x_str), float(y_str)))
        except ValueError:
            continue

    return points


def _points_to_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 0.0 or h <= 0.0:
        return None
    return x1, y1, w, h


_SSU_RE = re.compile(r"type:ssu;\s*id:([^;]+);")


def _extract_ssu_id(custom: str) -> str | None:
    if not custom:
        return None
    m = _SSU_RE.search(custom)
    if not m:
        return None
    return m.group(1).strip()


def _get_font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def _color_for_ssu(ssu_id: str) -> tuple[int, int, int]:
    h = abs(hash(ssu_id))
    return ((h >> 0) % 200 + 30, (h >> 8) % 200 + 30, (h >> 16) % 200 + 30)


def _draw_ssu_boxes(image_path: Path, regions: list[dict], show_labels: bool) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(14)

    for r in regions:
        x, y, w, h = r["x"], r["y"], r["width"], r["height"]
        ssu_id = r.get("ssu_id") or "ssu_unknown"
        color = _color_for_ssu(ssu_id)
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        if show_labels:
            label = ssu_id
            label_y = max(0, y1 - 16)
            bbox = draw.textbbox((x1, label_y), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, label_y), label, fill=(255, 255, 255), font=font)

    return img


def _downsample_image_and_regions(
    img: Image.Image,
    regions: list[dict],
    *,
    max_dim: int | None,
    scale: float | None,
) -> tuple[Image.Image, list[dict]]:
    if scale is not None:
        if scale <= 0:
            raise ValueError("--scale must be > 0")
        factor = float(scale)
    elif max_dim is not None:
        if max_dim <= 0:
            raise ValueError("--max-dim must be > 0")
        factor = min(1.0, float(max_dim) / float(max(img.width, img.height)))
    else:
        return img, regions

    if factor >= 1.0:
        return img, regions

    new_w = max(1, int(round(img.width * factor)))
    new_h = max(1, int(round(img.height * factor)))
    img_small = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    scaled_regions: list[dict] = []
    for r in regions:
        r2 = dict(r)
        r2["x"] = float(r2["x"]) * factor
        r2["y"] = float(r2["y"]) * factor
        r2["width"] = float(r2["width"]) * factor
        r2["height"] = float(r2["height"]) * factor
        scaled_regions.append(r2)

    return img_small, scaled_regions


def _guess_image_path(images_dir: Path, xml_page: ET.Element, xml_path: Path) -> Path | None:
    image_filename = xml_page.get("imageFilename") if xml_page is not None else None
    candidates: list[Path] = []

    if image_filename:
        candidates.append(images_dir / image_filename)
        candidates.append(images_dir / Path(image_filename).name)

    stem = xml_path.stem
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidates.append(images_dir / f"{stem}{ext}")

    for p in candidates:
        if p.exists():
            return p
    return None


def _load_ssu_regions_from_page_xml(xml_path: Path) -> tuple[Path | None, list[dict]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns_uri = _detect_namespace(root)
    ns = {"page": ns_uri} if ns_uri else {}

    page = root.find("page:Page", ns) if ns else root.find("Page")

    regions_out: list[dict] = []

    if page is None:
        return None, regions_out

    text_regions = page.findall("page:TextRegion", ns) if ns else page.findall("TextRegion")

    for tr in text_regions:
        ssu_id = _extract_ssu_id(tr.get("custom", ""))
        coords = tr.find("page:Coords", ns) if ns else tr.find("Coords")
        if coords is None:
            continue

        bbox = _points_to_bbox(_extract_points_from_coords(coords, ns))
        if bbox is None:
            continue

        x, y, w, h = bbox
        regions_out.append(
            {
                "id": tr.get("id", ""),
                "type": tr.get("type", ""),
                "ssu_id": ssu_id,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            }
        )

    return page, regions_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize SSU ground truth boxes for HNLA2013 PAGE XML"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing HNLA images",
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        required=True,
        help="Directory containing PAGE XML with SSU custom tags (e.g. data/HNLA2013/groundtruth_with_ssu)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/visualizations_hnla_ssu"), help="Output directory"
    )
    parser.add_argument(
        "--num-samples", type=int, default=20, help="How many pages to render (0 = all)"
    )
    parser.add_argument("--show-labels", action="store_true", help="Draw SSU ids as labels")
    parser.add_argument(
        "--max-dim",
        type=int,
        default=2500,
        help="Downsample output so max(width,height) is <= this value (set 0 to disable)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Optional explicit downsample factor (e.g. 0.25). Overrides --max-dim.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s - %(message)s",
    )

    images_dir: Path = args.images_dir
    xml_dir: Path = args.xml_dir
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    max_dim = None if args.max_dim == 0 else args.max_dim

    if not xml_dir.is_dir():
        raise SystemExit(f"XML dir not found: {xml_dir}")
    if not images_dir.is_dir():
        raise SystemExit(f"Images dir not found: {images_dir}")

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No .xml files found in {xml_dir}")

    if args.num_samples > 0:
        xml_files = xml_files[: args.num_samples]

    rendered = 0
    skipped = 0

    for xml_path in xml_files:
        page, regions = _load_ssu_regions_from_page_xml(xml_path)
        image_path = _guess_image_path(images_dir, page, xml_path)
        if image_path is None:
            skipped += 1
            logger.info("No image match for %s", xml_path.name)
            continue

        if not regions:
            logger.warning("No regions extracted from %s", xml_path.name)

        base_img = Image.open(image_path).convert("RGB")
        base_img, regions = _downsample_image_and_regions(
            base_img,
            regions,
            max_dim=max_dim,
            scale=args.scale,
        )
        tmp_path = image_path
        img = base_img

        draw = ImageDraw.Draw(img)
        font = _get_font(14)
        for r in regions:
            x, y, w, h = r["x"], r["y"], r["width"], r["height"]
            ssu_id = r.get("ssu_id") or "ssu_unknown"
            color = _color_for_ssu(ssu_id)
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            if args.show_labels:
                label = ssu_id
                label_y = max(0, y1 - 16)
                bbox = draw.textbbox((x1, label_y), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, label_y), label, fill=(255, 255, 255), font=font)

        out_path = output_dir / f"{image_path.stem}_ssu.png"
        img.save(out_path)
        rendered += 1

        logger.info(
            "Rendered %s (regions=%d, size=%dx%d)",
            tmp_path.name,
            len(regions),
            img.width,
            img.height,
        )

    print(f"Done. Rendered: {rendered}, Skipped (no image match): {skipped}, Output: {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
