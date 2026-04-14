#!/usr/bin/env python3
"""
Visualize document layout model predictions on HNLA2013 images.
Supports multiple model types: DocLayout-YOLO, Docling Heron, etc.
"""

import argparse
import sys
import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.doclayout_yolo import DocLayoutYOLO
from models.docling_heron import DoclingLayoutHeron
from cotescore.dataset import HNLA2013Dataset
from cotescore.adapters import compute_canvas, boxes_to_gt_ssu_map, boxes_to_pred_masks
from cotescore.metrics import mean_iou, coverage, overlap, trespass, excess, cote_score

EVAL_MAX_DIM = 2000

COLORS = {
    "ground_truth": (0, 255, 0),
    "prediction": (255, 0, 0),
    "overlap": (255, 165, 0),
    "text": (255, 255, 255),
    "background": (40, 40, 40),
}


def get_font(size=20):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def draw_boxes_pil(image_path, boxes, color, label_prefix="", thickness=3, font_size=20):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)

    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        if w <= 0 or h <= 0:
            continue

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

        label = f"{label_prefix}{box.get('class', 'unknown')}"
        label_y = max(0, y1 - font_size - 5)
        bbox = draw.textbbox((x1, label_y), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, label_y), label, fill=(255, 255, 255), font=font)

    return img


def create_comparison_image(
    image_path, ground_truth, predictions, metrics, output_path, model_name=None
):
    gt_img = draw_boxes_pil(image_path, ground_truth, COLORS["ground_truth"], "GT: ", 2, 16)
    pred_img = draw_boxes_pil(image_path, predictions, COLORS["prediction"], "P: ", 2, 16)

    width, height = gt_img.size
    padding = 300
    comparison = Image.new("RGB", (width * 2 + 40, height + padding), COLORS["background"])

    comparison.paste(gt_img, (10, padding))
    comparison.paste(pred_img, (width + 30, padding))

    draw = ImageDraw.Draw(comparison)
    title_font = get_font(24)
    metric_font = get_font(18)
    small_font = get_font(16)

    filename = Path(image_path).name
    draw.text((20, 20), f"Image: {filename}", fill=COLORS["text"], font=title_font)

    if model_name:
        draw.text((20, 50), f"Model: {model_name}", fill=(180, 180, 180), font=small_font)

    draw.text(
        (width // 2 - 100, padding - 35),
        f"Ground Truth ({len(ground_truth)})",
        fill=COLORS["ground_truth"],
        font=title_font,
    )
    draw.text(
        (width + width // 2 - 100, padding - 35),
        f"Predictions ({len(predictions)})",
        fill=COLORS["prediction"],
        font=title_font,
    )

    y_offset = 85 if model_name else 60
    draw.text((20, y_offset), "Metrics:", fill=COLORS["text"], font=metric_font)
    y_offset += 30

    descriptions = {
        "mean_iou": "Intersection over Union",
        "coverage": "GT area covered",
        "overlap": "Prediction duplicates",
        "trespass": "Wrong GT overlap",
        "excess": "Background coverage",
        "cote_score": "Overall quality (C-O-T)",
    }

    metric_order = ["cote_score", "coverage", "overlap", "trespass", "excess", "mean_iou"]
    sorted_metrics = {k: metrics[k] for k in metric_order if k in metrics}

    for metric, score in sorted_metrics.items():
        if metric == "cote_score":
            if score > 0.7:
                color = COLORS["ground_truth"]
            elif score < 0:
                color = COLORS["prediction"]
            else:
                color = COLORS["text"]
        elif metric in ["overlap", "trespass", "excess"]:
            color = (
                COLORS["ground_truth"]
                if score < 0.2
                else (COLORS["prediction"] if score > 0.5 else COLORS["text"])
            )
        else:
            color = (
                COLORS["ground_truth"]
                if score > 0.8
                else (COLORS["prediction"] if score < 0.5 else COLORS["text"])
            )

        if metric == "cote_score":
            score_str = f"{score:+.4f}"
        else:
            score_str = f"{score:.4f}"

        draw.text((20, y_offset), f"  {metric.upper()}: {score_str}", fill=color, font=metric_font)
        y_offset += 25

        desc = descriptions.get(metric, "")
        if desc:
            draw.text((40, y_offset), f"  ({desc})", fill=(180, 180, 180), font=metric_font)
            y_offset += 20

    comparison.save(output_path)
    print(f"Saved: {output_path}")


def create_overlay_image(image_path, ground_truth, predictions, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    for box in ground_truth:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        if w > 0 and h > 0:
            draw.rectangle([x, y, x + w, y + h], outline=COLORS["ground_truth"] + (200,), width=3)

    for box in predictions:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        if w > 0 and h > 0:
            draw.rectangle([x, y, x + w, y + h], outline=COLORS["prediction"] + (200,), width=3)

    legend_height = 80
    legend = Image.new("RGB", (img.width, legend_height), COLORS["background"])
    legend_draw = ImageDraw.Draw(legend)
    font = get_font(18)

    legend_draw.rectangle([20, 20, 60, 40], outline=COLORS["ground_truth"], width=3)
    legend_draw.text(
        (70, 22), f"Ground Truth ({len(ground_truth)})", fill=COLORS["ground_truth"], font=font
    )

    legend_draw.rectangle([300, 20, 340, 40], outline=COLORS["prediction"], width=3)
    legend_draw.text(
        (350, 22), f"Predictions ({len(predictions)})", fill=COLORS["prediction"], font=font
    )

    result = Image.new("RGB", (img.width, img.height + legend_height))
    result.paste(legend, (0, 0))
    result.paste(img, (0, legend_height))

    result.save(output_path)
    print(f"Saved overlay: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize predictions from document layout models on HNLA2013"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to flat directory containing HNLA2013 PNG images",
    )
    parser.add_argument(
        "--groundtruth",
        type=str,
        required=True,
        help="Path to groundtruth_with_ssu/ directory containing PAGE XML files",
    )
    parser.add_argument(
        "--output", type=str, default="visualizations/hnla2013", help="Output directory"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["yolo", "heron"],
        default="yolo",
        help="Model architecture type (yolo=DocLayout-YOLO, heron=Docling Heron)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path for inference (displayed on visualizations). "
        "Defaults: yolo='juliozhao/DocLayout-YOLO-DocStructBench', heron='ds4sd/docling-layout-heron'",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Samples to visualize")
    parser.add_argument("--indices", nargs="+", type=int, help="Specific indices")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument(
        "--image-ext",
        type=str,
        default="png",
        help="Image file extension (default: png)",
    )
    parser.add_argument("--overlay", action="store_true", help="Create overlay image")

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.model is None:
        if args.model_type == "yolo":
            args.model = "juliozhao/DocLayout-YOLO-DocStructBench"
        elif args.model_type == "heron":
            args.model = "ds4sd/docling-layout-heron"

    print(f"Loading dataset: {args.dataset}")
    print(f"Ground truth: {args.groundtruth}")
    dataset = HNLA2013Dataset(
        images_path=Path(args.dataset),
        groundtruth_path=Path(args.groundtruth),
        image_ext=args.image_ext,
    )
    dataset.load()
    print(f"Dataset loaded: {len(dataset)} images")

    print(f"Initializing {args.model_type} model: {args.model}")
    if args.model_type == "yolo":
        model = DocLayoutYOLO(
            model_name=args.model, conf_threshold=args.conf, imgsz=1024, device=args.device
        )
    elif args.model_type == "heron":
        model = DoclingLayoutHeron(model_name=args.model, threshold=args.conf, device=args.device)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model.load()

    if args.indices:
        indices = [i for i in args.indices if 0 <= i < len(dataset)]
    else:
        step = max(1, len(dataset) // args.num_samples)
        indices = list(range(0, len(dataset), step))[: args.num_samples]

    print(f"Visualizing {len(indices)} images: {indices}\n")
    results_summary = []

    for idx in indices:
        sample = dataset[idx]
        image_path = Path(sample["image_path"])
        ground_truth = sample["annotations"]
        filename = sample["filename"]

        print(f"Processing {filename}...")
        predictions = model.predict(image_path)

        with Image.open(image_path) as img:
            image_width, image_height = img.size

        canvas_w, canvas_h = compute_canvas(image_width, image_height, EVAL_MAX_DIM)
        gt_ssu_map = boxes_to_gt_ssu_map(ground_truth, image_width, image_height, canvas_w, canvas_h)
        pred_masks = boxes_to_pred_masks(predictions, image_width, image_height, canvas_w, canvas_h)

        metrics = {
            "mean_iou": mean_iou(predictions, ground_truth),
            "coverage": coverage(gt_ssu_map, pred_masks),
            "overlap": overlap(gt_ssu_map, pred_masks),
            "trespass": trespass(gt_ssu_map, pred_masks),
            "excess": excess(gt_ssu_map, pred_masks),
            "cote_score": cote_score(gt_ssu_map, pred_masks)[0],
        }

        print(f"  GT: {len(ground_truth)}, Preds: {len(predictions)}")
        print(
            f"  Metrics: COT={metrics['cote_score']:+.3f}, Cov={metrics['coverage']:.3f}, "
            f"Ovlp={metrics['overlap']:.3f}, Tres={metrics['trespass']:.3f}, "
            f"Excess={metrics['excess']:.3f}, IoU={metrics['mean_iou']:.3f}"
        )

        base_name = image_path.stem
        comparison_path = output_path / f"{base_name}_comparison.png"
        create_comparison_image(
            image_path, ground_truth, predictions, metrics, comparison_path, model_name=args.model
        )

        if args.overlay:
            overlay_path = output_path / f"{base_name}_overlay.png"
            create_overlay_image(image_path, ground_truth, predictions, overlay_path)

        results_summary.append(
            {
                "filename": filename,
                "metrics": metrics,
                "visualization": str(comparison_path),
            }
        )
        print()

    summary_path = output_path / "visualization_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
