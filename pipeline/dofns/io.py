from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cotescore.types import TokenPositions

NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
BoxRecord = Dict[str, Any]


def parse_alto_xml(
    alto_path: Path,
    image_path: Path,
) -> Tuple[List[BoxRecord], TokenPositions, List[str]]:
    """Parse an ALTO XML file and return GT box records, token positions, and per-line texts.

    Returns:
        (gt_box_records, token_positions, gt_textline_texts)
        - gt_box_records: one BoxRecord per TextLine (source="gt")
        - token_positions: TokenPositions from all String elements
        - gt_textline_texts: list of per-TextLine concatenated content strings
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()

    image_id = image_path.stem
    gt_boxes: List[BoxRecord] = []
    all_tokens: List[str] = []
    all_xs: List[float] = []
    all_ys: List[float] = []
    tl_texts: List[str] = []

    for tl in root.findall(".//alto:TextLine", NS):
        ssu = tl.get("SSU", "")
        hpos = float(tl.get("HPOS", 0))
        vpos = float(tl.get("VPOS", 0))
        width = float(tl.get("WIDTH", 0))
        height = float(tl.get("HEIGHT", 0))

        gt_boxes.append({
            "image_id": image_id,
            "image_path": str(image_path),
            "box_id": ssu,
            "source": "gt",
            "x": hpos,
            "y": vpos,
            "width": width,
            "height": height,
            "class": "text",
            "confidence": 1.0,
            "ocr_text": None,
            "ocr_model": None,
            "image_crop": None,
        })

        line_chars: List[str] = []
        for string_el in tl.findall("alto:String", NS):
            content = string_el.get("CONTENT", "")
            s_hpos = float(string_el.get("HPOS", hpos))
            s_vpos = float(string_el.get("VPOS", vpos))
            s_width = float(string_el.get("WIDTH", 0))
            s_height = float(string_el.get("HEIGHT", height))
            mid_x = s_hpos + s_width / 2.0
            mid_y = s_vpos + s_height / 2.0
            for ch in content:
                all_tokens.append(ch)
                all_xs.append(mid_x)
                all_ys.append(mid_y)
            line_chars.append(content)

        tl_texts.append(" ".join(line_chars))

    token_positions = TokenPositions(
        tokens=np.array(all_tokens, dtype=object),
        xs=np.array(all_xs, dtype=int),
        ys=np.array(all_ys, dtype=int),
    )
    return gt_boxes, token_positions, tl_texts
