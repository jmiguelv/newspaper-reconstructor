"""Loaders for the various input formats consumed by the etl stage."""

import json
import math
import re

from jawi_pipeline.types import ArticleReconstructionInput

from newspaper_reconstructor.module import input_to_fragments


def load_article_json(path: str) -> list[dict]:
    """Load a {region_id: text} JSON file into a fragment list.

    Args:
        path: Path to a JSON file mapping region IDs to OCR text strings.

    Returns:
        List of dicts with 'id' and 'text' keys.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [{"id": k, "text": v} for k, v in data.items()]


def load_ocr_output_json(path: str, slim: bool = False) -> list[dict]:
    """Load a module-format OcrOutput JSON file ({page, regions}) into fragments.

    Uses module.input_to_fragments, so fragment semantics (line joining,
    whitespace normalization, image-region skipping, bbox derivation) match
    the jawi-pipeline module exactly. With slim=True, keeps only 'id' and
    'text', matching the fragment fields documented in the clustering
    prompts.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    fragments = input_to_fragments(ArticleReconstructionInput.model_validate(data))
    return _slim_fragments(fragments) if slim else fragments


def load_baseline_ocr_file(path: str) -> tuple[list[dict], list[dict]]:
    """Load baseline candidates and layout from current or legacy module OCR."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "page" not in data or "regions" not in data:
        raise ValueError(
            "baseline layout context requires module-format OCR with page and regions"
        )
    if not isinstance(data["regions"], list):
        raise TypeError("module-format OCR regions must be a list")

    fragments = []
    layout_regions = []
    for region in data["regions"]:
        fragment, layout_region = _baseline_region(region)
        layout_regions.append(layout_region)
        if fragment is not None:
            fragments.append(fragment)
    return fragments, layout_regions


def _baseline_region(region: object) -> tuple[dict | None, dict]:
    if not isinstance(region, dict):
        raise TypeError("module-format OCR regions must be objects")
    region_id = region.get("id")
    region_type = region.get("t")
    bbox = region.get("bbox")
    if not isinstance(region_id, str) or not region_id:
        raise ValueError("module-format OCR region id must be a non-empty string")
    if not isinstance(region_type, str) or not isinstance(bbox, dict):
        raise TypeError(f"invalid region type or bbox for region {region_id}")
    try:
        xs = tuple(float(bbox[key]) for key in ("x1", "x2", "x3", "x4"))
        ys = tuple(float(bbox[key]) for key in ("y1", "y2", "y3", "y4"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid geometry for region {region_id}") from error
    if not all(math.isfinite(value) for value in (*xs, *ys)):
        raise ValueError(f"invalid geometry for region {region_id}")

    layout_region = {
        "id": region_id,
        "text": "",
        "type": region_type,
        "hpos": int(min(xs)),
        "vpos": int(min(ys)),
        "width": int(max(xs) - min(xs)),
        "height": int(max(ys) - min(ys)),
    }
    lines = region.get("line_ocr")
    if lines is None:
        lines = []
    elif not isinstance(lines, list):
        raise TypeError(f"line_ocr must be a list for region {region_id}")
    try:
        text = re.sub(r"\s+", " ", " ".join(line["text"] for line in lines)).strip()
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid OCR text for region {region_id}") from error
    if region_type == "image" or not text:
        return None, layout_region

    fragment = {**layout_region, "text": text}
    return fragment, fragment


def load_fragments_file(path: str, slim: bool = False) -> list[dict]:
    """Load a fragments file, auto-detecting the input format.

    Supported formats:
    - module OcrOutput JSON: dict with "page" and "regions" keys
    - article JSON: {region_id: ocr_text}
    - fragment list: [{id, text, ...}] (returned as-is)

    With slim=True, fragments are reduced to 'id' and 'text'.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        fragments = data
    elif isinstance(data, dict) and "page" in data and "regions" in data:
        fragments = input_to_fragments(ArticleReconstructionInput.model_validate(data))
    else:
        fragments = [{"id": k, "text": v} for k, v in data.items()]
    return _slim_fragments(fragments) if slim else fragments


def _slim_fragments(fragments: list[dict]) -> list[dict]:
    return [{"id": f["id"], "text": f["text"]} for f in fragments]
