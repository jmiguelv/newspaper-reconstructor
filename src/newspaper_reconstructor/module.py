"""Article reconstruction as a jawi-pipeline stage.

Implements `Module[ArticleReconstructionInput, ArticleReconstructionOutput]`:
pipeline OCR regions are converted to the fragment format used by the LLM
reconstruction stage, and the LLM's grouped items are mapped to the
pipeline's article contract.
"""

import re

from jawi_pipeline.types import (
    ArticleReconstructionInput,
    ImageRegion,
    RegionWithOcr,
)


def input_to_fragments(data: ArticleReconstructionInput) -> list[dict]:
    """Convert pipeline OCR regions into fragment dicts for the LLM stage."""
    fragments = []
    for region in data.regions:
        fragment = _region_to_fragment(region)
        if fragment is not None:
            fragments.append(fragment)
    return fragments


def _region_to_fragment(region: RegionWithOcr) -> dict | None:
    """Map a region to a fragment dict, or None if it carries no text."""
    if isinstance(region, ImageRegion):
        return None
    text = re.sub(r"\s+", " ", " ".join(lo.text for lo in region.line_ocr)).strip()
    if not text:
        return None
    xs = (region.bbox.x1, region.bbox.x2, region.bbox.x3, region.bbox.x4)
    ys = (region.bbox.y1, region.bbox.y2, region.bbox.y3, region.bbox.y4)
    return {
        "id": region.id,
        "text": text,
        "type": region.t,
        "hpos": int(min(xs)),
        "vpos": int(min(ys)),
        "width": int(max(xs) - min(xs)),
        "height": int(max(ys) - min(ys)),
    }
