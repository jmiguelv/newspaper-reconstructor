"""Loaders for the various input formats consumed by the etl stage."""

import json

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
