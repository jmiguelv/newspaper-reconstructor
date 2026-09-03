"""Article reconstruction as a jawi-pipeline stage.

Implements `Module[ArticleReconstructionInput, ArticleReconstructionOutput]`:
pipeline OCR regions are converted to the fragment format used by the LLM
reconstruction stage, and the LLM's grouped items are mapped to the
pipeline's article contract.
"""

import re
from collections.abc import Iterable

from jawi_pipeline import Config, InputRow, Module
from jawi_pipeline.types import (
    ArticleReconstructionInput,
    ArticleReconstructionOutput,
    ImageRegion,
    RegionWithOcr,
)

from src.newspaper_reconstructor.llm import LLMClient, make_client
from src.newspaper_reconstructor.prompts import load_prompt
from src.newspaper_reconstructor.reconstruct import reconstruct_articles


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


class ArticleReconstructionConfig(Config):
    """Configuration for the article reconstruction module.

    LLM fields default to None and fall back to LLM_* environment
    variables via make_client.
    """

    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    provider: str | None = None
    timeout: float = 300.0
    prompt_file: str = "prompts/v01.md"
    max_retries: int = 3
    max_workers: int = 1
    article_id_prefix: str = "article_"


class ArticleReconstructionModule(
    Module[ArticleReconstructionInput, ArticleReconstructionOutput]
):
    """Reconstructs articles from OCR regions of a single page."""

    config_cls = ArticleReconstructionConfig

    def __init__(self, config: Config | None = None) -> None:
        super().__init__(config)
        self.client: LLMClient = make_client(
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            provider=self.config.provider,
        )
        self.system_prompt, self.user_prompt = load_prompt(self.config.prompt_file)

    def process(self, data: ArticleReconstructionInput) -> ArticleReconstructionOutput:
        fragments = input_to_fragments(data)
        if not fragments:
            return ArticleReconstructionOutput(articles={})
        items = reconstruct_articles(
            fragments,
            self.client,
            self.system_prompt,
            self.user_prompt,
            max_retries=self.config.max_retries,
        )
        if items is None:
            raise RuntimeError(f"LLM reconstruction failed for page {data.page.id}")
        articles = {
            f"{self.config.article_id_prefix}{i}": item["fragment_ids"]
            for i, item in enumerate(items, start=1)
        }
        return ArticleReconstructionOutput(articles=articles)

    def _input_rows(self) -> Iterable[InputRow]:
        return []
