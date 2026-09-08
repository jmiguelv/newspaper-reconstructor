"""Shared test fixtures."""

import pytest
from jawi_pipeline.types import (
    ArticleReconstructionInput,
    BBox,
    Line,
    LineOcr,
    Page,
    PageMetadata,
    TextRegionWithOcr,
)
from jawi_pipeline.types.module import BaseLine, Coordinate


def make_ocr_output_dict(pid="p1", regions=None) -> dict:
    """Build a module-format OcrOutput JSON dict ({page, regions}) for one page."""
    line_ocr = [
        LineOcr(
            line=Line(
                id="l0",
                baseline=BaseLine(x1=0.0, y1=0.0, x2=10.0, y2=0.0),
                boundaries=[Coordinate(x=0.0, y=0.0), Coordinate(x=10.0, y=0.0)],
            ),
            script="jawi",
            text="Hello world",
            conf=0.9,
            glyph_loc=[],
        )
    ]
    if regions is None:
        regions = [
            TextRegionWithOcr(
                t="text",
                id="r_1",
                bbox=BBox(
                    x1=10.0,
                    y1=20.0,
                    x2=210.0,
                    y2=20.0,
                    x3=210.0,
                    y3=120.0,
                    x4=10.0,
                    y4=120.0,
                ),
                line_ocr=line_ocr,
            )
        ]
    data = ArticleReconstructionInput(
        page=Page(
            id=pid,
            metadata=PageMetadata(height=2000, width=1000),
            url=f"https://example.com/{pid}.png",
        ),
        regions=regions,
    )
    return data.model_dump()


@pytest.fixture
def ocr_output_json() -> dict:
    return make_ocr_output_dict()
