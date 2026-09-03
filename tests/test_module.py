"""Tests for the jawi-pipeline module: adapter, process, bulk_process, CLI."""

from jawi_pipeline.types import (
    ArticleReconstructionInput,
    BBox,
    Coordinate,
    ImageRegion,
    Line,
    LineOcr,
    Page,
    PageMetadata,
    TextImageRegionWithOcr,
    TextRegionWithOcr,
)
from jawi_pipeline.types.module import BaseLine

from src.newspaper_reconstructor.module import input_to_fragments


def make_bbox(x=10.0, y=20.0, w=200.0, h=100.0) -> BBox:
    return BBox(x1=x, y1=y, x2=x + w, y2=y, x3=x + w, y3=y + h, x4=x, y4=y + h)


def make_line_ocr(i: int, text: str, conf: float = 0.9) -> LineOcr:
    return LineOcr(
        line=Line(
            id=f"l{i}",
            baseline=BaseLine(x1=0.0, y1=0.0, x2=10.0, y2=0.0),
            boundaries=[Coordinate(x=0.0, y=0.0), Coordinate(x=10.0, y=0.0)],
        ),
        script="jawi",
        text=text,
        conf=conf,
        glyph_loc=[],
    )


def make_text_region(rid, texts, t="text", bbox=None) -> TextRegionWithOcr:
    return TextRegionWithOcr(
        t=t,
        id=rid,
        bbox=bbox or make_bbox(),
        line_ocr=[make_line_ocr(i, text) for i, text in enumerate(texts)],
    )


def make_text_image_region(rid, texts, bbox=None) -> TextImageRegionWithOcr:
    return TextImageRegionWithOcr(
        t="text-image",
        id=rid,
        bbox=bbox or make_bbox(),
        line_ocr=[make_line_ocr(i, text) for i, text in enumerate(texts)],
    )


def make_image_region(rid, bbox=None) -> ImageRegion:
    return ImageRegion(t="image", id=rid, bbox=bbox or make_bbox())


def make_page(pid="p1") -> Page:
    return Page(
        id=pid,
        metadata=PageMetadata(height=2000, width=1000),
        url=f"https://example.com/{pid}.png",
    )


def make_input(regions, pid="p1") -> ArticleReconstructionInput:
    return ArticleReconstructionInput(page=make_page(pid), regions=regions)


class TestInputToFragments:
    def test_joins_line_texts_in_order(self):
        region = make_text_region("r_1", ["Hello", "World"])
        fragments = input_to_fragments(make_input([region]))
        assert fragments == [
            {
                "id": "r_1",
                "text": "Hello World",
                "type": "text",
                "hpos": 10,
                "vpos": 20,
                "width": 200,
                "height": 100,
            }
        ]

    def test_whitespace_normalized(self):
        region = make_text_region("r_1", ["Hello \n world", "  foo   bar  "])
        fragments = input_to_fragments(make_input([region]))
        assert fragments[0]["text"] == "Hello world foo bar"

    def test_image_region_skipped(self):
        regions = [make_image_region("r_img"), make_text_region("r_1", ["text"])]
        fragments = input_to_fragments(make_input(regions))
        assert [f["id"] for f in fragments] == ["r_1"]

    def test_empty_text_region_skipped(self):
        regions = [
            make_text_region("r_empty", ["", "   "]),
            make_text_region("r_1", ["text"]),
        ]
        fragments = input_to_fragments(make_input(regions))
        assert [f["id"] for f in fragments] == ["r_1"]

    def test_region_without_line_ocr_skipped(self):
        regions = [make_text_region("r_none", []), make_text_region("r_1", ["text"])]
        fragments = input_to_fragments(make_input(regions))
        assert [f["id"] for f in fragments] == ["r_1"]

    def test_bbox_min_max_derivation(self):
        bbox = BBox(
            x1=110.0, y1=30.0, x2=210.0, y2=30.0, x3=210.0, y3=130.0, x4=100.0, y4=140.0
        )
        region = make_text_region("r_1", ["text"], bbox=bbox)
        fragments = input_to_fragments(make_input([region]))
        assert fragments[0]["hpos"] == 100
        assert fragments[0]["vpos"] == 30
        assert fragments[0]["width"] == 110
        assert fragments[0]["height"] == 110

    def test_float_coords_truncated_to_int(self):
        bbox = BBox(
            x1=10.9, y1=20.4, x2=50.5, y2=20.4, x3=50.5, y3=80.7, x4=10.9, y4=80.7
        )
        region = make_text_region("r_1", ["text"], bbox=bbox)
        fragment = input_to_fragments(make_input([region]))[0]
        assert fragment["hpos"] == 10
        assert fragment["vpos"] == 20
        assert fragment["width"] == 39
        assert fragment["height"] == 60

    def test_type_mapping(self):
        regions = [
            make_text_region("r_1", ["a"], t="header"),
            make_text_region("r_2", ["b"], t="footer"),
            make_text_region("r_3", ["c"], t="headline"),
            make_text_image_region("r_4", ["d"]),
        ]
        fragments = input_to_fragments(make_input(regions))
        assert [f["type"] for f in fragments] == [
            "header",
            "footer",
            "headline",
            "text-image",
        ]

    def test_region_order_preserved(self):
        regions = [make_text_region("r_2", ["b"]), make_text_region("r_1", ["a"])]
        fragments = input_to_fragments(make_input(regions))
        assert [f["id"] for f in fragments] == ["r_2", "r_1"]

    def test_no_text_regions_yields_empty_list(self):
        regions = [make_image_region("r_img")]
        assert input_to_fragments(make_input(regions)) == []
