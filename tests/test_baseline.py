import json

import pytest
from typer.testing import CliRunner

from newspaper_reconstructor.baseline import (
    THETA_V,
    _axis_order,
    _tau_d,
    baseline_items,
    detect_cross_layout,
    group_columns,
    ncd,
    ncd_pass,
    reading_order,
)
from newspaper_reconstructor.cli import app
from newspaper_reconstructor.ingest import load_baseline_ocr_file

runner = CliRunner()

MALAY_A = (
    "persidangan bahasa melayu telah diadakan di universiti malaya pada "
    "bulan september dengan hadirin yang ramai sekali"
)
MALAY_A2 = (
    "persidangan bahasa melayu diadakan di universiti malaya pada bulan "
    "september bersama hadirin yang ramai"
)
MALAY_B = (
    "kerajaan telah mengumumkan kenaikan harga gula dan tepung di seluruh "
    "negara mulai bulan hadapan"
)


def frag(fid, x, y, w, h, text=""):
    return {"id": fid, "text": text, "hpos": x, "vpos": y, "width": w, "height": h}


def module_region(fid, x, y, w, h, text=None):
    region = {
        "id": fid,
        "t": "image" if text is None else "text",
        "bbox": {
            "x1": x,
            "y1": y,
            "x2": x + w,
            "y2": y,
            "x3": x + w,
            "y3": y + h,
            "x4": x,
            "y4": y + h,
        },
    }
    if text is not None:
        region["line_ocr"] = [
            {
                "line": {
                    "id": f"{fid}_line",
                    "baseline": {"x1": x, "y1": y, "x2": x + w, "y2": y},
                    "boundaries": [],
                },
                "script": "jawi",
                "text": text,
                "conf": 0.9,
                "glyph_loc": [],
            }
        ]
    return region


def module_page(page_id, regions):
    return {
        "page": {
            "id": page_id,
            "metadata": {"height": 2000, "width": 1000},
            "url": "",
        },
        "regions": regions,
    }


class TestGroupColumns:
    def test_two_disjoint_columns_rtl(self):
        left = [frag("l1", 0, 0, 100, 200), frag("l2", 0, 300, 100, 200)]
        right = [frag("r1", 500, 0, 100, 200), frag("r2", 500, 300, 100, 200)]
        columns = group_columns(left + right)
        assert len(columns) == 2
        assert [f["id"] for f in columns[0]] == ["r1", "r2"]
        assert [f["id"] for f in columns[1]] == ["l1", "l2"]

    def test_overlapping_fragments_same_column(self):
        a = frag("a", 0, 0, 100, 200)
        b = frag("b", 20, 300, 100, 200)
        assert len(group_columns([a, b])) == 1

    def test_within_column_top_to_bottom(self):
        frags = [frag("low", 0, 300, 100, 200), frag("top", 0, 0, 100, 200)]
        assert [f["id"] for f in group_columns(frags)[0]] == ["top", "low"]

    def test_partial_overlap_normalized_by_narrower(self):
        a = frag("a", 0, 0, 100, 200)
        half = frag("half", 50, 300, 100, 200)
        under = frag("under", 51, 300, 100, 200)
        assert len(group_columns([a, half], x_overlap_threshold=0.5)) == 1
        assert len(group_columns([a, under], x_overlap_threshold=0.5)) == 2

    def test_later_bridge_joins_previously_skipped_fragment(self):
        first = frag("first", 0, 0, 100, 100)
        skipped = frag("skipped", 160, 100, 100, 100)
        bridge = frag("bridge", 80, 200, 100, 100)
        columns = group_columns([first, skipped, bridge], x_overlap_threshold=0.1)
        assert [[f["id"] for f in column] for column in columns] == [
            ["first", "skipped", "bridge"]
        ]


class TestReadingOrder:
    def test_rtl_then_top_to_bottom(self):
        frags = [
            frag("l2", 0, 300, 100, 200),
            frag("r2", 500, 300, 100, 200),
            frag("l1", 0, 0, 100, 200),
            frag("r1", 500, 0, 100, 200),
        ]
        assert [f["id"] for f in reading_order(frags)] == ["r1", "r2", "l1", "l2"]


class TestGeometryVariant:
    def test_adjacent_fragments_merge(self):
        items = baseline_items(
            [frag("a", 0, 0, 100, 200, "x"), frag("b", 0, 210, 100, 200, "y")],
            variant="geometry",
            gap_k=1.0,
        )
        assert items == [{"fragment_ids": ["a", "b"], "title": "", "class": ""}]

    def test_large_gap_splits(self):
        items = baseline_items(
            [frag("a", 0, 0, 100, 200, "x"), frag("b", 0, 1000, 100, 200, "y")],
            variant="geometry",
            gap_k=1.0,
        )
        assert [i["fragment_ids"] for i in items] == [["a"], ["b"]]

    def test_headline_does_not_bridge_columns(self):
        headline = frag("h", 0, 0, 600, 50, "headline")
        t1 = frag("t1", 0, 60, 280, 100, "top left")
        t2 = frag("t2", 320, 70, 280, 100, "top right")
        b1 = frag("b1", 0, 200, 280, 100, "body left")
        b2 = frag("b2", 320, 210, 280, 100, "body right")
        items = baseline_items(
            [headline, t1, t2, b1, b2], variant="geometry", gap_k=1.0
        )
        assert [i["fragment_ids"] for i in items] == [["h", "t1", "b1"], ["t2", "b2"]]

    def test_cross_column_never_merges(self):
        items = baseline_items(
            [frag("a", 0, 0, 100, 200, "x"), frag("b", 500, 210, 100, 200, "y")],
            variant="geometry",
            gap_k=10.0,
        )
        assert [i["fragment_ids"] for i in items] == [["b"], ["a"]]

    def test_same_vertical_position_does_not_merge_by_id_order(self):
        items = baseline_items(
            [frag("a", 0, 0, 100, 100), frag("b", 0, 0, 100, 100)],
            variant="geometry",
            gap_k=1.0,
        )
        assert [item["fragment_ids"] for item in items] == [["a"], ["b"]]

    def test_tied_successor_candidates_do_not_link(self):
        def run(left_id, right_id):
            headline = frag("h", 0, 0, 600, 50)
            left = frag(left_id, 0, 60, 280, 100)
            right = frag(right_id, 320, 60, 280, 100)
            items = baseline_items(
                [headline, left, right], variant="geometry", gap_k=1.0
            )
            return [item["fragment_ids"] for item in items]

        assert run("l", "r") == [["h"], ["l"], ["r"]]
        assert run("r", "l") == [["h"], ["r"], ["l"]]

    def test_every_fragment_appears_once(self):
        frags = [
            frag("a", 0, 0, 100, 200, "x"),
            frag("b", 0, 210, 100, 200, "y"),
            frag("c", 500, 0, 100, 200, "z"),
        ]
        items = baseline_items(frags, variant="geometry", gap_k=1.0)
        seen = [fid for i in items for fid in i["fragment_ids"]]
        assert sorted(seen) == ["a", "b", "c"]

    def test_empty_input(self):
        assert baseline_items([], variant="geometry") == []

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            baseline_items([frag("a", 0, 0, 1, 1)], variant="nope")

    def test_unknown_variant_raises_for_empty_input(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            baseline_items([], variant="nope")

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"gap_k": -0.1}, "gap_k"),
            ({"x_overlap_threshold": -0.1}, "x_overlap_threshold"),
            ({"x_overlap_threshold": 1.1}, "x_overlap_threshold"),
            ({"ncd_threshold": -0.1}, "ncd_threshold"),
            ({"ncd_min_chars": -1}, "ncd_min_chars"),
        ],
    )
    def test_invalid_numeric_parameter_raises(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            baseline_items([], **kwargs)

    def test_duplicate_candidate_ids_raise(self):
        with pytest.raises(ValueError, match="duplicate candidate fragment ids: same"):
            baseline_items([frag("same", 0, 0, 1, 1), frag("same", 0, 2, 1, 1)])

    def test_duplicate_layout_ids_raise(self):
        candidate = frag("candidate", 0, 0, 1, 1)
        layout = [candidate, frag("same", 0, 2, 1, 1), frag("same", 0, 4, 1, 1)]
        with pytest.raises(ValueError, match="duplicate layout region ids: same"):
            baseline_items([candidate], layout_regions=layout)

    def test_missing_geometry_raises(self):
        with pytest.raises(ValueError, match="module-format OCR"):
            baseline_items([{"id": "a", "text": "x"}], variant="geometry")

    def test_trivial_variants_also_require_geometry(self):
        with pytest.raises(ValueError, match="missing keys: height, hpos"):
            baseline_items([{"id": "a", "text": "x"}], variant="singletons")

    def test_missing_candidate_id_raises(self):
        fragment = {"text": "x", "hpos": 0, "vpos": 0, "width": 1, "height": 1}
        with pytest.raises(ValueError, match="candidate fragment"):
            baseline_items([fragment])

    def test_missing_layout_id_raises(self):
        candidate = frag("a", 0, 0, 1, 1)
        layout = [
            candidate,
            {"text": "", "hpos": 0, "vpos": 5, "width": 1, "height": 1},
        ]
        with pytest.raises(ValueError, match="layout region"):
            baseline_items([candidate], layout_regions=layout)

    def test_geometry_ncd_requires_candidate_text(self):
        fragment = {"id": "a", "hpos": 0, "vpos": 0, "width": 1, "height": 1}
        with pytest.raises(ValueError, match="text"):
            baseline_items([fragment], variant="geometry-ncd")

    def test_geometry_variant_does_not_require_text(self):
        fragment = {"id": "a", "hpos": 0, "vpos": 0, "width": 1, "height": 1}
        assert baseline_items([fragment], variant="geometry") == [
            {"fragment_ids": ["a"], "title": "", "class": ""}
        ]


class TestLayoutContext:
    def test_image_blocks_geometry_raycast_without_becoming_an_output_fragment(self):
        top = frag("top", 0, 0, 100, 100, "top text")
        bottom = frag("bottom", 0, 140, 100, 100, "bottom text")
        image = frag("image", 0, 110, 100, 20)

        items = baseline_items(
            [top, bottom],
            variant="geometry",
            gap_k=1.0,
            layout_regions=[top, image, bottom],
        )

        assert [item["fragment_ids"] for item in items] == [["top"], ["bottom"]]

    def test_tied_obstacle_blocks_without_using_id_order(self):
        top = frag("top", 0, 0, 100, 100, "top text")
        target = frag("a-target", 0, 110, 100, 20, "target text")
        image = frag("z-image", 0, 110, 100, 20)

        items = baseline_items(
            [top, target],
            variant="geometry",
            gap_k=1.0,
            layout_regions=[top, target, image],
        )

        assert [item["fragment_ids"] for item in items] == [["top"], ["a-target"]]

    @pytest.mark.parametrize("variant", ["xy-cut", "xy-cut-plus"])
    def test_image_occupies_whitespace_without_becoming_an_output_fragment(
        self, variant
    ):
        top = frag("top", 0, 0, 100, 100, "top text")
        bottom = frag("bottom", 0, 200, 100, 100, "bottom text")
        image = frag("image", 0, 100, 100, 100)

        items = baseline_items(
            [top, bottom],
            variant=variant,
            gap_k=0.5,
            layout_regions=[top, image, bottom],
        )

        assert [item["fragment_ids"] for item in items] == [["top", "bottom"]]

    def test_image_can_be_detected_as_a_cross_layout_spanner(self):
        image = frag("image", 0, 0, 600, 50)
        left = frag("left", 0, 100, 250, 200, "left")
        right = frag("right", 350, 100, 250, 200, "right")
        assert detect_cross_layout([image, left, right]) == {"image"}

    def test_raw_ocr_loader_separates_candidates_from_layout(self, tmp_path):
        path = tmp_path / "p1.json"
        path.write_text(
            json.dumps(
                module_page(
                    "p1",
                    [
                        module_region("text", 0, 0, 100, 100, "article text"),
                        module_region("empty", 0, 100, 100, 100, ""),
                        module_region("image", 0, 200, 100, 100),
                    ],
                )
            ),
            encoding="utf-8",
        )

        candidates, layout = load_baseline_ocr_file(str(path))

        assert [fragment["id"] for fragment in candidates] == ["text"]
        assert [region["id"] for region in layout] == ["text", "empty", "image"]

    def test_loader_accepts_legacy_lines_without_boundaries(self, tmp_path):
        page = module_page(
            "p1", [module_region("text", 0, 0, 100, 100, "article text")]
        )
        del page["regions"][0]["line_ocr"][0]["line"]["boundaries"]
        path = tmp_path / "p1.json"
        path.write_text(json.dumps(page), encoding="utf-8")

        candidates, layout = load_baseline_ocr_file(str(path))

        assert candidates[0]["text"] == "article text"
        assert layout[0]["id"] == "text"

    def test_loader_rejects_missing_baseline_geometry(self, tmp_path):
        page = module_page(
            "p1", [module_region("text", 0, 0, 100, 100, "article text")]
        )
        del page["regions"][0]["bbox"]["x1"]
        path = tmp_path / "p1.json"
        path.write_text(json.dumps(page), encoding="utf-8")

        with pytest.raises(ValueError, match="invalid geometry for region text"):
            load_baseline_ocr_file(str(path))

    def test_loader_rejects_non_list_line_ocr(self, tmp_path):
        page = module_page(
            "p1", [module_region("text", 0, 0, 100, 100, "article text")]
        )
        page["regions"][0]["line_ocr"] = {}
        path = tmp_path / "p1.json"
        path.write_text(json.dumps(page), encoding="utf-8")

        with pytest.raises(TypeError, match="line_ocr must be a list"):
            load_baseline_ocr_file(str(path))

    def test_loader_rejects_non_finite_baseline_geometry(self, tmp_path):
        page = module_page(
            "p1", [module_region("text", 0, 0, 100, 100, "article text")]
        )
        page["regions"][0]["bbox"]["x1"] = float("nan")
        path = tmp_path / "p1.json"
        path.write_text(json.dumps(page), encoding="utf-8")

        with pytest.raises(ValueError, match="invalid geometry for region text"):
            load_baseline_ocr_file(str(path))

    def test_empty_text_region_blocks_raycast_after_raw_ocr_loading(self, tmp_path):
        path = tmp_path / "p1.json"
        path.write_text(
            json.dumps(
                module_page(
                    "p1",
                    [
                        module_region("top", 0, 0, 100, 100, "top text"),
                        module_region("empty", 0, 110, 100, 20, ""),
                        module_region("bottom", 0, 140, 100, 100, "bottom text"),
                    ],
                )
            ),
            encoding="utf-8",
        )
        candidates, layout = load_baseline_ocr_file(str(path))

        items = baseline_items(
            candidates, variant="geometry", gap_k=1.0, layout_regions=layout
        )

        assert [item["fragment_ids"] for item in items] == [["top"], ["bottom"]]


class TestTrivialVariants:
    def test_singletons(self):
        frags = [frag("a", 0, 0, 100, 200), frag("b", 500, 0, 100, 200)]
        items = baseline_items(frags, variant="singletons")
        assert [i["fragment_ids"] for i in items] == [["b"], ["a"]]

    def test_one_cluster(self):
        frags = [frag("a", 0, 300, 100, 200), frag("b", 500, 0, 100, 200)]
        items = baseline_items(frags, variant="one-cluster")
        assert items == [{"fragment_ids": ["b", "a"], "title": "", "class": ""}]


class TestNcd:
    def test_self_below_different(self):
        assert ncd(MALAY_A, MALAY_A) < 0.1
        assert ncd(MALAY_A, MALAY_A) < ncd(MALAY_A, MALAY_B)

    def test_symmetric(self):
        assert ncd(MALAY_A, MALAY_B) == pytest.approx(ncd(MALAY_B, MALAY_A))

    def test_similar_lower_than_different(self):
        assert ncd(MALAY_A, MALAY_A2) < ncd(MALAY_A, MALAY_B)

    def test_short_texts_pass_neutrally(self):
        assert ncd_pass("abc", "xyz", threshold=0.1, min_chars=20)
        assert not ncd_pass(MALAY_A, MALAY_B, threshold=0.1, min_chars=20)
        assert ncd_pass(MALAY_A, MALAY_A2, threshold=1.0, min_chars=20)


class TestGeometryNcdVariant:
    def test_similar_adjacent_texts_merge(self):
        items = baseline_items(
            [frag("a", 0, 0, 100, 200, MALAY_A), frag("b", 0, 210, 100, 200, MALAY_A2)],
            variant="geometry-ncd",
            gap_k=1.0,
            ncd_threshold=1.0,
        )
        assert [i["fragment_ids"] for i in items] == [["a", "b"]]

    def test_unrelated_adjacent_texts_stay_split(self):
        items = baseline_items(
            [frag("a", 0, 0, 100, 200, MALAY_A), frag("b", 0, 210, 100, 200, MALAY_B)],
            variant="geometry-ncd",
            gap_k=1.0,
            ncd_threshold=0.2,
        )
        assert [i["fragment_ids"] for i in items] == [["a"], ["b"]]

    def test_short_fragments_not_vetoed(self):
        items = baseline_items(
            [frag("a", 0, 0, 100, 200, "abc"), frag("b", 0, 210, 100, 200, "xyz")],
            variant="geometry-ncd",
            gap_k=1.0,
            ncd_threshold=0.01,
        )
        assert [i["fragment_ids"] for i in items] == [["a", "b"]]

    def test_rejected_nearest_fragment_blocks_farther_similar_fragment(self):
        items = baseline_items(
            [
                frag("top", 0, 0, 100, 100, MALAY_A),
                frag("near", 0, 110, 100, 100, MALAY_B),
                frag("far", 0, 220, 100, 100, MALAY_A),
            ],
            variant="geometry-ncd",
            gap_k=2.0,
            ncd_threshold=0.1,
            ncd_min_chars=1,
        )
        assert [item["fragment_ids"] for item in items] == [
            ["top"],
            ["near"],
            ["far"],
        ]


class TestBaselineCli:
    def _make_dataset(self, tmp_path):
        frags_dir = tmp_path / "fragments"
        frags_dir.mkdir()
        page = module_page(
            "p1",
            [
                module_region("p1_r_1", 500, 0, 100, 200, "text one"),
                module_region("p1_r_2", 500, 210, 100, 200, "text two"),
            ],
        )
        (frags_dir / "p1.json").write_text(json.dumps(page), encoding="utf-8")
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        gt_dir.joinpath("p1.xml").write_text(
            """<?xml version='1.0' encoding='UTF-8'?>
<Articles pageId="p1">
  <Article uuid="u1" class="article">
    <Topics />
    <Notes />
    <Regions>
      <Region ref="p1_r_1" seq="1" />
      <Region ref="p1_r_2" seq="2" />
    </Regions>
  </Article>
</Articles>
""",
            encoding="utf-8",
        )
        return frags_dir, gt_dir

    def test_baseline_then_evaluate(self, tmp_path):
        frags_dir, gt_dir = self._make_dataset(tmp_path)
        out_dir = tmp_path / "recon"
        eval_dir = tmp_path / "evals"

        result = runner.invoke(
            app,
            [
                "baseline",
                "-i",
                str(frags_dir),
                "-o",
                str(out_dir),
                "--variant",
                "geometry",
                "--gap-k",
                "1.0",
            ],
        )
        assert result.exit_code == 0, result.output
        items = json.loads((out_dir / "p1.json").read_text())
        assert [i["fragment_ids"] for i in items] == [["p1_r_1", "p1_r_2"]]
        metadata = json.loads((out_dir / "_metadata.json").read_text())
        assert metadata["model"] == "baseline-geometry"
        assert metadata["model_kwargs"]["gap_k"] == 1.0

        result = runner.invoke(
            app,
            [
                "evaluate",
                "-i",
                str(out_dir),
                "-g",
                str(gt_dir),
                "--task",
                "reconstruction",
                "--eval-dir",
                str(eval_dir),
                "--experiment-id",
                "p1_baseline-geometry",
            ],
        )
        assert result.exit_code == 0, result.output
        log = json.loads((eval_dir / "p1_baseline-geometry.json").read_text())
        assert log["config"]["model"] == "baseline-geometry"
        assert log["pages"][0]["metrics"]["clustering_f1"] == 1.0

    def test_unknown_variant_fails(self, tmp_path):
        frags_dir, _ = self._make_dataset(tmp_path)
        result = runner.invoke(
            app,
            [
                "baseline",
                "-i",
                str(frags_dir),
                "-o",
                str(tmp_path / "out"),
                "--variant",
                "nope",
            ],
        )
        assert result.exit_code == 1

    def test_geometryless_fragments_fail_with_clear_error(self, tmp_path):
        frags_dir = tmp_path / "articles"
        frags_dir.mkdir()
        (frags_dir / "p1.json").write_text(
            json.dumps({"r1": "teks", "r2": "teks lain"}), encoding="utf-8"
        )
        result = runner.invoke(
            app,
            ["baseline", "-i", str(frags_dir), "-o", str(tmp_path / "out")],
        )
        assert result.exit_code == 1
        assert "module-format OCR" in result.output


class TestXYCutVariant:
    def test_two_columns_split_by_gutter(self):
        a1 = frag("a1", 0, 0, 100, 100)
        a2 = frag("a2", 0, 100, 100, 100)
        b1 = frag("b1", 300, 0, 100, 200)
        items = baseline_items([a1, a2, b1], variant="xy-cut", gap_k=1.0)
        assert [i["fragment_ids"] for i in items] == [["b1"], ["a1", "a2"]]

    def test_vertical_stack_split_by_whitespace(self):
        top = frag("t", 0, 0, 100, 100)
        bottom = frag("b", 0, 150, 100, 100)
        items = baseline_items([top, bottom], variant="xy-cut", gap_k=0.5)
        assert [i["fragment_ids"] for i in items] == [["t"], ["b"]]

    def test_touching_fragments_stay_one_leaf(self):
        a = frag("a", 0, 0, 100, 100)
        b = frag("b", 0, 100, 100, 100)
        items = baseline_items([a, b], variant="xy-cut", gap_k=0.5)
        assert [i["fragment_ids"] for i in items] == [["a", "b"]]

    def test_banner_blocks_x_cut(self):
        banner = frag("h", 0, 0, 400, 50)
        left_top = frag("l", 0, 60, 180, 90)
        right_top = frag("r", 220, 60, 180, 90)
        items = baseline_items(
            [banner, left_top, right_top], variant="xy-cut", gap_k=1.0
        )
        assert [i["fragment_ids"] for i in items] == [["h", "l", "r"]]

    def test_three_columns_multiway_split(self):
        cols = [
            frag("c1", 0, 0, 100, 200),
            frag("c2", 300, 0, 100, 200),
            frag("c3", 600, 0, 100, 200),
        ]
        items = baseline_items(cols, variant="xy-cut", gap_k=1.0)
        assert [i["fragment_ids"] for i in items] == [["c3"], ["c2"], ["c1"]]

    def test_thin_gutter_below_threshold_not_cut(self):
        a = frag("a", 0, 0, 100, 200)
        b = frag("b", 130, 0, 100, 200)
        items = baseline_items([a, b], variant="xy-cut", gap_k=0.5)
        assert [i["fragment_ids"] for i in items] == [["b", "a"]]

    def test_every_fragment_appears_once(self):
        frags = [
            frag("a1", 0, 0, 100, 100),
            frag("a2", 0, 100, 100, 100),
            frag("b1", 300, 0, 100, 200),
            frag("b2", 300, 250, 100, 100),
        ]
        items = baseline_items(frags, variant="xy-cut", gap_k=1.0)
        seen = [fid for i in items for fid in i["fragment_ids"]]
        assert sorted(seen) == ["a1", "a2", "b1", "b2"]


class TestXYCutPlusVariant:
    def test_spanner_detection_masks_wide_fragment(self):
        # banner spans both columns; two column tops below
        banner = frag("h", 0, 0, 600, 50)
        left = frag("l", 0, 100, 250, 200)
        right = frag("r", 350, 100, 250, 200)
        cross = detect_cross_layout([banner, left, right])
        assert cross == {"h"}

    def test_masking_opens_the_gutter(self):
        banner = frag("h", 0, 0, 600, 50)
        left = frag("l", 0, 100, 250, 200)
        right = frag("r", 350, 100, 250, 200)
        items = baseline_items([banner, left, right], variant="xy-cut-plus", gap_k=0.0)
        # unlike classic xy-cut, the columns separate; the masked banner is
        # remapped into exactly one of them (IoU tie -> nearest centre)
        groups = [i["fragment_ids"] for i in items]
        assert len(groups) == 2
        assert not any({"l", "r"} <= set(g) for g in groups)
        assert sum("h" in g for g in groups) == 1
        assert sorted(f for g in groups for f in g) == ["h", "l", "r"]

    def test_no_spanner_behaves_like_xy_cut(self):
        a = frag("a", 0, 0, 100, 100)
        b = frag("b", 0, 150, 100, 100)
        items = baseline_items([a, b], variant="xy-cut-plus", gap_k=0.5)
        assert [i["fragment_ids"] for i in items] == [["a"], ["b"]]

    def test_touching_fragments_stay_one_leaf(self):
        a = frag("a", 0, 0, 100, 100)
        b = frag("b", 0, 100, 100, 100)
        items = baseline_items([a, b], variant="xy-cut-plus", gap_k=0.5)
        assert [i["fragment_ids"] for i in items] == [["a", "b"]]

    def test_tau_d_and_axis_order(self):
        """XY-Cut++ density internals, tested directly.

        The leaf partition of axis-aligned fragments converges regardless
        of axis order (only cross-fragment remapping interleaves), so the
        density-driven priority is not reliably observable through
        baseline_items and is pinned here instead.
        """
        singles = [
            frag("s1", 0, 0, 200, 50),
            frag("s2", 0, 100, 200, 50),
            frag("s3", 300, 200, 200, 150),
        ]
        spanning = frag("c", 0, 0, 600, 100)
        cross = detect_cross_layout(singles + [spanning])
        assert cross == {"c"}
        low = _tau_d(singles, set())
        high = _tau_d(singles + [spanning], cross)
        assert low == 0.0
        assert high > THETA_V
        assert _axis_order(singles, set()) == ("x", "y")
        assert _axis_order(singles + [spanning], cross) == ("y", "x")

    def test_every_fragment_appears_once(self):
        frags = [
            frag("h", 0, 0, 600, 50),
            frag("a1", 0, 100, 250, 200),
            frag("a2", 0, 350, 250, 200),
            frag("b1", 350, 100, 250, 450),
        ]
        items = baseline_items(frags, variant="xy-cut-plus", gap_k=0.0)
        seen = [fid for i in items for fid in i["fragment_ids"]]
        assert sorted(seen) == ["a1", "a2", "b1", "h"]

    def test_empty_and_single(self):
        assert baseline_items([], variant="xy-cut-plus") == []
        one = baseline_items([frag("a", 0, 0, 10, 10)], variant="xy-cut-plus")
        assert one == [{"fragment_ids": ["a"], "title": "", "class": ""}]
