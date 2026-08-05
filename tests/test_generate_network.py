"""Tests for generate_network.py — CSV export of evaluation data for the network visualizer."""

import csv
import json

import pytest

from generate_network import (
    build_segment_maps,
    derive_eval_name,
    export_eval_log,
    export_page,
    load_fragments,
)


def _make_fragments():
    return [
        {"id": "r_1", "text": "fragment one", "type": "text", "hpos": 10, "vpos": 20, "width": 100, "height": 200},
        {"id": "r_2", "text": "fragment two", "type": "text", "hpos": 200, "vpos": 400, "width": 50, "height": 60},
        {"id": "r_3", "text": "fragment three", "type": "text", "hpos": 300, "vpos": 500, "width": 75, "height": 80},
    ]


def _make_eval_log(
    pages=None,
    prompt_name="v03",
    model="arc:lite",
    sample_size=16,
    seed=42,
):
    if pages is None:
        pages = []
    return {
        "run_id": "test_run",
        "timestamp": "2026-01-01T00:00:00",
        "config": {
            "provider": "openai",
            "model": model,
            "prompt_name": prompt_name,
            "sample_size": sample_size,
            "seed": seed,
        },
        "pages": pages,
        "aggregate": {},
    }


def _write_fragment_cache(tmp_path, page_id, fragments):
    frag_dir = tmp_path / "interim" / "fragments"
    frag_dir.mkdir(parents=True, exist_ok=True)
    with open(frag_dir / f"{page_id}.json", "w", encoding="utf-8") as f:
        json.dump(fragments, f, ensure_ascii=False)


def _read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ─── derive_eval_name ───────────────────────────────────────────────────────


class TestDeriveEvalName:
    def test_default_name(self):
        config = {"prompt_name": "v03", "model": "arc:lite", "sample_size": 16, "seed": 42}
        assert derive_eval_name(config) == "v03_arc:lite_sample16_seed42"

    def test_no_sample_size(self):
        config = {"prompt_name": "v01", "model": "arc:nexus", "sample_size": None, "seed": 42}
        assert derive_eval_name(config) == "v01_arc:nexus_seed42"

    def test_no_seed(self):
        config = {"prompt_name": "v01", "model": "arc:nexus", "sample_size": 8, "seed": None}
        assert derive_eval_name(config) == "v01_arc:nexus_sample8"

    def test_override(self):
        config = {"prompt_name": "v03", "model": "arc:lite", "sample_size": 16, "seed": 42}
        assert derive_eval_name(config, override="custom_name") == "custom_name"

    def test_missing_fields(self):
        config = {}
        assert derive_eval_name(config) == "unknown_unknown"


# ─── build_segment_maps ─────────────────────────────────────────────────────


class TestBuildSegmentMaps:
    def test_predicted_items(self):
        predicted = [
            {"fragment_ids": ["r_1", "r_2"], "title": "A", "class": "article"},
            {"fragment_ids": ["r_3"], "title": "B", "class": "advertisement"},
        ]
        llm_map, gt_map = build_segment_maps(predicted, None)
        assert llm_map == {"r_1": "0", "r_2": "0", "r_3": "1"}
        assert gt_map == {}

    def test_ground_truth_items(self):
        gt = [
            {"uuid": "uuid-aaa", "class": "article", "fragment_ids": ["r_1", "r_2"], "topics": []},
            {"uuid": "uuid-bbb", "class": "advertisement", "fragment_ids": ["r_3"], "topics": []},
        ]
        llm_map, gt_map = build_segment_maps(None, gt)
        assert llm_map == {}
        assert gt_map == {"r_1": "uuid-aaa", "r_2": "uuid-aaa", "r_3": "uuid-bbb"}

    def test_both(self):
        predicted = [{"fragment_ids": ["r_1", "r_2"], "title": "A", "class": "article"}]
        gt = [{"uuid": "uuid-aaa", "class": "article", "fragment_ids": ["r_1"], "topics": []}]
        llm_map, gt_map = build_segment_maps(predicted, gt)
        assert llm_map == {"r_1": "0", "r_2": "0"}
        assert gt_map == {"r_1": "uuid-aaa"}

    def test_empty(self):
        llm_map, gt_map = build_segment_maps(None, None)
        assert llm_map == {}
        assert gt_map == {}


# ─── load_fragments ──────────────────────────────────────────────────────────


class TestLoadFragments:
    def test_loads_existing(self, tmp_path):
        _write_fragment_cache(tmp_path, "test_page", _make_fragments())
        result = load_fragments("test_page", str(tmp_path / "interim"))
        assert len(result) == 3
        assert result[0]["id"] == "r_1"

    def test_returns_none_if_missing(self, tmp_path):
        result = load_fragments("nonexistent", str(tmp_path / "interim"))
        assert result is None


# ─── export_page ────────────────────────────────────────────────────────────


class TestExportPage:
    def test_nodes_csv_columns(self, tmp_path):
        fragments = _make_fragments()
        page = {
            "page_id": "UM-1956-01-09-6",
            "metrics": None,
            "predicted_items": [{"fragment_ids": ["r_1", "r_2"], "title": "A", "class": "article"}],
            "ground_truth_items": [{"uuid": "uuid-aaa", "class": "article", "fragment_ids": ["r_1"], "topics": []}],
        }
        nodes_dir = tmp_path / "nodes"
        edges_dir = tmp_path / "edges"
        nodes_dir.mkdir()
        edges_dir.mkdir()

        export_page(page, fragments, "https://example.com/scans", nodes_dir, edges_dir, "arc:lite")

        rows = _read_csv(nodes_dir / "UM-1956-01-09-6.csv")
        assert len(rows) == 3
        header = list(rows[0].keys())
        assert header == [
            "Image_URL", "Page_ID", "Region_ID", "Region_Text",
            "x1", "y1", "x2", "y2", "arc:lite_segment", "ground_truth_segment",
        ]

    def test_nodes_coords_and_segments(self, tmp_path):
        fragments = _make_fragments()
        page = {
            "page_id": "test_page",
            "metrics": None,
            "predicted_items": [
                {"fragment_ids": ["r_1", "r_2"], "title": "A", "class": "article"},
                {"fragment_ids": ["r_3"], "title": "B", "class": "ad"},
            ],
            "ground_truth_items": [
                {"uuid": "uuid-aaa", "class": "article", "fragment_ids": ["r_1"], "topics": []},
            ],
        }
        nodes_dir = tmp_path / "nodes"
        edges_dir = tmp_path / "edges"
        nodes_dir.mkdir()
        edges_dir.mkdir()

        export_page(page, fragments, "https://example.com/scans", nodes_dir, edges_dir, "arc:lite")

        rows = _read_csv(nodes_dir / "test_page.csv")
        assert rows[0]["Region_ID"] == "r_1"
        assert rows[0]["x1"] == "10"
        assert rows[0]["y1"] == "20"
        assert rows[0]["x2"] == "110"
        assert rows[0]["y2"] == "220"
        assert rows[0]["arc:lite_segment"] == "0"
        assert rows[0]["ground_truth_segment"] == "uuid-aaa"

        assert rows[1]["Region_ID"] == "r_2"
        assert rows[1]["arc:lite_segment"] == "0"
        assert rows[1]["ground_truth_segment"] == ""

        assert rows[2]["Region_ID"] == "r_3"
        assert rows[2]["arc:lite_segment"] == "1"
        assert rows[2]["ground_truth_segment"] == ""

    def test_image_url(self, tmp_path):
        fragments = _make_fragments()
        page = {
            "page_id": "test_page",
            "metrics": None,
            "predicted_items": [],
            "ground_truth_items": [],
        }
        nodes_dir = tmp_path / "nodes"
        edges_dir = tmp_path / "edges"
        nodes_dir.mkdir()
        edges_dir.mkdir()

        export_page(page, fragments, "https://example.com/scans/", nodes_dir, edges_dir, "arc:lite")

        rows = _read_csv(nodes_dir / "test_page.csv")
        assert rows[0]["Image_URL"] == "https://example.com/scans/test_page.jpg"

    def test_unassigned_fragments_appear(self, tmp_path):
        fragments = _make_fragments()
        page = {
            "page_id": "test_page",
            "metrics": None,
            "predicted_items": [{"fragment_ids": ["r_1"], "title": "A", "class": "article"}],
            "ground_truth_items": [],
        }
        nodes_dir = tmp_path / "nodes"
        edges_dir = tmp_path / "edges"
        nodes_dir.mkdir()
        edges_dir.mkdir()

        export_page(page, fragments, "https://example.com/scans", nodes_dir, edges_dir, "arc:lite")

        rows = _read_csv(nodes_dir / "test_page.csv")
        assert len(rows) == 3
        assert rows[1]["arc:lite_segment"] == ""
        assert rows[1]["ground_truth_segment"] == ""
        assert rows[2]["arc:lite_segment"] == ""
        assert rows[2]["ground_truth_segment"] == ""

    def test_edges_csv_same_item_only(self, tmp_path):
        fragments = _make_fragments()
        page = {
            "page_id": "test_page",
            "metrics": None,
            "predicted_items": [
                {"fragment_ids": ["r_1", "r_2"], "title": "A", "class": "article"},
                {"fragment_ids": ["r_3"], "title": "B", "class": "ad"},
            ],
            "ground_truth_items": [],
        }
        nodes_dir = tmp_path / "nodes"
        edges_dir = tmp_path / "edges"
        nodes_dir.mkdir()
        edges_dir.mkdir()

        export_page(page, fragments, "https://example.com/scans", nodes_dir, edges_dir, "arc:lite")

        edge_rows = _read_csv(edges_dir / "test_page.csv")
        edge_cols = list(edge_rows[0].keys())
        assert edge_cols == ["Image_URL", "Page_ID", "Source_Region_ID", "Target_Region_ID", "Hop_Distance"]

        pairs = {(r["Source_Region_ID"], r["Target_Region_ID"]) for r in edge_rows}
        assert ("r_1", "r_2") in pairs
        assert ("r_2", "r_1") in pairs
        assert len(edge_rows) == 2
        assert all(r["Hop_Distance"] == "1" for r in edge_rows)

    def test_edges_empty_for_single_fragment_items(self, tmp_path):
        fragments = _make_fragments()
        page = {
            "page_id": "test_page",
            "metrics": None,
            "predicted_items": [
                {"fragment_ids": ["r_1"], "title": "A", "class": "article"},
                {"fragment_ids": ["r_2"], "title": "B", "class": "ad"},
            ],
            "ground_truth_items": [],
        }
        nodes_dir = tmp_path / "nodes"
        edges_dir = tmp_path / "edges"
        nodes_dir.mkdir()
        edges_dir.mkdir()

        export_page(page, fragments, "https://example.com/scans", nodes_dir, edges_dir, "arc:lite")

        rows = _read_csv(edges_dir / "test_page.csv")
        assert len(rows) == 0

    def test_edges_empty_when_no_predicted_items(self, tmp_path):
        fragments = _make_fragments()
        page = {
            "page_id": "test_page",
            "metrics": None,
            "predicted_items": None,
            "ground_truth_items": None,
        }
        nodes_dir = tmp_path / "nodes"
        edges_dir = tmp_path / "edges"
        nodes_dir.mkdir()
        edges_dir.mkdir()

        export_page(page, fragments, "https://example.com/scans", nodes_dir, edges_dir, "arc:lite")

        edge_rows = _read_csv(edges_dir / "test_page.csv")
        assert len(edge_rows) == 0

    def test_jawi_text_quoted(self, tmp_path):
        fragments = [
            {"id": "r_1", "text": 'د راديوم "اتو اوبة', "type": "text", "hpos": 0, "vpos": 0, "width": 10, "height": 10},
        ]
        page = {
            "page_id": "test_page",
            "metrics": None,
            "predicted_items": [],
            "ground_truth_items": [],
        }
        nodes_dir = tmp_path / "nodes"
        edges_dir = tmp_path / "edges"
        nodes_dir.mkdir()
        edges_dir.mkdir()

        export_page(page, fragments, "https://example.com/scans", nodes_dir, edges_dir, "arc:lite")

        with open(nodes_dir / "test_page.csv", encoding="utf-8") as f:
            content = f.read()
        assert '"د راديوم ""اتو اوبة"' in content


# ─── export_eval_log ────────────────────────────────────────────────────────


class TestExportEvalLog:
    def test_creates_directory_structure(self, tmp_path):
        fragments = _make_fragments()
        _write_fragment_cache(tmp_path, "test_page", fragments)

        log = _make_eval_log(
            pages=[{
                "page_id": "test_page",
                "metrics": None,
                "predicted_items": [{"fragment_ids": ["r_1", "r_2"], "title": "A", "class": "article"}],
                "ground_truth_items": [{"uuid": "uuid-aaa", "class": "article", "fragment_ids": ["r_1"], "topics": []}],
            }]
        )
        log_path = tmp_path / "eval.json"
        log_path.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")

        result = export_eval_log(
            str(log_path),
            str(tmp_path / "output"),
            "https://example.com/scans",
            str(tmp_path / "interim"),
        )

        eval_dir = tmp_path / "output" / "v03_arc:lite_sample16_seed42"
        assert eval_dir.exists()
        assert (eval_dir / "nodes" / "test_page.csv").exists()
        assert (eval_dir / "edges" / "test_page.csv").exists()
        assert result == str(eval_dir)

    def test_multi_page(self, tmp_path):
        _write_fragment_cache(tmp_path, "page_a", _make_fragments())
        _write_fragment_cache(tmp_path, "page_b", _make_fragments())

        log = _make_eval_log(pages=[
            {
                "page_id": "page_a",
                "metrics": None,
                "predicted_items": [{"fragment_ids": ["r_1", "r_2"], "title": "A", "class": "article"}],
                "ground_truth_items": [],
            },
            {
                "page_id": "page_b",
                "metrics": None,
                "predicted_items": [{"fragment_ids": ["r_1"], "title": "B", "class": "article"}],
                "ground_truth_items": [],
            },
        ])
        log_path = tmp_path / "eval.json"
        log_path.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")

        export_eval_log(
            str(log_path),
            str(tmp_path / "output"),
            "https://example.com/scans",
            str(tmp_path / "interim"),
        )

        eval_dir = tmp_path / "output" / "v03_arc:lite_sample16_seed42"
        assert (eval_dir / "nodes" / "page_a.csv").exists()
        assert (eval_dir / "nodes" / "page_b.csv").exists()
        assert (eval_dir / "edges" / "page_a.csv").exists()
        assert (eval_dir / "edges" / "page_b.csv").exists()

    def test_eval_name_override(self, tmp_path):
        _write_fragment_cache(tmp_path, "test_page", _make_fragments())

        log = _make_eval_log(pages=[{
            "page_id": "test_page",
            "metrics": None,
            "predicted_items": [],
            "ground_truth_items": [],
        }])
        log_path = tmp_path / "eval.json"
        log_path.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")

        export_eval_log(
            str(log_path),
            str(tmp_path / "output"),
            "https://example.com/scans",
            str(tmp_path / "interim"),
            eval_name="custom_eval",
        )

        assert (tmp_path / "output" / "custom_eval" / "nodes" / "test_page.csv").exists()

    def test_skips_page_with_missing_fragments(self, tmp_path):
        log = _make_eval_log(pages=[{
            "page_id": "missing_page",
            "metrics": None,
            "predicted_items": [],
            "ground_truth_items": [],
        }])
        log_path = tmp_path / "eval.json"
        log_path.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")

        export_eval_log(
            str(log_path),
            str(tmp_path / "output"),
            "https://example.com/scans",
            str(tmp_path / "interim"),
        )

        eval_dir = tmp_path / "output" / "v03_arc:lite_sample16_seed42"
        assert not (eval_dir / "nodes" / "missing_page.csv").exists()

    def test_failed_page_no_predicted_items(self, tmp_path):
        _write_fragment_cache(tmp_path, "failed_page", _make_fragments())

        log = _make_eval_log(pages=[{
            "page_id": "failed_page",
            "metrics": None,
            "predicted_items": None,
            "ground_truth_items": None,
        }])
        log_path = tmp_path / "eval.json"
        log_path.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")

        export_eval_log(
            str(log_path),
            str(tmp_path / "output"),
            "https://example.com/scans",
            str(tmp_path / "interim"),
        )

        eval_dir = tmp_path / "output" / "v03_arc:lite_sample16_seed42"
        node_rows = _read_csv(eval_dir / "nodes" / "failed_page.csv")
        assert len(node_rows) == 3
        assert all(r["arc:lite_segment"] == "" for r in node_rows)
        assert all(r["ground_truth_segment"] == "" for r in node_rows)

        edge_rows = _read_csv(eval_dir / "edges" / "failed_page.csv")
        assert len(edge_rows) == 0


# ─── CLI (main) ──────────────────────────────────────────────────────────────


class TestCli:
    def test_help(self):
        from generate_network import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_missing_eval_log(self, tmp_path):
        from generate_network import main
        ret = main(["--eval-log", str(tmp_path / "nonexistent.json")])
        assert ret == 1

    def test_env_var_image_base_url(self, tmp_path, monkeypatch):
        _write_fragment_cache(tmp_path, "test_page", _make_fragments())

        log = _make_eval_log(pages=[{
            "page_id": "test_page",
            "metrics": None,
            "predicted_items": [],
            "ground_truth_items": [],
        }])
        log_path = tmp_path / "eval.json"
        log_path.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")

        monkeypatch.setenv("IMAGE_BASE_URL", "https://env.example.com/imgs")
        from generate_network import main
        main([
            "--eval-log", str(log_path),
            "--output-dir", str(tmp_path / "output"),
            "--interim-dir", str(tmp_path / "interim"),
        ])

        rows = _read_csv(tmp_path / "output" / "v03_arc:lite_sample16_seed42" / "nodes" / "test_page.csv")
        assert rows[0]["Image_URL"] == "https://env.example.com/imgs/test_page.jpg"

    def test_cli_image_base_url_flag(self, tmp_path):
        _write_fragment_cache(tmp_path, "test_page", _make_fragments())

        log = _make_eval_log(pages=[{
            "page_id": "test_page",
            "metrics": None,
            "predicted_items": [],
            "ground_truth_items": [],
        }])
        log_path = tmp_path / "eval.json"
        log_path.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")

        from generate_network import main
        main([
            "--eval-log", str(log_path),
            "--output-dir", str(tmp_path / "output"),
            "--image-base-url", "https://flag.example.com/scans",
            "--interim-dir", str(tmp_path / "interim"),
        ])

        rows = _read_csv(tmp_path / "output" / "v03_arc:lite_sample16_seed42" / "nodes" / "test_page.csv")
        assert rows[0]["Image_URL"] == "https://flag.example.com/scans/test_page.jpg"

    def test_cli_eval_name_override(self, tmp_path):
        _write_fragment_cache(tmp_path, "test_page", _make_fragments())

        log = _make_eval_log(pages=[{
            "page_id": "test_page",
            "metrics": None,
            "predicted_items": [],
            "ground_truth_items": [],
        }])
        log_path = tmp_path / "eval.json"
        log_path.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")

        from generate_network import main
        main([
            "--eval-log", str(log_path),
            "--output-dir", str(tmp_path / "output"),
            "--eval-name", "my_eval",
            "--interim-dir", str(tmp_path / "interim"),
        ])

        assert (tmp_path / "output" / "my_eval" / "nodes" / "test_page.csv").exists()
