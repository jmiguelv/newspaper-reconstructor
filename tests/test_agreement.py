import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from main import app
from src.newspaper_reconstructor.agreement import (
    AgreementError,
    Article,
    calculate_agreement,
    copy_agreed_pages,
    format_summary,
    load_annotations,
    match_articles,
    page_agreement,
    validate_page_sets,
    write_agreement_reports,
)

DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/0_external/ds_article_20260902",
)


def _write_xml(path, articles):
    """Write an annotation XML file.

    articles: list of (uuid, [region refs])
    """
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<Articles>"]
    for uuid, regions in articles:
        parts.append(f'  <Article uuid="{uuid}" class="article">')
        parts.append("    <Topics></Topics>")
        parts.append("    <Regions>")
        for i, ref in enumerate(regions, start=1):
            parts.append(f'      <Region ref="{ref}" seq="{i}"/>')
        parts.append("    </Regions>")
        parts.append("  </Article>")
    parts.append("</Articles>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _make_dir(tmp_path, name, pages):
    d = tmp_path / name
    d.mkdir()
    for page, articles in pages.items():
        _write_xml(d / f"{page}.xml", articles)
    return str(d)


# ─── Task 1: loader + validation ─────────────────────────────────────────────


class TestLoadAnnotations:
    def test_parses_uuid_and_regions(self, tmp_path):
        d = _make_dir(
            tmp_path,
            "a",
            {
                "p1": [
                    ("uuid-1", ["r_1", "r_2"]),
                    ("uuid-2", ["r_3"]),
                ]
            },
        )

        ann = load_annotations(d)

        assert list(ann.keys()) == ["p1"]
        assert ann["p1"] == [
            Article(uuid="uuid-1", regions=("r_1", "r_2")),
            Article(uuid="uuid-2", regions=("r_3",)),
        ]

    def test_ignores_non_xml_files(self, tmp_path):
        d = tmp_path / "a"
        d.mkdir()
        _write_xml(d / "p1.xml", [("uuid-1", ["r_1"])])
        (d / "notes.txt").write_text("hello")

        ann = load_annotations(str(d))

        assert list(ann.keys()) == ["p1"]

    def test_duplicate_region_ref_within_page_raises(self, tmp_path):
        d = _make_dir(
            tmp_path,
            "a",
            {
                "p1": [
                    ("uuid-1", ["r_1"]),
                    ("uuid-2", ["r_1", "r_2"]),
                ]
            },
        )

        with pytest.raises(AgreementError, match="r_1.*p1"):
            load_annotations(d)

    def test_malformed_xml_raises(self, tmp_path):
        d = tmp_path / "a"
        d.mkdir()
        (d / "p1.xml").write_text("<Articles><Article></Articles>")

        with pytest.raises(AgreementError, match="p1.xml"):
            load_annotations(str(d))

    def test_missing_region_ref_attribute_raises(self, tmp_path):
        d = tmp_path / "a"
        d.mkdir()
        (d / "p1.xml").write_text(
            '<Articles><Article uuid="u1"><Regions><Region seq="1"/></Regions>'
            "</Article></Articles>"
        )

        with pytest.raises(AgreementError, match="ref"):
            load_annotations(str(d))


class TestValidatePageSets:
    def test_matching_sets_pass(self, tmp_path):
        pages = {"p1": [("u1", ["r_1"])]}
        a = load_annotations(_make_dir(tmp_path, "a", pages))
        b = load_annotations(_make_dir(tmp_path, "b", pages))

        validate_page_sets(a, b)

    def test_missing_page_raises_with_names(self, tmp_path):
        a = load_annotations(
            _make_dir(tmp_path, "a", {"p1": [("u1", ["r_1"])], "p2": [("u2", ["r_2"])]})
        )
        b = load_annotations(_make_dir(tmp_path, "b", {"p1": [("u3", ["r_1"])]}))

        with pytest.raises(AgreementError, match="p2.xml"):
            validate_page_sets(a, b)


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(DATASET, "article_xml_frial")),
    reason="ds_article_20260902 dataset not available",
)
class TestLoadAnnotationsRealData:
    def test_loads_both_annotators(self):
        frial = load_annotations(os.path.join(DATASET, "article_xml_frial"))
        syafiq = load_annotations(os.path.join(DATASET, "article_xml_syafiq"))

        assert len(frial) == 80
        assert len(syafiq) == 80
        validate_page_sets(frial, syafiq)

        total_frial = sum(len(arts) for arts in frial.values())
        total_syafiq = sum(len(arts) for arts in syafiq.values())
        assert total_frial == 708
        assert total_syafiq == 807


# ─── Task 2: greedy Jaccard matcher ──────────────────────────────────────────


class TestMatchArticles:
    def test_identical_region_sets_match(self):
        a = [Article("u1", ("r_1", "r_2")), Article("u2", ("r_3",))]
        b = [Article("v1", ("r_2", "r_1")), Article("v2", ("r_3",))]

        result = match_articles(a, b)

        assert [(pa.uuid, pb.uuid) for pa, pb, _ in result.matched] == [
            ("u1", "v1"),
            ("u2", "v2"),
        ]
        assert result.partial == []
        assert result.unmatched_a == []
        assert result.unmatched_b == []

    def test_partial_overlap_listed_not_matched(self):
        a = [Article("u1", ("r_1", "r_2", "r_3"))]
        b = [Article("v1", ("r_1", "r_2"))]

        result = match_articles(a, b)

        assert result.matched == []
        assert len(result.partial) == 1
        pa, pb, j = result.partial[0]
        assert (pa.uuid, pb.uuid) == ("u1", "v1")
        assert j == pytest.approx(2 / 3)
        assert result.unmatched_a == []
        assert result.unmatched_b == []

    def test_one_sided_articles_unmatched(self):
        a = [Article("u1", ("r_1",)), Article("u2", ("r_2",))]
        b = [Article("v1", ("r_1",))]

        result = match_articles(a, b)

        assert [(pa.uuid, pb.uuid) for pa, pb, _ in result.matched] == [("u1", "v1")]
        assert result.partial == []
        assert [art.uuid for art in result.unmatched_a] == ["u2"]
        assert result.unmatched_b == []

    def test_greedy_takes_highest_jaccard_below_threshold(self):
        a = [Article("u1", ("r_1", "r_2", "r_3")), Article("u2", ("r_3", "r_4"))]
        b = [Article("v1", ("r_1", "r_2"))]

        result = match_articles(a, b, threshold=0.5)

        # u1-v1 has J=2/3 >= 0.5 and beats u2-v1 (J=1/3)
        assert [(pa.uuid, pb.uuid) for pa, pb, _ in result.matched] == [("u1", "v1")]
        assert result.unmatched_a == [a[1]]

    def test_greedy_tie_break_by_uuid(self):
        a = [Article("u2", ("r_1", "r_2")), Article("u1", ("r_1", "r_2"))]
        b = [Article("v1", ("r_1", "r_2"))]

        result = match_articles(a, b)

        assert [(pa.uuid, pb.uuid) for pa, pb, _ in result.matched] == [("u1", "v1")]
        assert [art.uuid for art in result.unmatched_a] == ["u2"]

    def test_empty_region_articles_never_matched(self):
        a = [Article("u1", ()), Article("u2", ("r_1",))]
        b = [Article("v1", ()), Article("v2", ("r_1",))]

        result = match_articles(a, b)

        assert [(pa.uuid, pb.uuid) for pa, pb, _ in result.matched] == [("u2", "v2")]
        assert [art.uuid for art in result.no_regions_a] == ["u1"]
        assert [art.uuid for art in result.no_regions_b] == ["v1"]

    def test_split_shows_two_partials(self):
        a = [Article("u1", ("r_1", "r_2", "r_3"))]
        b = [Article("v1", ("r_1",)), Article("v2", ("r_2", "r_3"))]

        result = match_articles(a, b)

        assert result.matched == []
        assert result.unmatched_a == []
        assert result.unmatched_b == []
        assert len(result.partial) == 2
        js = {pb.uuid: j for _, pb, j in result.partial}
        assert js["v1"] == pytest.approx(1 / 3)
        assert js["v2"] == pytest.approx(2 / 3)


# ─── Task 3: page-level metrics ──────────────────────────────────────────────


class TestPageAgreement:
    def test_identical_pages_score_perfectly(self):
        a = [Article("u1", ("r_1", "r_2")), Article("u2", ("r_3",))]
        b = [Article("v1", ("r_1", "r_2")), Article("v2", ("r_3",))]

        m = page_agreement(a, b)

        assert m["matched"] == 2
        assert m["clustering_f1"] == 1.0
        assert m["bcubed_f1"] == 1.0
        assert m["full_agreement_rate_a"] == 1.0
        assert m["full_agreement_rate_b"] == 1.0
        assert m["regions_both"] == 3
        assert m["regions_only_a"] == 0
        assert m["regions_only_b"] == 0

    def test_partial_overlap_hand_computed(self):
        a = [Article("u1", ("r_1", "r_2")), Article("u2", ("r_3",))]
        b = [Article("v1", ("r_1", "r_2")), Article("v2", ("r_3", "r_4"))]

        m = page_agreement(a, b)

        assert m["matched"] == 1
        assert m["clustering_precision"] == 1.0
        assert m["clustering_recall"] == 0.5
        assert m["clustering_f1"] == pytest.approx(2 / 3)
        assert m["bcubed_precision"] == 1.0
        assert m["bcubed_recall"] == 0.75
        assert m["bcubed_f1"] == pytest.approx(6 / 7)
        assert m["full_agreement_rate_a"] == 0.5
        assert m["full_agreement_rate_b"] == 0.5
        assert m["regions_both"] == 3
        assert m["regions_only_a"] == 0
        assert m["regions_only_b"] == 1

    def test_swapped_partitions_score_zero(self):
        a = [Article("u1", ("r_1", "r_2")), Article("u2", ("r_3",))]
        b = [Article("v1", ("r_1",)), Article("v2", ("r_2", "r_3"))]

        m = page_agreement(a, b)

        assert m["matched"] == 0
        assert m["clustering_f1"] == 0.0
        assert m["bcubed_f1"] == pytest.approx(2 / 3)

    def test_metrics_symmetric(self):
        a = [Article("u1", ("r_1", "r_2")), Article("u2", ("r_3", "r_4"))]
        b = [Article("v1", ("r_1", "r_2", "r_3")), Article("v2", ("r_4",))]

        forward = page_agreement(a, b)
        backward = page_agreement(b, a)

        assert forward["clustering_f1"] == pytest.approx(backward["clustering_f1"])
        assert forward["bcubed_f1"] == pytest.approx(backward["bcubed_f1"])

    def test_zero_matched_pairs_full_rate_is_none(self):
        a = [Article("u1", ("r_1",))]
        b = [Article("v1", ("r_2",))]

        m = page_agreement(a, b)

        assert m["matched"] == 0
        assert m["full_agreement_rate_a"] is None
        assert m["full_agreement_rate_b"] is None

    def test_disagreement_records(self):
        a = [
            Article("u1", ("r_1", "r_2")),
            Article("u2", ("r_3",)),
            Article("u3", ("r_9",)),
            Article("u4", ()),
        ]
        b = [
            Article("v1", ("r_1", "r_2", "r_3")),
            Article("v2", ("r_8",)),
            Article("v3", ()),
        ]

        m = page_agreement(a, b)

        assert m["matched"] == 0
        partials = {(d["uuid_a"], d["uuid_b"]): d for d in m["partial"]}
        assert set(partials) == {("u1", "v1"), ("u2", "v1")}
        assert partials[("u1", "v1")]["only_a_regions"] == []
        assert partials[("u1", "v1")]["only_b_regions"] == ["r_3"]
        assert [d["uuid"] for d in m["unmatched_a"]] == ["u3"]
        assert [d["uuid"] for d in m["unmatched_b"]] == ["v2"]
        assert [d["uuid"] for d in m["no_regions_a"]] == ["u4"]
        assert [d["uuid"] for d in m["no_regions_b"]] == ["v3"]


# ─── Task 4: aggregation + reports ───────────────────────────────────────────


class TestComputeAgreement:
    def test_mismatched_page_sets_raise(self, tmp_path):
        a = load_annotations(_make_dir(tmp_path, "a", {"p1": [("u1", ["r_1"])]}))
        b = load_annotations(
            _make_dir(tmp_path, "b", {"p1": [("v1", ["r_1"])], "p2": [("v2", ["r_2"])]})
        )

        with pytest.raises(AgreementError, match="p2.xml"):
            calculate_agreement(a, b, name_a="A", name_b="B")

    def test_aggregate_over_pages(self, tmp_path):
        pages_a = {
            "p1": [("u1", ["r_1", "r_2"]), ("u2", ["r_3"])],
            "p2": [("u3", ["r_4", "r_5"])],
            "p3": [("u5", ["r_8", "r_9"])],
        }
        pages_b = {
            "p1": [("v1", ["r_1", "r_2"]), ("v2", ["r_3"])],
            "p2": [("v3", ["r_4", "r_5", "r_6"])],
            "p3": [("v5", ["r_8"]), ("v6", ["r_9"])],
        }
        a = load_annotations(_make_dir(tmp_path, "a", pages_a))
        b = load_annotations(_make_dir(tmp_path, "b", pages_b))

        result = calculate_agreement(a, b, name_a="frial", name_b="syafiq")

        assert result["name_a"] == "frial"
        assert result["name_b"] == "syafiq"
        assert result["threshold"] == 1.0
        assert [p["page_id"] for p in result["pages"]] == ["p1", "p2", "p3"]

        agg = result["aggregate"]
        assert agg["pages"] == 3
        assert agg["articles_a"] == 4
        assert agg["articles_b"] == 5
        assert agg["matched"] == 2
        assert agg["partial"] == 3
        # micro: matched articles / all articles with regions
        assert agg["full_agreement_rate_a"] == 0.5
        assert agg["full_agreement_rate_b"] == 0.4
        # macro: only pages with at least one matched pair (p1: 2/2)
        assert agg["mean_full_agreement_rate_a"] == 1.0
        assert agg["mean_full_agreement_rate_b"] == 1.0
        # p1 f1=1.0, p2 f1=0.5 (B merges r_6 into the pair → 2 fp pairs), p3 f1=0.0
        assert agg["mean_clustering_f1"] == pytest.approx(0.5)
        # pooled: p1 both=3, p2 both=2 + only_b r_6, p3 both=2
        assert agg["regions_both"] == 7
        assert agg["regions_only_a"] == 0
        assert agg["regions_only_b"] == 1

    def test_zero_matched_page_excluded_from_rate_average(self, tmp_path):
        pages_a = {
            "p1": [("u1", ["r_1", "r_2"])],
            "p2": [("u2", ["r_3"])],
        }
        pages_b = {
            "p1": [("v1", ["r_1", "r_2"])],
            "p2": [("v2", ["r_4"])],
        }
        a = load_annotations(_make_dir(tmp_path, "a", pages_a))
        b = load_annotations(_make_dir(tmp_path, "b", pages_b))

        result = calculate_agreement(a, b, name_a="A", name_b="B")

        # micro counts the zero-match page (1/2); macro excludes it (1/1)
        assert result["aggregate"]["full_agreement_rate_a"] == 0.5
        assert result["aggregate"]["full_agreement_rate_b"] == 0.5
        assert result["aggregate"]["mean_full_agreement_rate_a"] == 1.0
        assert result["aggregate"]["mean_full_agreement_rate_b"] == 1.0


class TestWriteAgreementReports:
    def test_writes_json_and_markdown(self, tmp_path):
        pages_a = {
            "p1": [
                ("u1", ["r_1", "r_2"]),
                ("u2", ["r_3"]),
                ("u3", ["r_9"]),
                ("u4", []),
            ]
        }
        pages_b = {
            "p1": [
                ("v1", ["r_1", "r_2", "r_3"]),
                ("v2", ["r_8"]),
                ("v3", []),
            ]
        }
        a = load_annotations(_make_dir(tmp_path, "a", pages_a))
        b = load_annotations(_make_dir(tmp_path, "b", pages_b))
        result = calculate_agreement(a, b, name_a="frial", name_b="syafiq")

        out = tmp_path / "reports"
        json_path, md_path = write_agreement_reports(result, str(out))

        assert out.is_dir()
        assert json_path.startswith(str(out / "frial_vs_syafiq_"))
        assert json_path.endswith(".json")
        assert md_path.endswith(".md")

        log = json.loads(Path(json_path).read_text())
        assert log["name_a"] == "frial"
        assert log["pages"][0]["page_id"] == "p1"
        assert len(log["pages"][0]["partial"]) == 2

        md = Path(md_path).read_text()
        assert "# Annotation agreement: frial vs syafiq" in md
        assert "## Summary" in md
        assert "## Per-page breakdown" in md
        assert "## Disagreements" in md
        assert "### Unmatched articles — frial only (1)" in md
        assert "### Unmatched articles — syafiq only (1)" in md
        assert "### Articles without regions (frial: 1, syafiq: 1)" in md
        assert "### Partial matches — region sets differ (2)" in md
        assert "u3" in md and "r_9" in md

    def test_perfect_agreement_has_no_disagreement_entries(self, tmp_path):
        pages = {"p1": [("u1", ["r_1", "r_2"])]}
        a = load_annotations(_make_dir(tmp_path, "a", pages))
        b = load_annotations(_make_dir(tmp_path, "b", pages))
        result = calculate_agreement(a, b, name_a="A", name_b="B")

        _, md_path = write_agreement_reports(result, str(tmp_path / "out"))
        md = Path(md_path).read_text()

        assert "## Disagreements" in md
        assert "Unmatched articles" not in md
        assert "Partial matches" not in md


class TestFormatSummary:
    def test_summary_mentions_key_numbers(self, tmp_path):
        pages_a = {"p1": [("u1", ["r_1", "r_2"]), ("u2", ["r_3"])]}
        pages_b = {"p1": [("v1", ["r_1", "r_2"]), ("v2", ["r_3"])]}
        a = load_annotations(_make_dir(tmp_path, "a", pages_a))
        b = load_annotations(_make_dir(tmp_path, "b", pages_b))
        result = calculate_agreement(a, b, name_a="frial", name_b="syafiq")

        summary = format_summary(result)

        assert "frial" in summary and "syafiq" in summary
        assert "Pages: 1" in summary
        assert "100.0%" in summary


# ─── Task 5: CLI e2e ──────────────────────────────────────────────────────────

runner = CliRunner()


class TestAgreeCommand:
    def test_agree_writes_reports_and_prints_summary(self, tmp_path):
        pages_a = {"p1": [("u1", ["r_1", "r_2"]), ("u2", ["r_3"])]}
        pages_b = {"p1": [("v1", ["r_1", "r_2"]), ("v2", ["r_3", "r_4"])]}
        dir_a = _make_dir(tmp_path, "frial", pages_a)
        dir_b = _make_dir(tmp_path, "syafiq", pages_b)
        out = tmp_path / "reports"

        result = runner.invoke(
            app,
            [
                "agree",
                "--annotator-a",
                dir_a,
                "--annotator-b",
                dir_b,
                "--output",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Annotation agreement: frial vs syafiq" in result.output
        assert "Fully agreeing pairs: 1" in result.output
        files = list(out.iterdir())
        assert len(files) == 2
        assert sum(1 for f in files if f.suffix == ".json") == 1
        assert sum(1 for f in files if f.suffix == ".md") == 1

    def test_agree_custom_names_in_report(self, tmp_path):
        pages = {"p1": [("u1", ["r_1"])]}
        dir_a = _make_dir(tmp_path, "annA", pages)
        dir_b = _make_dir(tmp_path, "annB", pages)

        result = runner.invoke(
            app,
            [
                "agree",
                "--annotator-a",
                dir_a,
                "--annotator-b",
                dir_b,
                "--name-a",
                "alice",
                "--name-b",
                "bob",
                "--output",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "alice vs bob" in result.output
        assert any(
            f.name.startswith("alice_vs_bob") for f in (tmp_path / "out").iterdir()
        )

    def test_agree_missing_directory_exits_1(self, tmp_path):
        dir_a = _make_dir(tmp_path, "a", {"p1": [("u1", ["r_1"])]})

        result = runner.invoke(
            app,
            [
                "agree",
                "--annotator-a",
                dir_a,
                "--annotator-b",
                str(tmp_path / "nope"),
            ],
        )

        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_agree_mismatched_files_exits_1(self, tmp_path):
        dir_a = _make_dir(tmp_path, "a", {"p1": [("u1", ["r_1"])]})
        dir_b = _make_dir(
            tmp_path, "b", {"p1": [("v1", ["r_1"])], "p2": [("v2", ["r_2"])]}
        )

        result = runner.invoke(
            app,
            [
                "agree",
                "--annotator-a",
                dir_a,
                "--annotator-b",
                dir_b,
                "--output",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "p2.xml" in result.output

    def test_agree_duplicate_region_exits_1(self, tmp_path):
        dir_a = _make_dir(tmp_path, "a", {"p1": [("u1", ["r_1"]), ("u2", ["r_1"])]})
        dir_b = _make_dir(tmp_path, "b", {"p1": [("v1", ["r_1"])]})

        result = runner.invoke(
            app,
            [
                "agree",
                "--annotator-a",
                dir_a,
                "--annotator-b",
                dir_b,
                "--output",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "r_1" in result.output

    def test_agree_invalid_threshold_exits_1(self, tmp_path):
        pages = {"p1": [("u1", ["r_1"])]}
        dir_a = _make_dir(tmp_path, "a", pages)
        dir_b = _make_dir(tmp_path, "b", pages)

        result = runner.invoke(
            app,
            [
                "agree",
                "--annotator-a",
                dir_a,
                "--annotator-b",
                dir_b,
                "--match-threshold",
                "1.5",
                "--output",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "match-threshold" in result.output


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(DATASET, "article_xml_frial")),
    reason="ds_article_20260902 dataset not available",
)
class TestAgreeRealData:
    def test_agree_on_real_annotations(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "agree",
                "--annotator-a",
                os.path.join(DATASET, "article_xml_frial"),
                "--annotator-b",
                os.path.join(DATASET, "article_xml_syafiq"),
                "--output",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Pages: 80" in result.output
        files = list((tmp_path / "out").iterdir())
        assert len(files) == 2


# ─── Agreed pages (B-cubed F1 threshold) ─────────────────────────────────────


class TestAgreedPages:
    def test_default_threshold_lists_completely_agreed_pages(self, tmp_path):
        pages_a = {
            "p1": [("u1", ["r_1", "r_2"]), ("u2", ["r_3"])],
            "p2": [("u3", ["r_4", "r_5"])],
            "p3": [("u5", ["r_6"])],
        }
        pages_b = {
            "p1": [("v2", ["r_3"]), ("v1", ["r_1", "r_2"])],
            "p2": [("v3", ["r_4", "r_5", "r_6"])],
            "p3": [("v5", ["r_7"])],
        }
        a = load_annotations(_make_dir(tmp_path, "a", pages_a))
        b = load_annotations(_make_dir(tmp_path, "b", pages_b))

        result = calculate_agreement(a, b, name_a="A", name_b="B")

        # p1: identical partitions despite different order/uuids;
        # p3 excluded: disjoint region sets score a trivial B-cubed F1 of 1.0
        assert result["aggregate"]["agreed_pages"] == ["p1"]
        assert result["agreement_threshold"] == 1.0

    def test_threshold_below_one_includes_imperfect_pages(self, tmp_path):
        pages_a = {
            "p1": [("u1", ["r_1", "r_2"]), ("u2", ["r_3"])],
            "p2": [("u3", ["r_4", "r_5"])],
            "p3": [("u5", ["r_6"])],
        }
        pages_b = {
            "p1": [("v2", ["r_3"]), ("v1", ["r_1", "r_2"])],
            "p2": [("v3", ["r_4", "r_5", "r_6"])],
            "p3": [("v5", ["r_7"])],
        }
        a = load_annotations(_make_dir(tmp_path, "a", pages_a))
        b = load_annotations(_make_dir(tmp_path, "b", pages_b))

        result = calculate_agreement(
            a, b, name_a="A", name_b="B", agreement_threshold=0.5
        )

        # pure B-cubed rule below 1.0: p1=1.0, p2≈0.78, p3=1.0 (trivially)
        assert result["aggregate"]["agreed_pages"] == ["p1", "p2", "p3"]

    def test_report_lists_agreed_pages(self, tmp_path):
        pages_a = {
            "p1": [("u1", ["r_1", "r_2"])],
            "p2": [("u2", ["r_3", "r_4"])],
        }
        pages_b = {
            "p1": [("v1", ["r_1", "r_2"])],
            "p2": [("v1", ["r_3"]), ("v2", ["r_4"])],
        }
        a = load_annotations(_make_dir(tmp_path, "a", pages_a))
        b = load_annotations(_make_dir(tmp_path, "b", pages_b))
        result = calculate_agreement(a, b, name_a="A", name_b="B")

        _, md_path = write_agreement_reports(result, str(tmp_path / "out"))
        md = Path(md_path).read_text()

        assert "## Pages with agreement (B-cubed F1 ≥ 1.00) (1)" in md
        assert "- p1" in md

    def test_copy_agreed_pages_copies_both_annotators(self, tmp_path):
        pages_a = {
            "p1": [("u1", ["r_1", "r_2"])],
            "p2": [("u2", ["r_3", "r_4"])],
        }
        pages_b = {
            "p1": [("v1", ["r_1", "r_2"])],
            "p2": [("v1", ["r_3"]), ("v2", ["r_4"])],
        }
        dir_a = _make_dir(tmp_path, "a", pages_a)
        dir_b = _make_dir(tmp_path, "b", pages_b)
        result = calculate_agreement(
            load_annotations(dir_a),
            load_annotations(dir_b),
            name_a="frial",
            name_b="syafiq",
        )

        dest = tmp_path / "agreed"
        copied = copy_agreed_pages(result, dir_a, dir_b, str(dest))

        assert len(copied) == 2
        assert (dest / "frial" / "p1.xml").exists()
        assert (dest / "syafiq" / "p1.xml").exists()
        # copy, not move: originals stay in place
        assert (Path(dir_a) / "p1.xml").exists()
        assert (Path(dir_b) / "p1.xml").exists()
        assert (Path(dir_a) / "p2.xml").exists()
        assert (Path(dir_b) / "p2.xml").exists()

    def test_copy_agreed_pages_nothing_agreed_is_noop(self, tmp_path):
        pages_a = {"p1": [("u1", ["r_1"])]}
        pages_b = {"p1": [("v1", ["r_2"])]}
        dir_a = _make_dir(tmp_path, "a", pages_a)
        dir_b = _make_dir(tmp_path, "b", pages_b)
        result = calculate_agreement(
            load_annotations(dir_a), load_annotations(dir_b), name_a="A", name_b="B"
        )

        dest = tmp_path / "agreed"
        copied = copy_agreed_pages(result, dir_a, dir_b, str(dest))

        assert copied == []
        assert not dest.exists()
        assert (Path(dir_a) / "p1.xml").exists()


class TestAgreeCopyAgreedCli:
    def test_copy_agreed_copies_files(self, tmp_path):
        pages_a = {
            "p1": [("u1", ["r_1", "r_2"])],
            "p2": [("u2", ["r_3", "r_4"])],
        }
        pages_b = {
            "p1": [("v1", ["r_1", "r_2"])],
            "p2": [("v1", ["r_3"]), ("v2", ["r_4"])],
        }
        dir_a = _make_dir(tmp_path, "frial", pages_a)
        dir_b = _make_dir(tmp_path, "syafiq", pages_b)
        dest = tmp_path / "agreed"

        result = runner.invoke(
            app,
            [
                "agree",
                "--annotator-a",
                dir_a,
                "--annotator-b",
                dir_b,
                "--name-a",
                "frial",
                "--name-b",
                "syafiq",
                "--output",
                str(tmp_path / "out"),
                "--copy-agreed",
                str(dest),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Copied 1 agreed page" in result.output
        assert (dest / "frial" / "p1.xml").exists()
        assert (dest / "syafiq" / "p1.xml").exists()
        assert (Path(dir_a) / "p1.xml").exists()
        assert (Path(dir_a) / "p2.xml").exists()

    def test_copy_agreed_with_agreement_threshold(self, tmp_path):
        pages_a = {
            "p1": [("u1", ["r_1", "r_2"])],
            "p2": [("u2", ["r_3", "r_4"])],
        }
        pages_b = {
            "p1": [("v1", ["r_1", "r_2"])],
            "p2": [("v1", ["r_3"]), ("v2", ["r_4"])],
        }
        dir_a = _make_dir(tmp_path, "a", pages_a)
        dir_b = _make_dir(tmp_path, "b", pages_b)
        dest = tmp_path / "agreed"

        result = runner.invoke(
            app,
            [
                "agree",
                "--annotator-a",
                dir_a,
                "--annotator-b",
                dir_b,
                "--output",
                str(tmp_path / "out"),
                "--agreement-threshold",
                "0.5",
                "--copy-agreed",
                str(dest),
            ],
        )

        assert result.exit_code == 0, result.output
        # p2 has B-cubed F1 = 2/3 >= 0.5, so both pages qualify
        assert "Copied 2 agreed pages" in result.output
        assert (dest / "a" / "p1.xml").exists()
        assert (dest / "a" / "p2.xml").exists()

    def test_copy_agreed_with_no_agreed_pages(self, tmp_path):
        pages_a = {"p1": [("u1", ["r_1"])]}
        pages_b = {"p1": [("v1", ["r_2"])]}
        dir_a = _make_dir(tmp_path, "a", pages_a)
        dir_b = _make_dir(tmp_path, "b", pages_b)
        dest = tmp_path / "agreed"

        result = runner.invoke(
            app,
            [
                "agree",
                "--annotator-a",
                dir_a,
                "--annotator-b",
                dir_b,
                "--output",
                str(tmp_path / "out"),
                "--copy-agreed",
                str(dest),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "No pages met the agreement threshold" in result.output
        assert not dest.exists()

    def test_invalid_agreement_threshold_exits_1(self, tmp_path):
        pages = {"p1": [("u1", ["r_1"])]}
        dir_a = _make_dir(tmp_path, "a", pages)
        dir_b = _make_dir(tmp_path, "b", pages)

        result = runner.invoke(
            app,
            [
                "agree",
                "--annotator-a",
                dir_a,
                "--annotator-b",
                dir_b,
                "--agreement-threshold",
                "1.5",
                "--output",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "--agreement-threshold" in result.output
