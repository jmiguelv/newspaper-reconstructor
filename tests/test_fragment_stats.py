"""Tests for fragment_stats.py — corpus statistics and script detection."""

import json
import re

import pytest

from newspaper_reconstructor.fragment_stats import (
    PageStats,
    iter_fragment_files,
    page_stats,
    render_report,
    summarize,
)

JAWI_TEXT = "اين اداله ساتو لافورن بریت درفد سورات خبر ملايو يڠ دتربيتن"
ASCII_TEXT = "THE STRAITS TIMES, FRIDAY, 11 FEBRUARY 1956 - PAGE SIX"
# Real OCR garbage from UM-1956-02-11-6_r_0009: Latin/digits interleaved with Jawi.
GIBBERISH_TEXT = (
    "م م٢O3 E (H4T511) 1.19104 110751 ک1 4کO 1O19, 1 ,124 BEECEE \u200fT001aڠ 551151TOSCP2 "
    "9a 11ݢ1111 OES4 O 11 O1, Oa 1HO OT HaEH12Ta TaH HCTOaCOO 213 -Ea OaIaH aH a 2OT "
    "12ڠ3O O2 OCI17721 STH[ݢ .OTsT,H 1I1 OITOOHT HCaaHIII7 H, SCa .22 a HaBa 11H12I1, "
    "111HI52عS ڠEݢ9O18 3L 521 OO2111201 .6375O O511 OO5 لم1ݢ2 2ڽݢق57 دکدل"
)


def write_fragments(tmp_path, name, fragments):
    path = tmp_path / name
    path.write_text(json.dumps(fragments, ensure_ascii=False), encoding="utf-8")
    return path


class TestPageStats:
    def test_counts_fragments_chars_and_prompt_size(self):
        fragments = [{"id": "r1", "text": JAWI_TEXT}, {"id": "r2", "text": "فندق"}]
        stats = page_stats("page-1", fragments)
        assert stats.page == "page-1"
        assert stats.n_fragments == 2
        assert stats.chars == len(JAWI_TEXT) + len("فندق")
        assert stats.prompt_chars == len(
            json.dumps(fragments, ensure_ascii=False, indent=2)
        )

    def test_jawi_page_has_high_jawi_ratio(self):
        stats = page_stats("page-1", [{"id": "r1", "text": JAWI_TEXT}])
        assert stats.jawi_ratio > 0.9
        assert stats.ascii_ratio < 0.05

    def test_ascii_page_is_flagged_as_ascii_dominant(self):
        stats = page_stats("page-1", [{"id": "r1", "text": ASCII_TEXT}])
        assert stats.ascii_ratio > 0.9
        assert stats.jawi_ratio == 0.0
        assert stats.is_ascii_dominant() is True

    def test_jawi_page_is_not_flagged_as_ascii_dominant(self):
        stats = page_stats("page-1", [{"id": "r1", "text": JAWI_TEXT}])
        assert stats.is_ascii_dominant() is False

    def test_ascii_dominant_threshold_is_configurable(self):
        fragments = [{"id": "r1", "text": f"{ASCII_TEXT} {JAWI_TEXT}"}]
        stats = page_stats("page-1", fragments)
        assert stats.is_ascii_dominant(threshold=0.2) is True
        assert stats.is_ascii_dominant(threshold=0.8) is False

    def test_detects_gibberish_fragment(self):
        stats = page_stats("page-1", [{"id": "r1", "text": GIBBERISH_TEXT}])
        assert stats.gibberish_fragments == 1
        assert stats.gibberish_chars == len(GIBBERISH_TEXT)

    def test_clean_jawi_fragment_is_not_gibberish(self):
        stats = page_stats("page-1", [{"id": "r1", "text": JAWI_TEXT}])
        assert stats.gibberish_fragments == 0
        assert stats.gibberish_chars == 0

    def test_counts_duplicate_fragment_texts(self):
        fragments = [
            {"id": "r1", "text": JAWI_TEXT},
            {"id": "r2", "text": JAWI_TEXT},
            {"id": "r3", "text": "لاين"},
        ]
        stats = page_stats("page-1", fragments)
        assert stats.duplicate_fragments == 1

    def test_measures_fragment_sizes_and_words(self):
        stats = page_stats("page-1", [{"id": "r1", "text": JAWI_TEXT}])
        assert stats.max_fragment_chars == len(JAWI_TEXT)
        assert stats.mean_fragment_chars == pytest.approx(float(len(JAWI_TEXT)))
        assert stats.longest_word_chars == max(len(w) for w in JAWI_TEXT.split())
        assert 0.0 < stats.whitespace_ratio < 1.0

    def test_detects_repeated_window(self):
        repeated = "ساتو ساتو ساتو ساتو ساتو ساتو ساتو ساتو"
        stats = page_stats("page-1", [{"id": "r1", "text": repeated}])
        assert stats.max_repeat_window >= 5


class TestSummarize:
    def test_reports_mean_median_min_max(self):
        pages = [
            page_stats("a", [{"id": "r1", "text": JAWI_TEXT}]),
            page_stats("b", [{"id": "r1", "text": JAWI_TEXT * 3}]),
        ]
        summary = summarize(pages)
        chars = summary["chars"]
        assert chars["min"] < chars["median"] < chars["max"]
        assert chars["mean"] == pytest.approx((chars["min"] + chars["max"]) / 2)

    def test_counts_ascii_dominant_and_gibberish_pages(self):
        pages = [
            page_stats("a", [{"id": "r1", "text": ASCII_TEXT}]),
            page_stats("b", [{"id": "r1", "text": JAWI_TEXT}]),
            page_stats("c", [{"id": "r1", "text": GIBBERISH_TEXT}]),
        ]
        summary = summarize(pages)
        # Both the ASCII page and the ASCII-heavy OCR garbage count as ASCII-dominant.
        assert summary["ascii_dominant_pages"] == 2
        assert summary["pages_with_gibberish"] == 1


class TestIterFragmentFiles:
    def test_accepts_directory_and_files_and_skips_underscore_files(self, tmp_path):
        write_fragments(tmp_path, "page-1.json", [{"id": "r1", "text": JAWI_TEXT}])
        write_fragments(tmp_path, "_metadata.json", [{"id": "r1", "text": JAWI_TEXT}])
        single = write_fragments(
            tmp_path, "single.json", [{"id": "r1", "text": JAWI_TEXT}]
        )

        found = iter_fragment_files([str(tmp_path), str(single)])
        names = sorted(f.split("/")[-1] for f in found)
        assert names == ["page-1.json", "single.json", "single.json"]

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            iter_fragment_files([str(tmp_path / "nope.json")])


class TestRenderReport:
    def test_includes_corpus_summary_and_flagged_pages(self):
        pages = [
            page_stats("page-ascii", [{"id": "r1", "text": ASCII_TEXT}]),
            page_stats("page-jawi", [{"id": "r1", "text": JAWI_TEXT}]),
        ]
        report = render_report(pages, ascii_threshold=0.5, top=5)
        assert "page-ascii" in report
        assert "page-jawi" in report
        assert "ascii" in report.lower()

    def test_limits_table_to_top_n(self):
        pages = [
            page_stats(f"p{i}", [{"id": "r1", "text": JAWI_TEXT}]) for i in range(10)
        ]
        report = render_report(pages, ascii_threshold=0.5, top=2)
        rows = [ln for ln in report.splitlines() if re.match(r"^p\d", ln)]
        assert len(rows) == 2

    def test_renders_group_comparison(self):
        pages = [
            page_stats("p1", [{"id": "r1", "text": JAWI_TEXT}]),
            page_stats("p2", [{"id": "r1", "text": JAWI_TEXT * 4}]),
        ]
        report = render_report(pages, ascii_threshold=0.5, top=5, group={"p2"})
        assert "group" in report.lower()
        assert "p2" in report

    def test_group_ids_match_despite_json_extension_or_path(self):
        pages = [
            page_stats("p1", [{"id": "r1", "text": JAWI_TEXT}]),
            page_stats("p2", [{"id": "r1", "text": JAWI_TEXT * 4}]),
        ]
        report = render_report(
            pages, ascii_threshold=0.5, top=5, group={"p2.json", "/tmp/p2.json"}
        )
        assert "group" in report.lower()

    def test_group_lists_group_pages_instead_of_top_n(self):
        pages = [
            page_stats(f"p{i}", [{"id": "r1", "text": JAWI_TEXT * (i + 1)}])
            for i in range(10)
        ]
        report = render_report(pages, ascii_threshold=0.5, top=2, group={"p3", "p7"})
        assert "Top" not in report
        assert "Group pages" in report
        rows = [ln for ln in report.splitlines() if re.match(r"^p\d", ln)]
        assert [r.split()[0] for r in rows] == ["p7", "p3"]

    def test_without_group_keeps_top_n_table(self):
        pages = [
            page_stats(f"p{i}", [{"id": "r1", "text": JAWI_TEXT}]) for i in range(10)
        ]
        report = render_report(pages, ascii_threshold=0.5, top=2)
        assert "Top 2 pages by chars" in report

    def test_group_matching_no_pages_raises(self):
        pages = [page_stats("p1", [{"id": "r1", "text": JAWI_TEXT}])]
        with pytest.raises(ValueError, match="no pages match"):
            render_report(pages, ascii_threshold=0.5, top=5, group={"nope.json"})

    def test_group_matching_all_pages_raises(self):
        pages = [
            page_stats("p1", [{"id": "r1", "text": JAWI_TEXT}]),
            page_stats("p2", [{"id": "r1", "text": JAWI_TEXT}]),
        ]
        with pytest.raises(ValueError, match="matches all"):
            render_report(pages, ascii_threshold=0.5, top=5, group={"p1", "p2"})


def test_page_stats_is_dataclass_exported_for_reporting():
    assert isinstance(page_stats("p", [{"id": "r1", "text": JAWI_TEXT}]), PageStats)
