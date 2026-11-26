import json
from pathlib import Path

import pytest

from newspaper_reconstructor.continuation_data import (
    LabeledPage,
    LabeledRegion,
    build_continuation_dataset,
    build_examples,
    load_pages,
    split_pages,
    split_region_text,
)


def _page(page_id: str) -> LabeledPage:
    return LabeledPage(
        page_id=page_id,
        regions=(
            LabeledRegion("a", "first", "article-1", 1),
            LabeledRegion("b", "second", "article-1", 2),
            LabeledRegion("c", "third", "article-1", 3),
            LabeledRegion("d", "other", "article-2", 1),
        ),
    )


def _write_page(
    articles_dir: Path,
    regions_dir: Path,
    page_id: str,
    texts: dict[str, str],
    articles: list[list[tuple[str, int]]],
) -> None:
    articles_dir.mkdir(exist_ok=True)
    regions_dir.mkdir(exist_ok=True)
    (articles_dir / f"{page_id}.json").write_text(json.dumps(texts), encoding="utf-8")
    xml = "".join(
        f'<Article uuid="article-{article_index}"><Regions>'
        + "".join(
            f'<Region ref="{region_id}" seq="{sequence}" />'
            for region_id, sequence in regions
        )
        + "</Regions></Article>"
        for article_index, regions in enumerate(articles)
    )
    (regions_dir / f"{page_id}.xml").write_text(
        f"<Articles>{xml}</Articles>", encoding="utf-8"
    )


def test_split_region_text_short_text_is_unchanged():
    assert split_region_text("short text", 400) == ["short text"]
    assert split_region_text("", 400) == [""]


def test_split_region_text_rejects_small_max_chars():
    with pytest.raises(ValueError):
        split_region_text("abc", 1)


def test_split_region_text_parts_stay_within_bounds_and_recombine():
    text = " ".join(f"word{index}" for index in range(500))

    parts = split_region_text(text, 400)

    assert len(parts) > 1
    assert all(len(part) <= 400 for part in parts)
    assert all(len(part) >= 200 for part in parts)
    assert "".join(parts) == text


def test_split_region_text_prefers_sentence_ends():
    text = ("This is a sentence. " * 40).strip()

    parts = split_region_text(text, 400)

    assert len(parts) == 2
    assert parts[0].endswith(". ")
    assert parts[1].startswith("This")


def test_split_region_text_falls_back_to_whitespace_then_hard_cut():
    whitespace_text = "aaaa bbbb " * 100
    parts = split_region_text(whitespace_text, 400)
    assert all(part.endswith(" ") for part in parts[:-1])

    hard_text = "a" * 900
    parts = split_region_text(hard_text, 400)
    assert [len(part) for part in parts] == [400, 250, 250]


def test_split_region_text_handles_uneven_sentence_drift():
    text = ("x" * 210 + ". ") * 3 + "x" * 500

    parts = split_region_text(text, 400)

    assert [len(part) for part in parts] == [212, 212, 212, 250, 250]
    assert "".join(parts) == text


def test_split_region_text_final_part_never_exceeds_max_chars():
    text = "a" * 401

    parts = split_region_text(text, 400)

    assert [len(part) for part in parts] == [200, 201]


def test_build_examples_keeps_all_positives_and_one_negative_per_positive():
    examples = build_examples([_page("page-1")], seed=42)

    assert len(examples) == 4
    assert sum(example["label"] for example in examples) == 2
    assert {
        (example["a_id"], example["b_id"]) for example in examples if example["label"]
    } == {
        ("a", "b"),
        ("b", "c"),
    }
    assert all(
        set(example)
        == {
            "page_id",
            "a_id",
            "b_id",
            "text_a",
            "text_b",
            "label",
        }
        for example in examples
    )


def test_split_pages_is_deterministic_and_keeps_pages_disjoint():
    pages = [_page(f"page-{index:02d}") for index in range(20)]

    first = split_pages(pages, seed=42)
    second = split_pages(pages, seed=42)

    assert first == second
    assert {name: len(split) for name, split in first.items()} == {
        "train": 14,
        "validation": 3,
        "test": 3,
    }
    page_sets = {
        name: {page.page_id for page in split} for name, split in first.items()
    }
    assert page_sets["train"].isdisjoint(page_sets["validation"])
    assert page_sets["train"].isdisjoint(page_sets["test"])
    assert page_sets["validation"].isdisjoint(page_sets["test"])


def test_load_pages_excludes_invalid_and_blank_pages(tmp_path):
    articles_dir = tmp_path / "articles"
    regions_dir = tmp_path / "regions"
    _write_page(
        articles_dir,
        regions_dir,
        "valid",
        {"a": "first", "b": "second"},
        [[("a", 1), ("b", 2)]],
    )
    _write_page(
        articles_dir,
        regions_dir,
        "mismatch",
        {"a": "first", "extra": "other"},
        [[("a", 1)]],
    )
    _write_page(
        articles_dir,
        regions_dir,
        "blank",
        {"a": " "},
        [[("a", 1)]],
    )

    pages, exclusions = load_pages(articles_dir, regions_dir)

    assert [page.page_id for page in pages] == ["valid"]
    assert [(item.page_id, item.reason) for item in exclusions] == [
        ("blank", "region text is empty"),
        ("mismatch", "region IDs differ between JSON and XML"),
    ]


def test_build_examples_with_split_adds_intra_and_cross_region_positives():
    page = LabeledPage(
        "page-1",
        regions=(
            LabeledRegion("a", "x" * 500, "article-1", 1),
            LabeledRegion("b", "b text", "article-1", 2),
            LabeledRegion("c", "c text", "article-2", 1),
        ),
    )

    examples = build_examples([page], seed=42, max_region_chars=400)

    positives = {
        (example["a_id"], example["b_id"]) for example in examples if example["label"]
    }
    assert positives == {("a#1", "a#2"), ("a#2", "b")}
    assert len(examples) == 2 * len(positives)
    texts = {(example["a_id"], example["b_id"]): example for example in examples}
    assert texts[("a#1", "a#2")]["text_a"] == "x" * 250
    assert texts[("a#1", "a#2")]["text_b"] == "x" * 250


def test_build_examples_with_split_keeps_sequence_gap_pairs_negative():
    page = LabeledPage(
        "page-1",
        regions=(
            LabeledRegion("a", "x" * 500, "article-1", 1),
            LabeledRegion("b", "b text", "article-1", 3),
        ),
    )

    examples = build_examples([page], seed=42, max_region_chars=400)

    positives = {
        (example["a_id"], example["b_id"]) for example in examples if example["label"]
    }
    assert positives == {("a#1", "a#2")}


def test_build_continuation_dataset_records_split_stats(tmp_path):
    articles_dir = tmp_path / "articles"
    regions_dir = tmp_path / "regions"
    for index in range(10):
        _write_page(
            articles_dir,
            regions_dir,
            f"page-{index:02d}",
            {"a": "x" * 500, "b": "second", "c": "other"},
            [[("a", 1), ("b", 2)], [("c", 1)]],
        )

    manifest = build_continuation_dataset(
        articles_dir,
        regions_dir,
        tmp_path / "split",
        seed=42,
        max_region_chars=400,
    )

    assert manifest["max_region_chars"] == 400
    assert manifest["counts"]["fragments"] == 40
    for split in manifest["splits"].values():
        assert split["positive_intra_region"] == split["pages"]
        assert (
            split["positive_cross_region"]
            == split["positive_examples"] - split["positive_intra_region"]
        )
        assert split["text_length_chars"]["text_a"]["max"] <= 400


def test_build_examples_cross_region_negatives_drop_same_region_negatives():
    page = LabeledPage(
        "page-1",
        regions=(
            LabeledRegion("a", "x" * 900, "article-1", 1),
            LabeledRegion("b", "b text", "article-1", 2),
            LabeledRegion("c", "c text", "article-2", 1),
        ),
    )

    restricted = build_examples(
        [page], seed=42, max_region_chars=400, cross_region_negatives=True
    )
    unrestricted = build_examples(
        [page], seed=42, max_region_chars=400, cross_region_negatives=False
    )

    def region_of(fragment_id):
        return fragment_id.split("#", 1)[0]

    same_region_negatives = [
        example
        for example in restricted
        if not example["label"]
        and region_of(example["a_id"]) == region_of(example["b_id"])
    ]
    assert same_region_negatives == []
    assert {
        (example["a_id"], example["b_id"]) for example in restricted if example["label"]
    } == {("a#1", "a#2"), ("a#2", "a#3"), ("a#3", "b")}
    assert len(restricted) == 2 * 3
    assert any(
        not example["label"]
        and region_of(example["a_id"]) == region_of(example["b_id"])
        for example in unrestricted
    )


def test_build_continuation_dataset_records_negative_same_region_count(tmp_path):
    articles_dir = tmp_path / "articles"
    regions_dir = tmp_path / "regions"
    for index in range(10):
        _write_page(
            articles_dir,
            regions_dir,
            f"page-{index:02d}",
            {"a": "x" * 500, "b": "second", "c": "other"},
            [[("a", 1), ("b", 2)], [("c", 1)]],
        )

    restricted = build_continuation_dataset(
        articles_dir,
        regions_dir,
        tmp_path / "restricted",
        seed=42,
        max_region_chars=400,
        cross_region_negatives=True,
    )
    unrestricted = build_continuation_dataset(
        articles_dir,
        regions_dir,
        tmp_path / "unrestricted",
        seed=42,
        max_region_chars=400,
    )

    for split in ("train", "validation", "test"):
        assert restricted["splits"][split]["negative_same_region"] == 0
        assert "negative_same_region" in unrestricted["splits"][split]


def test_build_examples_with_gibberish_filter_drops_gibberish_regions():
    gibberish = "Ab1 " * 30
    page = LabeledPage(
        "page-1",
        regions=(
            LabeledRegion("a", gibberish, "article-1", 1),
            LabeledRegion("b", "b text", "article-1", 2),
        ),
    )

    filtered = build_examples([page], seed=42, filter_gibberish=True)
    unfiltered = build_examples([page], seed=42, filter_gibberish=False)

    assert filtered == []
    assert len(unfiltered) == 2


def test_build_continuation_dataset_counts_dropped_gibberish(tmp_path):
    articles_dir = tmp_path / "articles"
    regions_dir = tmp_path / "regions"
    for index in range(10):
        _write_page(
            articles_dir,
            regions_dir,
            f"page-{index:02d}",
            {"a": "Ab1 " * 30, "b": "second", "c": "other", "d": "fourth"},
            [[("a", 1), ("b", 2), ("d", 3)], [("c", 1)]],
        )

    manifest = build_continuation_dataset(
        articles_dir,
        regions_dir,
        tmp_path / "filtered",
        seed=42,
        filter_gibberish=True,
    )

    assert manifest["filter_gibberish"] is True
    assert manifest["counts"]["gibberish_regions_dropped"] == 10
    assert manifest["counts"]["fragments"] == 30


def test_build_continuation_dataset_writes_balanced_reproducible_splits(tmp_path):
    articles_dir = tmp_path / "articles"
    regions_dir = tmp_path / "regions"
    for index in range(10):
        _write_page(
            articles_dir,
            regions_dir,
            f"page-{index:02d}",
            {"a": "first", "b": "second", "c": "other"},
            [[("a", 1), ("b", 2)], [("c", 1)]],
        )

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = build_continuation_dataset(articles_dir, regions_dir, first_dir, seed=42)
    second = build_continuation_dataset(articles_dir, regions_dir, second_dir, seed=42)

    assert first == second
    assert first["splits"]["train"]["examples"] == 14
    assert first["splits"]["validation"]["examples"] == 2
    assert first["splits"]["test"]["examples"] == 4
    for split in ("train", "validation", "test"):
        assert (
            first["splits"][split]["positive_examples"]
            == first["splits"][split]["negative_examples"]
        )
        assert (first_dir / f"{split}.jsonl").read_bytes() == (
            second_dir / f"{split}.jsonl"
        ).read_bytes()
    assert (first_dir / "manifest.json").read_bytes() == (
        second_dir / "manifest.json"
    ).read_bytes()
