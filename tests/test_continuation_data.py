import json
from pathlib import Path

from newspaper_reconstructor.continuation_data import (
    LabeledPage,
    LabeledRegion,
    build_continuation_dataset,
    build_examples,
    load_pages,
    split_pages,
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
