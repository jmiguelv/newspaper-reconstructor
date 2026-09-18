"""Build balanced text-pair data for continuation classification."""

from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class LabeledRegion:
    region_id: str
    text: str
    article_id: str
    sequence: int


@dataclass(frozen=True)
class LabeledPage:
    page_id: str
    regions: tuple[LabeledRegion, ...]


@dataclass(frozen=True)
class ExcludedPage:
    page_id: str
    reason: str


def build_continuation_dataset(
    articles_dir: str | Path,
    regions_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> dict:
    """Write deterministic, balanced train, validation, and test JSONL files."""
    pages, exclusions = load_pages(articles_dir, regions_dir)
    splits = split_pages(pages, seed=seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    examples_by_split = {
        name: build_examples(split_pages, seed=seed)
        for name, split_pages in splits.items()
    }
    for name, examples in examples_by_split.items():
        _write_jsonl(output_path / f"{name}.jsonl", examples)

    manifest = {
        "seed": seed,
        "split_fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "negative_ratio": 1,
        "counts": {
            "pages": len(pages),
            "excluded_pages": len(exclusions),
            "regions": sum(len(page.regions) for page in pages),
        },
        "exclusions": [
            {"page_id": exclusion.page_id, "reason": exclusion.reason}
            for exclusion in exclusions
        ],
        "splits": {
            name: _split_manifest(splits[name], examples_by_split[name])
            for name in SPLIT_NAMES
        },
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_pages(
    articles_dir: str | Path, regions_dir: str | Path
) -> tuple[tuple[LabeledPage, ...], tuple[ExcludedPage, ...]]:
    """Load pages whose JSON text and XML labels contain the same regions."""
    article_paths = {path.stem: path for path in Path(articles_dir).glob("*.json")}
    region_paths = {path.stem: path for path in Path(regions_dir).glob("*.xml")}
    pages = []
    exclusions = []

    for page_id in sorted(article_paths.keys() | region_paths.keys()):
        if page_id not in article_paths:
            exclusions.append(ExcludedPage(page_id, "article JSON is missing"))
            continue
        if page_id not in region_paths:
            exclusions.append(ExcludedPage(page_id, "region XML is missing"))
            continue

        texts = _load_texts(article_paths[page_id])
        labels = _load_labels(region_paths[page_id])
        if set(texts) != set(labels):
            exclusions.append(
                ExcludedPage(page_id, "region IDs differ between JSON and XML")
            )
            continue
        if any(not text.strip() for text in texts.values()):
            exclusions.append(ExcludedPage(page_id, "region text is empty"))
            continue

        regions = tuple(
            LabeledRegion(region_id, texts[region_id], *labels[region_id])
            for region_id in sorted(texts)
        )
        pages.append(LabeledPage(page_id, regions))

    return tuple(pages), tuple(exclusions)


def split_pages(
    pages: list[LabeledPage] | tuple[LabeledPage, ...],
    seed: int = 42,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> dict[str, tuple[LabeledPage, ...]]:
    """Split whole pages so no OCR region appears in more than one split."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")
    if len(pages) < 3:
        raise ValueError("at least three pages are required")

    shuffled = sorted(pages, key=lambda page: page.page_id)
    random.Random(seed).shuffle(shuffled)
    train_end = min(max(round(len(shuffled) * train_fraction), 1), len(shuffled) - 2)
    validation_end = min(
        max(
            round(len(shuffled) * (train_fraction + validation_fraction)), train_end + 1
        ),
        len(shuffled) - 1,
    )
    splits = {
        "train": tuple(shuffled[:train_end]),
        "validation": tuple(shuffled[train_end:validation_end]),
        "test": tuple(shuffled[validation_end:]),
    }
    if any(not build_examples(split, seed=seed) for split in splits.values()):
        raise ValueError("every split must contain at least one positive pair")
    return splits


def build_examples(
    pages: list[LabeledPage] | tuple[LabeledPage, ...], seed: int = 42
) -> list[dict]:
    """Keep every positive pair and sample one page-local negative per positive."""
    examples = []
    for page in pages:
        positives = []
        negatives = []
        for region_a in page.regions:
            for region_b in page.regions:
                if region_a.region_id == region_b.region_id:
                    continue
                example = _example(page, region_a, region_b)
                (positives if example["label"] else negatives).append(example)

        if len(negatives) < len(positives):
            raise ValueError(f"page {page.page_id} has too few negative pairs")
        random.Random(f"{seed}:{page.page_id}").shuffle(negatives)
        examples.extend(positives)
        examples.extend(negatives[: len(positives)])

    return sorted(
        examples,
        key=lambda example: (
            example["page_id"],
            example["a_id"],
            example["b_id"],
        ),
    )


def _example(
    page: LabeledPage, region_a: LabeledRegion, region_b: LabeledRegion
) -> dict:
    label = int(
        region_a.article_id == region_b.article_id
        and region_b.sequence == region_a.sequence + 1
    )
    return {
        "page_id": page.page_id,
        "a_id": region_a.region_id,
        "b_id": region_b.region_id,
        "text_a": region_a.text,
        "text_b": region_b.text,
        "label": label,
    }


def _load_texts(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path.name}: article JSON must be an object")
    if any(not isinstance(region_id, str) or not region_id for region_id in data):
        raise ValueError(f"{path.name}: region IDs must be non-empty strings")
    if any(not isinstance(text, str) for text in data.values()):
        raise TypeError(f"{path.name}: region text must be a string")
    return data


def _load_labels(path: Path) -> dict[str, tuple[str, int]]:
    labels = {}
    for article in ET.parse(path).getroot().findall(".//Article"):
        article_id = article.get("uuid")
        if not article_id:
            raise ValueError(f"{path.name}: article UUID is missing")
        for region in article.findall("./Regions/Region"):
            region_id = region.get("ref")
            if not region_id:
                raise ValueError(f"{path.name}: region reference is missing")
            if region_id in labels:
                raise ValueError(f"{path.name}: duplicate region reference {region_id}")
            try:
                sequence = int(region.get("seq", ""))
            except ValueError as error:
                raise ValueError(
                    f"{path.name}: invalid sequence for region {region_id}"
                ) from error
            labels[region_id] = (article_id, sequence)
    return labels


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n"
            for example in examples
        ),
        encoding="utf-8",
    )


def _split_manifest(pages: tuple[LabeledPage, ...], examples: list[dict]) -> dict:
    positives = sum(example["label"] for example in examples)
    return {
        "page_ids": sorted(page.page_id for page in pages),
        "pages": len(pages),
        "examples": len(examples),
        "positive_examples": positives,
        "negative_examples": len(examples) - positives,
    }
