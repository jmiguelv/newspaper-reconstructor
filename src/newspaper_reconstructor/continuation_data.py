"""Build balanced text-pair data for continuation classification."""

from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from newspaper_reconstructor.fragment_stats import is_gibberish

SPLIT_NAMES = ("train", "validation", "test")

_SENTENCE_END = re.compile(r"[.?!]\s+")
_WHITESPACE = re.compile(r"\s+")


def split_region_text(text: str, max_chars: int) -> list[str]:
    """Split a long region into ordered parts, each at most max_chars long.

    Cuts land at up to max_chars from the previous cut, snapped backward to
    sentence ends (".?!" + whitespace), then whitespace, then a hard cut. A
    tiny final tail is rebalanced into the previous part's span. Every part
    except possibly the last is at least half of max_chars; no part exceeds
    max_chars.
    """
    if max_chars < 2:
        raise ValueError("max_chars must be at least two")
    if len(text) <= max_chars:
        return [text]

    length = len(text)
    min_part = max(1, max_chars // 2)
    parts = []
    start = 0
    while length - start > max_chars:
        cut = _snap_cut(text, start + min_part, start + max_chars)
        parts.append(text[start:cut])
        start = cut
    parts.append(text[start:])

    if len(parts) >= 2 and len(parts[-1]) < min_part:
        span = text[start - len(parts[-2]) :]
        if len(span) >= 2 * min_part:
            cut = _snap_cut(span, min_part, round(len(span) / 2))
            parts[-2:] = [span[:cut], span[cut:]]
    return parts


def _snap_cut(text: str, low: int, high: int) -> int:
    """Best cut position in [low, high]: sentence end, whitespace, then high."""
    if high < low:
        return low
    for pattern in (_SENTENCE_END, _WHITESPACE):
        candidates = [
            match.end()
            for match in pattern.finditer(text)
            if low <= match.end() <= high
        ]
        if candidates:
            return max(candidates)
    return high


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
class Fragment:
    fragment_id: str
    region_id: str
    text: str
    article_id: str
    sequence: int
    part: int
    is_last_part: bool


@dataclass(frozen=True)
class ExcludedPage:
    page_id: str
    reason: str


def build_continuation_dataset(
    articles_dir: str | Path,
    regions_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    max_region_chars: int | None = None,
    filter_gibberish: bool = False,
    cross_region_negatives: bool = False,
) -> dict:
    """Write deterministic, balanced train, validation, and test JSONL files."""
    pages, exclusions = load_pages(articles_dir, regions_dir)
    splits = split_pages(
        pages,
        seed=seed,
        max_region_chars=max_region_chars,
        filter_gibberish=filter_gibberish,
        cross_region_negatives=cross_region_negatives,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    examples_by_split = {
        name: build_examples(
            split_pages,
            seed=seed,
            max_region_chars=max_region_chars,
            filter_gibberish=filter_gibberish,
            cross_region_negatives=cross_region_negatives,
        )
        for name, split_pages in splits.items()
    }
    for name, examples in examples_by_split.items():
        _write_jsonl(output_path / f"{name}.jsonl", examples)

    manifest = {
        "seed": seed,
        "max_region_chars": max_region_chars,
        "filter_gibberish": filter_gibberish,
        "cross_region_negatives": cross_region_negatives,
        "split_fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "negative_ratio": 1,
        "counts": {
            "pages": len(pages),
            "excluded_pages": len(exclusions),
            "regions": sum(len(page.regions) for page in pages),
            "fragments": sum(
                len(_page_fragments(page, max_region_chars, filter_gibberish))
                for page in pages
            ),
            "gibberish_regions_dropped": (
                sum(
                    is_gibberish(region.text)
                    for page in pages
                    for region in page.regions
                )
                if filter_gibberish
                else 0
            ),
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
    max_region_chars: int | None = None,
    filter_gibberish: bool = False,
    cross_region_negatives: bool = False,
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
    if any(
        not build_examples(
            split,
            seed=seed,
            max_region_chars=max_region_chars,
            filter_gibberish=filter_gibberish,
            cross_region_negatives=cross_region_negatives,
        )
        for split in splits.values()
    ):
        raise ValueError("every split must contain at least one positive pair")
    return splits


def build_examples(
    pages: list[LabeledPage] | tuple[LabeledPage, ...],
    seed: int = 42,
    max_region_chars: int | None = None,
    filter_gibberish: bool = False,
    cross_region_negatives: bool = False,
) -> list[dict]:
    """Keep every positive pair and sample one page-local negative per positive."""
    examples = []
    for page in pages:
        fragments = _page_fragments(page, max_region_chars, filter_gibberish)
        positives = []
        negatives = []
        for fragment_a in fragments:
            for fragment_b in fragments:
                if fragment_a.fragment_id == fragment_b.fragment_id:
                    continue
                example = _example(page, fragment_a, fragment_b)
                if (
                    cross_region_negatives
                    and not example["label"]
                    and fragment_a.region_id == fragment_b.region_id
                ):
                    continue
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


def _page_fragments(
    page: LabeledPage,
    max_region_chars: int | None,
    filter_gibberish: bool = False,
) -> tuple[Fragment, ...]:
    return tuple(
        fragment
        for region in page.regions
        if not (filter_gibberish and is_gibberish(region.text))
        for fragment in _region_fragments(region, max_region_chars)
    )


def _region_fragments(
    region: LabeledRegion, max_region_chars: int | None
) -> tuple[Fragment, ...]:
    if max_region_chars is None or len(region.text) <= max_region_chars:
        return (
            Fragment(
                region.region_id,
                region.region_id,
                region.text,
                region.article_id,
                region.sequence,
                0,
                True,
            ),
        )
    parts = split_region_text(region.text, max_region_chars)
    return tuple(
        Fragment(
            f"{region.region_id}#{index + 1}",
            region.region_id,
            part,
            region.article_id,
            region.sequence,
            index,
            index == len(parts) - 1,
        )
        for index, part in enumerate(parts)
    )


def _continues(fragment_a: Fragment, fragment_b: Fragment) -> bool:
    if fragment_a.article_id != fragment_b.article_id:
        return False
    if fragment_a.region_id == fragment_b.region_id:
        return fragment_b.part == fragment_a.part + 1
    return (
        fragment_a.is_last_part
        and fragment_b.part == 0
        and fragment_b.sequence == fragment_a.sequence + 1
    )


def _example(page: LabeledPage, fragment_a: Fragment, fragment_b: Fragment) -> dict:
    label = int(_continues(fragment_a, fragment_b))
    return {
        "page_id": page.page_id,
        "a_id": fragment_a.fragment_id,
        "b_id": fragment_b.fragment_id,
        "text_a": fragment_a.text,
        "text_b": fragment_b.text,
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
    positives = [example for example in examples if example["label"]]
    negatives = [example for example in examples if not example["label"]]
    intra_region = sum(
        1
        for example in positives
        if _region_of(example["a_id"]) == _region_of(example["b_id"])
    )
    negative_same_region = sum(
        1
        for example in negatives
        if _region_of(example["a_id"]) == _region_of(example["b_id"])
    )
    return {
        "page_ids": sorted(page.page_id for page in pages),
        "pages": len(pages),
        "examples": len(examples),
        "positive_examples": len(positives),
        "negative_examples": len(examples) - len(positives),
        "positive_intra_region": intra_region,
        "positive_cross_region": len(positives) - intra_region,
        "negative_same_region": negative_same_region,
        "text_length_chars": {
            "text_a": _length_percentiles(
                [len(example["text_a"]) for example in examples]
            ),
            "text_b": _length_percentiles(
                [len(example["text_b"]) for example in examples]
            ),
        },
    }


def _region_of(fragment_id: str) -> str:
    return fragment_id.split("#", 1)[0]


def _length_percentiles(values: list[int]) -> dict:
    ordered = sorted(values)
    count = len(ordered)
    return {
        "min": ordered[0],
        "p25": ordered[round(0.25 * (count - 1))],
        "median": ordered[round(0.50 * (count - 1))],
        "p75": ordered[round(0.75 * (count - 1))],
        "p90": ordered[round(0.90 * (count - 1))],
        "max": ordered[-1],
    }
