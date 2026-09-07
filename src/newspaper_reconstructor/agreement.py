"""Inter-annotator agreement on region-to-article assignment.

Compares two annotators' article XML directories purely on region
membership: which regions belong to the same article. Topics, classes,
notes, and continuation attributes are out of scope.
"""

import json
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime

from src.newspaper_reconstructor.evaluate import _bcubed_f1, clustering_f1


class AgreementError(Exception):
    """Raised on invalid annotator input (bad XML, duplicate refs, file mismatch)."""


@dataclass(frozen=True)
class Article:
    uuid: str
    regions: tuple[str, ...]


@dataclass
class MatchResult:
    matched: list[tuple[Article, Article, float]]
    partial: list[tuple[Article, Article, float]]
    unmatched_a: list[Article]
    unmatched_b: list[Article]
    no_regions_a: list[Article]
    no_regions_b: list[Article]


def parse_annotation_xml(path: str) -> list[Article]:
    """Parse one annotator XML file into Articles (uuid + region refs)."""
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        raise AgreementError(f"Cannot parse {os.path.basename(path)}: {e}") from e

    articles = []
    for art in tree.getroot().findall(".//Article"):
        uuid = art.get("uuid")
        regions = []
        for region in art.findall(".//Region"):
            ref = region.get("ref")
            if not ref:
                raise AgreementError(
                    f"{os.path.basename(path)}: <Region> without ref attribute "
                    f"in article {uuid}"
                )
            regions.append(ref)
        articles.append(Article(uuid=uuid, regions=tuple(regions)))
    return articles


def load_annotations(annotation_dir: str) -> dict[str, list[Article]]:
    """Load all annotation XML files from a directory.

    Returns a dict keyed by page name (filename without extension).
    Fails fast on unparseable XML and duplicate region refs within a page.
    """
    result = {}
    for fname in sorted(os.listdir(annotation_dir)):
        if not fname.endswith(".xml"):
            continue
        page_name = os.path.splitext(fname)[0]
        articles = parse_annotation_xml(os.path.join(annotation_dir, fname))

        seen = {}
        for art in articles:
            for ref in art.regions:
                if ref in seen:
                    raise AgreementError(
                        f"Region {ref} used in two articles of page {page_name}: "
                        f"{seen[ref]} and {art.uuid}"
                    )
                seen[ref] = art.uuid

        result[page_name] = articles
    return result


def validate_page_sets(
    ann_a: dict[str, list[Article]], ann_b: dict[str, list[Article]]
):
    """Fail fast if the two annotators cover different pages."""
    pages_a = set(ann_a)
    pages_b = set(ann_b)
    if pages_a != pages_b:
        only_a = sorted(pages_a - pages_b)
        only_b = sorted(pages_b - pages_a)
        missing = []
        if only_a:
            missing.append(f"only in A: {', '.join(p + '.xml' for p in only_a)}")
        if only_b:
            missing.append(f"only in B: {', '.join(p + '.xml' for p in only_b)}")
        raise AgreementError(
            "Annotator directories contain different files — " + "; ".join(missing)
        )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def match_articles(
    page_a: list[Article], page_b: list[Article], threshold: float = 1.0
) -> MatchResult:
    """Greedy one-to-one match of articles between two annotators' pages.

    Articles are matched by Jaccard similarity of their region-ref sets.
    A pair is accepted (greedy, highest similarity first) when both articles
    are still unmatched and J >= threshold. Remaining articles with a
    best-overlapping counterpart (0 < J) become partial matches; the rest are
    unmatched. Empty-region articles never participate.
    """
    no_regions_a = [a for a in page_a if not a.regions]
    no_regions_b = [b for b in page_b if not b.regions]
    candidates_a = [a for a in page_a if a.regions]
    candidates_b = [b for b in page_b if b.regions]

    sets_a = {a.uuid: frozenset(a.regions) for a in candidates_a}
    sets_b = {b.uuid: frozenset(b.regions) for b in candidates_b}

    pairs = []
    for a in candidates_a:
        for b in candidates_b:
            j = _jaccard(sets_a[a.uuid], sets_b[b.uuid])
            if j > 0:
                pairs.append((j, a, b))
    pairs.sort(key=lambda p: (-p[0], p[1].uuid, p[2].uuid))

    matched = []
    taken_a, taken_b = set(), set()
    for j, a, b in pairs:
        if a.uuid in taken_a or b.uuid in taken_b:
            continue
        if j >= threshold:
            matched.append((a, b, j))
            taken_a.add(a.uuid)
            taken_b.add(b.uuid)

    remaining_a = [a for a in candidates_a if a.uuid not in taken_a]
    remaining_b = [b for b in candidates_b if b.uuid not in taken_b]

    overlaps = {}
    for a in remaining_a:
        for b in remaining_b:
            j = _jaccard(sets_a[a.uuid], sets_b[b.uuid])
            if j > 0:
                overlaps[(a.uuid, b.uuid)] = (a, b, j)
    partial = [overlaps[key] for key in sorted(overlaps)]

    partial_a = {a.uuid for a, _, _ in partial}
    partial_b = {b.uuid for _, b, _ in partial}
    unmatched_a = [a for a in remaining_a if a.uuid not in partial_a]
    unmatched_b = [b for b in remaining_b if b.uuid not in partial_b]

    return MatchResult(
        matched=matched,
        partial=partial,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
        no_regions_a=no_regions_a,
        no_regions_b=no_regions_b,
    )


def _items(articles: list[Article]) -> list[dict]:
    return [{"fragment_ids": list(a.regions)} for a in articles if a.regions]


def _full_rate(matched: int, articles_with_regions: int) -> float | None:
    if matched == 0 or articles_with_regions == 0:
        return None
    return matched / articles_with_regions


def page_agreement(
    page_a: list[Article], page_b: list[Article], threshold: float = 1.0
) -> dict:
    """Compute region-based agreement metrics for one page."""
    result = match_articles(page_a, page_b, threshold)

    items_a = _items(page_a)
    items_b = _items(page_b)
    cluster = clustering_f1(items_a, items_b)
    bcubed = _bcubed_f1(items_a, items_b)

    regions_a = {ref for art in page_a for ref in art.regions}
    regions_b = {ref for art in page_b for ref in art.regions}

    partial_records = [
        {
            "uuid_a": pa.uuid,
            "uuid_b": pb.uuid,
            "jaccard": round(j, 4),
            "regions_a": sorted(pa.regions),
            "regions_b": sorted(pb.regions),
            "only_a_regions": sorted(set(pa.regions) - set(pb.regions)),
            "only_b_regions": sorted(set(pb.regions) - set(pa.regions)),
        }
        for pa, pb, j in result.partial
    ]

    return {
        "articles_a": len(page_a),
        "articles_b": len(page_b),
        "matched": len(result.matched),
        "clustering_precision": cluster["precision"],
        "clustering_recall": cluster["recall"],
        "clustering_f1": cluster["f1"],
        "bcubed_precision": bcubed["precision"],
        "bcubed_recall": bcubed["recall"],
        "bcubed_f1": bcubed["f1"],
        "full_agreement_rate_a": _full_rate(len(result.matched), len(items_a)),
        "full_agreement_rate_b": _full_rate(len(result.matched), len(items_b)),
        "regions_total": len(regions_a | regions_b),
        "regions_both": len(regions_a & regions_b),
        "regions_only_a": len(regions_a - regions_b),
        "regions_only_b": len(regions_b - regions_a),
        "partial": partial_records,
        "unmatched_a": [
            {"uuid": a.uuid, "regions": sorted(a.regions)} for a in result.unmatched_a
        ],
        "unmatched_b": [
            {"uuid": b.uuid, "regions": sorted(b.regions)} for b in result.unmatched_b
        ],
        "no_regions_a": [{"uuid": a.uuid} for a in result.no_regions_a],
        "no_regions_b": [{"uuid": b.uuid} for b in result.no_regions_b],
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


_EPS = 1e-9


def _agreed_pages(pages: list[dict], threshold: float) -> list[str]:
    """Pages whose B-cubed F1 meets the agreement threshold.

    B-cubed F1 = 1.0 iff every region's cluster is identical on both sides,
    regardless of article order, uuids, or empty-region articles. At the
    default threshold of 1.0 (complete agreement) pages whose region sets
    differ are excluded too — disjoint singleton partitions also score a
    trivial B-cubed F1 of 1.0. Below 1.0 the threshold is a pure B-cubed
    F1 cutoff.
    """
    if threshold >= 1.0 - _EPS:
        return [
            p["page_id"]
            for p in pages
            if p["bcubed_f1"] >= threshold - _EPS
            and p["regions_only_a"] == 0
            and p["regions_only_b"] == 0
        ]
    return [p["page_id"] for p in pages if p["bcubed_f1"] >= threshold - _EPS]


def _rates(pages: list[dict], key: str) -> list[float]:
    return [p[key] for p in pages if p[key] is not None]


def _aggregate(pages: list[dict], agreement_threshold: float) -> dict:
    matched = sum(p["matched"] for p in pages)
    with_regions_a = sum(p["articles_a"] - len(p["no_regions_a"]) for p in pages)
    with_regions_b = sum(p["articles_b"] - len(p["no_regions_b"]) for p in pages)
    regions_both = sum(p["regions_both"] for p in pages)
    regions_only_a = sum(p["regions_only_a"] for p in pages)
    regions_only_b = sum(p["regions_only_b"] for p in pages)
    regions_total = regions_both + regions_only_a + regions_only_b

    return {
        "pages": len(pages),
        "agreed_pages": _agreed_pages(pages, agreement_threshold),
        "articles_a": sum(p["articles_a"] for p in pages),
        "articles_b": sum(p["articles_b"] for p in pages),
        "matched": matched,
        "partial": sum(len(p["partial"]) for p in pages),
        "unmatched_a": sum(len(p["unmatched_a"]) for p in pages),
        "unmatched_b": sum(len(p["unmatched_b"]) for p in pages),
        "no_regions_a": sum(len(p["no_regions_a"]) for p in pages),
        "no_regions_b": sum(len(p["no_regions_b"]) for p in pages),
        "full_agreement_rate_a": matched / with_regions_a if with_regions_a else None,
        "full_agreement_rate_b": matched / with_regions_b if with_regions_b else None,
        "mean_full_agreement_rate_a": _mean(_rates(pages, "full_agreement_rate_a")),
        "mean_full_agreement_rate_b": _mean(_rates(pages, "full_agreement_rate_b")),
        "mean_clustering_precision": _mean([p["clustering_precision"] for p in pages]),
        "mean_clustering_recall": _mean([p["clustering_recall"] for p in pages]),
        "mean_clustering_f1": _mean([p["clustering_f1"] for p in pages]),
        "mean_bcubed_f1": _mean([p["bcubed_f1"] for p in pages]),
        "regions_both": regions_both,
        "regions_only_a": regions_only_a,
        "regions_only_b": regions_only_b,
        "coverage_both_pct": regions_both / regions_total * 100
        if regions_total
        else 0.0,
    }


def calculate_agreement(
    ann_a: dict[str, list[Article]],
    ann_b: dict[str, list[Article]],
    name_a: str,
    name_b: str,
    threshold: float = 1.0,
    agreement_threshold: float = 1.0,
) -> dict:
    """Compare two annotators page by page and aggregate the results.

    Args:
        threshold: Jaccard cutoff for one-to-one article matching.
        agreement_threshold: B-cubed F1 cutoff for a page to count as agreed.
    """
    validate_page_sets(ann_a, ann_b)

    pages = []
    for page_id in sorted(ann_a):
        metrics = page_agreement(ann_a[page_id], ann_b[page_id], threshold)
        pages.append({"page_id": page_id, **metrics})

    return {
        "name_a": name_a,
        "name_b": name_b,
        "threshold": threshold,
        "agreement_threshold": agreement_threshold,
        "pages": pages,
        "aggregate": _aggregate(pages, agreement_threshold),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _fmt_regions(regions: list[str], limit: int = 6) -> str:
    shown = ", ".join(regions[:limit])
    if len(regions) > limit:
        shown += f", … (+{len(regions) - limit})"
    return f"[{shown}]"


def format_summary(result: dict) -> str:
    """Short console summary of an agreement run."""
    name_a, name_b = result["name_a"], result["name_b"]
    agg = result["aggregate"]
    headline = (
        f"Fully agreeing pairs: {agg['matched']} "
        f"({_pct(agg['full_agreement_rate_a'])} / {_pct(agg['full_agreement_rate_b'])})"
    )
    means = (
        f"Mean clustering F1: {agg['mean_clustering_f1'] * 100:.1f}% | "
        f"B-cubed F1: {agg['mean_bcubed_f1'] * 100:.1f}%"
    )
    coverage = (
        f"Region coverage: both {agg['coverage_both_pct']:.1f}% | "
        f"only {name_a} {agg['regions_only_a']} | "
        f"only {name_b} {agg['regions_only_b']}"
    )
    disagreements = (
        "Disagreements: "
        f"{agg['partial']} partial, "
        f"{agg['unmatched_a']} + {agg['unmatched_b']} unmatched, "
        f"{agg['no_regions_a']} + {agg['no_regions_b']} without regions"
    )
    return "\n".join(
        [
            f"Annotation agreement: {name_a} vs {name_b}",
            f"Pages: {agg['pages']} | Articles: {agg['articles_a']} / {agg['articles_b']}",
            headline,
            means,
            coverage,
            disagreements,
            (
                f"Pages with agreement (B-cubed F1 ≥ {result['agreement_threshold']:.2f}): "
                f"{len(agg['agreed_pages'])}"
            ),
        ]
    )


def _markdown_report(result: dict) -> str:
    name_a, name_b = result["name_a"], result["name_b"]
    agg = result["aggregate"]

    lines = [
        f"# Annotation agreement: {name_a} vs {name_b}",
        "",
        (
            f"Pages: {agg['pages']} | Articles: {agg['articles_a']} / {agg['articles_b']} | "
            f"Fully agreeing pairs: {agg['matched']} "
            f"({_pct(agg['full_agreement_rate_a'])} / {_pct(agg['full_agreement_rate_b'])})"
        ),
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| ------ | ----- |",
        f"| Region clustering F1 (mean) | {agg['mean_clustering_f1'] * 100:.1f}% |",
        (
            f"| Clustering precision / recall (mean) | "
            f"{agg['mean_clustering_precision'] * 100:.1f}% / "
            f"{agg['mean_clustering_recall'] * 100:.1f}% |"
        ),
        f"| B-cubed F1 (mean) | {agg['mean_bcubed_f1'] * 100:.1f}% |",
        (
            f"| Full agreement rate — micro ({name_a} / {name_b}) | "
            f"{_pct(agg['full_agreement_rate_a'])} / {_pct(agg['full_agreement_rate_b'])} |"
        ),
        (
            f"| Full agreement rate — macro ({name_a} / {name_b}) | "
            f"{_pct(agg['mean_full_agreement_rate_a'])} / "
            f"{_pct(agg['mean_full_agreement_rate_b'])} |"
        ),
        (
            f"| Region coverage: both / only {name_a} / only {name_b} | "
            f"{agg['coverage_both_pct']:.1f}% / {agg['regions_only_a']} / "
            f"{agg['regions_only_b']} |"
        ),
        "",
        "## Per-page breakdown",
        "",
        "| Page | Arts A | Arts B | Matched | Clust. F1 |",
        "| ---- | ------ | ------ | ------- | --------- |",
    ]
    for p in result["pages"]:
        lines.append(
            f"| {p['page_id']} | {p['articles_a']} | {p['articles_b']} | "
            f"{p['matched']} | {p['clustering_f1'] * 100:.1f}% |"
        )

    agreed = agg["agreed_pages"]
    lines += [
        "",
        (
            f"## Pages with agreement "
            f"(B-cubed F1 ≥ {result['agreement_threshold']:.2f}) ({len(agreed)})"
        ),
        "",
    ]
    if agreed:
        lines += [f"- {page}" for page in agreed]
    else:
        lines += ["None — no page meets the agreement threshold."]
    lines += [""]

    lines += ["", "## Disagreements", ""]

    has_disagreements = bool(
        agg["unmatched_a"]
        or agg["unmatched_b"]
        or agg["partial"]
        or agg["no_regions_a"]
        or agg["no_regions_b"]
    )
    if has_disagreements:
        lines.append(f"### Unmatched articles — {name_a} only ({agg['unmatched_a']})")
        lines.append("")
        for p in result["pages"]:
            for d in p["unmatched_a"]:
                lines.append(
                    f"- {p['page_id']} · uuid={d['uuid']} · regions={_fmt_regions(d['regions'])}"
                )
        lines.append("")
        lines.append(f"### Unmatched articles — {name_b} only ({agg['unmatched_b']})")
        lines.append("")
        for p in result["pages"]:
            for d in p["unmatched_b"]:
                lines.append(
                    f"- {p['page_id']} · uuid={d['uuid']} · regions={_fmt_regions(d['regions'])}"
                )
        lines.append("")
        lines.append(
            f"### Articles without regions ({name_a}: {agg['no_regions_a']}, "
            f"{name_b}: {agg['no_regions_b']})"
        )
        lines.append("")
        for p in result["pages"]:
            for d in p["no_regions_a"]:
                lines.append(f"- {p['page_id']} · {name_a} · uuid={d['uuid']}")
            for d in p["no_regions_b"]:
                lines.append(f"- {p['page_id']} · {name_b} · uuid={d['uuid']}")
        lines.append("")
        lines.append(f"### Partial matches — region sets differ ({agg['partial']})")
        lines.append("")
        for p in result["pages"]:
            for d in p["partial"]:
                lines.append(
                    f"- {p['page_id']} · J={d['jaccard']:.2f} · "
                    f"{name_a} uuid={d['uuid_a']} vs {name_b} uuid={d['uuid_b']} · "
                    f"{name_a}-only regions={_fmt_regions(d['only_a_regions'])}, "
                    f"{name_b}-only regions={_fmt_regions(d['only_b_regions'])}"
                )
        lines.append("")
    else:
        lines += [
            "No disagreements found — all articles have identical region sets.",
            "",
        ]

    return "\n".join(lines)


def copy_agreed_pages(
    result: dict, annotator_a_dir: str, annotator_b_dir: str, dest_root: str
) -> list[tuple[str, str]]:
    """Copy both annotators' XML files for pages meeting the agreement threshold.

    Files land in dest_root/<annotator name>/<page>.xml; the originals stay
    in place. Returns the list of (source, destination) copies; a no-op (no
    directory created) when no page qualifies.
    """
    agreed = result["aggregate"]["agreed_pages"]
    if not agreed:
        return []

    copied = []
    for name, src_dir in (
        (result["name_a"], annotator_a_dir),
        (result["name_b"], annotator_b_dir),
    ):
        dest_dir = os.path.join(dest_root, name)
        os.makedirs(dest_dir, exist_ok=True)
        for page in agreed:
            src = os.path.join(src_dir, f"{page}.xml")
            dest = os.path.join(dest_dir, f"{page}.xml")
            shutil.copy2(src, dest)
            copied.append((src, dest))
    return copied


def write_agreement_reports(result: dict, output_dir: str) -> tuple[str, str]:
    """Write the JSON log and Markdown report for an agreement run."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    base = f"{result['name_a']}_vs_{result['name_b']}_{timestamp}"

    json_path = os.path.join(output_dir, f"{base}.json")
    log = {"timestamp": datetime.now().astimezone().isoformat(), **result}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(output_dir, f"{base}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_markdown_report(result))

    return json_path, md_path
