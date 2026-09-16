"""Sweep baseline parameters against ground truth and write a report.

Usage:
    uv run python scripts/baseline_sweep.py \
        --ocr-dir data/0_external/ds-filteredUM1956alto/json \
        --gt-dir data/0_external/ds-filteredUM1956alto/article_xml \
        --out-dir reports/baselines

Writes sweep.csv (one row per parameter cell) and sweep.md (summary tables).
"""

import argparse
import csv
import os

from newspaper_reconstructor.baseline import baseline_items
from newspaper_reconstructor.evaluate import (
    evaluate_reconstruction_page,
    parse_article_xml,
)
from newspaper_reconstructor.ingest import load_baseline_ocr_file

DEFAULT_GAP_KS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.3]
DEFAULT_NCD_THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 1.0]
METRIC_COLUMNS = (
    ("pw_f1", "clustering_f1"),
    ("pw_precision", "clustering_precision"),
    ("pw_recall", "clustering_recall"),
    ("bcubed_f1", "bcubed_f1"),
    ("ari", "ari"),
    ("coverage", "coverage"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--out-dir", default="reports/baselines")
    parser.add_argument(
        "--gap-ks",
        type=float,
        nargs="+",
        default=DEFAULT_GAP_KS,
        help="gap_k values to sweep",
    )
    parser.add_argument(
        "--ncd-thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_NCD_THRESHOLDS,
        help="ncd thresholds to sweep (geometry-ncd)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["geometry", "geometry-ncd", "xy-cut", "xy-cut-plus"],
        choices=["geometry", "geometry-ncd", "xy-cut", "xy-cut-plus"],
    )
    args = parser.parse_args()

    pages = load_pages(args.ocr_dir, args.gt_dir)
    if not pages:
        raise SystemExit("No pages with ground truth found")

    rows = []
    for variant in args.variants:
        for gap_k in args.gap_ks:
            thresholds = (
                args.ncd_thresholds
                if variant == "geometry-ncd"
                else [DEFAULT_NCD_THRESHOLDS[-1]]
            )
            rows.extend(run_cells(pages, variant, gap_k, thresholds))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(args.out_dir, "sweep.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Baseline parameter sweep\n\n")
        f.write(f"Pages: {len(pages)} · module OCR: `{args.ocr_dir}`\n\n")
        f.write(
            "| variant | gap_k | ncd | pwF1 | pwP | pwR | B³F1 | ARI | coverage |\n"
        )
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        f.writelines(
            f"| {r['variant']} | {r['gap_k']} | {r['ncd_threshold']} "
            f"| {r['pw_f1']:.3f} | {r['pw_precision']:.3f} "
            f"| {r['pw_recall']:.3f} | {r['bcubed_f1']:.3f} | {r['ari']:.3f} "
            f"| {r['coverage']:.3f} |\n"
            for r in rows
        )
    print(f"Wrote {csv_path} and {md_path} ({len(rows)} cells)")


def load_pages(ocr_dir: str, gt_dir: str) -> list[tuple[str, list, list, list]]:
    pages = []
    for fname in sorted(os.listdir(ocr_dir)):
        if not fname.endswith(".json"):
            continue
        page_id = fname[:-5]
        gt_path = os.path.join(gt_dir, f"{page_id}.xml")
        if not os.path.exists(gt_path):
            continue
        fragments, layout_regions = load_baseline_ocr_file(os.path.join(ocr_dir, fname))
        pages.append((page_id, fragments, layout_regions, parse_article_xml(gt_path)))
    return pages


def run_cells(pages, variant, gap_k, ncd_thresholds):
    """Evaluate a (variant, gap_k) cell across ncd thresholds.

    Pages loop outermost so the same fragment pairs recur across threshold
    cells while still hot in baseline.ncd's cache.
    """
    acc = {t: {m: [] for _, m in METRIC_COLUMNS} for t in ncd_thresholds}
    for _, fragments, layout_regions, gt in pages:
        for t in ncd_thresholds:
            items = baseline_items(
                fragments,
                variant=variant,
                gap_k=gap_k,
                ncd_threshold=t,
                layout_regions=layout_regions,
            )
            metrics = evaluate_reconstruction_page(items, gt)
            for _, m in METRIC_COLUMNS:
                acc[t][m].append(metrics[m])
    n = len(pages)
    rows = []
    for t in ncd_thresholds:
        row = {
            "variant": variant,
            "gap_k": gap_k,
            "ncd_threshold": t if variant == "geometry-ncd" else "",
            "pages": n,
        }
        for name, m in METRIC_COLUMNS:
            xs = acc[t][m]
            row[name] = sum(xs) / n if n else 0.0
        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
