"""Report statistics for OCR fragment files or directories.

Usage:
    uv run python scripts/fragment_stats_report.py data/1_interim/<dataset>/fragments
    uv run python scripts/fragment_stats_report.py page.json --csv pages.csv
    uv run python scripts/fragment_stats_report.py <dir> --group failing_pages.txt
"""

import argparse
import csv
import os
import sys
from dataclasses import fields

from newspaper_reconstructor.fragment_stats import (
    ASCII_DOMINANT_THRESHOLD,
    PageStats,
    iter_fragment_files,
    load_page_stats,
    render_report,
    to_csv_rows,
)

SORTABLE = tuple(f.name for f in fields(PageStats) if f.name != "page")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Fragment JSON files or directories")
    parser.add_argument(
        "--ascii-threshold",
        type=float,
        default=ASCII_DOMINANT_THRESHOLD,
        help="ascii_ratio at or above which a page is flagged ASCII-dominant",
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Rows in the per-page tables"
    )
    parser.add_argument("--sort-by", default="chars", choices=SORTABLE)
    parser.add_argument("--csv", help="Write per-page rows to this CSV path")
    parser.add_argument(
        "--group",
        help="File of page ids (one per line) to compare against the remaining pages",
    )
    args = parser.parse_args()

    try:
        files = iter_fragment_files(args.paths)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)

    pages = [load_page_stats(f) for f in files]
    if not pages:
        print("Error: no fragment JSON files found.", file=sys.stderr)
        raise SystemExit(1)

    group = None
    if args.group:
        with open(args.group, encoding="utf-8") as f:
            group = {line.strip() for line in f if line.strip()}

    try:
        report = render_report(
            pages,
            ascii_threshold=args.ascii_threshold,
            top=args.top,
            sort_by=args.sort_by,
            group=group,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(report)

    if args.csv:
        rows = to_csv_rows(pages)
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
