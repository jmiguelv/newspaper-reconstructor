"""Build balanced train, validation, and test data for continuation classification."""

import argparse

from newspaper_reconstructor.continuation_data import build_continuation_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles-dir", required=True)
    parser.add_argument("--regions-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-region-chars",
        type=int,
        default=None,
        help="Split regions longer than this many characters into sub-fragments",
    )
    parser.add_argument(
        "--filter-gibberish",
        action="store_true",
        help="Drop regions flagged as mixed-script OCR garbage",
    )
    parser.add_argument(
        "--cross-region-negatives",
        action="store_true",
        help="Sample negatives only from pairs of different regions",
    )
    args = parser.parse_args()

    manifest = build_continuation_dataset(
        args.articles_dir,
        args.regions_dir,
        args.output_dir,
        seed=args.seed,
        max_region_chars=args.max_region_chars,
        filter_gibberish=args.filter_gibberish,
        cross_region_negatives=args.cross_region_negatives,
    )
    for name, split in manifest["splits"].items():
        print(
            f"{name}: {split['pages']} pages, {split['examples']} examples "
            f"({split['positive_examples']} positive, "
            f"{split['negative_examples']} negative)"
        )
    print(f"Excluded pages: {manifest['counts']['excluded_pages']}")


if __name__ == "__main__":
    main()
