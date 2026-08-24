#!/usr/bin/env python3
"""Migrate existing evaluation logs to include dataset name in experiment_id.

Reads each JSON file under reports/evaluations/<dataset>/, prefixes the
experiment_id with the dataset name, renames the file, and rewrites the JSON.

Usage:
    uv run python scripts/migrate_experiment_ids.py [--dry-run]
"""

import json
import os
import sys


def migrate(eval_root: str, dry_run: bool = False) -> None:
    if not os.path.isdir(eval_root):
        print(f"Directory not found: {eval_root}", file=sys.stderr)
        sys.exit(1)

    migrated = 0
    skipped = 0

    for dataset in sorted(os.listdir(eval_root)):
        dataset_dir = os.path.join(eval_root, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        for fname in sorted(os.listdir(dataset_dir)):
            if not fname.endswith(".json"):
                continue

            fpath = os.path.join(dataset_dir, fname)
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)

            old_id = data.get("experiment_id", "")

            if old_id.startswith(f"{dataset}_"):
                skipped += 1
                continue

            new_id = f"{dataset}_{old_id}"
            new_fname = f"{new_id}.json"
            new_fpath = os.path.join(dataset_dir, new_fname)

            if dry_run:
                print(f"[DRY RUN] {fname} -> {new_fname}")
            else:
                data["experiment_id"] = new_id
                with open(new_fpath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                if new_fpath != fpath:
                    os.remove(fpath)

                print(f"Migrated: {fname} -> {new_fname}")
            migrated += 1

    print(f"\nDone. Migrated: {migrated}, Skipped (already prefixed): {skipped}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    migrate("reports/evaluations", dry_run=dry_run)
