"""Export evaluation logs to nodes/edges CSV for the article-network-visualizer.

Usage:
    uv run python generate_network.py --eval-log reports/evaluations/<file>.json
    uv run python generate_network.py --eval-log <file>.json --output-dir reports/networks
    uv run python generate_network.py --eval-log <file>.json --image-base-url https://example.com/images

Options:
    --eval-log        Path to evaluation log JSON (required)
    --output-dir      Base output directory (default: reports/networks, env: OUTPUT_DIR)
    --image-base-url  Base URL for page scan images (default: https://jawi.sgp1.digitaloceanspaces.com/page_scans, env: IMAGE_BASE_URL)
    --interim-dir     Directory for cached fragments (default: data/1_interim)
    --eval-name       Override the evaluation subdirectory name
"""

import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_IMAGE_BASE_URL = "https://jawi.sgp1.digitaloceanspaces.com/page_scans"
DEFAULT_OUTPUT_DIR = "reports/networks"
DEFAULT_INTERIM_DIR = "data/1_interim"

NODE_BASE_COLUMNS = [
    "Image_URL",
    "Page_ID",
    "Region_ID",
    "Region_Text",
    "x1",
    "y1",
    "x2",
    "y2",
]

EDGE_METRIC_COLUMNS = [
    "clustering_f1",
    "bcubed_f1",
    "coverage",
    "class_accuracy",
    "tp",
    "fp",
    "fn",
]

EDGE_COLUMNS = [
    "Image_URL",
    "Page_ID",
    "Source_Region_ID",
    "Target_Region_ID",
    "Hop_Distance",
    "edge_weight",
    *EDGE_METRIC_COLUMNS,
]


def derive_eval_name(config: dict, override: str | None = None) -> str:
    if override:
        return override

    # Parse missing configuration fields from run_id (since they were decoupled)
    import re

    run_id = config.get("run_id", "")
    if "model" not in config and "create_" in run_id:
        m = re.search(r"create_(.+?)_c-", run_id)
        if m:
            config["model"] = m.group(1)
    if "prompt_name" not in config and "_r-" in run_id:
        m = re.search(r"_r-(.+?)(?:_|$)", run_id)
        if m:
            config["prompt_name"] = m.group(1)
    if "sample_size" not in config and "_sample" in run_id:
        m = re.search(r"_sample(\d+)", run_id)
        if m:
            config["sample_size"] = int(m.group(1))
    if "seed" not in config and "_seed" in run_id:
        m = re.search(r"_seed(\d+)", run_id)
        if m:
            config["seed"] = int(m.group(1))

    parts = [config.get("prompt_name", "unknown")]
    model = config.get("model", "unknown")
    parts.append(model)
    sample = config.get("sample_size")
    if sample is not None:
        parts.append(f"sample{sample}")
    seed = config.get("seed")
    if seed is not None:
        parts.append(f"seed{seed}")
    return "_".join(parts)


def build_segment_maps(
    predicted_items: list[dict] | None,
    ground_truth_items: list[dict] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    llm_map: dict[str, str] = {}
    if predicted_items:
        for i, item in enumerate(predicted_items):
            for fid in item.get("fragment_ids", []):
                llm_map[fid] = str(i)

    gt_map: dict[str, str] = {}
    if ground_truth_items:
        for item in ground_truth_items:
            uuid = item.get("uuid", "")
            for fid in item.get("fragment_ids", []):
                gt_map[fid] = uuid

    return llm_map, gt_map


def export_page(
    page: dict,
    fragments: list[dict],
    image_base_url: str,
    nodes_dir: Path,
    edges_dir: Path,
    model: str,
) -> None:
    page_id = page["page_id"]
    image_url = f"{image_base_url.rstrip('/')}/{page_id}.jpg"

    predicted_items = page.get("predicted_items")
    ground_truth_items = page.get("ground_truth_items")
    llm_map, gt_map = build_segment_maps(predicted_items, ground_truth_items)

    model_segment_col = f"{model}_segment"
    node_columns = [*NODE_BASE_COLUMNS, model_segment_col, "ground_truth_segment"]

    nodes_path = nodes_dir / f"{page_id}.csv"
    edges_path = edges_dir / f"{page_id}.csv"

    with open(nodes_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(node_columns)
        for frag in fragments:
            fid = frag["id"]
            hpos = frag.get("hpos", 0)
            vpos = frag.get("vpos", 0)
            width = frag.get("width", 0)
            height = frag.get("height", 0)
            writer.writerow(
                [
                    image_url,
                    page_id,
                    fid,
                    frag.get("text", ""),
                    hpos,
                    vpos,
                    hpos + width,
                    vpos + height,
                    llm_map.get(fid, ""),
                    gt_map.get(fid, ""),
                ]
            )

    with open(edges_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(EDGE_COLUMNS)
        metrics = page.get("metrics")
        metric_vals = (
            [metrics.get(c, "") for c in EDGE_METRIC_COLUMNS]
            if metrics
            else [""] * len(EDGE_METRIC_COLUMNS)
        )
        if predicted_items:
            for item in predicted_items:
                fids = item.get("fragment_ids", [])
                if len(fids) < 2:
                    continue
                for src, tgt in itertools.permutations(fids, 2):
                    src_gt = gt_map.get(src)
                    tgt_gt = gt_map.get(tgt)
                    weight = 1.0 if (src_gt and tgt_gt and src_gt == tgt_gt) else -1.0
                    writer.writerow(
                        [image_url, page_id, src, tgt, 1, weight, *metric_vals]
                    )


def export_eval_log(
    eval_log_path: str,
    output_dir: str,
    image_base_url: str,
    interim_dir: str,
    eval_name: str | None = None,
) -> str:
    with open(eval_log_path, encoding="utf-8") as f:
        log = json.load(f)

    config = log.get("config", {})
    name = derive_eval_name(config, eval_name)
    model = config.get("model", "unknown")
    eval_dir = Path(output_dir) / name
    nodes_dir = eval_dir / "nodes"
    edges_dir = eval_dir / "edges"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)

    # Infer fragments directory from input_folder if available
    input_folder = config.get("input_folder")
    if input_folder:
        # e.g., data/1_interim/ds-filteredUM1956alto/reconstructions/run_id
        # We want data/1_interim/ds-filteredUM1956alto/fragments
        fragments_dir = Path(input_folder).parent.parent / "fragments"
    else:
        fragments_dir = Path(interim_dir) / "fragments"

    pages = log.get("pages", [])
    exported = 0
    for page in pages:
        page_id = page["page_id"]

        path = fragments_dir / f"{page_id}.json"
        if not path.exists():
            fragments = None
        else:
            with open(path, encoding="utf-8") as f:
                fragments = json.load(f)

        if fragments is None:
            print(
                f"[{page_id}] WARN: fragment cache missing ({path}), skipping",
                file=sys.stderr,
            )
            continue
        export_page(page, fragments, image_base_url, nodes_dir, edges_dir, model)
        exported += 1

    print(f"Exported {exported} page(s) to {eval_dir}", file=sys.stderr)
    return str(eval_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate nodes/edges CSV from evaluation logs for the article-network-visualizer."
    )
    parser.add_argument("--eval-log", required=True, help="Path to evaluation log JSON")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--image-base-url",
        default=os.environ.get("IMAGE_BASE_URL", DEFAULT_IMAGE_BASE_URL),
        help="Base URL for page scan images (env: IMAGE_BASE_URL)",
    )
    parser.add_argument(
        "--interim-dir",
        default=DEFAULT_INTERIM_DIR,
        help=f"Directory for cached fragments (default: {DEFAULT_INTERIM_DIR})",
    )
    parser.add_argument(
        "--eval-name",
        default=None,
        help="Override the evaluation subdirectory name",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.eval_log):
        print(f"Error: eval log not found: {args.eval_log}", file=sys.stderr)
        return 1

    eval_dir = export_eval_log(
        args.eval_log,
        args.output_dir,
        args.image_base_url,
        args.interim_dir,
        args.eval_name,
    )
    print(eval_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
