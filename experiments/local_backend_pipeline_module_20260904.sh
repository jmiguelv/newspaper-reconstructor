#!/usr/bin/env bash
# local_backend_pipeline_module_20260904.sh: Runs the jawi-pipeline module
# (pipeline_main.py bulk-process) over sampled pages from a legacy ALTO dataset
# using the LOCAL transformers backend, then evaluates against ground truth.
#
# Requires the opt-in dependency group: uv sync --group local
# Models are HF hub ids or local paths (MODELS array below).
# Sampled pages match main.py's sampling (same seed => same pages as the
# sample16 experiments driven by pipeline.sh).
set -euo pipefail

DATASET=""
TAG=""
SAMPLE_SIZE="16"
SEED="42"
FORCE=0
IMAGE_BASE_URL="${IMAGE_BASE_URL:-https://jawi.sgp1.digitaloceanspaces.com/page_scans}"

CLUSTER_PROMPTS=(
  "prompts/v01.01.02.md"
)

MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --tag) TAG="$2"; shift ;;
        --sample-size) SAMPLE_SIZE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --force) FORCE=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$DATASET" ]; then
    echo "Usage: $0 --dataset <dataset_name> [--tag <tag>] [--sample-size <N>] [--seed <S>] [--force]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

ALTO_DIR="data/0_external/${DATASET}/alto"
GROUND_TRUTH_DIR="data/0_external/${DATASET}/article_xml"
EVAL_DIR="reports/evaluations/${DATASET}"

if [ ! -d "$ALTO_DIR" ]; then
    echo "Error: ${ALTO_DIR} not found (legacy ALTO dataset required)."
    exit 1
fi

total=$(( ${#CLUSTER_PROMPTS[@]} * ${#MODELS[@]} ))
count=0

echo "Starting $total pipeline-module experiments (local backend) on dataset: $DATASET..."

for model in "${MODELS[@]}"; do
  for cluster_prompt in "${CLUSTER_PROMPTS[@]}"; do
    count=$(( count + 1 ))
    echo "=== Experiment [$count/$total] ==="

    cluster_prompt_name=$(basename "$cluster_prompt" .md)
    MODEL_PREFIX="${model}"
    if [ -n "$TAG" ]; then
        MODEL_PREFIX="${MODEL_PREFIX}_${TAG}"
    fi
    experiment_id="${DATASET}_${MODEL_PREFIX}_module_r-${cluster_prompt_name}_sample${SAMPLE_SIZE}_seed${SEED}"
    safe_experiment_id=$(echo "$experiment_id" | tr ':|/' '_')

    inputs_dir="data/1_interim/${DATASET}/pipeline_module/${safe_experiment_id}/inputs"
    outputs_dir="data/1_interim/${DATASET}/pipeline_module/${safe_experiment_id}/outputs"
    reconstructions_dir="data/1_interim/${DATASET}/reconstructions/${safe_experiment_id}"

    if [ -n "$(find "$inputs_dir" -maxdepth 1 -name '*.json' 2>/dev/null)" ]; then
        echo "Module inputs for ${safe_experiment_id} already exist. Skipping conversion."
    else
        echo "Building module inputs (${SAMPLE_SIZE} pages, seed ${SEED})..."
        ALTO_DIR="$ALTO_DIR" INPUTS_DIR="$inputs_dir" SAMPLE_SIZE="$SAMPLE_SIZE" \
        SEED="$SEED" IMAGE_BASE_URL="$IMAGE_BASE_URL" \
        uv run python - <<'PY'
import json
import os
import random
import re
import xml.etree.ElementTree as ET

ALTO_NS = {"a": "http://www.loc.gov/standards/alto/ns-v4#"}
alto_dir = os.environ["ALTO_DIR"]
inputs_dir = os.environ["INPUTS_DIR"]
sample_size = int(os.environ["SAMPLE_SIZE"])
seed = int(os.environ["SEED"])
image_base_url = os.environ["IMAGE_BASE_URL"]

os.makedirs(inputs_dir, exist_ok=True)
files = sorted(
    f for f in os.listdir(alto_dir)
    if f.endswith(".xml") and not f.startswith("_")
)
if sample_size < len(files):
    random.seed(seed)
    files = random.sample(files, sample_size)


def quad(hpos, vpos, width, height):
    return {
        "x1": hpos, "y1": vpos,
        "x2": hpos + width, "y2": vpos,
        "x3": hpos + width, "y3": vpos + height,
        "x4": hpos, "y4": vpos + height,
    }


for fname in files:
    page_id = os.path.splitext(fname)[0]
    root = ET.parse(os.path.join(alto_dir, fname)).getroot()
    page_el = root.find(".//a:Page", ALTO_NS)

    regions = []
    for tb in root.findall(".//a:TextBlock", ALTO_NS):
        lines = []
        for i, tl in enumerate(tb.findall(".//a:TextLine", ALTO_NS)):
            text = " ".join(
                s.get("CONTENT", "") for s in tl.findall(".//a:String", ALTO_NS)
            )
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            hpos = float(tl.get("HPOS", 0))
            vpos = float(tl.get("VPOS", 0))
            width = float(tl.get("WIDTH", 0))
            height = float(tl.get("HEIGHT", 0))
            lines.append(
                {
                    "line": {
                        "id": tl.get("ID") or f"{tb.get('ID')}_l{i}",
                        "baseline": {
                            "x1": hpos, "y1": vpos,
                            "x2": hpos + width, "y2": vpos,
                        },
                        "boundaries": [
                            {"x": hpos, "y": vpos},
                            {"x": hpos + width, "y": vpos},
                            {"x": hpos + width, "y": vpos + height},
                            {"x": hpos, "y": vpos + height},
                        ],
                    },
                    "script": "jawi",
                    "text": text,
                    "conf": 1.0,
                    "glyph_loc": [],
                }
            )
        if not lines:
            continue
        regions.append(
            {
                "t": tb.get("TYPE") or "text",
                "id": tb.get("ID"),
                "bbox": quad(
                    float(tb.get("HPOS", 0)),
                    float(tb.get("VPOS", 0)),
                    float(tb.get("WIDTH", 0)),
                    float(tb.get("HEIGHT", 0)),
                ),
                "line_ocr": lines,
            }
        )
    for ill in root.findall(".//a:Illustration", ALTO_NS):
        regions.append(
            {
                "t": "image",
                "id": ill.get("ID"),
                "bbox": quad(
                    float(ill.get("HPOS", 0)),
                    float(ill.get("VPOS", 0)),
                    float(ill.get("WIDTH", 0)),
                    float(ill.get("HEIGHT", 0)),
                ),
            }
        )

    page = {
        "page": {
            "id": page_id,
            "metadata": {
                "height": float(page_el.get("HEIGHT")),
                "width": float(page_el.get("WIDTH")),
            },
            "url": f"{image_base_url}/{page_id}.png",
        },
        "regions": regions,
    }
    with open(os.path.join(inputs_dir, f"{page_id}.json"), "w", encoding="utf-8") as f:
        json.dump(page, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(files)} module input files to {inputs_dir}")
PY
    fi

    echo "Running pipeline module (bulk-process, local backend, ${model})..."
    if [ "$FORCE" -eq 1 ]; then
        uv run --group local python pipeline_main.py bulk-process \
            --input "$inputs_dir" \
            --output "$outputs_dir" \
            --config "{\"model\": \"${model}\", \"backend\": \"local\", \"prompt_file\": \"${cluster_prompt}\", \"max_workers\": 1}" \
            --force
    else
        uv run --group local python pipeline_main.py bulk-process \
            --input "$inputs_dir" \
            --output "$outputs_dir" \
            --config "{\"model\": \"${model}\", \"backend\": \"local\", \"prompt_file\": \"${cluster_prompt}\", \"max_workers\": 1}"
    fi

    echo "Converting module outputs to cluster format for evaluation..."
    OUTPUTS_DIR="$outputs_dir" RECONSTRUCTIONS_DIR="$reconstructions_dir" \
    uv run python - <<'PY'
import json
import os

outputs_dir = os.environ["OUTPUTS_DIR"]
reconstructions_dir = os.environ["RECONSTRUCTIONS_DIR"]
os.makedirs(reconstructions_dir, exist_ok=True)

count = 0
for fname in sorted(os.listdir(outputs_dir)):
    if not fname.endswith(".json") or fname.startswith("."):
        continue
    with open(os.path.join(outputs_dir, fname), encoding="utf-8") as f:
        out = json.load(f)
    items = []
    for article in out.get("articles", {}).values():
        item = {
            "fragment_ids": article.get("region_ids", []),
            "title": article.get("title"),
            "class": article.get("item_class"),
        }
        if article.get("title_en") is not None:
            item["title_en"] = article["title_en"]
        items.append(item)
    with open(os.path.join(reconstructions_dir, fname), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    count += 1

print(f"Converted {count} module outputs to {reconstructions_dir}")
PY

    uv run python main.py evaluate \
        -i "$reconstructions_dir" \
        -g "$GROUND_TRUTH_DIR" \
        --eval-dir "$EVAL_DIR" \
        --experiment-id "$safe_experiment_id" \
        --task reconstruction
  done
done

echo "All $count experiments completed successfully!"
