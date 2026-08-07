#!/usr/bin/env bash
# pipeline.sh: Runs a single end-to-end evaluation pipeline
set -euo pipefail

MODEL=""
PROMPT_FILE=""
SAMPLE_SIZE=""
SEED="42"
PAGE_ID=""
DATASET=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --prompt) PROMPT_FILE="$2"; shift ;;
        --sample-size) SAMPLE_SIZE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --page-id) PAGE_ID="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$MODEL" ] || [ -z "$PROMPT_FILE" ] || [ -z "$DATASET" ]; then
    echo "Usage: $0 --dataset <dataset_name> --model <model> --prompt <prompt_file> [--sample-size <N>] [--seed <S>] [--page-id <ID>]"
    exit 1
fi

if [ -n "$PAGE_ID" ] && [ -n "$SAMPLE_SIZE" ]; then
    echo "Warning: Both --page-id and --sample-size provided. Ignoring --sample-size."
    SAMPLE_SIZE=""
fi

ALTO_DIR="data/0_external/${DATASET}/alto"
FRAGMENTS_DIR="data/1_interim/${DATASET}/fragments"
GROUND_TRUTH_DIR="data/0_external/${DATASET}/article_xml"
EVAL_DIR="reports/evaluations/${DATASET}"
TIMEOUT=60

prompt_name=$(basename "$PROMPT_FILE" .md)

if [ -n "$PAGE_ID" ]; then
    run_id="create_${MODEL}_${prompt_name}_page_${PAGE_ID}"
else
    run_id="create_${MODEL}_${prompt_name}_sample${SAMPLE_SIZE}_seed${SEED}"
fi
safe_run_id=$(echo "$run_id" | tr ':' '_')
reconstructions_dir="data/1_interim/${DATASET}/reconstructions/$safe_run_id"

echo "=== Running Pipeline for: $run_id (Dataset: $DATASET) ==="

PAGE_ARG=""
if [ -n "$PAGE_ID" ]; then
    PAGE_ARG="--page-id $PAGE_ID"
fi
SAMPLE_ARG=""
if [ -n "$SAMPLE_SIZE" ]; then
    SAMPLE_ARG="--sample-size $SAMPLE_SIZE"
fi

# Step 1: Parse ALTO to JSON (if not already done)
if [ ! -d "$FRAGMENTS_DIR" ]; then
    echo "Parsing ALTO..."
    # Intentionally don't pass PAGE_ARG here so we parse everything once
    uv run python main.py parse -i "$ALTO_DIR" -o "$FRAGMENTS_DIR"
fi

# Step 2: Classify
echo "Classifying..."
uv run python main.py classify \
    -i "$FRAGMENTS_DIR" \
    -p "prompts/classify_v00.md" \
    -o "data/1_interim/${DATASET}/classified/$safe_run_id" \
    --model "$MODEL" \
    --seed "$SEED" \
    --timeout "$TIMEOUT" \
    ${SAMPLE_ARG:+$SAMPLE_ARG} \
    ${PAGE_ARG:+$PAGE_ARG}

# Step 3: Cluster
echo "Clustering..."
uv run python main.py cluster \
    -i "data/1_interim/${DATASET}/classified/$safe_run_id" \
    -o "$reconstructions_dir" \
    -p "$PROMPT_FILE" \
    --model "$MODEL" \
    --seed "$SEED" \
    --timeout "$TIMEOUT" \
    ${SAMPLE_ARG:+$SAMPLE_ARG} \
    ${PAGE_ARG:+$PAGE_ARG}

# Step 4: Evaluate
echo "Evaluating..."
uv run python main.py evaluate \
    -i "$reconstructions_dir" \
    -g "$GROUND_TRUTH_DIR" \
    --eval-dir "$EVAL_DIR" \
    --run-id "$run_id" \
    ${PAGE_ARG:+$PAGE_ARG}

echo "Pipeline complete!"
echo "----------------------------------------"
