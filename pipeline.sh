#!/usr/bin/env bash
# pipeline.sh: Runs a single end-to-end evaluation pipeline
set -euo pipefail

MODEL=""
CLASSIFY_PROMPT_FILE=""
CLUSTER_PROMPT_FILE=""
SAMPLE_SIZE=""
SEED="42"
PAGE_ID=""
DATASET=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --classify-prompt) CLASSIFY_PROMPT_FILE="$2"; shift ;;
        --cluster-prompt) CLUSTER_PROMPT_FILE="$2"; shift ;;
        --sample-size) SAMPLE_SIZE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --page-id) PAGE_ID="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$MODEL" ] || [ -z "$CLASSIFY_PROMPT_FILE" ] || [ -z "$CLUSTER_PROMPT_FILE" ] || [ -z "$DATASET" ]; then
    echo "Usage: $0 --dataset <dataset_name> --model <model> --classify-prompt <file> --cluster-prompt <file> [--sample-size <N>] [--seed <S>] [--page-id <ID>]"
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

classify_prompt_name=$(basename "$CLASSIFY_PROMPT_FILE" .md)
cluster_prompt_name=$(basename "$CLUSTER_PROMPT_FILE" .md)

if [ -n "$PAGE_ID" ]; then
    classify_run_id="create_${MODEL}_c-${classify_prompt_name}_page_${PAGE_ID}"
    cluster_run_id="create_${MODEL}_c-${classify_prompt_name}_r-${cluster_prompt_name}_page_${PAGE_ID}"
else
    classify_run_id="create_${MODEL}_c-${classify_prompt_name}_sample${SAMPLE_SIZE}_seed${SEED}"
    cluster_run_id="create_${MODEL}_c-${classify_prompt_name}_r-${cluster_prompt_name}_sample${SAMPLE_SIZE}_seed${SEED}"
fi

safe_classify_run_id=$(echo "$classify_run_id" | tr ':' '_')
safe_cluster_run_id=$(echo "$cluster_run_id" | tr ':' '_')

classified_dir="data/1_interim/${DATASET}/classified/$safe_classify_run_id"
reconstructions_dir="data/1_interim/${DATASET}/reconstructions/$safe_cluster_run_id"

echo "=== Running Pipeline for: $cluster_run_id (Dataset: $DATASET) ==="

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
if [ ! -d "$classified_dir" ] || [ -z "$(ls -A "$classified_dir" 2>/dev/null)" ]; then
    echo "Classifying..."
    uv run python main.py classify \
        -i "$FRAGMENTS_DIR" \
        -p "$CLASSIFY_PROMPT_FILE" \
        -o "$classified_dir" \
        --model "$MODEL" \
        --seed "$SEED" \
        --timeout "$TIMEOUT" \
        ${SAMPLE_ARG:+$SAMPLE_ARG} \
        ${PAGE_ARG:+$PAGE_ARG}
else
    echo "Classification for $safe_classify_run_id already exists. Skipping classification."
fi

# Step 3: Cluster
echo "Clustering..."
uv run python main.py cluster \
    -i "$classified_dir" \
    -o "$reconstructions_dir" \
    -p "$CLUSTER_PROMPT_FILE" \
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
    --run-id "$cluster_run_id" \
    ${PAGE_ARG:+$PAGE_ARG}

echo "Pipeline complete!"
echo "----------------------------------------"
