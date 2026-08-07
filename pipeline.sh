#!/usr/bin/env bash
# pipeline.sh: Runs a single end-to-end evaluation pipeline
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 <model> <prompt_file> <sample_size> <seed> [page_id]"
    exit 1
fi

MODEL=$1
PROMPT_FILE=$2
SAMPLE_SIZE=$3
SEED=$4
PAGE_ID=${5:-""}

ALTO_DIR="data/0_external/alto"
FRAGMENTS_DIR="data/1_interim/fragments"
GROUND_TRUTH_DIR="data/0_external/article_xml"
EVAL_DIR="reports/evaluations"
TIMEOUT=60

prompt_name=$(basename "$PROMPT_FILE" .md)
run_id="openai_${MODEL}_${prompt_name}_sample${SAMPLE_SIZE}_seed${SEED}"
safe_run_id=$(echo "$run_id" | tr ':' '_')
reconstructions_dir="data/1_interim/reconstructions/$safe_run_id"

echo "=== Running Pipeline for: $run_id ==="

# Setup optional page_id argument
PAGE_ARG=""
if [ -n "$PAGE_ID" ]; then
    PAGE_ARG="--page-id $PAGE_ID"
fi

# Step 1: Parse ALTO to JSON (if not already done)
if [ ! -d "$FRAGMENTS_DIR" ]; then
    echo "Parsing ALTO..."
    uv run python main.py parse -i "$ALTO_DIR" -o "$FRAGMENTS_DIR" $PAGE_ARG
fi

# Step 2: Classify (placeholder - skipping for now since no prompt)
# uv run python main.py classify -i "$FRAGMENTS_DIR" -p "prompts/classify.md" -o "data/1_interim/classified/$safe_run_id" $PAGE_ARG

# Step 3: Cluster
echo "Clustering..."
uv run python main.py cluster \
    -i "$FRAGMENTS_DIR" \
    -o "$reconstructions_dir" \
    -p "$PROMPT_FILE" \
    --model "$MODEL" \
    --sample-size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    --timeout "$TIMEOUT" \
    $PAGE_ARG

# Step 4: Evaluate
echo "Evaluating..."
uv run python main.py evaluate \
    -i "$reconstructions_dir" \
    -g "$GROUND_TRUTH_DIR" \
    --eval-dir "$EVAL_DIR" \
    --run-id "$run_id" \
    $PAGE_ARG

echo "Pipeline complete!"
echo "----------------------------------------"
