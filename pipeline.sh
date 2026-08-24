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
PROVIDER=""
SAVE_PROMPTS=""
SKIP_CLASSIFICATION=0

TIMEOUT="300"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --classify-prompt) CLASSIFY_PROMPT_FILE="$2"; shift ;;
        --cluster-prompt) CLUSTER_PROMPT_FILE="$2"; shift ;;
        --sample-size) SAMPLE_SIZE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --page-id) PAGE_ID="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --provider) PROVIDER="$2"; shift ;;
        --timeout) TIMEOUT="$2"; shift ;;
        --save-prompts) SAVE_PROMPTS="--save-prompts" ;;
        --skip-classification) SKIP_CLASSIFICATION=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$MODEL" ] || [ -z "$CLUSTER_PROMPT_FILE" ] || [ -z "$DATASET" ]; then
    echo "Usage: $0 --dataset <dataset_name> --model <model> --cluster-prompt <file> [--classify-prompt <file>] [--skip-classification] [--sample-size <N>] [--seed <S>] [--page-id <ID>] [--timeout <T>]"
    exit 1
fi

if [ "$SKIP_CLASSIFICATION" -eq 0 ] && [ -z "$CLASSIFY_PROMPT_FILE" ]; then
    echo "Error: --classify-prompt is required unless --skip-classification is used."
    exit 1
fi

if [ -n "$PAGE_ID" ] && [ -n "$SAMPLE_SIZE" ]; then
    echo "Warning: Both --page-id and --sample-size provided. Ignoring --sample-size."
    SAMPLE_SIZE=""
fi

FRAGMENTS_DIR="data/1_interim/${DATASET}/fragments"
EVAL_DIR="reports/evaluations/${DATASET}"

# Auto-detect dataset format
if [ -d "data/0_external/${DATASET}/articles" ]; then
    INPUT_DIR="data/0_external/${DATASET}/articles"
    GROUND_TRUTH_DIR="data/0_external/${DATASET}/regions"
    INPUT_FORMAT="json"
elif [ -d "data/0_external/${DATASET}/alto" ]; then
    INPUT_DIR="data/0_external/${DATASET}/alto"
    GROUND_TRUTH_DIR="data/0_external/${DATASET}/article_xml"
    INPUT_FORMAT="alto"
else
    echo "Error: Could not detect dataset format for ${DATASET}. Expected 'articles/' or 'alto/' directory."
    exit 1
fi

classify_prompt_name=""
if [ -n "$CLASSIFY_PROMPT_FILE" ]; then
    classify_prompt_name=$(basename "$CLASSIFY_PROMPT_FILE" .md)
fi
cluster_prompt_name=$(basename "$CLUSTER_PROMPT_FILE" .md)

MODEL_PREFIX="${MODEL}"
if [ -n "$PROVIDER" ]; then
    MODEL_PREFIX="${PROVIDER}_${MODEL}"
fi

if [ -n "$PAGE_ID" ]; then
    if [ "$SKIP_CLASSIFICATION" -eq 1 ]; then
        cluster_experiment_id="${DATASET}_${MODEL_PREFIX}_r-${cluster_prompt_name}_page_${PAGE_ID}"
    else
        classify_experiment_id="${DATASET}_${MODEL_PREFIX}_c-${classify_prompt_name}_page_${PAGE_ID}"
        cluster_experiment_id="${DATASET}_${MODEL_PREFIX}_c-${classify_prompt_name}_r-${cluster_prompt_name}_page_${PAGE_ID}"
    fi
else
    if [ "$SKIP_CLASSIFICATION" -eq 1 ]; then
        cluster_experiment_id="${DATASET}_${MODEL_PREFIX}_r-${cluster_prompt_name}_sample${SAMPLE_SIZE}_seed${SEED}"
    else
        classify_experiment_id="${DATASET}_${MODEL_PREFIX}_c-${classify_prompt_name}_sample${SAMPLE_SIZE}_seed${SEED}"
        cluster_experiment_id="${DATASET}_${MODEL_PREFIX}_c-${classify_prompt_name}_r-${cluster_prompt_name}_sample${SAMPLE_SIZE}_seed${SEED}"
    fi
fi

safe_cluster_experiment_id=$(echo "$cluster_experiment_id" | tr ':|/' '_')
reconstructions_dir="data/1_interim/${DATASET}/reconstructions/$safe_cluster_experiment_id"

if [ "$SKIP_CLASSIFICATION" -eq 0 ]; then
    safe_classify_experiment_id=$(echo "$classify_experiment_id" | tr ':|/' '_')
    classified_dir="data/1_interim/${DATASET}/classified/$safe_classify_experiment_id"
    cluster_input_dir="$classified_dir"
else
    cluster_input_dir="$FRAGMENTS_DIR"
    safe_classify_experiment_id="skipped"
    classified_dir=""
fi

echo "=== Running Pipeline for: $cluster_experiment_id (Dataset: $DATASET, Format: $INPUT_FORMAT) ==="

PAGE_ARG=""
if [ -n "$PAGE_ID" ]; then
    PAGE_ARG="--page-id $PAGE_ID"
fi
SAMPLE_ARG=""
if [ -n "$SAMPLE_SIZE" ]; then
    SAMPLE_ARG="--sample-size $SAMPLE_SIZE"
fi
PROVIDER_ARG=""
if [ -n "$PROVIDER" ]; then
    PROVIDER_ARG="--provider $PROVIDER"
fi

# Step 1: Extract fragments (if not already done)
if [ ! -d "$FRAGMENTS_DIR" ]; then
    if [ "$INPUT_FORMAT" = "json" ]; then
        echo "Converting article JSON to fragments..."
        uv run python main.py etl -i "$INPUT_DIR" -o "$FRAGMENTS_DIR"
    else
        echo "Parsing ALTO XML..."
        uv run python main.py parse -i "$INPUT_DIR" -o "$FRAGMENTS_DIR"
    fi
fi

# Step 2: Classify
if [ "$SKIP_CLASSIFICATION" -eq 0 ]; then
    if [ ! -d "$classified_dir" ] || [ "$(find "$classified_dir" -maxdepth 1 -name "*.json" ! -name "_*.json" 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]; then
        echo "Classifying..."
        uv run python main.py classify \
            -i "$FRAGMENTS_DIR" \
            -p "$CLASSIFY_PROMPT_FILE" \
            -o "$classified_dir" \
            --model "$MODEL" \
            --seed "$SEED" \
            --timeout "$TIMEOUT" \
            ${SAMPLE_ARG:+$SAMPLE_ARG} \
            ${PAGE_ARG:+$PAGE_ARG} \
            ${PROVIDER_ARG:+$PROVIDER_ARG} \
            ${SAVE_PROMPTS:+$SAVE_PROMPTS}
    else
        echo "Classification for $safe_classify_experiment_id already exists. Skipping classification."
    fi

    # Step 2.5: Evaluate Classification
    echo "Evaluating Classification..."
    uv run python main.py evaluate \
        -i "$classified_dir" \
        -g "$GROUND_TRUTH_DIR" \
        --eval-dir "$EVAL_DIR" \
        --experiment-id "$safe_classify_experiment_id" \
        --task classification \
        ${PAGE_ARG:+$PAGE_ARG}
fi


# Step 3: Cluster
echo "Clustering..."
uv run python main.py cluster \
    -i "$cluster_input_dir" \
    -o "$reconstructions_dir" \
    -p "$CLUSTER_PROMPT_FILE" \
    --model "$MODEL" \
    --seed "$SEED" \
    --timeout "$TIMEOUT" \
    ${SAMPLE_ARG:+$SAMPLE_ARG} \
    ${PAGE_ARG:+$PAGE_ARG} \
    ${PROVIDER_ARG:+$PROVIDER_ARG} \
    ${SAVE_PROMPTS:+$SAVE_PROMPTS}

# Step 4: Evaluate
echo "Evaluating..."
uv run python main.py evaluate \
    -i "$reconstructions_dir" \
    -g "$GROUND_TRUTH_DIR" \
    --eval-dir "$EVAL_DIR" \
    --experiment-id "$safe_cluster_experiment_id" \
    --task reconstruction \
    ${PAGE_ARG:+$PAGE_ARG}

echo "Pipeline complete!"
echo "----------------------------------------"
