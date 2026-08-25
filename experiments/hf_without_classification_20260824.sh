#!/usr/bin/env bash
# hf_without_classification_20260824.sh: Orchestrates multiple evaluation runs across combinations of prompts, models, and sample sizes.
set -euo pipefail

DATASET=""
PROVIDER=""
TIMEOUT="300"

SKIP_CLASSIFICATION=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --provider) PROVIDER="$2"; shift ;;
        --timeout) TIMEOUT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$DATASET" ]; then
    echo "Usage: $0 --dataset <dataset_name> [--provider <provider>] [--timeout <timeout>]"
    exit 1
fi

if [ "$SKIP_CLASSIFICATION" -eq 1 ]; then
    CLASSIFY_PROMPTS=("")
else
    CLASSIFY_PROMPTS=(
      "prompts/classify_v01.md"
    )
fi

CLUSTER_PROMPTS=(
  "prompts/v01.01.md"
  "prompts/v05.md"
)

MODELS=(
  "aisingapore/Qwen-SEA-LION-v4.5-27B-IT"
  "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT"
  "gemma4-31b-it-bnb"
  "aisingapore/Gemma-SEA-LION-v4-27B-IT"
)

SAMPLE_SIZES=(
  16
)

SEED=42

total=$(( ${#CLASSIFY_PROMPTS[@]} * ${#CLUSTER_PROMPTS[@]} * ${#MODELS[@]} * ${#SAMPLE_SIZES[@]} ))
count=0

echo "Starting $total experiments on dataset: $DATASET..."

for model in "${MODELS[@]}"; do
  echo "==========================================================="
  echo ">>> Next model to test: $model"
  echo "==========================================================="
  read -p "Please turn ON the endpoint for '$model'. Press Enter when ready..."

  for classify_prompt in "${CLASSIFY_PROMPTS[@]}"; do
    for cluster_prompt in "${CLUSTER_PROMPTS[@]}"; do
      for sample_size in "${SAMPLE_SIZES[@]}"; do
        count=$(( count + 1 ))
        echo "=== Experiment [$count/$total] ==="
        # Prepare arguments for pipeline
        PIPELINE_ARGS=(
            "--dataset" "$DATASET"
            "--model" "$model"
            "--cluster-prompt" "$cluster_prompt"
            "--sample-size" "$sample_size"
            "--seed" "$SEED"
            "--timeout" "$TIMEOUT"
        )

        if [ "$SKIP_CLASSIFICATION" -eq 1 ]; then
            PIPELINE_ARGS+=("--skip-classification")
        else
            PIPELINE_ARGS+=("--classify-prompt" "$classify_prompt")
        fi

        if [ -n "$PROVIDER" ]; then
            PIPELINE_ARGS+=("--provider" "$PROVIDER")
        fi

        # Find the path to pipeline.sh relative to this script
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        ROOT_DIR="$(dirname "$SCRIPT_DIR")"

        # Execute pipeline in the background to parallelize
        (cd "$ROOT_DIR" && ./pipeline.sh "${PIPELINE_ARGS[@]}") &
      done
    done
  done

  echo ">>> Waiting for all parallel jobs for '$model' to finish..."
  wait

  echo "==========================================================="
  echo ">>> ✅ Finished all jobs for '$model'!"
  echo ">>> 🛑 REMINDER: Don't forget to turn OFF the endpoint for '$model'!"
  echo "==========================================================="
  echo ""
done

echo "All $count experiments completed successfully!"
