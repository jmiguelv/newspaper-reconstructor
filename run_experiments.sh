#!/usr/bin/env bash
# run_experiments.sh: Orchestrates multiple evaluation runs across combinations of prompts, models, and sample sizes.
set -euo pipefail

DATASET=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$DATASET" ]; then
    echo "Usage: $0 --dataset <dataset_name>"
    exit 1
fi

CLASSIFY_PROMPTS=(
  "prompts/classify_v01.md"
)

CLUSTER_PROMPTS=(
  "prompts/v00.md"
  "prompts/v01.md"
  "prompts/v02.md"
  "prompts/v03.md"
  "prompts/v04.md"
)

MODELS=(
  "arc:lite"
  "arc:nexus"
)

SAMPLE_SIZES=(
  16
  32
)

SEED=42

total=$(( ${#CLASSIFY_PROMPTS[@]} * ${#CLUSTER_PROMPTS[@]} * ${#MODELS[@]} * ${#SAMPLE_SIZES[@]} ))
count=0

echo "Starting $total experiments on dataset: $DATASET..."

for classify_prompt in "${CLASSIFY_PROMPTS[@]}"; do
  for cluster_prompt in "${CLUSTER_PROMPTS[@]}"; do
    for model in "${MODELS[@]}"; do
      for sample_size in "${SAMPLE_SIZES[@]}"; do
        count=$(( count + 1 ))
        echo "=== Experiment [$count/$total] ==="
        ./pipeline.sh \
            --dataset "$DATASET" \
            --model "$model" \
            --classify-prompt "$classify_prompt" \
            --cluster-prompt "$cluster_prompt" \
            --sample-size "$sample_size" \
            --seed "$SEED"
      done
    done
  done
done

echo "All $count experiments completed successfully!"
