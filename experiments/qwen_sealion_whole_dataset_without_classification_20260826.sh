#!/usr/bin/env bash
# qwen_sealion_whole_dataset_20260826.sh: Orchestrates multiple evaluation runs for Qwen-SEA-LION models on the entire dataset without sampling.
set -euo pipefail

DATASET=""
PROVIDER=""
TIMEOUT="300"

SKIP_CLASSIFICATION=1

PARALLEL=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --provider) PROVIDER="$2"; shift ;;
        --timeout) TIMEOUT="$2"; shift ;;
        --parallel) PARALLEL=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$DATASET" ]; then
    echo "Usage: $0 --dataset <dataset_name> [--provider <provider>] [--timeout <timeout>] [--parallel]"
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
  "prompts/v01.01.01.md"
  # "prompts/v05.md"
)

MODELS=(
  "aisingapore/Qwen-SEA-LION-v4.5-27B-IT"
)

# No sampling sizes, process the whole dataset

SEED=42

total=$(( ${#CLASSIFY_PROMPTS[@]} * ${#CLUSTER_PROMPTS[@]} * ${#MODELS[@]} ))
count=0

echo "Starting $total experiments on dataset: $DATASET..."

for model in "${MODELS[@]}"; do
  echo "==========================================================="
  echo ">>> Next model to test: $model"
  echo "==========================================================="
  read -p "Please turn ON the endpoint for '$model'. Press Enter when ready..."

  for classify_prompt in "${CLASSIFY_PROMPTS[@]}"; do
    for cluster_prompt in "${CLUSTER_PROMPTS[@]}"; do
      count=$(( count + 1 ))
      echo "=== Experiment [$count/$total] ==="
      # Prepare arguments for pipeline
      PIPELINE_ARGS=(
          "--dataset" "$DATASET"
          "--model" "$model"
          "--cluster-prompt" "$cluster_prompt"
          "--seed" "$SEED"
          "--timeout" "$TIMEOUT"
          "--max-workers" "32"
          "--max-tokens" "16384"
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

      if [ "$PARALLEL" -eq 1 ]; then
          # Execute pipeline in the background to parallelize
          (cd "$ROOT_DIR" && ./pipeline.sh "${PIPELINE_ARGS[@]}") &
      else
          # Execute pipeline synchronously
          (cd "$ROOT_DIR" && ./pipeline.sh "${PIPELINE_ARGS[@]}")
      fi
    done
  done

  if [ "$PARALLEL" -eq 1 ]; then
      echo ">>> Waiting for all parallel jobs for '$model' to finish..."
      wait
  fi

  echo "==========================================================="
  echo ">>> ✅ Finished all jobs for '$model'!"
  echo ">>> 🛑 REMINDER: Don't forget to turn OFF the endpoint for '$model'!"
  echo "==========================================================="
  echo ""
done

echo "All $count experiments completed successfully!"
