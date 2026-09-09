#!/usr/bin/env bash
# qwen_sealion_1956_without_classification.sh: Runs the main.py pipeline
# (pipeline.sh: etl -> cluster -> evaluate) over module-format OCR inputs
# (per-page OcrOutput JSON: {page, regions}) for Qwen-SEA-LION models,
# without classification.
#
# Module-format inputs are auto-detected and converted to fragment lists by
# `main.py etl`; clustering runs through `main.py cluster --module-format`,
# which processes pages concurrently (as-completed, per-file output, automatic
# resume on re-run by skipping existing outputs) and writes each page as an
# ArticleReconstructionOutput JSON ({articles: ...}). Ground truth is
# auto-detected from regions/ (new format) or article_xml/ (legacy);
# evaluation is skipped with a warning if absent. Inputs are read from
# data/0_external/<dataset>/.
set -euo pipefail

DATASET=""
PROVIDER="huggingface"
TIMEOUT="600"
FREQUENCY_PENALTY=""
MAX_WORKERS="64"
MAX_TOKENS="16384"
SAVE_RAW=0
SLIM=0

CLUSTER_PROMPTS=(
  "prompts/v01.01.02.md"
)

MODELS=(
  "aisingapore/Qwen-SEA-LION-v4.5-27B-IT"
)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --provider) PROVIDER="$2"; shift ;;
        --timeout) TIMEOUT="$2"; shift ;;
        --frequency-penalty) FREQUENCY_PENALTY="$2"; shift ;;
        --max-workers) MAX_WORKERS="$2"; shift ;;
        --max-tokens) MAX_TOKENS="$2"; shift ;;
        --save-raw) SAVE_RAW=1 ;;
        --slim) SLIM=1;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$DATASET" ]; then
    echo "Usage: $0 --dataset <dataset_name> [--provider <provider>] [--timeout <T>] [--max-workers <N>] [--max-tokens <N>] [--slim] [--save-raw]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

total=$(( ${#CLUSTER_PROMPTS[@]} * ${#MODELS[@]} ))
count=0

echo "Starting $total experiments on dataset: $DATASET..."

for model in "${MODELS[@]}"; do
  echo "==========================================================="
  echo ">>> Next model to test: $model"
  echo "==========================================================="
  read -r -p "Please turn ON the endpoint for '$model'. Press Enter when ready..."

  for cluster_prompt in "${CLUSTER_PROMPTS[@]}"; do
    count=$(( count + 1 ))
    echo "=== Experiment [$count/$total] ==="

    PIPELINE_ARGS=(
        --dataset "$DATASET"
        --model "$model"
        --cluster-prompt "$cluster_prompt"
        --skip-classification
        --provider "$PROVIDER"
        --timeout "$TIMEOUT"
        --max-workers "$MAX_WORKERS"
        --max-tokens "$MAX_TOKENS"
        --module-format
    )
    if [ -n "$FREQUENCY_PENALTY" ]; then
        PIPELINE_ARGS+=(--frequency-penalty "$FREQUENCY_PENALTY")
    fi
    if [ "$SLIM" -eq 1 ]; then
        PIPELINE_ARGS+=(--slim)
    fi
    if [ "$SAVE_RAW" -eq 1 ]; then
        PIPELINE_ARGS+=(--save-raw)
    fi

    (cd "$ROOT_DIR" && ./scripts/pipeline.sh "${PIPELINE_ARGS[@]}")
  done

  echo "==========================================================="
  echo ">>> ✅ Finished all jobs for '$model'!"
  echo ">>> 🛑 REMINDER: Don't forget to turn OFF the endpoint for '$model'!"
  echo "==========================================================="
  echo ""
done

echo "All $count experiments completed successfully!"
